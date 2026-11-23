"""Use LightRAG's document index and its structured retrieval response."""

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from .index_store import index_key, working_index

EXAMPLE_DIR = Path(__file__).resolve().parent
DEEPSEEK_URL = "https://api.deepseek.com"
QUERY_OPTIONS = {"mode": "naive", "top_k": 5, "chunk_top_k": 5, "enable_rerank": False}


@dataclass(frozen=True)
class Passage:
    source: str
    text: str
    reference_id: str
    chunk_id: str


class QueryBackend(Protocol):
    async def aquery_data(self, query: str, *, param: Any) -> dict[str, Any]: ...


@dataclass
class Retriever:
    backend: QueryBackend
    param: Any
    index_reused: bool | None = None

    async def retrieve(self, query: str) -> list[Passage]:
        response = await self.backend.aquery_data(query, param=self.param)
        if response["status"] != "success":
            if response.get("metadata", {}).get("failure_reason") == "no_results":
                return []
            raise RuntimeError(
                f"LightRAG retrieval failed: {response.get('message', 'unknown error')}"
            )

        data = response["data"]
        sources = {ref["reference_id"]: ref["file_path"] for ref in data["references"]}
        passages = []
        for chunk in data["chunks"]:
            reference_id = chunk["reference_id"]
            source = sources.get(reference_id)
            if not source or source != chunk["file_path"]:
                raise RuntimeError(
                    "LightRAG returned a chunk without matching source metadata"
                )
            passages.append(
                Passage(source, chunk["content"], reference_id, chunk["chunk_id"])
            )
        return passages


def read_manuals(directory: Path = EXAMPLE_DIR / "manuals") -> dict[str, str]:
    manuals = {}
    for path in sorted(directory.glob("*.md")):
        if path.is_file():
            content = path.read_text(encoding="utf-8-sig")
            if content.strip():
                manuals[path.name] = content
    if not manuals:
        raise ValueError(f"{directory} 中没有可索引的 Markdown 手册")
    return manuals


class FixtureBackend:
    """Return known candidates for an offline check of evidence handling."""

    def __init__(self, documents: Mapping[str, str]) -> None:
        self.documents = documents

    async def aquery_data(self, query: str, *, param: Any) -> dict[str, Any]:
        candidates = list(self.documents.items())[: param.chunk_top_k]
        references = []
        chunks = []
        for index, (source, text) in enumerate(candidates, start=1):
            reference = {"reference_id": str(index), "file_path": source}
            references.append(reference)
            chunks.append(
                {**reference, "chunk_id": f"fixture-{index}", "content": text.strip()}
            )
        return {
            "status": "success",
            "data": {"references": references, "chunks": chunks},
        }


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str
    base_url: str
    embedding_model: str
    api_key: str | None = field(repr=False)

    def require_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "首次建立手册索引需要 SENSOR_LLM_API_KEY，默认 DeepSeek 端点也支持 "
                "DEEPSEEK_API_KEY。可用 --logs-only 继续复核现场数据。"
            )

    @classmethod
    def from_env(cls) -> "ModelConfig":
        from dotenv import load_dotenv

        load_dotenv(EXAMPLE_DIR / ".env", override=False)
        base_url = (os.getenv("SENSOR_LLM_BASE_URL") or DEEPSEEK_URL).rstrip("/")
        key = os.getenv("SENSOR_LLM_API_KEY")
        if not key and base_url == DEEPSEEK_URL:
            key = os.getenv("DEEPSEEK_API_KEY")
        return cls(
            chat_model=os.getenv("SENSOR_LLM_MODEL") or "deepseek-flash",
            base_url=base_url,
            embedding_model=os.getenv("SENSOR_EMBEDDING_MODEL")
            or "BAAI/bge-small-zh-v1.5",
            api_key=key,
        )


@asynccontextmanager
async def open_retriever(
    documents: Mapping[str, str],
    config: ModelConfig | None,
    *,
    index_dir: Path | None = None,
) -> AsyncIterator[Retriever]:
    if config is None:
        yield Retriever(FixtureBackend(documents), SimpleNamespace(**QUERY_OPTIONS))
        return

    key = index_key(
        documents, config.chat_model, config.base_url, config.embedding_model
    )
    with working_index(index_dir, key) as (working_dir, reused):
        if not reused:
            config.require_key()
        async with open_lightrag(working_dir, documents, config, reused) as retriever:
            yield retriever


@asynccontextmanager
async def open_lightrag(
    working_dir: Path, documents: Mapping[str, str], config: ModelConfig, reused: bool
) -> AsyncIterator[Retriever]:
    import numpy as np
    from fastembed import TextEmbedding
    from lightrag import LightRAG, QueryParam
    from lightrag.base import DocStatus
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import wrap_embedding_func_with_attrs

    dimension = TextEmbedding.get_embedding_size(config.embedding_model)
    embedding_model = await asyncio.to_thread(
        TextEmbedding, config.embedding_model, threads=2
    )

    @wrap_embedding_func_with_attrs(
        embedding_dim=dimension, max_token_size=512, model_name=config.embedding_model
    )
    async def embed(texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(
            lambda: np.asarray(list(embedding_model.embed(texts)))
        )

    chat_options = {"temperature": 0}
    if config.base_url == DEEPSEEK_URL:
        chat_options["extra_body"] = {"thinking": {"type": "disabled"}}

    rag = LightRAG(
        working_dir=str(working_dir),
        embedding_func=embed,
        llm_model_func=partial(openai_complete_if_cache, config.chat_model),
        llm_model_name=config.chat_model,
        llm_model_kwargs={
            "base_url": config.base_url,
            "api_key": config.api_key,
            **chat_options,
        },
        addon_params={"language": "Chinese"},
        chunk_token_size=350,
        entity_extract_max_gleaning=0,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.20},
    )
    await rag.initialize_storages()
    try:
        if not reused:
            sources = list(documents)
            ids = [f"sensor-doc-{index}" for index in range(len(sources))]
            track_id = await rag.ainsert(
                list(documents.values()), ids=ids, file_paths=sources
            )
            statuses = await rag.aget_docs_by_track_id(track_id)
            failed = [
                source
                for doc_id, source in zip(ids, sources, strict=True)
                if doc_id not in statuses
                or statuses[doc_id].status != DocStatus.PROCESSED
            ]
            if failed:
                raise RuntimeError(f"LightRAG indexing failed for: {', '.join(failed)}")
        yield Retriever(rag, QueryParam(**QUERY_OPTIONS), index_reused=reused)
    finally:
        await rag.finalize_storages()
