"""Publish successful LightRAG indexes and serialize access to each snapshot."""

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

# Bump when changing index layout, chunking, or provider embedding behavior.
INDEX_VERSION = 1


def default_index_dir() -> Path:
    explicit = os.getenv("SENSOR_INDEX_DIR")
    if explicit:
        return Path(explicit)
    app_data = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".cache")
    return app_data / "SensorAnomalyRAG" / "indexes"


def index_key(documents: Mapping[str, str], *model_settings: str) -> str:
    data = json.dumps(
        [INDEX_VERSION, model_settings, list(documents.items())], ensure_ascii=False
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@contextmanager
def working_index(root: Path | None, key: str) -> Iterator[tuple[Path, bool]]:
    if root is None:
        with TemporaryDirectory(prefix="sensor-anomaly-rag-") as temporary:
            yield Path(temporary), False
        return

    from filelock import FileLock, Timeout

    slot = root.resolve() / key
    slot.mkdir(parents=True, exist_ok=True)
    marker = slot / "ready"
    try:
        with FileLock(str(slot / "index.lock"), timeout=120):
            if marker.is_file():
                snapshot = marker.read_text(encoding="utf-8").strip()
                if snapshot.startswith("index-") and Path(snapshot).name == snapshot:
                    published = slot / snapshot
                    if published.is_dir():
                        yield published, True
                        return

            candidate = Path(mkdtemp(prefix="index-", dir=slot))
            try:
                yield candidate, False
                pending = slot / "ready.tmp"
                pending.write_text(candidate.name + "\n", encoding="utf-8")
                pending.replace(marker)
            except BaseException:
                shutil.rmtree(candidate, ignore_errors=True)
                raise
    except Timeout as exc:
        raise RuntimeError("手册索引正由另一个进程使用，请稍后重试") from exc
