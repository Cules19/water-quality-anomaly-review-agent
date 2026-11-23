import asyncio
import sys
from contextlib import closing
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from examples.sensor_anomaly_rag.__main__ import build_parser, run_review
from examples.sensor_anomaly_rag.database import (
    DEMO_DATABASE,
    build_fixture,
    find_site_logs,
    open_database,
    parse_time,
)
from examples.sensor_anomaly_rag.evaluation import CASES, evaluate
from examples.sensor_anomaly_rag.retrieval import (
    QUERY_OPTIONS,
    FixtureBackend,
    ModelConfig,
    Retriever,
    open_retriever,
)
from examples.sensor_anomaly_rag.review import (
    INSUFFICIENT,
    Policy,
    Request,
    collect_events,
    render_report,
    review_events,
)

pytestmark = pytest.mark.offline


def test_seed_reproduces_shipped_sqlite_database(tmp_path):
    rebuilt_path = tmp_path / "rebuilt.sqlite3"
    build_fixture(rebuilt_path)
    with (
        closing(open_database(rebuilt_path)) as rebuilt,
        closing(open_database(DEMO_DATABASE)) as shipped,
    ):
        assert rebuilt.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert list(rebuilt.iterdump()) == list(shipped.iterdump())
    with pytest.raises(FileExistsError):
        build_fixture(rebuilt_path)


def test_event_facts_and_logs_enter_report_through_retrieval_adapter():
    request, policy = CASES[0].request(), Policy()
    with closing(open_database(DEMO_DATABASE)) as db:
        count, events = collect_events(db, request, policy)
    assert count == 3
    (event,) = events
    assert event.observation.site == "A"
    assert event.observation.timestamp == int(
        parse_time("2026-01-15T22:00:00+08:00").timestamp()
    )
    assert event.observation.observed_ph == Decimal("5.85")
    assert event.observation.baseline_ph == Decimal("7.20")
    assert event.observation.deviation == Decimal("-1.35")
    assert [log.id for log in event.logs] == ["A-0115-clean"]

    returned_documents = {
        "manual.md": "替身返回的原文，核对独立水样。",
        "foreign-log.md": "错误来源",
    }
    fixture = FixtureBackend(returned_documents)
    backend = SimpleNamespace(aquery_data=AsyncMock(side_effect=fixture.aquery_data))
    params = SimpleNamespace(**QUERY_OPTIONS)
    retriever = Retriever(backend, params)
    reviews = asyncio.run(
        review_events(request, events, retriever.retrieve, {"manual.md"})
    )

    backend.aquery_data.assert_awaited_once()
    args, kwargs = backend.aquery_data.await_args
    assert request.question in args[0]
    assert all(text in args[0] for text in ("pH 突降", "5.85", "7.20", "-1.35"))
    assert kwargs == {"param": params}
    report = render_report(request, policy, count, reviews)
    assert "**manual.md**（引用 `1`，片段 `fixture-1`）" in report
    assert "> 替身返回的原文，核对独立水样。" in report
    assert "清洗了 pH 探头" in report
    assert INSUFFICIENT in report
    assert "错误来源" not in report
    assert "B-0115-cal" not in report and "A-0108-cal" not in report


def test_log_window_has_inclusive_edges_and_exact_site_matching():
    event_time = int(parse_time("2026-01-15T14:00:00Z").timestamp())
    with closing(open_database(DEMO_DATABASE)) as db:
        assert [log.id for log in find_site_logs(db, "A", event_time, 15)] == [
            "A-0115-clean"
        ]
        assert not find_site_logs(db, "A", event_time, 14)
        assert not find_site_logs(db, "a", event_time, 90)


def test_observation_range_is_half_open_and_threshold_is_inclusive():
    request = Request(
        "A",
        parse_time("2026-01-15T14:00:00Z"),
        parse_time("2026-01-15T14:30:00Z"),
        "核对 pH 突降",
    )
    with closing(open_database(DEMO_DATABASE)) as db:
        count, events = collect_events(db, request, Policy(Decimal("1.35")))
        assert count == len(events) == 1
        count, events = collect_events(db, request, Policy(Decimal("1.36")))
        assert count == 1 and not events


def test_manual_only_report_leaves_cause_unknown():
    args = build_parser().parse_args(
        [
            "review",
            "--offline",
            "--site",
            "C",
            "--start",
            "2026-01-15T21:00:00+08:00",
            "--end",
            "2026-01-15T23:00:00+08:00",
        ]
    )
    report = asyncio.run(run_review(args))
    assert "该站在事件窗口内没有维护或校准记录" in report
    assert "ph-drop.md" in report
    assert INSUFFICIENT in report


@pytest.mark.parametrize(
    ("site", "day", "status"),
    [
        ("A", "2026-01-16", "no_observations"),
        ("B", "2026-01-15", "no_anomaly"),
    ],
)
def test_empty_or_normal_period_skips_model_configuration(
    monkeypatch, site, day, status
):
    def fail_if_called():
        pytest.fail("No-anomaly review must not initialize a model")

    monkeypatch.setattr(
        "examples.sensor_anomaly_rag.__main__.ModelConfig.from_env", fail_if_called
    )
    args = build_parser().parse_args(
        [
            "review",
            "--site",
            site,
            "--start",
            f"{day}T21:00:00+08:00",
            "--end",
            f"{day}T23:00:00+08:00",
        ]
    )
    assert f"`{status}`" in asyncio.run(run_review(args))


def test_retrieval_distinguishes_no_results_from_failure():
    backend = SimpleNamespace(
        aquery_data=AsyncMock(
            side_effect=[
                {"status": "failure", "metadata": {"failure_reason": "no_results"}},
                {"status": "failure", "message": "index unavailable"},
            ]
        )
    )
    retriever = Retriever(backend, SimpleNamespace(**QUERY_OPTIONS))
    assert not asyncio.run(retriever.retrieve("pH 突降复核建议"))
    with pytest.raises(RuntimeError, match="index unavailable"):
        asyncio.run(retriever.retrieve("pH 突降复核建议"))


def test_live_backend_setup_binds_the_provider_model_and_checks_the_batch(monkeypatch):
    async def complete(model, prompt):
        return model, prompt

    class SimulatedSDK(FixtureBackend):
        def __init__(self, *, llm_model_func, **kwargs):
            super().__init__({})
            self.complete = llm_model_func

        initialize_storages = AsyncMock()
        finalize_storages = AsyncMock()
        aget_docs_by_track_id = AsyncMock(
            return_value={"sensor-doc-0": SimpleNamespace(status="processed")}
        )

        async def ainsert(self, contents, *, ids, file_paths):
            assert await self.complete("extraction prompt") == (
                "test-model",
                "extraction prompt",
            )
            assert ids == ["sensor-doc-0"]
            self.documents = dict(zip(file_paths, contents, strict=True))
            return "test-track-id"

    for name, module in {
        "numpy": SimpleNamespace(ndarray=object),
        "fastembed": SimpleNamespace(
            TextEmbedding=Mock(get_embedding_size=Mock(return_value=1))
        ),
        "lightrag": SimpleNamespace(LightRAG=SimulatedSDK, QueryParam=SimpleNamespace),
        "lightrag.base": SimpleNamespace(
            DocStatus=SimpleNamespace(PROCESSED="processed")
        ),
        "lightrag.llm.openai": SimpleNamespace(openai_complete_if_cache=complete),
        "lightrag.utils": SimpleNamespace(
            wrap_embedding_func_with_attrs=lambda **kw: lambda func: func
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    async def run():
        config = ModelConfig(
            "test-model", "https://example.invalid", "test-embedding", "fake"
        )
        async with open_retriever({"source.md": "manual text"}, config) as retriever:
            return await retriever.retrieve("pH 突降复核建议")

    (passage,) = asyncio.run(run())
    assert (passage.source, passage.text) == ("source.md", "manual text")
    SimulatedSDK.aget_docs_by_track_id.assert_awaited_once_with("test-track-id")
    SimulatedSDK.initialize_storages.assert_awaited_once()
    SimulatedSDK.finalize_storages.assert_awaited_once()


def test_evaluation_scores_shared_corpus_and_retains_wrong_citations():
    result = asyncio.run(evaluate(config=None))
    summary = result["summary"]
    baseline, agent = summary["fixed_retrieval"], summary["event_review"]
    assert (
        baseline["cases_with_wrong_site_logs"]
        == baseline["cases_with_outside_window_logs"]
        == 3
    )
    assert (
        agent["cases_with_wrong_site_logs"]
        == agent["cases_with_outside_window_logs"]
        == 0
    )
    assert agent["correct_observation_cases"] == agent["abstained_on_event_cases"] == 2
    assert agent["detected_no_observations"] and agent["manual_only_case_has_manual"]
    assert agent["exact_log_sets"] == 3
