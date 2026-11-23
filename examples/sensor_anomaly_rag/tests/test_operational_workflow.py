import asyncio
import json
from contextlib import closing

import pytest

from examples.sensor_anomaly_rag.__main__ import build_parser, run_review
from examples.sensor_anomaly_rag.database import (
    all_site_logs,
    open_database,
    read_observations,
)
from examples.sensor_anomaly_rag.import_data import import_site_data
from examples.sensor_anomaly_rag.index_store import index_key, working_index

pytestmark = pytest.mark.offline

OBS_HEADER = "site,timestamp,observed_ph,baseline_ph\n"
LOG_HEADER = "id,site,timestamp,kind,note\n"


def test_imported_data_can_be_reviewed_without_a_model(tmp_path, monkeypatch):
    observations = tmp_path / "observations.csv"
    logs = tmp_path / "logs.csv"
    database = tmp_path / "station data.sqlite3"
    observations.write_text(
        OBS_HEADER + "East,2026-11-23T14:00:00Z,5.84,7.20\n", encoding="utf-8-sig"
    )
    logs.write_text(
        LOG_HEADER
        + 'east-1,East,2026-11-23T21:50:00+08:00,维护,"现场清洗,记录完毕\n缺少同期复测"\n'
        + "west-1,West,2026-11-23T21:50:00+08:00,校准,另站记录\n",
        encoding="utf-8",
        newline="\r\n",
    )
    assert import_site_data(database, observations, logs) == {
        "observations": {"inserted": 1, "already_present": 0},
        "site_logs": {"inserted": 2, "already_present": 0},
    }
    logs.write_text(logs.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    assert import_site_data(database, observations, logs) == {
        "observations": {"inserted": 0, "already_present": 1},
        "site_logs": {"inserted": 0, "already_present": 2},
    }

    def fail_if_called():
        pytest.fail("logs-only must not load model configuration")

    monkeypatch.setattr(
        "examples.sensor_anomaly_rag.__main__.ModelConfig.from_env", fail_if_called
    )
    args = build_parser().parse_args(
        [
            "review",
            "--logs-only",
            "--format",
            "json",
            "--database",
            str(database),
            "--site",
            "East",
            "--start",
            "2026-11-23T21:00:00+08:00",
            "--end",
            "2026-11-23T23:00:00+08:00",
        ]
    )
    report = json.loads(asyncio.run(run_review(args)))
    assert report["status"] == "events_found"
    assert report["data_source"] == {"file": database.name, "demo": False}
    assert report["manual_retrieval"] == {"backend": "logs_only", "index_reused": None}
    (event,) = report["events"]
    (log,) = event["site_logs"]
    assert log["id"] == "east-1" and log["note"] == "现场清洗,记录完毕\n缺少同期复测"
    assert event["deviation"] == "-1.36"
    assert event["cause_status"] == "insufficient_evidence"
    assert event["manual_passages"] == []


def test_conflicting_log_rolls_back_observations_and_preserves_existing_log(tmp_path):
    observations = tmp_path / "observations.csv"
    logs = tmp_path / "logs.csv"
    database = tmp_path / "data.sqlite3"
    logs.write_text(
        LOG_HEADER + "id-1,A,2026-11-23T14:00:00Z,维护,原始记录\n", encoding="utf-8"
    )
    import_site_data(database, None, logs)
    observations.write_text(
        OBS_HEADER + "A,2026-11-23T14:00:00Z,6.00,7.00\n", encoding="utf-8"
    )
    logs.write_text(
        LOG_HEADER + "id-1,A,2026-11-23T22:00:00+08:00,维护,更改后的记录\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"logs.csv 第 2 行.*内容不同"):
        import_site_data(database, observations, logs)
    with closing(open_database(database)) as db:
        assert not read_observations(db, "A", 0, 2_000_000_000)
        assert [log.text for log in all_site_logs(db)] == ["原始记录"]


def test_invalid_ph_reports_the_csv_line(tmp_path):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        OBS_HEADER + "A,2026-11-23T14:00:00Z,7.0001,7.20\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"第 2 行.*observed_ph.*百分位"):
        import_site_data(tmp_path / "data.sqlite3", observations, None)


def test_index_reuses_only_successful_snapshots_for_the_same_inputs(tmp_path):
    key = index_key({"guide.md": "version one"}, "chat", "endpoint", "embedding")
    with pytest.raises(RuntimeError, match="indexing failed"):
        with working_index(tmp_path, key) as (failed, reused):
            assert not reused
            raise RuntimeError("indexing failed")
    assert not failed.exists()

    with working_index(tmp_path, key) as (built, reused):
        assert not reused
        (built / "sample").write_text("indexed", encoding="utf-8")
    with working_index(tmp_path, key) as (opened, reused):
        assert reused and opened == built
        assert (opened / "sample").read_text(encoding="utf-8") == "indexed"

    changed = index_key({"guide.md": "version two"}, "chat", "endpoint", "embedding")
    with working_index(tmp_path, changed) as (rebuilt, reused):
        assert not reused and rebuilt != built
