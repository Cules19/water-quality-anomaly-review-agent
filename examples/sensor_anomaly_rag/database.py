"""Read the demo SQLite data using exact site and UTC time bounds."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "fixtures"
DEMO_DATABASE = DATA_DIR / "demo.sqlite3"


@dataclass(frozen=True)
class Observation:
    site: str
    timestamp: int
    observed_ph: Decimal
    baseline_ph: Decimal

    @property
    def deviation(self) -> Decimal:
        return self.observed_ph - self.baseline_ph


@dataclass(frozen=True)
class SiteLog:
    id: str
    site: str
    timestamp: int
    kind: str
    text: str


def open_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 SQLite 数据库 {path}，请先运行 import-data")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def read_observations(
    db: sqlite3.Connection, site: str, start: int, end: int
) -> list[Observation]:
    rows = db.execute(
        """SELECT site, timestamp_utc, observed_ph_centi, baseline_ph_centi
           FROM observations
           WHERE site = ? AND timestamp_utc >= ? AND timestamp_utc < ?
           ORDER BY timestamp_utc""",
        (site, start, end),
    )
    return [
        Observation(
            site_id, timestamp, Decimal(observed) / 100, Decimal(baseline) / 100
        )
        for site_id, timestamp, observed, baseline in rows
    ]


def find_site_logs(
    db: sqlite3.Connection, site: str, event_time: int, window_minutes: int
) -> list[SiteLog]:
    window_seconds = window_minutes * 60
    rows = db.execute(
        """SELECT id, site, timestamp_utc, kind, note FROM site_logs
           WHERE site = ? AND timestamp_utc BETWEEN ? AND ?
           ORDER BY timestamp_utc, id""",
        (site, event_time - window_seconds, event_time + window_seconds),
    )
    return [SiteLog(*row) for row in rows]


def all_site_logs(db: sqlite3.Connection) -> list[SiteLog]:
    """Export evaluation fixtures without filtering by case."""
    return [
        SiteLog(*row)
        for row in db.execute(
            "SELECT id, site, timestamp_utc, kind, note FROM site_logs ORDER BY id"
        )
    ]


def parse_time(value: str) -> datetime:
    try:
        instant = datetime.fromisoformat(value)
        if instant.utcoffset() is not None and not instant.microsecond:
            return instant
    except ValueError:
        pass
    raise ValueError(
        "时间须为带时区的 ISO 8601 整秒时间，例如 2026-01-15T21:00:00+08:00"
    )


def utc_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


def build_fixture(path: Path) -> None:
    """Create a new fixture or fail if the destination already exists."""
    path.touch(exist_ok=False)
    try:
        with closing(sqlite3.connect(path)) as db:
            schema = (DATA_DIR / "schema.sql").read_text(encoding="utf-8")
            data = (DATA_DIR / "demo.sql").read_text(encoding="utf-8")
            db.executescript(f"BEGIN;\n{schema}\n{data}\nCOMMIT;")
    except Exception:
        path.unlink()
        raise
