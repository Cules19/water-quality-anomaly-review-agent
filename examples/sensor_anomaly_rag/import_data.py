"""Import observation and field-log CSV exports in one SQLite transaction."""

import csv
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .database import DATA_DIR, DEMO_DATABASE, parse_time

CsvRow = dict[str, str]
DatabaseRow = tuple[str | int, ...]


def csv_rows(path: Path, fields: tuple[str, ...]) -> Iterator[tuple[int, CsvRow]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        try:
            headers = reader.fieldnames or []
            missing = set(fields).difference(headers)
            if missing:
                raise ValueError(
                    f"{path.name} 缺少 CSV 列 {', '.join(sorted(missing))}"
                )
            if len(headers) != len(set(headers)):
                raise ValueError(f"{path.name} 的 CSV 表头有重复列名")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"{path.name} 第 {reader.line_num} 行的列数与表头不符"
                    )
                yield reader.line_num, row
        except csv.Error as exc:
            raise ValueError(
                f"{path.name} 第 {reader.line_num} 行 CSV 格式有误"
            ) from exc


def ph_centi(value: str, field: str) -> int:
    try:
        ph = Decimal(value)
        if ph.is_finite() and 0 <= ph <= 14:
            rounded = ph.quantize(Decimal("0.01"))
            if ph == rounded:
                return int(rounded * 100)
    except (InvalidOperation, ValueError):
        pass
    raise ValueError(f"{field} 须为 0–14 且精确至百分位的 pH 数值")


def observation_row(row: CsvRow) -> DatabaseRow:
    site = row["site"].strip()
    if not site:
        raise ValueError("site 不能为空")
    return (
        site,
        int(parse_time(row["timestamp"].strip()).timestamp()),
        ph_centi(row["observed_ph"], "observed_ph"),
        ph_centi(row["baseline_ph"], "baseline_ph"),
    )


def log_row(row: CsvRow) -> DatabaseRow:
    log_id, site, kind = (row[field].strip() for field in ("id", "site", "kind"))
    note = row["note"].replace("\r\n", "\n").replace("\r", "\n")
    if not log_id or not site or not note.strip():
        raise ValueError("id、site 和 note 不能为空")
    if kind not in ("维护", "校准"):
        raise ValueError("kind 须为「维护」或「校准」")
    return (
        log_id,
        site,
        int(parse_time(row["timestamp"].strip()).timestamp()),
        kind,
        note,
    )


@dataclass(frozen=True)
class ImportTable:
    name: str
    csv_fields: tuple[str, ...]
    db_fields: tuple[str, ...]
    key_size: int
    parse: Callable[[CsvRow], DatabaseRow]


OBSERVATIONS = ImportTable(
    "observations",
    ("site", "timestamp", "observed_ph", "baseline_ph"),
    ("site", "timestamp_utc", "observed_ph_centi", "baseline_ph_centi"),
    2,
    observation_row,
)
LOGS = ImportTable(
    "site_logs",
    ("id", "site", "timestamp", "kind", "note"),
    ("id", "site", "timestamp_utc", "kind", "note"),
    1,
    log_row,
)


def insert_csv(
    db: sqlite3.Connection, path: Path, table: ImportTable
) -> dict[str, int]:
    # SQL identifiers come from the two schemas above. CSV values are bound parameters.
    keys = table.db_fields[: table.key_size]
    fields = ", ".join(table.db_fields)
    placeholders = ", ".join("?" for _ in table.db_fields)
    condition = " AND ".join(f"{key} = ?" for key in keys)
    insert = f"INSERT INTO {table.name} ({fields}) VALUES ({placeholders}) ON CONFLICT ({', '.join(keys)}) DO NOTHING"
    select = f"SELECT {fields} FROM {table.name} WHERE {condition}"
    counts = {"inserted": 0, "already_present": 0}
    for line, raw in csv_rows(path, table.csv_fields):
        try:
            row = table.parse(raw)
            if db.execute(insert, row).rowcount:
                counts["inserted"] += 1
            elif db.execute(select, row[: table.key_size]).fetchone() == row:
                counts["already_present"] += 1
            else:
                raise ValueError(f"标识 {row[: table.key_size]} 已存在且内容不同")
        except ValueError as exc:
            raise ValueError(f"{path.name} 第 {line} 行，{exc}") from exc
    return counts


def import_site_data(
    database: Path, observations: Path | None, logs: Path | None
) -> dict[str, dict[str, int]]:
    if observations is None and logs is None:
        raise ValueError("请提供 --observations 或 --logs")
    if database.resolve() == DEMO_DATABASE.resolve():
        raise ValueError("随附的演示库用于复现实验，导入请指定独立数据库文件")
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected_tables = {OBSERVATIONS.name, LOGS.name}
        if not tables:
            schema = (DATA_DIR / "schema.sql").read_text(encoding="utf-8")
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)
        elif not expected_tables.issubset(tables):
            raise ValueError("目标 SQLite 文件缺少复核表，请选择一个新的数据库文件")
        return {
            table.name: insert_csv(db, path, table)
            for table, path in ((OBSERVATIONS, observations), (LOGS, logs))
            if path is not None
        }
