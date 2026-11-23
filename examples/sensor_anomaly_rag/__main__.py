"""CLI entry point for review, evaluation, and fixture regeneration."""

import argparse
import asyncio
import json
import sqlite3
import sys
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from .database import DEMO_DATABASE, build_fixture, open_database, parse_time
from .evaluation import evaluate
from .import_data import import_site_data
from .index_store import default_index_dir
from .retrieval import EXAMPLE_DIR, ModelConfig, open_retriever, read_manuals
from .review import (
    EventReview,
    Policy,
    Request,
    collect_events,
    render_report,
    review_events,
    structured_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水质传感器 pH 突降证据复核")
    commands = parser.add_subparsers(dest="command", required=True)

    review = commands.add_parser("review", help="复核一个站点和时间段")
    review.add_argument("--site", required=True, help="精确站点标识，例如 A")
    review.add_argument(
        "--start", required=True, help="观测范围起点（含），带时区的 ISO 8601"
    )
    review.add_argument(
        "--end", required=True, help="观测范围终点（不含），带时区的 ISO 8601"
    )
    review.add_argument("--question", default="pH 突降，有什么记录可以解释？")
    review.add_argument(
        "--database",
        type=Path,
        default=DEMO_DATABASE,
        help="SQLite 文件，默认使用演示数据",
    )
    review.add_argument(
        "--threshold",
        type=Decimal,
        default=Policy.drop_threshold,
        help="pH 负偏差的绝对阈值，默认 0.80",
    )
    review.add_argument(
        "--window-minutes",
        type=int,
        default=Policy.log_window_minutes,
        help="现场日志前后窗口，默认各 90 分钟",
    )
    review.add_argument(
        "--manual-dir",
        type=Path,
        default=EXAMPLE_DIR / "manuals",
        help="UTF-8 Markdown 手册目录",
    )
    review.add_argument("--index-dir", type=Path, help="LightRAG 索引缓存目录")
    review.add_argument("--format", choices=("markdown", "json"), default="markdown")
    review.add_argument("--output", type=Path, help="将报告写入新的 UTF-8 文件")
    review_mode = review.add_mutually_exclusive_group()
    review_mode.add_argument("--offline", action="store_true", help="使用演示替身检索")
    review_mode.add_argument(
        "--logs-only", action="store_true", help="使用 SQLite 观测和现场日志，无需模型"
    )

    evaluation = commands.add_parser("evaluate", help="对比固定检索与事件驱动复核")
    evaluation.add_argument("--output", type=Path, help="将评测 JSON 写入指定文件")

    evaluation.add_argument(
        "--offline", action="store_true", help="使用替身检索，无需模型凭据"
    )

    importer = commands.add_parser("import-data", help="将观测和运维 CSV 导入 SQLite")
    importer.add_argument(
        "--database", type=Path, required=True, help="新建或追加到已有数据库"
    )
    importer.add_argument("--observations", type=Path, help="观测 CSV")
    importer.add_argument("--logs", type=Path, help="运维日志 CSV")

    seed = commands.add_parser("build-db", help="从固定 SQL 生成一份新的示例数据库")
    seed.add_argument("--output", type=Path, required=True)
    return parser


async def run_review(args: argparse.Namespace) -> str:
    request = Request(
        args.site, parse_time(args.start), parse_time(args.end), args.question
    )
    policy = Policy(args.threshold, args.window_minutes)
    with closing(open_database(args.database)) as db, db:
        db.execute("BEGIN")
        observation_count, events = collect_events(db, request, policy)

    index_reused = None
    if events and args.logs_only:
        reviews = [EventReview(event, []) for event in events]
        backend = "logs_only"
    elif events:
        manuals = read_manuals(args.manual_dir)
        config = None if args.offline else ModelConfig.from_env()
        async with open_retriever(
            manuals, config, index_dir=args.index_dir or default_index_dir()
        ) as retriever:
            reviews = await review_events(request, events, retriever.retrieve, manuals)
            index_reused = retriever.index_reused
        backend = "fixture" if args.offline else "lightrag"
    else:
        reviews = []
        backend = "skipped"

    is_demo = args.database.resolve() == DEMO_DATABASE.resolve()
    if args.format == "json":
        payload = structured_report(request, policy, observation_count, reviews)
        payload["data_source"] = {"file": args.database.name, "demo": is_demo}
        payload["manual_retrieval"] = {"backend": backend, "index_reused": index_reused}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    source = f"演示数据（{args.database.name}）" if is_demo else args.database.name
    labels = {
        "logs_only": "现场记录模式",
        "fixture": "离线替身",
        "lightrag": "LightRAG",
        "skipped": "跳过手册检索",
    }
    mode = labels[backend]
    if index_reused is not None:
        mode += "（复用手册索引）" if index_reused else "（新建手册索引）"
    report = render_report(
        request, policy, observation_count, reviews, manuals_enabled=not args.logs_only
    )
    return f"数据来源：{source}\n\n检索后端：{mode}\n\n{report}"


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if args.command == "build-db":
            build_fixture(args.output)
            print(args.output.resolve())
        elif args.command == "review":
            if args.output and args.output.exists():
                raise FileExistsError(f"报告文件已存在，请使用新路径 {args.output}")
            report = asyncio.run(run_review(args))
            if args.output:
                with args.output.open("x", encoding="utf-8", newline="\n") as file:
                    file.write(report + "\n")
                print(args.output.resolve(), file=sys.stderr)
            else:
                print(report)
        elif args.command == "import-data":
            counts = import_site_data(args.database, args.observations, args.logs)
            print(json.dumps(counts, ensure_ascii=False, indent=2))
        else:
            config = None if args.offline else ModelConfig.from_env()
            payload = json.dumps(
                asyncio.run(evaluate(config)), ensure_ascii=False, indent=2
            )
            if args.output:
                args.output.write_text(payload + "\n", encoding="utf-8")
            else:
                print(payload)
    except (
        ArithmeticError,
        ValueError,
        RuntimeError,
        OSError,
        sqlite3.Error,
        ImportError,
    ) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
