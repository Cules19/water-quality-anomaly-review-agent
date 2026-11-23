"""Turn observations into events and format an evidence-bounded review."""

import sqlite3
from collections.abc import Awaitable, Callable, Collection
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from html import escape

from .database import Observation, SiteLog, find_site_logs, read_observations, utc_time
from .retrieval import Passage

INSUFFICIENT = "当前证据不足以确定原因"


@dataclass(frozen=True)
class Request:
    site: str
    start: datetime
    end: datetime
    question: str

    def __post_init__(self) -> None:
        if not self.site or not self.question.strip():
            raise ValueError("站点标识和问题不能为空")
        if self.start.utcoffset() is None or self.end.utcoffset() is None:
            raise ValueError("查询时间须包含时区")
        if self.start >= self.end:
            raise ValueError("开始时间必须早于结束时间")


@dataclass(frozen=True)
class Policy:
    drop_threshold: Decimal = Decimal("0.80")
    log_window_minutes: int = 90

    def __post_init__(self) -> None:
        if not self.drop_threshold.is_finite() or not 0 < self.drop_threshold <= 14:
            raise ValueError("pH 突降阈值须大于 0 且不超过 14")
        if self.log_window_minutes < 0:
            raise ValueError("现场记录窗口分钟数不能为负")


@dataclass(frozen=True)
class DropEvent:
    observation: Observation
    logs: list[SiteLog]


@dataclass(frozen=True)
class EventReview:
    event: DropEvent
    manuals: list[Passage]


def collect_events(
    db: sqlite3.Connection, request: Request, policy: Policy
) -> tuple[int, list[DropEvent]]:
    observations = read_observations(
        db, request.site, int(request.start.timestamp()), int(request.end.timestamp())
    )
    events = [
        DropEvent(
            obs, find_site_logs(db, obs.site, obs.timestamp, policy.log_window_minutes)
        )
        for obs in observations
        if obs.deviation <= -policy.drop_threshold
    ]
    return len(observations), events


def event_query(request: Request, observation: Observation) -> str:
    return (
        f"{request.question}\n异常类型：在线 pH 突降。"
        f"站点 {observation.site}，观测 pH {observation.observed_ph:.2f}，"
        f"基线 {observation.baseline_ph:.2f}，偏差 {observation.deviation:+.2f}。"
        "查找维护和校准核对、独立水样复测与测量链路检查的运维手册建议。"
    )


async def review_events(
    request: Request,
    events: list[DropEvent],
    retrieve: Callable[[str], Awaitable[list[Passage]]],
    manual_sources: Collection[str],
) -> list[EventReview]:
    reviews = []
    for event in events:
        passages = await retrieve(event_query(request, event.observation))
        manuals = [passage for passage in passages if passage.source in manual_sources]
        reviews.append(EventReview(event, manuals))
    return reviews


def render_report(
    request: Request,
    policy: Policy,
    observation_count: int,
    reviews: list[EventReview],
    *,
    manuals_enabled: bool = True,
) -> str:
    timezone = request.start.tzinfo
    lines = [
        f"# {escape(request.site)} 站 pH 异常复核",
        "",
        f"观测范围 [{request.start.isoformat(timespec='seconds')}, "
        f"{request.end.astimezone(timezone).isoformat(timespec='seconds')})",
        f"观测数 {observation_count}，判定阈值 ΔpH ≤ −{policy.drop_threshold}。",
    ]
    if observation_count == 0:
        return "\n".join(
            [*lines, "", "所选站点在查询时间段内没有观测数据（`no_observations`）。"]
        )
    if not reviews:
        return "\n".join([*lines, "", "有观测数据，未达到突降阈值（`no_anomaly`）。"])

    for review in reviews:
        obs = review.event.observation
        timestamp = datetime.fromtimestamp(obs.timestamp, timezone).isoformat(
            timespec="seconds"
        )
        lines.extend(
            [
                "",
                f"## 事件 {timestamp}",
                "",
                "### 传感器事实",
                "",
                f"- 站点 `{obs.site}`，观测 pH **{obs.observed_ph:.2f}**，"
                f"基线 **{obs.baseline_ph:.2f}**，偏差 **{obs.deviation:+.2f}**。",
                "",
                "### 命中的现场记录",
                "",
                f"以事件时间为中心前后各 {policy.log_window_minutes} 分钟。",
                "",
            ]
        )
        for log in review.event.logs:
            log_time = datetime.fromtimestamp(log.timestamp, timezone).isoformat(
                timespec="seconds"
            )
            lines.extend(
                [
                    f"- `{escape(log.id)}` · {escape(log.site)} 站 · {log_time} · {log.kind}",
                    "",
                    *(
                        f"  > {escape(line)}" if line else "  >"
                        for line in log.text.splitlines()
                    ),
                    "",
                ]
            )
        if not review.event.logs:
            lines.append("该站在事件窗口内没有维护或校准记录。")

        lines.extend(["", "### 手册建议", ""])
        for passage in review.manuals:
            lines.extend(
                [
                    f"**{passage.source}**（引用 `{passage.reference_id}`，片段 `{passage.chunk_id}`）",
                    "",
                    *(
                        f"> {line}" if line else ">"
                        for line in passage.text.splitlines()
                    ),
                    "",
                ]
            )
        if not manuals_enabled:
            lines.extend(["本次复核仅使用观测数据和现场记录。", ""])
        elif not review.manuals:
            lines.extend(["未检索到可引用的手册片段。", ""])

        lines.extend(
            [
                "### 证据限制",
                "",
                f"**{INSUFFICIENT}**。",
                "原因判断仍需要同站同期的独立水样复测或可追溯的仪表检查结果。",
            ]
        )
    return "\n".join(lines)


def structured_report(
    request: Request, policy: Policy, observation_count: int, reviews: list[EventReview]
) -> dict:
    if reviews:
        status = "events_found"
    else:
        status = "no_anomaly" if observation_count else "no_observations"

    events = []
    for review in reviews:
        observation = review.event.observation
        events.append(
            {
                "site": observation.site,
                "time_utc": utc_time(observation.timestamp),
                "observed_ph": f"{observation.observed_ph:.2f}",
                "baseline_ph": f"{observation.baseline_ph:.2f}",
                "deviation": f"{observation.deviation:+.2f}",
                "site_logs": [
                    {
                        "id": log.id,
                        "site": log.site,
                        "time_utc": utc_time(log.timestamp),
                        "kind": log.kind,
                        "note": log.text,
                    }
                    for log in review.event.logs
                ],
                "manual_passages": [asdict(passage) for passage in review.manuals],
                "cause_status": "insufficient_evidence",
                "cause_assessment": INSUFFICIENT,
            }
        )
    return {
        "schema_version": 1,
        "status": status,
        "site": request.site,
        "start": request.start.isoformat(timespec="seconds"),
        "end": request.end.isoformat(timespec="seconds"),
        "question": request.question,
        "observation_count": observation_count,
        "drop_threshold": str(policy.drop_threshold),
        "log_window_minutes": policy.log_window_minutes,
        "events": events,
    }
