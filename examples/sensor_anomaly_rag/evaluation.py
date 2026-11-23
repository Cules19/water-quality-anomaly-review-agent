"""Compare fixed retrieval and event-driven review on the same indexed corpus."""

from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .database import (
    DEMO_DATABASE,
    SiteLog,
    all_site_logs,
    open_database,
    parse_time,
    utc_time,
)
from .retrieval import QUERY_OPTIONS, ModelConfig, Passage, open_retriever, read_manuals
from .review import (
    INSUFFICIENT,
    DropEvent,
    Policy,
    Request,
    collect_events,
    render_report,
    review_events,
)


@dataclass(frozen=True)
class Case:
    id: str
    site: str
    day: str
    event_time: str | None
    expected_ph: tuple[str, str, str] | None
    expected_logs: frozenset[str]

    def request(self) -> Request:
        return Request(
            self.site,
            parse_time(f"{self.day}T21:00:00+08:00"),
            parse_time(f"{self.day}T23:00:00+08:00"),
            f"{self.site} 站这个时段 pH 突降，有什么维护、校准或手册记录可以解释？",
        )


CASES = (
    Case(
        "same-site-log",
        "A",
        "2026-01-15",
        "2026-01-15T22:00:00+08:00",
        ("5.85", "7.20", "-1.35"),
        frozenset({"A-0115-clean"}),
    ),
    Case(
        "manual-only",
        "C",
        "2026-01-15",
        "2026-01-15T22:00:00+08:00",
        ("6.12", "7.05", "-0.93"),
        frozenset(),
    ),
    Case("no-observations", "A", "2026-01-16", None, None, frozenset()),
)


def log_document(log: SiteLog) -> str:
    return (
        f"# pH 运维日志 {log.id}\n\n"
        f"站点 {log.site}，记录时间 {utc_time(log.timestamp)}，类型 {log.kind}。\n{log.text}"
    )


def citation(passage: Passage) -> dict[str, str]:
    return {
        "source": passage.source,
        "reference_id": passage.reference_id,
        "chunk_id": passage.chunk_id,
        "excerpt": passage.text,
    }


def score_logs(case: Case, logs: list[SiteLog], policy: Policy) -> dict[str, Any]:
    request = case.request()
    if case.event_time:
        center = int(parse_time(case.event_time).timestamp())
        start = center - policy.log_window_minutes * 60
        end = center + policy.log_window_minutes * 60 + 1
    else:
        start, end = int(request.start.timestamp()), int(request.end.timestamp())
    return {
        "wrong_site_log_ids": sorted({log.id for log in logs if log.site != case.site}),
        "outside_window_log_ids": sorted(
            {log.id for log in logs if not start <= log.timestamp < end}
        ),
        "expected_log_set_match": {log.id for log in logs} == case.expected_logs,
    }


def score_observation(case: Case, events: list[DropEvent]) -> bool | None:
    if case.expected_ph is None or case.event_time is None:
        return None
    if len(events) != 1:
        return False
    observation = events[0].observation
    expected_time = int(parse_time(case.event_time).timestamp())
    return (
        observation.site == case.site
        and observation.timestamp == expected_time
        and (observation.observed_ph, observation.baseline_ph, observation.deviation)
        == tuple(Decimal(value) for value in case.expected_ph)
    )


async def evaluate(config: ModelConfig | None) -> dict[str, Any]:
    policy = Policy()
    manuals = read_manuals()
    with closing(open_database(DEMO_DATABASE)) as db:
        log_sources = {f"site-log-{log.id}.md": log for log in all_site_logs(db)}
        prepared = [collect_events(db, case.request(), policy) for case in CASES]
    corpus = {
        **manuals,
        **{source: log_document(log) for source, log in log_sources.items()},
    }

    trace = []
    async with open_retriever(corpus, config) as retriever:
        for case, (observation_count, events) in zip(CASES, prepared, strict=True):
            request = case.request()
            baseline_query = f"{request.question} 期间 {request.start.isoformat()} 至 {request.end.isoformat()}。"
            baseline_passages = await retriever.retrieve(baseline_query)
            baseline_logs = [
                log_sources[p.source]
                for p in baseline_passages
                if p.source in log_sources
            ]
            baseline = {
                "cited_passages": [citation(p) for p in baseline_passages],
                "site_log_ids": sorted({log.id for log in baseline_logs}),
                "cause_assessment": INSUFFICIENT,
                "observation_status": "not_checked",
                "observation_correct": None,
                **score_logs(case, baseline_logs, policy),
            }

            reviews = await review_events(request, events, retriever.retrieve, manuals)
            report = render_report(request, policy, observation_count, reviews)
            reviewed_logs = [log for item in reviews for log in item.event.logs]
            agent = {
                "events": [
                    {
                        "site": obs.site,
                        "time_utc": utc_time(obs.timestamp),
                        "observed_ph": f"{obs.observed_ph:.2f}",
                        "baseline_ph": f"{obs.baseline_ph:.2f}",
                        "deviation": f"{obs.deviation:+.2f}",
                    }
                    for obs in (item.event.observation for item in reviews)
                ],
                "cited_passages": [
                    citation(p) for item in reviews for p in item.manuals
                ],
                "site_log_ids": sorted({log.id for log in reviewed_logs}),
                "cause_assessment": INSUFFICIENT if INSUFFICIENT in report else None,
                "observation_status": "no_observations"
                if "`no_observations`" in report
                else "observed",
                "observation_correct": score_observation(case, events),
                **score_logs(case, reviewed_logs, policy),
            }
            trace.append(
                {
                    "case": case.id,
                    "expected_log_ids": sorted(case.expected_logs),
                    "fixed_retrieval": baseline,
                    "event_review": agent,
                }
            )

    summary = {}
    for workflow in ("fixed_retrieval", "event_review"):
        results = {row["case"]: row[workflow] for row in trace}
        summary[workflow] = {
            "cases_with_wrong_site_logs": sum(
                bool(row["wrong_site_log_ids"]) for row in results.values()
            ),
            "cases_with_outside_window_logs": sum(
                bool(row["outside_window_log_ids"]) for row in results.values()
            ),
            "exact_log_sets": sum(
                row["expected_log_set_match"] for row in results.values()
            ),
            "abstained_on_event_cases": sum(
                results[case.id]["cause_assessment"] == INSUFFICIENT
                for case in CASES
                if case.event_time
            ),
            "correct_observation_cases": sum(
                row["observation_correct"] is True for row in results.values()
            ),
            "detected_no_observations": results["no-observations"]["observation_status"]
            == "no_observations",
            "manual_only_case_has_manual": any(
                p["source"] in manuals for p in results["manual-only"]["cited_passages"]
            ),
        }
    return {
        "backend": "lightrag" if config else "offline_fixture",
        "retrieval": QUERY_OPTIONS,
        "chat_model": config.chat_model if config else None,
        "embedding_model": config.embedding_model if config else None,
        "corpus": list(corpus),
        "summary": summary,
        "cases": trace,
    }
