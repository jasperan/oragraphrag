"""Run a BenchSuite against one or more baseline systems and aggregate metrics.

Each per-question record carries: id, hops, correctness (0-4 judge score),
citation P/R, latency_ms, tokens. Each system aggregate carries: n, medians
across the per-question records, and the full per-question list for paper
appendix tables.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from oragraphrag.bench.baselines import REGISTRY
from oragraphrag.bench.metrics import _judge_call, citation_pr
from oragraphrag.config import Config
from oragraphrag.llm import LLM


@dataclass(slots=True)
class BenchSuite:
    items: list[dict]


def load_suite(path: str | Path) -> BenchSuite:
    """Load a JSONL bench suite. Blank lines are skipped."""
    raw = Path(path).read_text()
    items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return BenchSuite(items=items)


async def _run_one(system: str, question: str, cfg: Config) -> dict:
    """Dispatch one question to one baseline. Latency timed by the caller."""
    runner = REGISTRY[system]
    t0 = time.perf_counter()
    out = await runner.run(question, cfg)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


async def run_suite(
    cfg: Config,
    *,
    suite: str,
    systems: Iterable[str],
    limit: int | None,
) -> dict:
    """Run every baseline in `systems` over the suite; return per-system aggregates."""
    items = load_suite(suite).items
    if limit:
        items = items[: limit]

    async with LLM(cfg) as judge_llm:
        results: dict[str, dict] = {}
        for system in systems:
            per_q: list[dict] = []
            for it in items:
                out = await _run_one(system, it["question"], cfg)
                score = await _judge_call(
                    it["question"], it["gold_answer"], out["answer"], judge_llm
                )
                precision, recall = citation_pr(
                    gold=it.get("gold_doc_ids", []),
                    pred=out.get("citations", []),
                )
                per_q.append(
                    {
                        "id": it.get("id"),
                        "hops": it.get("hops", 1),
                        "correctness": score,
                        "citation_precision": precision,
                        "citation_recall": recall,
                        "latency_ms": out["latency_ms"],
                        "tokens": out.get("tokens", 0),
                    }
                )
            results[system] = _aggregate(per_q)

    return {"suite": suite, "systems": results}


def _aggregate(per_q: list[dict]) -> dict:
    """Compute medians for a single system's per-question results."""
    if not per_q:
        return {
            "n": 0,
            "median_correctness": 0,
            "median_latency_ms": 0,
            "median_tokens": 0,
            "median_citation_precision": 0,
            "median_citation_recall": 0,
            "per_question": [],
        }
    return {
        "n": len(per_q),
        "median_correctness": statistics.median(q["correctness"] for q in per_q),
        "median_latency_ms": statistics.median(q["latency_ms"] for q in per_q),
        "median_tokens": statistics.median(q["tokens"] for q in per_q),
        "median_citation_precision": statistics.median(
            q["citation_precision"] for q in per_q
        ),
        "median_citation_recall": statistics.median(q["citation_recall"] for q in per_q),
        "per_question": per_q,
    }
