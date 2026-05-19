import asyncio
import json

import pytest

from oragraphrag.bench.baselines import REGISTRY
from oragraphrag.bench.metrics import citation_pr, recall_at_k
from oragraphrag.bench.runner import BenchSuite, load_suite, run_suite
from oragraphrag.config import Config


def test_registry_has_all_four_baselines():
    assert set(REGISTRY) == {"naive_rag", "graphrag", "lightrag", "oragraphrag"}


def test_citation_pr_handles_partial_overlap():
    p, r = citation_pr(gold=["a.md#1", "b.md#2"], pred=["a.md#1", "c.md#3"])
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)


def test_citation_pr_perfect_match():
    p, r = citation_pr(gold=["a.md#1"], pred=["a.md#1"])
    assert p == 1.0
    assert r == 1.0


def test_citation_pr_empty_pred_and_gold():
    p, r = citation_pr(gold=[], pred=[])
    assert p == 1.0
    assert r == 1.0


def test_citation_pr_empty_pred_nonempty_gold():
    p, r = citation_pr(gold=["a.md#1"], pred=[])
    assert p == 1.0  # no pred → vacuously precise
    assert r == 0.0


def test_citation_pr_empty_gold_nonempty_pred():
    p, r = citation_pr(gold=[], pred=["a.md#1"])
    assert p == 0.0
    assert r == 1.0  # no gold to miss


def test_recall_at_k_matches_gold_in_topk():
    assert recall_at_k(gold=["a", "b"], retrieved=["x", "a", "y", "b"], k=4) == pytest.approx(1.0)


def test_recall_at_k_misses_when_gold_not_in_topk():
    assert recall_at_k(gold=["a", "b"], retrieved=["x", "y"], k=2) == pytest.approx(0.0)


def test_recall_at_k_partial():
    assert recall_at_k(gold=["a", "b"], retrieved=["x", "a", "y"], k=3) == pytest.approx(0.5)


def test_recall_at_k_empty_gold_returns_one():
    """Vacuously: if there's no gold to find, recall is 1.0."""
    assert recall_at_k(gold=[], retrieved=["a"], k=1) == 1.0


def test_load_suite_parses_jsonl(tmp_path):
    suite_path = tmp_path / "s.jsonl"
    suite_path.write_text(
        json.dumps({"id": "q1", "question": "Q?", "gold_answer": "A.",
                    "gold_doc_ids": ["d.md#s"], "hops": 1}) + "\n"
        + json.dumps({"id": "q2", "question": "Q2?", "gold_answer": "A2.",
                      "gold_doc_ids": ["e.md#s"], "hops": 2}) + "\n"
    )
    suite = load_suite(suite_path)
    assert isinstance(suite, BenchSuite)
    assert len(suite.items) == 2
    assert suite.items[0]["id"] == "q1"


def test_load_suite_skips_blank_lines(tmp_path):
    suite_path = tmp_path / "s.jsonl"
    suite_path.write_text(
        json.dumps({"id": "q1", "question": "Q?", "gold_answer": "A.",
                    "gold_doc_ids": [], "hops": 1}) + "\n"
        "\n"  # blank line
        + json.dumps({"id": "q2", "question": "Q2?", "gold_answer": "A.",
                      "gold_doc_ids": [], "hops": 1}) + "\n"
    )
    suite = load_suite(suite_path)
    assert len(suite.items) == 2


@pytest.mark.asyncio
async def test_run_suite_aggregates_per_system(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        json.dumps({"id": "q1", "question": "Q?", "gold_answer": "A.",
                    "gold_doc_ids": ["d.md#s"], "hops": 1}) + "\n"
    )
    cfg = Config()

    async def fake_run_one(system, question, cfg):
        return {
            "answer": "stubbed answer",
            "citations": ["d.md#s"],
            "latency_ms": 1.0,
            "tokens": 100,
        }

    async def fake_judge_call(question, gold, predicted, llm):
        return 3

    monkeypatch.setattr("oragraphrag.bench.runner._run_one", fake_run_one)
    monkeypatch.setattr("oragraphrag.bench.runner._judge_call", fake_judge_call)

    out = await run_suite(cfg, suite=str(suite_path), systems=["oragraphrag"], limit=None)
    assert "systems" in out
    assert "oragraphrag" in out["systems"]
    assert out["systems"]["oragraphrag"]["n"] == 1
    per_q = out["systems"]["oragraphrag"]["per_question"]
    assert len(per_q) == 1
    assert per_q[0]["correctness"] == 3
    assert per_q[0]["citation_precision"] == 1.0
    assert per_q[0]["citation_recall"] == 1.0


@pytest.mark.asyncio
async def test_run_suite_respects_limit(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.jsonl"
    lines = []
    for i in range(5):
        lines.append(
            json.dumps({"id": f"q{i}", "question": f"Q{i}?", "gold_answer": "A.",
                        "gold_doc_ids": [], "hops": 1})
        )
    suite_path.write_text("\n".join(lines))

    async def fake_run_one(system, question, cfg):
        return {"answer": "a", "citations": [], "latency_ms": 1.0, "tokens": 0}

    async def fake_judge_call(question, gold, predicted, llm):
        return 2

    monkeypatch.setattr("oragraphrag.bench.runner._run_one", fake_run_one)
    monkeypatch.setattr("oragraphrag.bench.runner._judge_call", fake_judge_call)
    out = await run_suite(cfg=Config(), suite=str(suite_path), systems=["naive_rag"], limit=2)
    assert out["systems"]["naive_rag"]["n"] == 2


@pytest.mark.asyncio
async def test_run_suite_records_median_correctness(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.jsonl"
    lines = []
    for i in range(3):
        lines.append(
            json.dumps({"id": f"q{i}", "question": "Q?", "gold_answer": "A.",
                        "gold_doc_ids": [], "hops": 1})
        )
    suite_path.write_text("\n".join(lines))

    scores = iter([4, 2, 3])

    async def fake_run_one(system, question, cfg):
        return {"answer": "a", "citations": [], "latency_ms": 1.0, "tokens": 0}

    async def fake_judge_call(question, gold, predicted, llm):
        return next(scores)

    monkeypatch.setattr("oragraphrag.bench.runner._run_one", fake_run_one)
    monkeypatch.setattr("oragraphrag.bench.runner._judge_call", fake_judge_call)
    out = await run_suite(cfg=Config(), suite=str(suite_path), systems=["naive_rag"], limit=None)
    assert out["systems"]["naive_rag"]["median_correctness"] == 3


def test_judge_prompt_template_includes_question_and_gold():
    from importlib.resources import files

    text = files("oragraphrag.prompts").joinpath("judge.j2").read_text()
    from jinja2 import Template

    rendered = Template(text).render(question="what?", gold="the answer", predicted="something")
    assert "what?" in rendered
    assert "the answer" in rendered
    assert "something" in rendered


def test_oracle_docs_qa_suite_exists_and_loads():
    """The smoke suite must be checked in and parseable."""
    suite = load_suite("benchmarks/suites/oracle_docs_qa.jsonl")
    assert len(suite.items) >= 5
    for item in suite.items:
        assert "id" in item
        assert "question" in item
        assert "gold_answer" in item


def test_graphrag_baseline_raises_clear_error_without_artifacts():
    """Either GraphRAG is wired (and surfaces a clear error when artifacts
    are missing) or it's still a stub. Both are acceptable bench contracts;
    the harness records the failure and moves on."""
    from oragraphrag.bench.baselines import graphrag

    with pytest.raises((NotImplementedError, ImportError, FileNotFoundError, RuntimeError)):
        asyncio.run(graphrag.run("q", Config()))


def test_lightrag_baseline_raises_clear_error_without_working_dir():
    """Same loose contract as GraphRAG: clear error when preconditions
    (working dir populated by `rag.ainsert`) are missing."""
    from oragraphrag.bench.baselines import lightrag

    with pytest.raises((NotImplementedError, ImportError, FileNotFoundError, RuntimeError)):
        asyncio.run(lightrag.run("q", Config()))


def test_full_suite_has_200_questions_with_correct_hop_distribution():
    """The full bench suite must have the 80/80/40 split per spec."""
    suite = load_suite("benchmarks/suites/oracle_docs_qa.jsonl")
    assert len(suite.items) == 200

    by_hops = {1: 0, 2: 0, 3: 0}
    for item in suite.items:
        by_hops[item["hops"]] += 1
    assert by_hops[1] == 80
    assert by_hops[2] == 80
    assert by_hops[3] == 40


def test_smoke_suite_still_available():
    """The original 5-question smoke suite remains for fast iteration."""
    suite = load_suite("benchmarks/suites/oracle_docs_qa.smoke.jsonl")
    assert len(suite.items) == 5
