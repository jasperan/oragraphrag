"""Bench metrics: LLM-as-judge correctness, citation P/R, recall@k.

`_judge_call` is the async coroutine the runner uses. `score_correctness`
is a synchronous wrapper for ad-hoc usage outside the harness. Citation
P/R and recall@k are pure-Python set arithmetic — no I/O.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from importlib.resources import files
from typing import Any

from jinja2 import Template

_JUDGE_TMPL = Template(
    files("oragraphrag.prompts").joinpath("judge.j2").read_text()
)


async def _judge_call(question: str, gold: str, predicted: str, llm: Any) -> int:
    """Render the judge prompt, call the LLM, parse a 0-4 integer.

    Robust to LLM responses that include trailing prose or markdown:
    extracts the first 0-4 integer in the response.
    """
    prompt = _JUDGE_TMPL.render(question=question, gold=gold, predicted=predicted)
    raw = await llm.complete(prompt, temperature=0.0)
    text = str(raw).strip()
    # Find the first single-digit 0-4.
    for token in text.split():
        token = token.strip("[](){}.,;:'\"-")
        if token.isdigit():
            score = int(token)
            if 0 <= score <= 4:
                return score
    return 0


def score_correctness(*, question: str, gold: str, predicted: str, llm: Any) -> int:
    """Synchronous wrapper around _judge_call for ad-hoc / monkeypatched use."""
    return asyncio.get_event_loop().run_until_complete(
        _judge_call(question, gold, predicted, llm)
    )


def _citation_match(gold_ref: str, pred_ref: str) -> bool:
    """Return True if a predicted citation matches a gold doc#section.

    Gold refs use a flat leaf section like `vectors.md#datatype`. Predicted
    refs may carry the full markdown section hierarchy emitted by Task 6's
    parser, e.g. `vectors.md#Vectors in Oracle Database 23ai / datatype`.
    A match is declared when the docs are identical and the pred's section
    path ends with the gold's section (case-insensitive, separator-tolerant).
    """
    if gold_ref == pred_ref:
        return True
    if "#" not in gold_ref or "#" not in pred_ref:
        return False
    gold_doc, gold_sec = gold_ref.split("#", 1)
    pred_doc, pred_sec = pred_ref.split("#", 1)
    if gold_doc != pred_doc:
        return False
    gold_leaf = gold_sec.split("/")[-1].strip().lower()
    pred_leaf = pred_sec.split("/")[-1].strip().lower()
    return gold_leaf == pred_leaf


def citation_pr(*, gold: Sequence[str], pred: Sequence[str]) -> tuple[float, float]:
    """Return (precision, recall) over predicted vs gold citation sets.

    Uses _citation_match for tolerant comparison so the deep section paths
    Task 6 emits (e.g. `vectors.md#Vectors in Oracle Database 23ai / datatype`)
    still match a flat gold reference (`vectors.md#datatype`).

    Sentinels:
    - Empty gold AND empty pred → (1.0, 1.0). Vacuously perfect.
    - Empty pred, nonempty gold → (1.0, 0.0). No false positives possible.
    - Empty gold, nonempty pred → (0.0, 1.0). No truth to miss.
    """
    g_list, p_list = list(gold), list(pred)
    if not g_list and not p_list:
        return 1.0, 1.0
    if not p_list:
        return 1.0, 0.0
    if not g_list:
        return 0.0, 1.0
    # Match each pred against any gold, each gold against any pred.
    matched_pred = sum(1 for p in p_list if any(_citation_match(g, p) for g in g_list))
    matched_gold = sum(1 for g in g_list if any(_citation_match(g, p) for p in p_list))
    precision = matched_pred / len(p_list)
    recall = matched_gold / len(g_list)
    return precision, recall


def recall_at_k(*, gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of gold items that appear in the top-k retrieved items."""
    g = set(gold)
    if not g:
        return 1.0
    top = set(retrieved[:k])
    return len(g & top) / len(g)
