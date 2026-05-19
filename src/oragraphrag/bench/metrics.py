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


def citation_pr(*, gold: Sequence[str], pred: Sequence[str]) -> tuple[float, float]:
    """Return (precision, recall) over predicted vs gold citation sets.

    Sentinels:
    - Empty gold AND empty pred → (1.0, 1.0). Vacuously perfect.
    - Empty pred, nonempty gold → (1.0, 0.0). No false positives possible.
    - Empty gold, nonempty pred → (0.0, 1.0). No truth to miss.
    """
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0, 1.0
    if not p:
        return 1.0, 0.0
    if not g:
        return 0.0, 1.0
    precision = len(g & p) / len(p)
    recall = len(g & p) / len(g)
    return precision, recall


def recall_at_k(*, gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of gold items that appear in the top-k retrieved items."""
    g = set(gold)
    if not g:
        return 1.0
    top = set(retrieved[:k])
    return len(g & top) / len(g)
