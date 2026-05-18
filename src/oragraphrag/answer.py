"""Render the answer prompt, call the LLM, map [P#] citations back to propositions.

Task 11's query orchestrator calls `Answerer.answer(question, propositions)`
after retrieving propositions from the graph walk. Returns an AnswerResult
with the LLM's text and a list of Citation objects that map [P#] markers
in the text back to the (proposition_id, source_doc, source_span) triple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from jinja2 import Template


@dataclass(slots=True)
class Citation:
    proposition_id: bytes
    source_doc: str
    source_span: str


@dataclass(slots=True)
class AnswerResult:
    text: str
    citations: list[Citation]


_NO_INFO = "I don't have information about that in the indexed corpus."


def _normalize_pid(pid: Any) -> bytes:
    """Accept bytes or hex string. Returns bytes."""
    if isinstance(pid, bytes):
        return pid
    if isinstance(pid, str):
        try:
            return bytes.fromhex(pid)
        except ValueError:
            return pid.encode()
    return bytes(pid)


class Answerer:
    """Renders the answer prompt, calls the LLM, parses [P#] citations."""

    def __init__(
        self,
        *,
        llm: Any,
        token_budget: int,
        template_text: str | None = None,
    ) -> None:
        self._llm = llm
        self._budget = token_budget
        text = template_text or (
            files("oragraphrag.prompts").joinpath("answer.j2").read_text()
        )
        self._tmpl = Template(text)

    def _truncate_to_budget(self, question: str, propositions: list[dict]) -> list[dict]:
        """Drop trailing propositions until the rendered prompt fits the budget.

        Uses a char/4 token estimate consistent with Task 6's ingest buffer
        sizing. Returns the (possibly shortened) propositions list; the input
        is not mutated.
        """
        budget_chars = self._budget * 4  # char/4 ≈ tokens
        # Cheap upper bound: question + per-prop overhead + each prop text.
        overhead_chars = len(question) + 200  # template scaffold ~ 50 tokens
        cumulative = overhead_chars
        keep: list[dict] = []
        for p in propositions:
            prop_chars = len(p.get("text", "")) + 100  # template per-prop scaffold
            if cumulative + prop_chars > budget_chars and keep:
                break
            cumulative += prop_chars
            keep.append(p)
        return keep

    async def answer(
        self,
        *,
        question: str,
        propositions: list[dict],
    ) -> AnswerResult:
        if not propositions:
            return AnswerResult(text=_NO_INFO, citations=[])

        # Truncate from the tail if the rendered prompt would exceed the budget.
        # Propositions arrive in score order from Task 9, so the tail carries the
        # weakest evidence — safe to drop. The char/4 token heuristic mirrors
        # Task 6's ingest buffering.
        propositions = self._truncate_to_budget(question, propositions)

        items = [
            {
                "id_hex": (
                    p["id"].hex() if isinstance(p["id"], bytes) else str(p["id"])
                ),
                "text": p["text"],
                "source_doc": p.get("source_doc", ""),
                "source_span": p.get("source_span", ""),
            }
            for p in propositions
        ]
        prompt = self._tmpl.render(question=question, propositions=items)
        text = await self._llm.complete(prompt, temperature=0.0)
        if not isinstance(text, str):
            text = str(text)

        citations: list[Citation] = []
        # Accept both singleton citations [P1] and grouped citations [P1, P2; P3].
        # The outer regex captures the whole bracketed group; the inner regex pulls
        # out each Pn number. Local LLMs commonly emit grouped citations and we
        # don't want them silently dropped.
        for m in re.finditer(r"\[(P\d+(?:\s*[,;]\s*P\d+)*)\]", text):
            for num_str in re.findall(r"P(\d+)", m.group(1)):
                n = int(num_str)
                if 1 <= n <= len(propositions):
                    p = propositions[n - 1]
                    citations.append(
                        Citation(
                            proposition_id=_normalize_pid(p["id"]),
                            source_doc=p.get("source_doc", ""),
                            source_span=p.get("source_span", ""),
                        )
                    )
        return AnswerResult(text=text, citations=citations)
