"""Fine-tune corpus exporter.

Iterates every Proposition + its supporting REL edges and emits a JSONL
training corpus shaped for SFT-style fine-tuning. Question synthesis is
template-based to avoid a per-row LLM round-trip; the resulting prompts
can be reused as evals or rewritten by a downstream LLM if richer
questions are wanted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_training_example(
    prop_row: dict,
    edge_rows: list[dict],
) -> dict[str, Any]:
    """Compose one JSONL row from a proposition + its supporting edges.

    ``prop_row`` carries id, text, source_doc, source_span, source_id.
    ``edge_rows`` is the list of REL rows whose support_propositions
    contains this proposition's id.
    """
    axes = sorted({e["ontology_axis"] for e in edge_rows if e.get("ontology_axis")})
    predicates = sorted({e["predicate"] for e in edge_rows if e.get("predicate")})

    # Question synthesis: prefer the first edge's predicate for a concrete
    # noun-anchored prompt; otherwise fall back to a generic "state the
    # fact" framing.
    subject = None
    for e in edge_rows:
        if e.get("predicate"):
            subject = e["predicate"]
            break

    text = prop_row.get("text", "") or ""
    if subject:
        prompt = (
            "Answer this question grounded in Oracle 23ai docs: "
            f"What does the corpus say about {subject}?"
        )
    else:
        snippet = text[:60].rstrip(".") + ("..." if len(text) > 60 else "")
        prompt = f"State the fact: {snippet}"

    pid_raw = prop_row.get("id")
    pid_hex = (
        bytes(pid_raw).hex() if isinstance(pid_raw, (bytes, bytearray)) else str(pid_raw)
    )

    source = f"{prop_row.get('source_doc', '')}#{prop_row.get('source_span', '')}"

    return {
        "id": pid_hex,
        "source": source,
        "source_id": prop_row.get("source_id", "default"),
        "prompt": prompt,
        "completion": text,
        "axes": axes,
        "predicates": predicates,
    }


def export_finetune(
    store: Any,
    out_path: Path,
    *,
    source_filter: str | None = None,
) -> int:
    """Stream every Proposition into a JSONL file. Returns the row count.

    Optionally filters to a single source_id.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = store.iter_propositions_with_edges(source_filter=source_filter)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for prop_row, edge_rows in rows:
            example = build_training_example(prop_row, edge_rows)
            f.write(json.dumps(example) + "\n")
            count += 1
    return count
