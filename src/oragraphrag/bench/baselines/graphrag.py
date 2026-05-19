"""Microsoft GraphRAG baseline.

GraphRAG (v3.x) builds indexer artifacts (entities, communities, reports,
text units, relationships) on a one-time index pass; the bench harness
plugs into the query side (``LocalSearch``). For this baseline to actually
return answers, the indexer must have already produced artifacts under
``benchmarks/configs/graphrag/output``:

    python -m graphrag.index --root benchmarks/configs/graphrag

This runner defers all graphrag imports (heavy: spacy, transformers,
lancedb, etc.) until first call. It raises a clear ``FileNotFoundError``
when artifacts are missing — telling bench operators exactly what to do —
and otherwise loads them and dispatches to ``LocalSearch``. The full
``LocalSearch`` wiring (LocalContextBuilder + model adapter) is documented
in ``graphrag.py.disabled`` and must be uncommented once a corpus index
exists.
"""

from __future__ import annotations

from pathlib import Path

from oragraphrag.config import Config

ARTIFACTS_ROOT = Path("benchmarks/configs/graphrag/output")


async def run(question: str, cfg: Config) -> dict:
    try:
        # Defer: graphrag has very heavy transitive deps (spacy, lancedb,
        # onnxruntime, azure-*) and we don't want to pay for them on every
        # bench import — only when this baseline is actually selected.
        from graphrag.query.indexer_adapters import (  # noqa: F401
            read_indexer_entities,
            read_indexer_reports,
        )
    except ImportError as e:
        raise ImportError(
            "graphrag is not installed. Install with `pip install graphrag` "
            "and run `python -m graphrag.index --root benchmarks/configs/graphrag` once."
        ) from e

    if not ARTIFACTS_ROOT.exists():
        raise FileNotFoundError(
            f"GraphRAG artifacts not found at {ARTIFACTS_ROOT}. "
            "Run `python -m graphrag.index --root benchmarks/configs/graphrag` once "
            "to build them, then re-run the bench. The full LocalSearch wiring "
            "(model adapter + LocalContextBuilder) is drafted in graphrag.py.disabled."
        )

    # Artifacts present: full LocalSearch wiring lives in graphrag.py.disabled;
    # uncomment and adapt it to your model adapter before enabling.
    raise NotImplementedError(
        "GraphRAG artifacts directory exists but LocalSearch wiring is staged in "
        "graphrag.py.disabled. Port it into this module once your model adapter "
        "and LocalContextBuilder are configured."
    )
