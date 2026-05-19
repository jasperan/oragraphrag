"""LightRAG baseline (dual-level retrieval).

LightRAG keeps its own working directory (KV store, vector store, graph
storage) under ``benchmarks/configs/lightrag/work``. The directory must
have been populated by a prior ``rag.ainsert(...)`` pass over the same
Oracle 23ai docs slice as the rest of the bench.

This runner defers the lightrag import (heavy: aiohttp, google-genai,
nano-vectordb, ascii-colors) until first call and raises a clear
``FileNotFoundError`` if the working directory is missing.
"""

from __future__ import annotations

from pathlib import Path

from oragraphrag.config import Config

WORKING_DIR = Path("benchmarks/configs/lightrag/work")


async def run(question: str, cfg: Config) -> dict:
    try:
        from lightrag import LightRAG, QueryParam
    except ImportError as e:
        raise ImportError(
            "lightrag-hku is not installed. Install with `pip install lightrag-hku`."
        ) from e

    if not WORKING_DIR.exists():
        raise FileNotFoundError(
            f"LightRAG working dir not found at {WORKING_DIR}. Insert the corpus "
            "first via `await rag.ainsert(docs)` against the same Oracle 23ai slice "
            "used by the rest of the bench. See lightrag.py.disabled for a wiring "
            "template that binds LightRAG to our LLM and embedding backends."
        )

    rag = LightRAG(working_dir=str(WORKING_DIR))
    # LightRAG validates that the working dir was populated by a prior
    # insert; if not, this aquery will surface a runtime error — which the
    # bench harness will record as the system's failure mode.
    result = await rag.aquery(question, param=QueryParam(mode="hybrid", top_k=10))
    return {
        "answer": str(result),
        "citations": [],  # LightRAG's response shape doesn't include structured citations.
        "tokens": 0,
        "latency_ms": 0.0,
    }
