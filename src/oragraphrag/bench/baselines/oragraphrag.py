"""OraGraphRAG baseline runner — wires the live system into the bench harness."""

from __future__ import annotations

from oragraphrag.config import Config


async def run(question: str, cfg: Config) -> dict:
    """Run one question through the full OraGraphRAG pipeline.

    Returns {answer, citations, latency_ms, tokens}. Caller (the bench
    runner) wraps the per-question call in latency timing and try/except.
    """
    from oragraphrag.embed import Embedder, build_axis_vectors
    from oragraphrag.embed_backends import build_embed_backend
    from oragraphrag.graph import GraphStore
    from oragraphrag.llm import LLM
    from oragraphrag.pipeline_query import QueryPipeline

    store = GraphStore(cfg)
    store.connect()
    try:
        emb = Embedder(cfg, backend=build_embed_backend(cfg, store))
        axes = await build_axis_vectors(emb)
        async with LLM(cfg) as llm:
            pipeline = QueryPipeline(
                cfg=cfg, graph=store, embedder=emb, llm=llm, axis_vectors=axes
            )
            result = await pipeline.query(question)
            return {
                "answer": result.answer.text,
                "citations": [
                    f"{c.source_doc}#{c.source_span}" for c in result.answer.citations
                ],
                "latency_ms": float(sum(result.latency_ms.values())),
                "tokens": len(result.answer.text.split()),  # rough; refine in Task 16
            }
    finally:
        store.close()
