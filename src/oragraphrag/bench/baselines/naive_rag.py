"""Naive RAG baseline: fixed-chunk vector retrieval + same-LLM answer.

Uses the same Oracle 23ai vector store and the same LLM adapter so the
comparison isolates retrieval quality, not LLM identity. Chunks are the
existing Proposition rows from Task 5's schema — naive RAG operates on
proposition-level retrieval without the graph walk.
"""

from __future__ import annotations

from oragraphrag.config import Config


async def run(question: str, cfg: Config) -> dict:
    from oragraphrag.embed import Embedder
    from oragraphrag.embed_backends import build_embed_backend
    from oragraphrag.graph import GraphStore
    from oragraphrag.llm import LLM

    store = GraphStore(cfg)
    store.connect()
    try:
        emb = Embedder(cfg, backend=build_embed_backend(cfg, store))
        q_vec = (await emb.embed([question]))[0]
        seeds = store.vector_search_propositions(query_vec=q_vec.tolist(), k=10)
        ids = [s["id"] for s in seeds]
        fetched = store.fetch_propositions(ids)
        context = "\n".join(
            f"- {p['text']} (src={p['source_doc']}#{p['source_span']})"
            for p in fetched
        )
        prompt = (
            "Answer the question STRICTLY from the context below. Cite sources "
            "as source_doc#source_span in your answer. If the context does not "
            "contain enough information, say so explicitly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer:"
        )
        async with LLM(cfg) as llm:
            answer = await llm.complete(prompt, temperature=0.0)
        if not isinstance(answer, str):
            answer = str(answer)
        citations = list({f"{p['source_doc']}#{p['source_span']}" for p in fetched})
        return {
            "answer": answer,
            "citations": citations,
            "latency_ms": 0.0,  # runner wraps with timing
            "tokens": len(prompt.split()),
        }
    finally:
        store.close()
