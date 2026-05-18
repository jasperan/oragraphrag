"""End-to-end query: embed -> seeds -> subgraph -> reweight -> ppr -> assemble -> answer.

Task 12's CLI calls `QueryPipeline.query(question)` per user query. The
pipeline reads the ontology axis vectors once at construction (loaded from
Oracle by the caller) so subsequent queries don't re-embed the canonical
axis descriptions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from oragraphrag.answer import Answerer, AnswerResult
from oragraphrag.config import Config
from oragraphrag.retrieve import (
    assemble_propositions,
    compute_amplitudes,
    reweight_edges,
    spreading_activation,
)


@dataclass(slots=True)
class QueryResult:
    answer: AnswerResult
    amplitudes: dict[str, float]
    edges_used: list[dict]
    latency_ms: dict[str, float]


class QueryPipeline:
    def __init__(
        self,
        *,
        cfg: Config,
        graph: Any,
        embedder: Any,
        llm: Any,
        axis_vectors: dict[str, np.ndarray],
    ) -> None:
        self.cfg = cfg
        self.graph = graph
        self.embedder = embedder
        self.llm = llm
        self.axis_vectors = axis_vectors
        self._answerer = Answerer(llm=llm, token_budget=cfg.answer.token_budget)

    async def query(self, question: str) -> QueryResult:
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        q_emb = (await self.embedder.embed([question]))[0]
        t["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ent_seeds = self.graph.vector_search_entities(
            query_vec=q_emb.tolist(),
            k=self.cfg.retrieval.seed_k_entities,
        )
        prop_seeds = self.graph.vector_search_propositions(
            query_vec=q_emb.tolist(),
            k=self.cfg.retrieval.seed_k_propositions,
        )
        t["seed_ms"] = (time.perf_counter() - t0) * 1000

        seed_ids = [s["id"] for s in ent_seeds]
        # Map proposition id -> similarity (1 - distance); used by assemble_propositions.
        seed_sims = {p["id"]: 1.0 - float(p["distance"]) for p in prop_seeds}

        t0 = time.perf_counter()
        edges = self.graph.pgql_subgraph(
            seed_ids=seed_ids,
            max_edges=self.cfg.retrieval.max_subgraph_edges,
        )
        t["subgraph_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        amps = compute_amplitudes(
            q_emb,
            self.axis_vectors,
            alpha=self.cfg.retrieval.amplitude.alpha,
            beta=self.cfg.retrieval.amplitude.beta,
        )
        # Degenerate-amplitude fallback (spec §6 Step 3 + §11). If every
        # axis amplitude is saturated at one extreme, the query carries no
        # differential signal across axes -- fall back to uniform 1.0 so
        # PR runs on base weights alone.
        if all(a < 0.05 for a in amps.values()) or all(a > 0.95 for a in amps.values()):
            amps = dict.fromkeys(amps, 1.0)
        reweighted = reweight_edges(edges, amps)
        t["reweight_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        activations = spreading_activation(
            reweighted,
            seed_ids=seed_ids,
            damping=self.cfg.retrieval.pagerank.damping,
        )
        t["ppr_ms"] = (time.perf_counter() - t0) * 1000

        picked = assemble_propositions(
            reweighted,
            activations,
            seed_sims,
            top_m=self.cfg.retrieval.pagerank.top_m_entities,
        )
        props = self.graph.fetch_propositions(picked) if picked else []

        t0 = time.perf_counter()
        ans = await self._answerer.answer(question=question, propositions=props)
        t["answer_ms"] = (time.perf_counter() - t0) * 1000

        return QueryResult(
            answer=ans,
            amplitudes=amps,
            edges_used=reweighted,
            latency_ms=t,
        )
