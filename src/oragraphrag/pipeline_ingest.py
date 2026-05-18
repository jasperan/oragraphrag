"""Orchestrates: buffers -> extract -> canonicalize entities -> embed -> graph upserts.

Idempotency lives in the IngestLedger (Task 5 schema): each buffer's span
hashes are checked before extraction; on success, all hashes are appended.
A buffer is skipped only when ALL its hashes are already ledgered.

Concurrency is bounded by `cfg.ingest.extract_concurrency` via an asyncio
Semaphore on the LLM-calling section. Graph upserts after extraction also
run inside the semaphore so we don't overrun the DB connection pool
(cfg.oracle.pool_max).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import numpy as np

from oragraphrag.config import Config
from oragraphrag.ingest import Buffer


class IngestPipeline:
    """Runs the extract+embed+upsert pipeline over a stream of Buffers."""

    def __init__(self, *, cfg: Config, graph: Any, embedder: Any, extractor: Any) -> None:
        self.cfg = cfg
        self.graph = graph
        self.embedder = embedder
        self.extractor = extractor

    async def run(self, buffers: Iterable[Buffer]) -> dict[str, int]:
        stats = {
            "buffers": 0,
            "skipped": 0,
            "propositions": 0,
            "rels": 0,
            "entities": 0,
        }
        sem = asyncio.Semaphore(self.cfg.ingest.extract_concurrency)

        async def process(buf: Buffer) -> None:
            stats["buffers"] += 1
            # Skip if EVERY span hash in the buffer is already ledgered.
            # A partially-ledgered buffer (overlap carry-forward + new content)
            # still needs extraction.
            if buf.span_hashes and all(self.graph.ledger_has(h) for h in buf.span_hashes):
                stats["skipped"] += 1
                return
            async with sem:
                await self._process_one(buf, stats)

        await asyncio.gather(*(process(b) for b in buffers))
        return stats

    async def _process_one(self, buf: Buffer, stats: dict[str, int]) -> None:
        payload = await self.extractor.extract(buf.text)
        propositions = payload.get("propositions", [])

        if not propositions:
            # No extractable content; still ledger so we don't reprocess.
            for h in buf.span_hashes:
                self.graph.ledger_add(h, doc_id=buf.doc_id, section_path=buf.section_path)
            return

        # Collect distinct entity surface forms across all triples.
        entity_set: list[str] = []
        seen: set[str] = set()
        for p in propositions:
            for t in p["triples"]:
                for k in (t["subject"], t["object"]):
                    if k not in seen:
                        seen.add(k)
                        entity_set.append(k)

        # Embed propositions and entities in two batches.
        prop_texts = [p["text"] for p in propositions]
        prop_embs = await self.embedder.embed(prop_texts)
        entity_embs = (
            await self.embedder.embed(entity_set)
            if entity_set
            else np.empty((0, self.cfg.embeddings.dim), dtype=np.float32)
        )

        # Upsert entities first so we have ids for the rel inserts.
        entity_ids: dict[str, bytes] = {}
        for name, vec in zip(entity_set, entity_embs, strict=True):
            eid = self.graph.upsert_entity(name=name, kind="concept", embedding=vec.tolist())
            entity_ids[name] = eid
            stats["entities"] += 1

        # Upsert each proposition, then its edges.
        for prop, vec in zip(propositions, prop_embs, strict=True):
            pid = self.graph.upsert_proposition(
                text=prop["text"],
                source_doc=buf.doc_id,
                source_span=buf.section_path,
                embedding=vec.tolist(),
            )
            stats["propositions"] += 1
            for t in prop["triples"]:
                s_id = entity_ids[t["subject"]]
                o_id = entity_ids[t["object"]]
                # base_weight starts as a confidence-anchored value; Task 9
                # reweights at query time so the absolute scale here is less
                # important than monotonicity in confidence.
                base = 0.5 + 0.5 * float(t["confidence"])
                self.graph.upsert_rel(
                    s_id,
                    o_id,
                    predicate=t["predicate"],
                    ontology_axis=t["ontology_axis"],
                    base_weight=min(1.0, base),
                    support_prop_id=pid,
                )
                stats["rels"] += 1

        # Ledger only after all upserts succeed.
        for h in buf.span_hashes:
            self.graph.ledger_add(h, doc_id=buf.doc_id, section_path=buf.section_path)
