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
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oragraphrag.config import Config
from oragraphrag.extract import ExtractionError
from oragraphrag.graph import GraphStoreError
from oragraphrag.ingest import Buffer
from oragraphrag.ingest_records import (
    IngestEntity,
    IngestProposition,
    IngestTriple,
    IngestUnit,
)
from oragraphrag.llm import LLMError

_FAILURE_LOG_PATH = Path("logs") / "extract-failures.jsonl"

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Runs the extract+embed+upsert pipeline over a stream of Buffers."""

    def __init__(
        self,
        *,
        cfg: Config,
        graph: Any,
        embedder: Any,
        extractor: Any,
        source_id: str = "default",
    ) -> None:
        self.cfg = cfg
        self.graph = graph
        self.embedder = embedder
        self.extractor = extractor
        self.source_id = source_id

    async def run(self, buffers: Iterable[Buffer]) -> dict[str, int]:
        stats = {
            "buffers": 0,
            "skipped": 0,
            "failed": 0,
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
                try:
                    await self._process_one(buf, stats)
                except (ExtractionError, LLMError, GraphStoreError) as e:
                    stats["failed"] += 1
                    self._log_failure(buf, e)
                    # Continue — one bad buffer must not abort the whole ingest.

        await asyncio.gather(*(process(b) for b in buffers))
        return stats

    def _log_failure(self, buf: Buffer, exc: Exception) -> None:
        """Append a JSONL record to logs/extract-failures.jsonl. Best-effort: if
        the log file cannot be written, log a warning and continue.
        """
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "doc_id": buf.doc_id,
            "section_path": buf.section_path,
            "span_hashes": list(buf.span_hashes),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        try:
            _FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _FAILURE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as log_err:
            logger.warning("could not write to %s: %s", _FAILURE_LOG_PATH, log_err)

    async def _process_one(self, buf: Buffer, stats: dict[str, int]) -> None:
        payload = await self.extractor.extract(buf.text)
        propositions = payload.get("propositions", [])

        if not propositions:
            # No extractable content; still ledger so we don't reprocess.
            self.graph.ingest_buffer(
                IngestUnit(
                    doc_id=buf.doc_id,
                    section_path=buf.section_path,
                    span_hashes=tuple(buf.span_hashes),
                    source_id=self.source_id,
                    entities=(),
                    propositions=(),
                )
            )
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

        # Single batch call to halve embedding round-trips. The schema validator
        # guarantees propositions is non-empty here (we returned at line 65 otherwise)
        # and each triple has subject + object, so entity_set is also non-empty.
        prop_texts = [p["text"] for p in propositions]
        combined = prop_texts + entity_set
        combined_embs = await self.embedder.embed(combined)
        prop_embs = combined_embs[: len(prop_texts)]
        entity_embs = combined_embs[len(prop_texts) :]

        entities = tuple(
            IngestEntity(
                name=name,
                kind="concept",
                embedding=vec.tolist(),
            )
            for name, vec in zip(entity_set, entity_embs, strict=True)
        )
        ingest_props = tuple(
            IngestProposition(
                text=prop["text"],
                embedding=vec.tolist(),
                triples=tuple(
                    IngestTriple(
                        subject=t["subject"],
                        object=t["object"],
                        predicate=t["predicate"],
                        ontology_axis=t["ontology_axis"],
                        confidence=float(t["confidence"]),
                    )
                    for t in prop["triples"]
                ),
            )
            for prop, vec in zip(propositions, prop_embs, strict=True)
        )

        written = self.graph.ingest_buffer(
            IngestUnit(
                doc_id=buf.doc_id,
                section_path=buf.section_path,
                span_hashes=tuple(buf.span_hashes),
                source_id=self.source_id,
                entities=entities,
                propositions=ingest_props,
            )
        )
        stats["entities"] += written.entities
        stats["propositions"] += written.propositions
        stats["rels"] += written.rels
