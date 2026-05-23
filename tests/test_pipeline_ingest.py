import numpy as np
import pytest

from oragraphrag.config import Config
from oragraphrag.graph import GraphStoreError
from oragraphrag.ingest import Buffer
from oragraphrag.ingest_records import IngestWriteStats
from oragraphrag.pipeline_ingest import IngestPipeline


class _StubExtractor:
    """Returns the same proposition shape for every buffer."""

    def __init__(self):
        self.calls: list[str] = []

    async def extract(self, passage: str) -> dict:
        self.calls.append(passage)
        return {
            "propositions": [
                {
                    "text": passage[:30],
                    "triples": [
                        {
                            "subject": "alpha",
                            "predicate": "rel",
                            "object": "beta",
                            "ontology_axis": "causal",
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
        }


class _StubEmbedder:
    dim = 8

    async def embed(self, texts, *, normalize: bool = True):
        return np.ones((len(texts), 8), dtype=np.float32)


class _SpyGraph:
    """In-memory stand-in for GraphStore."""

    def __init__(self):
        self.entities: dict[str, bytes] = {}
        self.props: list[bytes] = []
        self.rels: list[tuple] = []
        self.ledger: set[str] = set()
        self.ingest_units = []

    def upsert_entity(self, *, name, kind, embedding, source_id="default"):
        eid = self.entities.setdefault(name, f"E{len(self.entities)}".encode())
        return eid

    def upsert_proposition(
        self, *, text, source_doc, source_span, embedding, source_id="default"
    ):
        pid = f"P{len(self.props)}".encode()
        self.props.append(pid)
        return pid

    def upsert_rel(
        self,
        src,
        dst,
        *,
        predicate,
        ontology_axis,
        base_weight,
        support_prop_id,
        source_id="default",
    ):
        self.rels.append(
            (src, dst, predicate, ontology_axis, base_weight, support_prop_id, source_id)
        )

    def ledger_has(self, h):
        return h in self.ledger

    def ledger_add(self, h, *, doc_id, section_path):
        self.ledger.add(h)

    def ingest_buffer(self, unit):
        self.ingest_units.append(unit)
        entity_ids = {
            entity.name: self.upsert_entity(
                name=entity.name,
                kind=entity.kind,
                embedding=entity.embedding,
                source_id=unit.source_id,
            )
            for entity in unit.entities
        }
        rel_count = 0
        for prop in unit.propositions:
            pid = self.upsert_proposition(
                text=prop.text,
                source_doc=unit.doc_id,
                source_span=unit.section_path,
                embedding=prop.embedding,
                source_id=unit.source_id,
            )
            for triple in prop.triples:
                base = min(1.0, 0.5 + 0.5 * triple.confidence)
                self.upsert_rel(
                    entity_ids[triple.subject],
                    entity_ids[triple.object],
                    predicate=triple.predicate,
                    ontology_axis=triple.ontology_axis,
                    base_weight=base,
                    support_prop_id=pid,
                    source_id=unit.source_id,
                )
                rel_count += 1
        for span_hash in unit.span_hashes:
            self.ledger_add(span_hash, doc_id=unit.doc_id, section_path=unit.section_path)
        return IngestWriteStats(
            entities=len(unit.entities),
            propositions=len(unit.propositions),
            rels=rel_count,
        )


@pytest.mark.asyncio
async def test_pipeline_inserts_entities_props_and_rels():
    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=_StubExtractor())
    buf = Buffer(
        doc_id="d.md",
        section_path="Hello",
        text="alpha causes beta.",
        span_hashes=["h1"],
    )
    stats = await p.run([buf])
    assert set(g.entities) == {"alpha", "beta"}
    assert len(g.props) == 1
    assert len(g.rels) == 1
    assert g.rels[0][2] == "rel"
    assert g.rels[0][3] == "causal"
    assert stats["entities"] == 2
    assert stats["propositions"] == 1
    assert stats["rels"] == 1
    assert stats["skipped"] == 0


@pytest.mark.asyncio
async def test_pipeline_delegates_graph_writes_to_one_ingest_unit():
    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    p = IngestPipeline(
        cfg=cfg,
        graph=g,
        embedder=_StubEmbedder(),
        extractor=_StubExtractor(),
        source_id="src_test",
    )
    buf = Buffer(
        doc_id="d.md",
        section_path="Hello",
        text="alpha causes beta.",
        span_hashes=["h1"],
    )

    await p.run([buf])

    assert len(g.ingest_units) == 1
    unit = g.ingest_units[0]
    assert unit.doc_id == "d.md"
    assert unit.section_path == "Hello"
    assert unit.span_hashes == ("h1",)
    assert unit.source_id == "src_test"
    assert [entity.name for entity in unit.entities] == ["alpha", "beta"]
    assert [prop.text for prop in unit.propositions] == ["alpha causes beta."]


@pytest.mark.asyncio
async def test_pipeline_skips_already_ingested_spans():
    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    g.ledger.add("h1")
    extractor = _StubExtractor()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=extractor)
    buf = Buffer(
        doc_id="d.md",
        section_path="Hello",
        text="alpha causes beta.",
        span_hashes=["h1"],
    )
    stats = await p.run([buf])
    assert g.props == []
    assert g.rels == []
    assert extractor.calls == []  # never called the LLM
    assert stats["skipped"] == 1


@pytest.mark.asyncio
async def test_pipeline_skips_buffer_only_if_all_hashes_ledgered():
    """A buffer that contains an overlap-carry-forward hash (from Task 6 fix)
    PLUS a new hash should NOT skip — the new content needs extraction."""
    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    g.ledger.add("h1")  # only the carry-forward hash is ledgered
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=_StubExtractor())
    buf = Buffer(
        doc_id="d.md",
        section_path="Hello",
        text="alpha causes beta. and more.",
        span_hashes=["h1", "h2"],
    )
    stats = await p.run([buf])
    assert stats["skipped"] == 0
    assert len(g.props) == 1
    # Both hashes are recorded after successful ingest.
    assert "h1" in g.ledger
    assert "h2" in g.ledger


@pytest.mark.asyncio
async def test_pipeline_ledgers_spans_after_successful_ingest():
    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=_StubExtractor())
    buf = Buffer(
        doc_id="d.md",
        section_path="Hello",
        text="t",
        span_hashes=["h1", "h2"],
    )
    await p.run([buf])
    assert "h1" in g.ledger
    assert "h2" in g.ledger


@pytest.mark.asyncio
async def test_pipeline_empty_propositions_still_ledgers_spans():
    """If the LLM returns {'propositions': []} (no extractable content),
    the spans are still marked ingested so we don't re-process them."""

    class _EmptyExtractor:
        async def extract(self, passage):
            return {"propositions": []}

    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    p = IngestPipeline(
        cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=_EmptyExtractor()
    )
    buf = Buffer(
        doc_id="d.md",
        section_path="Hello",
        text="just whitespace",
        span_hashes=["h_empty"],
    )
    stats = await p.run([buf])
    assert g.props == []
    assert g.rels == []
    assert "h_empty" in g.ledger
    assert stats["propositions"] == 0


@pytest.mark.asyncio
async def test_pipeline_canonicalizes_repeated_entities():
    """The same entity name across propositions should produce one Entity row."""

    class _MultiExtractor:
        async def extract(self, passage):
            return {
                "propositions": [
                    {
                        "text": "first",
                        "triples": [
                            {
                                "subject": "alpha",
                                "predicate": "rel",
                                "object": "beta",
                                "ontology_axis": "causal",
                                "confidence": 0.9,
                            },
                            {
                                "subject": "alpha",
                                "predicate": "rel2",
                                "object": "gamma",
                                "ontology_axis": "taxonomic",
                                "confidence": 0.8,
                            },
                        ],
                    }
                ]
            }

    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=_MultiExtractor())
    buf = Buffer(doc_id="d.md", section_path="s", text="x", span_hashes=["h"])
    await p.run([buf])
    # 'alpha' appears in two triples but only one Entity row.
    assert set(g.entities) == {"alpha", "beta", "gamma"}
    assert len(g.rels) == 2


@pytest.mark.asyncio
async def test_pipeline_concurrency_bounded_by_config():
    """Verify the pipeline respects cfg.ingest.extract_concurrency."""

    import asyncio

    class _SlowExtractor:
        def __init__(self):
            self.peak_concurrent = 0
            self.active = 0
            self.lock = asyncio.Lock()

        async def extract(self, passage):
            async with self.lock:
                self.active += 1
                self.peak_concurrent = max(self.peak_concurrent, self.active)
            await asyncio.sleep(0.01)
            async with self.lock:
                self.active -= 1
            return {"propositions": []}

    cfg = Config()
    cfg.embeddings.dim = 8
    cfg.ingest.extract_concurrency = 3
    g = _SpyGraph()
    extractor = _SlowExtractor()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=extractor)
    bufs = [
        Buffer(doc_id="d", section_path="s", text=f"t{i}", span_hashes=[f"h{i}"])
        for i in range(20)
    ]
    await p.run(bufs)
    assert extractor.peak_concurrent <= 3


@pytest.mark.asyncio
async def test_pipeline_one_bad_buffer_does_not_abort_the_rest(tmp_path, monkeypatch):
    """Spec §11: on extractor failure, log to logs/extract-failures.jsonl
    and continue with the next buffer."""

    monkeypatch.chdir(tmp_path)

    class _FlakyExtractor:
        """First call raises ExtractionError; subsequent calls return empty."""

        def __init__(self):
            self.calls = 0

        async def extract(self, passage):
            self.calls += 1
            if self.calls == 1:
                from oragraphrag.extract import ExtractionError

                raise ExtractionError("synthetic")
            return {"propositions": []}

    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    extractor = _FlakyExtractor()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=extractor)
    bufs = [
        Buffer(doc_id="d.md", section_path="s", text=f"t{i}", span_hashes=[f"h{i}"])
        for i in range(3)
    ]
    stats = await p.run(bufs)

    # All 3 buffers processed; 1 failed, 2 succeeded with empty propositions.
    assert stats["buffers"] == 3
    assert stats["failed"] == 1
    assert extractor.calls == 3
    # The two successful buffers' spans are ledgered; the failed one's are NOT.
    assert "h1" in g.ledger
    assert "h2" in g.ledger
    assert "h0" not in g.ledger

    # The failure log file exists and contains one JSONL record.
    log_path = tmp_path / "logs" / "extract-failures.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    import json as _json
    rec = _json.loads(lines[0])
    assert rec["doc_id"] == "d.md"
    assert rec["span_hashes"] == ["h0"]
    assert rec["error_type"] == "ExtractionError"
    assert "synthetic" in rec["error_message"]


@pytest.mark.asyncio
async def test_pipeline_graph_store_error_does_not_ledger_failed_buffer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _FailingGraph(_SpyGraph):
        def ingest_buffer(self, unit):
            raise GraphStoreError("write failed")

    cfg = Config()
    cfg.embeddings.dim = 8
    g = _FailingGraph()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=_StubEmbedder(), extractor=_StubExtractor())
    buf = Buffer(doc_id="d.md", section_path="s", text="t", span_hashes=["h"])

    stats = await p.run([buf])

    assert stats["failed"] == 1
    assert "h" not in g.ledger


@pytest.mark.asyncio
async def test_pipeline_combines_embeddings_into_single_batch():
    """Propositions and entities should be embedded in ONE backend call."""

    class _CountingEmbedder:
        dim = 8

        def __init__(self):
            self.call_count = 0

        async def embed(self, texts, *, normalize: bool = True):
            self.call_count += 1
            return np.ones((len(texts), 8), dtype=np.float32)

    cfg = Config()
    cfg.embeddings.dim = 8
    g = _SpyGraph()
    embedder = _CountingEmbedder()
    p = IngestPipeline(cfg=cfg, graph=g, embedder=embedder, extractor=_StubExtractor())
    buf = Buffer(doc_id="d", section_path="s", text="t", span_hashes=["h"])
    await p.run([buf])
    assert embedder.call_count == 1
