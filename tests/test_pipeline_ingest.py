import numpy as np
import pytest

from oragraphrag.config import Config
from oragraphrag.ingest import Buffer
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

    def upsert_entity(self, *, name, kind, embedding):
        eid = self.entities.setdefault(name, f"E{len(self.entities)}".encode())
        return eid

    def upsert_proposition(self, *, text, source_doc, source_span, embedding):
        pid = f"P{len(self.props)}".encode()
        self.props.append(pid)
        return pid

    def upsert_rel(self, src, dst, *, predicate, ontology_axis, base_weight, support_prop_id):
        self.rels.append((src, dst, predicate, ontology_axis, base_weight, support_prop_id))

    def ledger_has(self, h):
        return h in self.ledger

    def ledger_add(self, h, *, doc_id, section_path):
        self.ledger.add(h)


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
