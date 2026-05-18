import numpy as np
import pytest

from oragraphrag.config import Config
from oragraphrag.pipeline_query import QueryPipeline, QueryResult


class _StubEmbedder:
    dim = 5

    async def embed(self, texts, *, normalize: bool = True):
        return np.array([[1.0, 0, 0, 0, 0]] * len(texts), dtype=np.float32)


class _StubGraph:
    """In-memory stand-in for GraphStore. Records call args for inspection."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.seed_entities_result = [{"id": b"\x01", "name": "a", "distance": 0.01}]
        self.seed_props_result = [
            {"id": b"\xaa", "source_doc": "d", "source_span": "s", "distance": 0.02}
        ]
        self.subgraph_result = [
            {
                "src": b"\x01",
                "dst": b"\x02",
                "predicate": "causes",
                "ontology_axis": "causal",
                "base_weight": 0.8,
                "support_propositions": [b"\xaa".hex()],
            }
        ]
        self.props_result = [
            {"id": b"\xaa", "text": "a causes b", "source_doc": "d", "source_span": "s"}
        ]

    def vector_search_entities(self, *, query_vec, k):
        self.calls.append(("vse", k))
        return self.seed_entities_result

    def vector_search_propositions(self, *, query_vec, k):
        self.calls.append(("vsp", k))
        return self.seed_props_result

    def pgql_subgraph(self, *, seed_ids, max_edges):
        self.calls.append(("pgql", list(seed_ids), max_edges))
        return self.subgraph_result

    def fetch_propositions(self, ids):
        self.calls.append(("fp", list(ids)))
        return self.props_result


class _StubLLM:
    async def complete(self, prompt, *, schema=None, temperature=0.0):
        return "yes [P1]"


def _axis_vectors() -> dict[str, np.ndarray]:
    """One-hot axis vectors; causal aligns with the embedder's query."""
    axes = {}
    names = ["causal", "taxonomic", "temporal", "definitional", "exemplification"]
    for i, n in enumerate(names):
        v = np.zeros(5, dtype=np.float32)
        v[i] = 1.0
        axes[n] = v
    return axes


@pytest.mark.asyncio
async def test_query_returns_answer_with_citations():
    cfg = Config()
    cfg.embeddings.dim = 5
    p = QueryPipeline(
        cfg=cfg,
        graph=_StubGraph(),
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    out = await p.query("does a cause b?")
    assert isinstance(out, QueryResult)
    assert "yes" in out.answer.text.lower()
    assert any(c.proposition_id == b"\xaa" for c in out.answer.citations)


@pytest.mark.asyncio
async def test_query_amplitudes_peak_on_aligned_axis():
    """Embedder returns [1,0,0,0,0]; the causal axis is also [1,0,0,0,0]."""
    cfg = Config()
    cfg.embeddings.dim = 5
    p = QueryPipeline(
        cfg=cfg,
        graph=_StubGraph(),
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    out = await p.query("does a cause b?")
    assert out.amplitudes["causal"] > 0.9
    for other in ("taxonomic", "temporal", "definitional", "exemplification"):
        assert 0.45 < out.amplitudes[other] < 0.55


@pytest.mark.asyncio
async def test_query_empty_subgraph_yields_no_info_answer():
    cfg = Config()
    cfg.embeddings.dim = 5
    g = _StubGraph()
    g.subgraph_result = []
    g.props_result = []
    p = QueryPipeline(
        cfg=cfg,
        graph=g,
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    out = await p.query("nothing about this in the corpus?")
    assert "don't have information" in out.answer.text.lower()
    assert out.answer.citations == []


@pytest.mark.asyncio
async def test_query_passes_configured_k_to_graph():
    cfg = Config()
    cfg.embeddings.dim = 5
    cfg.retrieval.seed_k_entities = 12
    cfg.retrieval.seed_k_propositions = 24
    cfg.retrieval.max_subgraph_edges = 1000
    g = _StubGraph()
    p = QueryPipeline(
        cfg=cfg,
        graph=g,
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    await p.query("q")
    # vector_search_entities should have been called with k=12.
    vse_calls = [c for c in g.calls if c[0] == "vse"]
    assert vse_calls[0][1] == 12
    vsp_calls = [c for c in g.calls if c[0] == "vsp"]
    assert vsp_calls[0][1] == 24
    pgql_calls = [c for c in g.calls if c[0] == "pgql"]
    assert pgql_calls[0][2] == 1000


@pytest.mark.asyncio
async def test_query_latency_dict_has_all_phase_keys():
    cfg = Config()
    cfg.embeddings.dim = 5
    p = QueryPipeline(
        cfg=cfg,
        graph=_StubGraph(),
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    out = await p.query("q")
    expected_keys = {"embed_ms", "seed_ms", "subgraph_ms", "reweight_ms", "ppr_ms", "answer_ms"}
    assert set(out.latency_ms.keys()) >= expected_keys
    for v in out.latency_ms.values():
        assert v >= 0


@pytest.mark.asyncio
async def test_query_degenerate_amplitudes_fall_back_to_uniform():
    """When the query is orthogonal to all axes (amps all ~0.5), reweighting
    is still meaningful. But when EVERY amp is at an extreme (all <0.05 or
    all >0.95), fall back to uniform 1.0 so PR runs on base weights."""
    cfg = Config()
    cfg.embeddings.dim = 5

    class _OrthogonalEmbedder:
        dim = 5

        async def embed(self, texts, *, normalize: bool = True):
            # Query orthogonal to every axis vector -> projection = 0 -> amp = 0.5.
            return np.array([[0.0, 0.0, 0.0, 0.0, 1.0]] * len(texts), dtype=np.float32)

    # Build axis vectors that are ALL aligned with the query so every amp > 0.95.
    saturated_axes = {n: np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                      for n in ["causal", "taxonomic", "temporal",
                                "definitional", "exemplification"]}

    p = QueryPipeline(
        cfg=cfg,
        graph=_StubGraph(),
        embedder=_OrthogonalEmbedder(),
        llm=_StubLLM(),
        axis_vectors=saturated_axes,
    )
    out = await p.query("q")
    # The pipeline must detect the degenerate case and fall back to uniform.
    assert all(a == 1.0 for a in out.amplitudes.values())


@pytest.mark.asyncio
async def test_query_edges_used_is_the_reweighted_subgraph():
    """edges_used should expose the reweighted edge list for the visualizer."""
    cfg = Config()
    cfg.embeddings.dim = 5
    p = QueryPipeline(
        cfg=cfg,
        graph=_StubGraph(),
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    out = await p.query("q")
    assert len(out.edges_used) == 1
    # Each edge dict carries a `weight` field after reweighting.
    assert "weight" in out.edges_used[0]


@pytest.mark.asyncio
async def test_query_seeds_propositions_pass_through_seed_sims():
    """Propositions returned from vector_search_propositions become the
    seed_sims dict that assemble_propositions consults."""
    cfg = Config()
    cfg.embeddings.dim = 5
    g = _StubGraph()
    # Two seed propositions with different distances.
    g.seed_props_result = [
        {"id": b"\xaa", "source_doc": "d", "source_span": "s", "distance": 0.1},
        {"id": b"\xbb", "source_doc": "d", "source_span": "s", "distance": 0.5},
    ]
    p = QueryPipeline(
        cfg=cfg,
        graph=g,
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    await p.query("q")
    # fetch_propositions was called with some ids (the assembled ones).
    fp_calls = [c for c in g.calls if c[0] == "fp"]
    assert len(fp_calls) == 1


@pytest.mark.asyncio
async def test_query_edges_nonempty_but_picked_empty_yields_no_info():
    """Subgraph has an edge between non-seed nodes; PPR mass doesn't reach
    them, so picked is empty. Answer must short-circuit to _NO_INFO without
    calling fetch_propositions; edges_used should still expose the graph
    evidence for the visualizer."""
    cfg = Config()
    cfg.embeddings.dim = 5
    g = _StubGraph()
    # Edge between two NON-seed nodes (seed is b"\x01" from the stub).
    g.subgraph_result = [
        {
            "src": b"\x99",
            "dst": b"\x88",
            "predicate": "p",
            "ontology_axis": "causal",
            "base_weight": 0.8,
            "support_propositions": [b"\xcc".hex()],
        }
    ]
    g.props_result = []
    p = QueryPipeline(
        cfg=cfg,
        graph=g,
        embedder=_StubEmbedder(),
        llm=_StubLLM(),
        axis_vectors=_axis_vectors(),
    )
    out = await p.query("q")
    assert "don't have information" in out.answer.text.lower()
    assert len(out.edges_used) == 1
    # No fetch_propositions call when picked is empty.
    assert not any(c[0] == "fp" for c in g.calls)
