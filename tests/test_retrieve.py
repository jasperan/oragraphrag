import numpy as np
import pytest

from oragraphrag.retrieve import (
    assemble_propositions,
    compute_amplitudes,
    reweight_edges,
    spreading_activation,
)


def _axis_vecs(dim: int = 5) -> dict[str, np.ndarray]:
    """One-hot axis vectors so projections are exact and easy to reason about."""
    vecs = {}
    names = ["causal", "taxonomic", "temporal", "definitional", "exemplification"]
    for i, name in enumerate(names):
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0
        vecs[name] = v
    return vecs


# ----- compute_amplitudes -----


def test_amplitudes_peak_on_aligned_axis():
    """A query exactly aligned to causal should amplify causal and attenuate others."""
    q = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    amps = compute_amplitudes(q, _axis_vecs(), alpha=8.0, beta=0.0)
    assert amps["causal"] > 0.9
    for other in ("taxonomic", "temporal", "definitional", "exemplification"):
        # Orthogonal projection → sigmoid(0) = 0.5
        assert 0.45 < amps[other] < 0.55


def test_amplitudes_symmetric_under_orthogonal_query():
    """A query orthogonal to all axes should give amplitude 0.5 to each."""
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Zero vector is degenerate; the impl normalizes to a safe baseline.
    amps = compute_amplitudes(q, _axis_vecs(), alpha=8.0, beta=0.0)
    for axis in amps:
        assert amps[axis] == pytest.approx(0.5, abs=1e-3)


def test_amplitudes_anti_aligned_query_attenuates():
    """A query exactly opposite to causal should give causal a low amplitude."""
    q = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    amps = compute_amplitudes(q, _axis_vecs(), alpha=8.0, beta=0.0)
    assert amps["causal"] < 0.05  # sigmoid(-8) ~ 0.0003
    for other in ("taxonomic", "temporal", "definitional", "exemplification"):
        assert 0.45 < amps[other] < 0.55


def test_amplitudes_alpha_controls_steepness():
    """Higher alpha → sharper amplification of aligned axes."""
    q = np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    soft = compute_amplitudes(q, _axis_vecs(), alpha=2.0, beta=0.0)
    sharp = compute_amplitudes(q, _axis_vecs(), alpha=16.0, beta=0.0)
    # Soft: sigmoid(2 * 0.5)=sigmoid(1)≈0.73; sharp: sigmoid(16 * 0.5)=sigmoid(8)≈0.9997
    assert sharp["causal"] > soft["causal"]


def test_amplitudes_beta_shifts_baseline():
    """Positive beta lifts all amplitudes; negative beta lowers."""
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    high = compute_amplitudes(q, _axis_vecs(), alpha=8.0, beta=2.0)
    low = compute_amplitudes(q, _axis_vecs(), alpha=8.0, beta=-2.0)
    for axis in high:
        assert high[axis] > 0.5
        assert low[axis] < 0.5


# ----- reweight_edges -----


def test_reweight_multiplies_per_axis():
    edges = [
        {"src": b"a", "dst": b"b", "ontology_axis": "causal", "base_weight": 0.8,
         "predicate": "causes", "support_propositions": []},
        {"src": b"a", "dst": b"c", "ontology_axis": "temporal", "base_weight": 0.8,
         "predicate": "before", "support_propositions": []},
    ]
    amps = {"causal": 0.99, "taxonomic": 0.1, "temporal": 0.1,
            "definitional": 0.1, "exemplification": 0.1}
    out = reweight_edges(edges, amps)
    by_pred = {e["predicate"]: e for e in out}
    assert by_pred["causes"]["weight"] == pytest.approx(0.8 * 0.99)
    assert by_pred["before"]["weight"] == pytest.approx(0.8 * 0.1)


def test_reweight_preserves_other_fields():
    edges = [
        {"src": b"x", "dst": b"y", "ontology_axis": "causal", "base_weight": 0.5,
         "predicate": "p", "support_propositions": ["aa", "bb"]},
    ]
    amps = {"causal": 0.7, "taxonomic": 0.5, "temporal": 0.5,
            "definitional": 0.5, "exemplification": 0.5}
    out = reweight_edges(edges, amps)
    assert out[0]["src"] == b"x"
    assert out[0]["dst"] == b"y"
    assert out[0]["predicate"] == "p"
    assert out[0]["support_propositions"] == ["aa", "bb"]
    assert out[0]["weight"] == pytest.approx(0.5 * 0.7)


def test_reweight_unknown_axis_defaults_to_half():
    """An edge with an axis not in the amplitude dict should default to 0.5."""
    edges = [
        {"src": b"a", "dst": b"b", "ontology_axis": "mystery", "base_weight": 0.8,
         "predicate": "p", "support_propositions": []},
    ]
    amps = {"causal": 0.9, "taxonomic": 0.5, "temporal": 0.5,
            "definitional": 0.5, "exemplification": 0.5}
    out = reweight_edges(edges, amps)
    assert out[0]["weight"] == pytest.approx(0.8 * 0.5)


def test_reweight_empty_edge_list():
    assert reweight_edges([], {"causal": 0.9}) == []


# ----- spreading_activation -----


def test_spreading_activation_concentrates_on_seed_neighborhood():
    """The seed and its direct neighbor should outrank a disconnected pair."""
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9},
        {"src": b"b", "dst": b"c", "weight": 0.9},
        {"src": b"x", "dst": b"y", "weight": 0.9},
    ]
    scores = spreading_activation(edges, seed_ids=[b"a"], damping=0.85)
    assert scores[b"a"] > scores[b"x"]
    assert scores[b"b"] > scores[b"y"]


def test_spreading_activation_returns_seed_only_when_no_edges():
    scores = spreading_activation([], seed_ids=[b"a"], damping=0.85)
    assert scores == {b"a": 1.0}


def test_spreading_activation_higher_weight_routes_more_mass():
    """An edge with higher weight should pull more PageRank mass than a lower one."""
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9},
        {"src": b"a", "dst": b"c", "weight": 0.1},
    ]
    scores = spreading_activation(edges, seed_ids=[b"a"], damping=0.85)
    assert scores[b"b"] > scores[b"c"]


def test_spreading_activation_handles_multiple_seeds():
    """Two seeds should both rank high."""
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9},
        {"src": b"x", "dst": b"y", "weight": 0.9},
    ]
    scores = spreading_activation(edges, seed_ids=[b"a", b"x"], damping=0.85)
    # Both seeds get reset mass; their successors get propagation mass.
    assert scores[b"a"] > scores[b"y"] or scores[b"x"] > scores[b"b"]
    assert b"a" in scores and b"x" in scores


# ----- assemble_propositions -----


def test_assemble_picks_top_propositions_by_activation_and_seed_similarity():
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9,
         "support_propositions": [b"p1".hex(), b"p2".hex()]},
        {"src": b"a", "dst": b"c", "weight": 0.1,
         "support_propositions": [b"p3".hex()]},
    ]
    activations = {b"a": 0.5, b"b": 0.4, b"c": 0.05}
    seed_sims = {b"p1": 0.9, b"p2": 0.8, b"p3": 0.5}
    picked = assemble_propositions(edges, activations, seed_sims, top_m=2)
    assert b"p1" in picked
    assert len(picked) <= 2


def test_assemble_dedupes_propositions_across_edges():
    """Same proposition supporting two edges should not be picked twice."""
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9,
         "support_propositions": [b"p1".hex()]},
        {"src": b"a", "dst": b"c", "weight": 0.5,
         "support_propositions": [b"p1".hex()]},
    ]
    activations = {b"a": 0.5, b"b": 0.4, b"c": 0.3}
    seed_sims = {b"p1": 0.9}
    picked = assemble_propositions(edges, activations, seed_sims, top_m=5)
    assert picked.count(b"p1") == 1


def test_assemble_handles_propositions_with_no_seed_similarity():
    """A proposition not in seed_sims should still be ranked (with seed_sim=0)."""
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9,
         "support_propositions": [b"p1".hex()]},
    ]
    activations = {b"a": 0.5, b"b": 0.4}
    seed_sims: dict[bytes, float] = {}
    picked = assemble_propositions(edges, activations, seed_sims, top_m=5)
    assert b"p1" in picked


def test_assemble_accepts_bytes_or_hex_string_support_propositions():
    """support_propositions can be a list of hex strings (from GraphStore) or bytes."""
    edges = [
        {"src": b"a", "dst": b"b", "weight": 0.9,
         "support_propositions": [b"p1".hex()]},
        {"src": b"a", "dst": b"c", "weight": 0.9,
         "support_propositions": [b"p2"]},  # raw bytes
    ]
    activations = {b"a": 0.5, b"b": 0.4, b"c": 0.4}
    seed_sims = {b"p1": 0.9, b"p2": 0.9}
    picked = assemble_propositions(edges, activations, seed_sims, top_m=5)
    assert b"p1" in picked
    assert b"p2" in picked


def test_assemble_empty_edges_returns_empty_list():
    assert assemble_propositions([], {}, {}, top_m=5) == []


def test_reweight_does_not_alias_support_propositions_list():
    """Mutating the output's support_propositions must not affect the input."""
    edges = [
        {
            "src": b"a",
            "dst": b"b",
            "ontology_axis": "causal",
            "base_weight": 0.5,
            "support_propositions": ["aa", "bb"],
        }
    ]
    amps = {"causal": 0.9, "taxonomic": 0.5, "temporal": 0.5,
            "definitional": 0.5, "exemplification": 0.5}
    out = reweight_edges(edges, amps)
    out[0]["support_propositions"].append("MUTATED")
    assert edges[0]["support_propositions"] == ["aa", "bb"]


def test_assemble_drops_propositions_with_zero_graph_activation():
    """Documented contract: graph evidence is required. A proposition whose
    supporting edges all have zero activation is excluded even when seed_sim
    is high. Pure-vector retrieval lives upstream in Task 11's seed search."""
    edges = [
        {
            "src": b"a",
            "dst": b"b",
            "weight": 0.9,
            "support_propositions": [b"p_orphan".hex()],
        }
    ]
    activations = {b"a": 0.0, b"b": 0.0}  # graph didn't activate
    seed_sims = {b"p_orphan": 0.99}  # but vector seed matched strongly
    picked = assemble_propositions(edges, activations, seed_sims, top_m=5)
    assert picked == []
