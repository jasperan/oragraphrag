"""Query-time reweighting + spreading activation. Pure-Python compute over a
bounded subgraph fetched from Oracle by Task 11's orchestrator.

The four entry points compose as:
    amps = compute_amplitudes(query_vec, axis_vectors, alpha, beta)
    rew = reweight_edges(edges_from_pgql, amps)
    activations = spreading_activation(rew, seed_ids, damping)
    prop_ids = assemble_propositions(rew, activations, seed_sims, top_m)

This is the heart of OraGraphRAG's novelty: query-conditioned dynamic edge
reweighting via sigmoid-squashed projection onto ontology-axis vectors.
"""

from __future__ import annotations

import igraph as ig
import numpy as np

from oragraphrag.axes import ONTOLOGY_AXIS_NAMES


def compute_amplitudes(
    query_vec: np.ndarray,
    axis_vectors: dict[str, np.ndarray],
    *,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """For each ontology axis, project the query onto the axis vector,
    then apply a sigmoid-squashed linear map to produce an amplitude
    in (0, 1).

    Returns {axis_name: amplitude}. Amplitudes are deterministic: same
    query + same axis vectors → same numbers every call.

    Degenerate inputs:
    - Zero query vector → all projections are 0 → sigmoid(beta) for each.
    - Zero axis vector → projection is 0 → same as above.
    Both are handled by clamping the norm to a small epsilon before division.
    """
    qn = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
    out: dict[str, float] = {}
    for name in ONTOLOGY_AXIS_NAMES:
        if name not in axis_vectors:
            # Missing axis vector: amplitude defaults to 0.5 (no signal).
            out[name] = 0.5
            continue
        av = axis_vectors[name]
        an = av / max(float(np.linalg.norm(av)), 1e-12)
        proj = float(np.dot(qn, an))
        out[name] = float(1.0 / (1.0 + np.exp(-(alpha * proj + beta))))
    return out


def reweight_edges(edges: list[dict], amplitudes: dict[str, float]) -> list[dict]:
    """Multiply each edge's base_weight by the amplitude of its ontology axis.

    Returns a new list (input edges are not mutated). The `weight` key on
    each output edge is `base_weight * amp[axis]`. Edges with an axis not
    in `amplitudes` default to amplitude 0.5 (no signal).
    """
    out: list[dict] = []
    for e in edges:
        amp = amplitudes.get(e["ontology_axis"], 0.5)
        new = dict(e)
        # Defensive deep-ish copy: shallow-copy the only mutable field we know
        # downstream code touches. cheaper than copy.deepcopy and matches actual
        # data flow.
        if "support_propositions" in e:
            new["support_propositions"] = list(e["support_propositions"])
        new["weight"] = float(e["base_weight"]) * amp
        out.append(new)
    return out


def spreading_activation(
    edges: list[dict],
    *,
    seed_ids: list[bytes],
    damping: float,
) -> dict[bytes, float]:
    """Run personalized PageRank over the reweighted subgraph.

    Each edge dict must contain `src`, `dst`, and `weight`. The PageRank
    reset distribution is uniform across `seed_ids` (so multi-seed queries
    don't double-count a single seed). The `damping` factor controls how
    much mass stays close to the seeds vs. propagates through the graph.

    Returns {node_id: pagerank_score} for every node that appears in either
    the seed list or any edge. Disconnected nodes will still receive a
    small share of reset mass if they are seeds.

    Empty edges + seeds: returns {s: 1/len(seeds) for s in seeds} as a
    degenerate fallback, so each seed contributes equally and downstream
    assemble_propositions has activations to work with. Empty seeds with
    no edges: returns {}.
    """
    node_ids: list[bytes] = []
    idx: dict[bytes, int] = {}

    def _idx(n: bytes) -> int:
        if n not in idx:
            idx[n] = len(node_ids)
            node_ids.append(n)
        return idx[n]

    # Pre-register seeds so they exist as nodes even if no edge touches them.
    for s in seed_ids:
        _idx(s)

    pairs = [(_idx(e["src"]), _idx(e["dst"])) for e in edges]
    weights = [float(e["weight"]) for e in edges]

    if not pairs:
        # No edges: degenerate result. Give each seed equal mass.
        if not seed_ids:
            return {}
        share = 1.0 / len(seed_ids)
        return {s: share for s in seed_ids}

    g = ig.Graph(n=len(node_ids), edges=pairs, directed=True)
    g.es["weight"] = weights

    seed_set = set(seed_ids)
    reset = [1.0 if n in seed_set else 0.0 for n in node_ids]
    total = sum(reset)
    reset = (
        [x / total for x in reset]
        if total > 0
        else [1.0 / len(node_ids)] * len(node_ids)
    )

    scores = g.personalized_pagerank(
        reset=reset,
        weights="weight",
        damping=damping,
        implementation="prpack",
    )
    return {node_ids[i]: float(scores[i]) for i in range(len(node_ids))}


def assemble_propositions(
    edges: list[dict],
    activations: dict[bytes, float],
    seed_sims: dict[bytes, float],
    *,
    top_m: int,
) -> list[bytes]:
    """Pick the top-M propositions for the answer prompt.

    Each edge in `edges` carries a `support_propositions` list (hex strings
    from Oracle's JSON column, or raw bytes if the caller pre-decoded).
    A proposition's score is the MAX activation of either endpoint of any
    supporting edge, multiplied by (0.5 + seed_sim).

    Returns the top-M proposition IDs as bytes, sorted by score descending.
    Empty edges → []. Propositions appearing in multiple edges are deduped:
    each prop's final score is the MAX across its supporting edges.

    Contract: graph evidence is REQUIRED. A proposition whose supporting
    edges all have zero activation is excluded from the result, even when
    seed_sim is high. This enforces spec §6 Step 5's intent — the
    assembled propositions are surfaced by the reweighted graph walk, not
    by pure vector matches. Pure-vector-only retrieval is handled by the
    seed retrieval at Task 11's Step 1, BEFORE the graph walk runs.
    """
    prop_score: dict[bytes, float] = {}
    for e in edges:
        act = max(activations.get(e["src"], 0.0), activations.get(e["dst"], 0.0))
        for ph in e.get("support_propositions", []):
            pid = bytes.fromhex(ph) if isinstance(ph, str) else ph
            seed = seed_sims.get(pid, 0.0)
            score = act * (0.5 + seed)
            if score > prop_score.get(pid, 0.0):
                prop_score[pid] = score
    return [
        p
        for p, _ in sorted(prop_score.items(), key=lambda kv: kv[1], reverse=True)[
            :top_m
        ]
    ]
