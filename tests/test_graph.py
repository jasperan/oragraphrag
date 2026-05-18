import pytest

from oragraphrag.config import Config
from oragraphrag.graph import GraphStore

pytestmark = pytest.mark.oracle


@pytest.fixture
def store():
    cfg = Config()
    s = GraphStore(cfg)
    s.connect()
    yield s
    s.close()


def _zero_axis_vectors(dim: int = 384) -> dict[str, list[float]]:
    return {
        "causal": [0.0] * dim,
        "taxonomic": [0.0] * dim,
        "temporal": [0.0] * dim,
        "definitional": [0.0] * dim,
        "exemplification": [0.0] * dim,
    }


def test_init_db_creates_tables_and_graph(store):
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    tables = store.list_tables()
    expected = {"ENTITY", "PROPOSITION", "REL", "MENTIONS", "ONTOLOGY_AXIS", "INGEST_LEDGER"}
    assert expected <= set(tables)
    assert "ORAGRAPH" in store.list_property_graphs()


def test_insert_and_query_entity(store):
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    eid = store.upsert_entity(name="VECTOR datatype", kind="feature", embedding=[0.1] * 384)
    rows = store.vector_search_entities(query_vec=[0.1] * 384, k=5)
    assert any(r["id"] == eid for r in rows)


def test_pgql_two_hop_returns_edges(store):
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    a = store.upsert_entity(name="A", kind="x", embedding=[0.1] * 384)
    b = store.upsert_entity(name="B", kind="x", embedding=[0.2] * 384)
    c = store.upsert_entity(name="C", kind="x", embedding=[0.3] * 384)
    p = store.upsert_proposition(
        text="A causes B", source_doc="doc1", source_span="0:10", embedding=[0.5] * 384
    )
    store.upsert_rel(
        a, b, predicate="causes", ontology_axis="causal", base_weight=0.7, support_prop_id=p
    )
    store.upsert_rel(
        b, c, predicate="depends_on", ontology_axis="causal", base_weight=0.5, support_prop_id=p
    )
    edges = store.pgql_subgraph(seed_ids=[a], max_edges=100)
    src_dst = {(e["src"], e["dst"]) for e in edges}
    assert (a, b) in src_dst


def test_upsert_rel_concurrent_no_lost_updates(store):
    """Hammer the same triple from many threads; assert all support_props recorded."""
    import concurrent.futures

    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    a = store.upsert_entity(name="ca", kind="x", embedding=[0.1] * 384)
    b = store.upsert_entity(name="cb", kind="x", embedding=[0.2] * 384)
    # Pre-create 16 distinct propositions so each upsert appends a unique sp_hex.
    prop_ids = [
        store.upsert_proposition(
            text=f"p{i}", source_doc="d", source_span=str(i), embedding=[0.3] * 384
        )
        for i in range(16)
    ]

    def hit(pid):
        store.upsert_rel(
            a, b,
            predicate="r",
            ontology_axis="causal",
            base_weight=0.5,
            support_prop_id=pid,
        )

    # 8 workers, same pool_max from Config.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(hit, prop_ids))

    # Read back the row and count support_propositions.
    with store._conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT support_propositions, support_axis_counts FROM Rel "
            "WHERE src_id = :s AND dst_id = :d AND predicate = :p",
            s=a, d=b, p="r",
        )
        row = cur.fetchone()
        assert row is not None
        props = store._read_json(row[0])
        counts = store._read_json(row[1])
        # All 16 distinct propositions must be recorded; counts must total 16.
        assert len(set(props)) == 16, f"lost updates: only {len(set(props))} props recorded"
        assert int(counts["causal"]) == 16, f"axis count short: {counts}"


def test_pgql_one_hop_does_not_return_far_neighbor(store):
    """The contract of pgql_subgraph is one hop from any seed.

    So a -> b -> c with seed=[a] returns (a, b) but NOT (b, c).
    """
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    a = store.upsert_entity(name="ha", kind="x", embedding=[0.1] * 384)
    b = store.upsert_entity(name="hb", kind="x", embedding=[0.2] * 384)
    c = store.upsert_entity(name="hc", kind="x", embedding=[0.3] * 384)
    p = store.upsert_proposition(
        text="t", source_doc="d", source_span="s", embedding=[0.5] * 384
    )
    store.upsert_rel(
        a, b, predicate="r", ontology_axis="causal", base_weight=0.7, support_prop_id=p
    )
    store.upsert_rel(
        b, c, predicate="r", ontology_axis="causal", base_weight=0.5, support_prop_id=p
    )
    edges = store.pgql_subgraph(seed_ids=[a], max_edges=100)
    src_dst = {(e["src"], e["dst"]) for e in edges}
    assert (a, b) in src_dst
    # (b, c) is two hops from seed `a` — must NOT appear in the result.
    assert (b, c) not in src_dst
