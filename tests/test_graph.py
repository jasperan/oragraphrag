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


def test_ledger_add_and_check(store):
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    assert not store.ledger_has("abc123")
    store.ledger_add("abc123", doc_id="d.md", section_path="s")
    assert store.ledger_has("abc123")
    # Idempotent: second add does not raise.
    store.ledger_add("abc123", doc_id="d.md", section_path="s")
    assert store.ledger_has("abc123")


def test_upsert_with_source_id(store):
    """The new source_id column is populated on all three writers."""
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    sid = "src_test_aaa"
    e1 = store.upsert_entity(name="se1", kind="x", embedding=[0.1] * 384, source_id=sid)
    e2 = store.upsert_entity(name="se2", kind="x", embedding=[0.2] * 384, source_id=sid)
    p = store.upsert_proposition(
        text="t", source_doc="d", source_span="s",
        embedding=[0.3] * 384, source_id=sid,
    )
    store.upsert_rel(
        e1, e2, predicate="r", ontology_axis="causal",
        base_weight=0.5, support_prop_id=p, source_id=sid,
    )

    with store._conn() as c, c.cursor() as cur:
        cur.execute("SELECT source_id FROM Entity WHERE id = :id", id=e1)
        assert cur.fetchone()[0] == sid
        cur.execute("SELECT source_id FROM Proposition WHERE id = :id", id=p)
        assert cur.fetchone()[0] == sid
        cur.execute(
            "SELECT source_id FROM Rel WHERE src_id = :s AND dst_id = :d "
            "AND predicate = :p",
            s=e1, d=e2, p="r",
        )
        assert cur.fetchone()[0] == sid


def test_vector_search_filters_by_source(store):
    """vector_search_* must scope to the requested source_id when supplied."""
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    sid_a = "src_aaa"
    sid_b = "src_bbb"
    e_a = store.upsert_entity(
        name="alpha entity", kind="x", embedding=[0.5] * 384, source_id=sid_a
    )
    e_b = store.upsert_entity(
        name="beta entity", kind="x", embedding=[0.5] * 384, source_id=sid_b
    )
    p_a = store.upsert_proposition(
        text="from a", source_doc="da", source_span="0",
        embedding=[0.5] * 384, source_id=sid_a,
    )
    p_b = store.upsert_proposition(
        text="from b", source_doc="db", source_span="0",
        embedding=[0.5] * 384, source_id=sid_b,
    )

    # Unfiltered: both surface.
    ent_rows = store.vector_search_entities(query_vec=[0.5] * 384, k=10)
    ent_ids = {r["id"] for r in ent_rows}
    assert e_a in ent_ids and e_b in ent_ids

    # Filtered to sid_a only: only the a-entity surfaces.
    ent_rows_a = store.vector_search_entities(
        query_vec=[0.5] * 384, k=10, source_filter=sid_a
    )
    ent_ids_a = {r["id"] for r in ent_rows_a}
    assert e_a in ent_ids_a
    assert e_b not in ent_ids_a

    # Same for propositions.
    prop_rows_b = store.vector_search_propositions(
        query_vec=[0.5] * 384, k=10, source_filter=sid_b
    )
    prop_ids_b = {r["id"] for r in prop_rows_b}
    assert p_b in prop_ids_b
    assert p_a not in prop_ids_b


def test_list_sources(store):
    """list_sources returns the distinct set of source_ids on Entity."""
    store.init_db(rebuild=True, axis_vectors=_zero_axis_vectors())
    store.upsert_entity(name="ls_a", kind="x", embedding=[0.1] * 384, source_id="src_one")
    store.upsert_entity(name="ls_b", kind="x", embedding=[0.2] * 384, source_id="src_one")
    store.upsert_entity(name="ls_c", kind="x", embedding=[0.3] * 384, source_id="src_two")
    store.upsert_entity(name="ls_d", kind="x", embedding=[0.4] * 384)  # default

    sources = set(store.list_sources())
    assert {"src_one", "src_two", "default"} <= sources
