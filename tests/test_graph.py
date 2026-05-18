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
