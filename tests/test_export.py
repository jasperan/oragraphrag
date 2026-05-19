"""Unit tests for the fine-tune exporter shaping logic.

These tests run without any Oracle dependency — they exercise
``build_training_example`` directly with crafted dicts.
"""

from __future__ import annotations

from oragraphrag.export import build_training_example


def test_build_training_example_with_edges():
    prop = {
        "id": b"\x01\x02",
        "text": "HNSW is a vector index for ANN search.",
        "source_doc": "indexes.md",
        "source_span": "HNSW",
        "source_id": "src_abc",
    }
    edges = [
        {"predicate": "accelerates", "ontology_axis": "causal", "base_weight": 0.8},
        {"predicate": "is_a", "ontology_axis": "taxonomic", "base_weight": 0.7},
    ]
    out = build_training_example(prop, edges)
    assert out["id"] == "0102"
    assert out["source"] == "indexes.md#HNSW"
    assert out["source_id"] == "src_abc"
    assert "accelerates" in out["prompt"]  # picked the first predicate
    assert out["completion"] == "HNSW is a vector index for ANN search."
    assert out["axes"] == ["causal", "taxonomic"]
    assert out["predicates"] == ["accelerates", "is_a"]


def test_build_training_example_with_no_edges():
    prop = {
        "id": b"\xaa",
        "text": "Some long fact about Oracle that is over sixty chars in length here.",
        "source_doc": "d.md",
        "source_span": "s",
        "source_id": "src_x",
    }
    out = build_training_example(prop, [])
    assert out["axes"] == []
    assert out["predicates"] == []
    assert "State the fact" in out["prompt"]


def test_build_training_example_handles_string_id():
    prop = {
        "id": "deadbeef",
        "text": "t",
        "source_doc": "d",
        "source_span": "s",
        "source_id": "src_x",
    }
    out = build_training_example(prop, [])
    assert out["id"] == "deadbeef"


def test_build_training_example_dedupes_axes():
    prop = {
        "id": b"\x01",
        "text": "t",
        "source_doc": "d",
        "source_span": "s",
        "source_id": "src_x",
    }
    edges = [
        {"predicate": "rel1", "ontology_axis": "causal", "base_weight": 0.5},
        {"predicate": "rel2", "ontology_axis": "causal", "base_weight": 0.5},
    ]
    out = build_training_example(prop, edges)
    assert out["axes"] == ["causal"]


def test_build_training_example_handles_long_text_truncation():
    prop = {
        "id": b"\x01",
        "text": "x" * 120,
        "source_doc": "d",
        "source_span": "s",
        "source_id": "src_x",
    }
    out = build_training_example(prop, [])
    assert "..." in out["prompt"]


def test_build_training_example_default_source_id_when_missing():
    prop = {"id": b"\x01", "text": "t", "source_doc": "d", "source_span": "s"}
    out = build_training_example(prop, [])
    assert out["source_id"] == "default"


def test_build_training_example_handles_none_text():
    prop = {
        "id": b"\x01",
        "text": None,
        "source_doc": "d",
        "source_span": "s",
        "source_id": "src_x",
    }
    out = build_training_example(prop, [])
    assert out["completion"] == ""


def test_export_finetune_writes_jsonl_via_fake_store(tmp_path):
    """End-to-end shape test using a fake GraphStore."""
    from oragraphrag.export import export_finetune

    class FakeStore:
        def iter_propositions_with_edges(self, *, source_filter=None):
            yield (
                {
                    "id": b"\xab",
                    "text": "fact one",
                    "source_doc": "a.md",
                    "source_span": "h1",
                    "source_id": "src_x",
                },
                [{"predicate": "is_a", "ontology_axis": "taxonomic", "base_weight": 0.5}],
            )
            yield (
                {
                    "id": b"\xcd",
                    "text": "fact two",
                    "source_doc": "b.md",
                    "source_span": "h2",
                    "source_id": "src_y",
                },
                [],
            )

    out = tmp_path / "train.jsonl"
    count = export_finetune(FakeStore(), out)
    assert count == 2
    import json as _json
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = _json.loads(lines[0])
    assert rec0["id"] == "ab"
    assert rec0["axes"] == ["taxonomic"]
    rec1 = _json.loads(lines[1])
    assert rec1["axes"] == []
