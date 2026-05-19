"""Unit tests for the MemoryLayer wrapper + source_id_for_folder helper.

These tests deliberately avoid touching live Oracle: oracleagentmemory's
real schema needs a DB. We assert on contract instead — lazy construction,
deterministic ids, delegation to GraphStore.list_sources.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oragraphrag.config import Config
from oragraphrag.memory import MemoryLayer, source_id_for_folder


def test_source_id_for_folder_is_deterministic(tmp_path):
    a = source_id_for_folder(tmp_path)
    b = source_id_for_folder(tmp_path)
    assert a == b
    assert a.startswith("src_")
    # 4-char prefix + 32 hex digits = 36 chars total, fits VARCHAR2(64).
    assert len(a) == 36


def test_source_id_for_folder_distinguishes_paths(tmp_path):
    sub1 = tmp_path / "alpha"
    sub2 = tmp_path / "beta"
    sub1.mkdir()
    sub2.mkdir()
    assert source_id_for_folder(sub1) != source_id_for_folder(sub2)


def test_source_id_for_folder_accepts_str(tmp_path):
    """Both Path and str inputs produce the same id."""
    assert source_id_for_folder(tmp_path) == source_id_for_folder(str(tmp_path))


def test_source_id_resolves_symlinks_consistently(tmp_path):
    """Relative + absolute references to the same dir collapse to one id."""
    (tmp_path / "x").mkdir()
    # Path.resolve() canonicalizes; both forms must hash to the same id.
    via_abs = source_id_for_folder(tmp_path / "x")
    via_concat = source_id_for_folder(str(tmp_path) + "/x/")
    assert via_abs == via_concat


def test_memory_layer_construction_is_lazy():
    """Constructing MemoryLayer must not import oracleagentmemory."""
    # Drop a sentinel that would crash if MemoryLayer.__init__ tried to import.
    sys.modules.pop("oracleagentmemory", None)
    sys.modules.pop("oracleagentmemory.core", None)
    cfg = Config()
    layer = MemoryLayer(cfg, graph_store=MagicMock())
    assert layer._mem is None


def test_memory_layer_list_sources_delegates_to_graph():
    """list_sources() must call through to the graph store, not the package."""
    cfg = Config()
    graph = MagicMock()
    graph.list_sources.return_value = ["src_abc", "src_def"]
    layer = MemoryLayer(cfg, graph_store=graph)
    assert layer.list_sources() == ["src_abc", "src_def"]
    graph.list_sources.assert_called_once()


def test_memory_layer_lazy_import_wires_pool(monkeypatch):
    """First call to _mem_lazy() must import oracleagentmemory and pass
    the graph store's pool + configured vector dim into OracleDBMemoryStore."""
    cfg = Config()
    graph = MagicMock()
    graph._pool = "fake-pool"
    layer = MemoryLayer(cfg, graph_store=graph)

    captured = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeAgentMemory:
        def __init__(self, *, store):
            self.store = store

    class FakeSchemaPolicy:
        CREATE_IF_NECESSARY = "create_if_necessary"

    # Build the stub oracleagentmemory package on the fly so the lazy
    # import inside _mem_lazy() resolves to our stand-ins without
    # touching the real (heavy) package.
    pkg = types.ModuleType("oracleagentmemory")
    core_mod = types.ModuleType("oracleagentmemory.core")
    core_mod.OracleAgentMemory = FakeAgentMemory
    core_mod.OracleDBMemoryStore = FakeStore
    policy_mod = types.ModuleType("oracleagentmemory.core.dbschemapolicy")
    policy_mod.SchemaPolicy = FakeSchemaPolicy
    pkg.core = core_mod

    monkeypatch.setitem(sys.modules, "oracleagentmemory", pkg)
    monkeypatch.setitem(sys.modules, "oracleagentmemory.core", core_mod)
    monkeypatch.setitem(
        sys.modules, "oracleagentmemory.core.dbschemapolicy", policy_mod
    )

    mem = layer._mem_lazy()
    assert mem is layer._mem  # cached
    # OracleDBMemoryStore was constructed with our pool + dim.
    assert captured["pool"] == "fake-pool"
    assert captured["vector_dim"] == cfg.embeddings.dim
    assert captured["schema_policy"] == FakeSchemaPolicy.CREATE_IF_NECESSARY
    assert captured["table_name_prefix"] == "ORAGRAPH_MEM_"
    assert captured["embedder"] is None


def test_memory_layer_lazy_import_connects_if_needed(monkeypatch):
    """If graph._pool is None, _mem_lazy() must call graph.connect() first."""
    cfg = Config()
    graph = MagicMock()
    graph._pool = None

    def _set_pool():
        graph._pool = "lazy-pool"

    graph.connect.side_effect = _set_pool
    layer = MemoryLayer(cfg, graph_store=graph)

    captured = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeAgentMemory:
        def __init__(self, *, store):
            self.store = store

    class FakeSchemaPolicy:
        CREATE_IF_NECESSARY = "create_if_necessary"

    pkg = types.ModuleType("oracleagentmemory")
    core_mod = types.ModuleType("oracleagentmemory.core")
    core_mod.OracleAgentMemory = FakeAgentMemory
    core_mod.OracleDBMemoryStore = FakeStore
    policy_mod = types.ModuleType("oracleagentmemory.core.dbschemapolicy")
    policy_mod.SchemaPolicy = FakeSchemaPolicy
    pkg.core = core_mod
    monkeypatch.setitem(sys.modules, "oracleagentmemory", pkg)
    monkeypatch.setitem(sys.modules, "oracleagentmemory.core", core_mod)
    monkeypatch.setitem(
        sys.modules, "oracleagentmemory.core.dbschemapolicy", policy_mod
    )

    layer._mem_lazy()
    graph.connect.assert_called_once()
    assert captured["pool"] == "lazy-pool"


def test_register_source_creates_thread(monkeypatch):
    """register_source returns the thread id on create."""
    cfg = Config()
    graph = MagicMock()
    graph._pool = "p"
    layer = MemoryLayer(cfg, graph_store=graph)

    created = {}

    class FakeThread:
        def __init__(self, tid):
            self.thread_id = tid

    class FakeAgentMemory:
        def __init__(self, *, store):
            pass

        def create_thread(self, *, thread_id, agent_id, metadata, **kwargs):
            created["thread_id"] = thread_id
            created["agent_id"] = agent_id
            created["metadata"] = metadata
            created["kwargs"] = kwargs
            return FakeThread(thread_id)

        def get_thread(self, tid):  # pragma: no cover — not called here
            raise AssertionError("get_thread should not be called on a fresh create")

    class FakeStore:
        def __init__(self, **kwargs):
            pass

    class FakeSchemaPolicy:
        CREATE_IF_NECESSARY = "create_if_necessary"

    pkg = types.ModuleType("oracleagentmemory")
    core_mod = types.ModuleType("oracleagentmemory.core")
    core_mod.OracleAgentMemory = FakeAgentMemory
    core_mod.OracleDBMemoryStore = FakeStore
    policy_mod = types.ModuleType("oracleagentmemory.core.dbschemapolicy")
    policy_mod.SchemaPolicy = FakeSchemaPolicy
    pkg.core = core_mod
    monkeypatch.setitem(sys.modules, "oracleagentmemory", pkg)
    monkeypatch.setitem(sys.modules, "oracleagentmemory.core", core_mod)
    monkeypatch.setitem(
        sys.modules, "oracleagentmemory.core.dbschemapolicy", policy_mod
    )

    tid = layer.register_source("src_test_id", "/some/folder")
    assert tid == "src_test_id"
    assert created == {
        "thread_id": "src_test_id",
        "agent_id": "oragraphrag",
        "metadata": {"folder": "/some/folder"},
        "kwargs": {"extract_memories": False},
    }


def test_register_source_falls_back_to_get_thread(monkeypatch):
    """If create_thread raises (thread already exists), fall back to get_thread."""
    cfg = Config()
    graph = MagicMock()
    graph._pool = "p"
    layer = MemoryLayer(cfg, graph_store=graph)

    class FakeThread:
        def __init__(self, tid):
            self.thread_id = tid

    class FakeAgentMemory:
        def __init__(self, *, store):
            pass

        def create_thread(self, **_kwargs):
            raise RuntimeError("already exists")

        def get_thread(self, tid):
            return FakeThread(tid)

    class FakeStore:
        def __init__(self, **kwargs):
            pass

    class FakeSchemaPolicy:
        CREATE_IF_NECESSARY = "create_if_necessary"

    pkg = types.ModuleType("oracleagentmemory")
    core_mod = types.ModuleType("oracleagentmemory.core")
    core_mod.OracleAgentMemory = FakeAgentMemory
    core_mod.OracleDBMemoryStore = FakeStore
    policy_mod = types.ModuleType("oracleagentmemory.core.dbschemapolicy")
    policy_mod.SchemaPolicy = FakeSchemaPolicy
    pkg.core = core_mod
    monkeypatch.setitem(sys.modules, "oracleagentmemory", pkg)
    monkeypatch.setitem(sys.modules, "oracleagentmemory.core", core_mod)
    monkeypatch.setitem(
        sys.modules, "oracleagentmemory.core.dbschemapolicy", policy_mod
    )

    tid = layer.register_source("src_existing", "/some/folder")
    assert tid == "src_existing"


@pytest.mark.parametrize("path_input", ["/tmp", Path("/tmp")])
def test_source_id_returns_str(path_input):
    sid = source_id_for_folder(path_input)
    assert isinstance(sid, str)
