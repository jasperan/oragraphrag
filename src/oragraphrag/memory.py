"""Per-source memory layer backed by ``oracleagentmemory``.

Each ``/graphify`` invocation creates one OracleAgentMemory thread keyed on
the source folder's path hash. Propositions are mirrored as semantic
memories under that thread so the package's search and context-card APIs
work across sources without us re-implementing them.

Our property graph remains the semantic-graph source of truth for the
reweighting math. This module is the metadata index.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from oragraphrag.config import Config


def source_id_for_folder(folder: Path | str) -> str:
    """Stable, file-system-path-derived source_id for a folder.

    Uses ``sha256(realpath)`` hex truncated to 32 chars + a ``src_`` prefix so
    it fits Oracle ``VARCHAR2(64)`` without collisions in practice.
    """
    real = str(Path(folder).resolve())
    digest = hashlib.sha256(real.encode()).hexdigest()[:32]
    return f"src_{digest}"


class MemoryLayer:
    """Thin wrapper around ``oracleagentmemory.OracleAgentMemory``.

    Construction is lazy: ``oracleagentmemory`` imports a lot of optional
    dependencies (embedder libs, etc.) so we defer until first call. The
    package connects to the same Oracle pool used by ``GraphStore`` so it
    shares the existing connection budget.
    """

    def __init__(self, cfg: Config, graph_store: Any):
        self.cfg = cfg
        self._graph = graph_store
        self._mem: Any | None = None

    def _mem_lazy(self) -> Any:
        if self._mem is not None:
            return self._mem
        # Import lazily; the package has heavy transitive deps.
        from oracleagentmemory.core import OracleAgentMemory, OracleDBMemoryStore
        from oracleagentmemory.core.dbschemapolicy import SchemaPolicy

        # Reuse our existing graph store's pool.
        if self._graph._pool is None:
            self._graph.connect()
        pool = self._graph._pool

        # We pass ``embedder=None`` since we manage embeddings ourselves;
        # oracleagentmemory will store text-only memories. If the package
        # rejects None, the caller sees a TypeError/ValueError and can
        # decide how to react.
        store = OracleDBMemoryStore(
            embedder=None,
            pool=pool,
            schema_policy=SchemaPolicy.CREATE_IF_NECESSARY,
            vector_dim=self.cfg.embeddings.dim,
            table_name_prefix="ORAGRAPH_MEM_",
        )
        self._mem = OracleAgentMemory(store=store)
        return self._mem

    def register_source(self, source_id: str, folder: str) -> str:
        """Create or retrieve a thread for this source. Returns ``thread_id``."""
        mem = self._mem_lazy()
        try:
            thread = mem.create_thread(
                thread_id=source_id,
                agent_id="oragraphrag",
                metadata={"folder": folder},
                extract_memories=False,
            )
        except Exception:
            # Thread already exists; fetch it.
            thread = mem.get_thread(source_id)
        return thread.thread_id

    def list_sources(self) -> list[str]:
        """Delegate to the graph store.

        ``oracleagentmemory``'s listing API requires iterating all threads
        which is heavier than a single ``SELECT DISTINCT`` against Entity.
        """
        return self._graph.list_sources()
