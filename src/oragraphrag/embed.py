"""Embedding adapter. Wraps Oracle 23ai VECTOR_EMBEDDING / Ollama / sentence-transformers.

Concrete backends are wired in Task 13 (`embed_backends.py`). Until then,
callers inject a backend that exposes `dim: int` and `async embed(texts) -> np.ndarray`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from oragraphrag.axes import AXIS_DESCRIPTIONS, ONTOLOGY_AXIS_NAMES
from oragraphrag.config import Config


class _EmbedBackend(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class Embedder:
    """Normalizes shape, optionally L2-normalizes, asserts dim matches config."""

    def __init__(self, cfg: Config, backend: _EmbedBackend):
        self.cfg = cfg
        self._backend = backend
        if backend.dim != cfg.embeddings.dim:
            raise ValueError(
                f"backend dim {backend.dim} != config dim {cfg.embeddings.dim}; "
                f"run `oragraphrag init-db --rebuild`"
            )

    async def embed(self, texts: Iterable[str], *, normalize: bool = True) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, self.cfg.embeddings.dim), dtype=np.float32)
        out = await self._backend.embed(text_list)
        if normalize:
            out = self._l2_normalize(out)
        return out

    @staticmethod
    def _l2_normalize(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n = np.where(n == 0, 1.0, n)
        return a / n


async def build_axis_vectors(embedder: _EmbedBackend | Embedder) -> dict[str, np.ndarray]:
    """Embed the canonical description of each ontology axis once.

    Returns a dict {axis_name: np.ndarray(dim,)} suitable for storage in the
    Oracle `Ontology_Axis` table at `init-db` time. The vectors are NOT
    L2-normalized here; Task 9 normalizes on the fly when computing the
    cosine projection of the query onto each axis. (This keeps the table
    storage closer to whatever the underlying model produces.)

    Accepts either a raw backend (for tests) or a full Embedder. Both must
    expose an async `embed(texts: list[str]) -> np.ndarray` method.
    """
    names = list(ONTOLOGY_AXIS_NAMES)
    descs = [AXIS_DESCRIPTIONS[n] for n in names]
    mat = await embedder.embed(descs)
    return {name: mat[i] for i, name in enumerate(names)}
