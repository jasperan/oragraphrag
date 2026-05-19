"""Concrete embedding backends: Oracle VECTOR_EMBEDDING, Ollama, sentence-transformers.

All three conform to the `_EmbedBackend` Protocol in `embed.py`: a `dim: int`
attribute and `async embed(texts: list[str]) -> np.ndarray`. The Embedder
wrapper validates dim consistency against config and optionally L2-normalizes
the output.

`build_embed_backend(cfg, graph)` is the factory the CLI uses.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx
import numpy as np

from oragraphrag.config import Config


class _HttpClient(Protocol):
    async def post(
        self,
        url: str,
        json: Any = None,
        timeout: float | None = None,
    ) -> Any: ...


class OllamaEmbedBackend:
    """One request per text against Ollama's /api/embeddings endpoint.

    Ollama doesn't expose a batch embeddings API today, so concurrent text
    embedding happens at the caller level (Task 8's 8-way pipeline) — this
    backend itself is sequential per call.
    """

    def __init__(self, cfg: Config, http: _HttpClient | None = None):
        self.cfg = cfg
        self.dim = cfg.embeddings.dim
        self._http = http if http is not None else httpx.AsyncClient()
        self._timeout_s = cfg.llm.request_timeout_s
        self._owns_http = http is None

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        url = f"{self.cfg.llm.ollama.base_url.rstrip('/')}/api/embeddings"
        for i, t in enumerate(texts):
            resp = await self._http.post(
                url,
                json={"model": self.cfg.embeddings.ollama.model, "prompt": t},
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
            out[i] = np.array(resp.json()["embedding"], dtype=np.float32)
        return out

    async def aclose(self) -> None:
        if self._owns_http and hasattr(self._http, "aclose"):
            await self._http.aclose()


class OracleEmbedBackend:
    """Uses Oracle 23ai's VECTOR_EMBEDDING(:m USING :t AS data) FROM dual.

    Synchronous Oracle calls wrapped via asyncio.to_thread so the async
    surface is uniform with the other backends. The connection pool on
    `graph` is reused — no separate connection management.
    """

    def __init__(self, cfg: Config, graph: Any):
        self.cfg = cfg
        self.dim = cfg.embeddings.dim
        self._graph = graph

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with self._graph._conn() as c, c.cursor() as cur:
            for i, t in enumerate(texts):
                cur.execute(
                    "SELECT VECTOR_EMBEDDING(:m USING :t AS data) FROM dual",
                    m=self.cfg.embeddings.oracle.model,
                    t=t,
                )
                row = cur.fetchone()
                out[i] = np.array(row[0], dtype=np.float32)
        return out


class SentenceTransformersBackend:
    """Local model load via the sentence-transformers extra.

    Only available when the optional extra is installed:
        pip install oragraphrag[sentence-transformers]
    The model is loaded once at construction. encode() is synchronous;
    wrapped via asyncio.to_thread for the async surface.
    """

    def __init__(
        self,
        cfg: Config,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Install with "
                "`pip install oragraphrag[sentence-transformers]` or set "
                "cfg.embeddings.provider to 'oracle' or 'ollama'."
            ) from e

        self.cfg = cfg
        self.dim = cfg.embeddings.dim
        self._model = SentenceTransformer(model_name)
        model_dim = self._model.get_sentence_embedding_dimension()
        if model_dim != cfg.embeddings.dim:
            raise ValueError(
                f"sentence-transformers model dim {model_dim} != config dim "
                f"{cfg.embeddings.dim}"
            )

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=False
        )
        return np.asarray(arr, dtype=np.float32)


def build_embed_backend(cfg: Config, graph: Any) -> Any:
    """Return the embedding backend keyed on cfg.embeddings.provider.

    For 'sentence_transformers', surfaces an ImportError if the optional
    extra isn't installed — better than a deep transitive crash later.
    """
    provider = cfg.embeddings.provider
    if provider == "oracle":
        return OracleEmbedBackend(cfg, graph=graph)
    if provider == "ollama":
        return OllamaEmbedBackend(cfg)
    if provider == "sentence_transformers":
        return SentenceTransformersBackend(cfg)
    raise ValueError(f"unknown embeddings.provider: {provider!r}")
