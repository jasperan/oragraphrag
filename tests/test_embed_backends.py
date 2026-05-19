import numpy as np
import pytest

from oragraphrag.config import Config
from oragraphrag.embed_backends import (
    OllamaEmbedBackend,
    OracleEmbedBackend,
    build_embed_backend,
)

# --- OllamaEmbedBackend ---


class _StubHttp:
    """Returns a constant embedding for any /api/embeddings POST."""

    def __init__(self, dim: int):
        self.dim = dim
        self.calls: list[dict] = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        outer_dim = self.dim

        class _Resp:
            status_code = 200

            def raise_for_status(self_):
                pass

            def json(self_):
                return {"embedding": [0.1] * outer_dim}

        return _Resp()


@pytest.mark.asyncio
async def test_ollama_embed_returns_correct_shape():
    cfg = Config()
    cfg.embeddings.provider = "ollama"
    cfg.embeddings.dim = 384
    backend = OllamaEmbedBackend(cfg, http=_StubHttp(384))
    arr = await backend.embed(["hello", "world"])
    assert arr.shape == (2, 384)
    assert arr.dtype == np.float32


@pytest.mark.asyncio
async def test_ollama_embed_posts_one_request_per_text():
    cfg = Config()
    cfg.embeddings.dim = 8
    client = _StubHttp(8)
    backend = OllamaEmbedBackend(cfg, http=client)
    await backend.embed(["a", "b", "c"])
    assert len(client.calls) == 3
    for call in client.calls:
        assert call["url"].endswith("/api/embeddings")


@pytest.mark.asyncio
async def test_ollama_embed_handles_empty_list():
    cfg = Config()
    cfg.embeddings.dim = 8
    client = _StubHttp(8)
    backend = OllamaEmbedBackend(cfg, http=client)
    arr = await backend.embed([])
    assert arr.shape == (0, 8)
    assert client.calls == []


@pytest.mark.asyncio
async def test_ollama_embed_uses_configured_base_url():
    cfg = Config()
    cfg.embeddings.dim = 8
    cfg.llm.ollama.base_url = "http://custom-host:9999/"  # trailing slash
    client = _StubHttp(8)
    backend = OllamaEmbedBackend(cfg, http=client)
    await backend.embed(["x"])
    assert client.calls[0]["url"] == "http://custom-host:9999/api/embeddings"


@pytest.mark.asyncio
async def test_ollama_embed_propagates_timeout_from_config():
    cfg = Config()
    cfg.embeddings.dim = 8
    cfg.llm.request_timeout_s = 12.5
    client = _StubHttp(8)
    backend = OllamaEmbedBackend(cfg, http=client)
    await backend.embed(["x"])
    assert client.calls[0]["timeout"] == 12.5


# --- OracleEmbedBackend ---


class _StubGraph:
    """Simulates GraphStore._conn().cursor() for VECTOR_EMBEDDING calls."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.execute_calls: list[dict] = []
        self._row_payload = [0.1] * dim

    def _conn(self):
        outer = self

        class _Conn:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                return False

            def cursor(self_inner):
                class _Cur:
                    def __enter__(s):
                        return s

                    def __exit__(s, *args):
                        return False

                    def execute(s, sql, **kwargs):
                        outer.execute_calls.append({"sql": sql, **kwargs})
                        s._row = (outer._row_payload,)

                    def fetchone(s):
                        return s._row

                return _Cur()

        return _Conn()


@pytest.mark.asyncio
async def test_oracle_embed_calls_vector_embedding_once_per_text():
    cfg = Config()
    cfg.embeddings.dim = 384
    g = _StubGraph(dim=384)
    backend = OracleEmbedBackend(cfg, graph=g)
    arr = await backend.embed(["hello", "world", "foo"])
    assert arr.shape == (3, 384)
    assert arr.dtype == np.float32
    assert len(g.execute_calls) == 3
    # The SQL must use VECTOR_EMBEDDING with the configured model.
    for call in g.execute_calls:
        assert "VECTOR_EMBEDDING" in call["sql"].upper()


@pytest.mark.asyncio
async def test_oracle_embed_handles_empty_list():
    cfg = Config()
    cfg.embeddings.dim = 384
    g = _StubGraph(dim=384)
    backend = OracleEmbedBackend(cfg, graph=g)
    arr = await backend.embed([])
    assert arr.shape == (0, 384)
    assert g.execute_calls == []


@pytest.mark.asyncio
async def test_oracle_embed_uses_configured_model():
    cfg = Config()
    cfg.embeddings.dim = 384
    cfg.embeddings.oracle.model = "ALL_MPNET_BASE_V2"
    g = _StubGraph(dim=384)
    backend = OracleEmbedBackend(cfg, graph=g)
    await backend.embed(["x"])
    # The model name should appear in the bind for each call.
    assert any(
        v == "ALL_MPNET_BASE_V2"
        for call in g.execute_calls
        for v in call.values()
    )


# --- build_embed_backend factory ---


def test_factory_returns_oracle_backend_when_provider_oracle():
    cfg = Config()
    cfg.embeddings.provider = "oracle"
    g = _StubGraph()
    backend = build_embed_backend(cfg, g)
    assert isinstance(backend, OracleEmbedBackend)


def test_factory_returns_ollama_backend_when_provider_ollama():
    cfg = Config()
    cfg.embeddings.provider = "ollama"
    g = _StubGraph()
    backend = build_embed_backend(cfg, g)
    assert isinstance(backend, OllamaEmbedBackend)


def test_factory_raises_on_unknown_provider():
    cfg = Config()
    # Bypass pydantic Literal validation for the test by directly poking the field.
    object.__setattr__(cfg.embeddings, "provider", "nonsense")
    g = _StubGraph()
    with pytest.raises(ValueError, match="unknown"):
        build_embed_backend(cfg, g)


def test_factory_raises_on_sentence_transformers_when_extra_not_installed(monkeypatch):
    """sentence-transformers is an optional extra. The factory must raise a
    clear error when the package isn't importable, NOT crash with ImportError
    deep inside a transitive import."""
    import sys

    cfg = Config()
    cfg.embeddings.provider = "sentence_transformers"
    g = _StubGraph()
    # Simulate missing sentence_transformers package.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises((ImportError, ValueError)):
        build_embed_backend(cfg, g)
