import numpy as np
import pytest

from oragraphrag.axes import ONTOLOGY_AXIS_NAMES
from oragraphrag.config import Config
from oragraphrag.embed import Embedder, build_axis_vectors


class _StubEmbedder:
    """Stub embedder used by the build_axis_vectors test."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        rng = np.random.default_rng(seed=42)
        return rng.standard_normal((len(texts), self.dim)).astype(np.float32)


def test_axis_names_are_five_in_order():
    assert ONTOLOGY_AXIS_NAMES == (
        "causal",
        "taxonomic",
        "temporal",
        "definitional",
        "exemplification",
    )


@pytest.mark.asyncio
async def test_build_axis_vectors_uses_canonical_descriptions():
    emb = _StubEmbedder()
    vecs = await build_axis_vectors(emb)
    assert set(vecs.keys()) == set(ONTOLOGY_AXIS_NAMES)
    assert all(v.shape == (384,) for v in vecs.values())
    # All five axes embedded in one batch call:
    assert len(emb.calls) == 1
    assert len(emb.calls[0]) == 5


def test_embedder_l2_normalize_static_method():
    arr = np.array([[3.0, 4.0], [0.0, 1.0]], dtype=np.float32)
    out = Embedder._l2_normalize(arr)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


@pytest.mark.asyncio
async def test_embedder_returns_normalized_vectors():
    cfg = Config()
    cfg.embeddings.dim = 384
    backend = _StubEmbedder(dim=384)
    emb = Embedder(cfg, backend=backend)
    out = await emb.embed(["hello", "world"], normalize=True)
    assert out.shape == (2, 384)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.asyncio
async def test_embedder_returns_unnormalized_when_asked():
    cfg = Config()
    cfg.embeddings.dim = 384
    backend = _StubEmbedder(dim=384)
    emb = Embedder(cfg, backend=backend)
    out = await emb.embed(["hello"], normalize=False)
    norm = float(np.linalg.norm(out[0]))
    # Stub returns standard-normal random; norm is essentially never 1.0.
    assert norm > 0
    assert not np.isclose(norm, 1.0, atol=1e-3)


@pytest.mark.asyncio
async def test_embedder_empty_input_returns_correct_shape():
    cfg = Config()
    cfg.embeddings.dim = 384
    backend = _StubEmbedder(dim=384)
    emb = Embedder(cfg, backend=backend)
    out = await emb.embed([])
    assert out.shape == (0, 384)


def test_embedder_rejects_dim_mismatch():
    cfg = Config()
    cfg.embeddings.dim = 384
    backend = _StubEmbedder(dim=768)  # wrong dim
    with pytest.raises(ValueError, match="dim"):
        Embedder(cfg, backend=backend)
