"""OraGraphRAG — Oracle-backed graph-augmented RAG with dynamic edge reweighting."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oragraphrag")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
