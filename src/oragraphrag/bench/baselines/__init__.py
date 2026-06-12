"""Baseline registry used by the bench runner.

REGISTRY is keyed on the system name (CLI flag value) and maps to a module
that exposes `async def run(question: str, cfg: Config) -> dict`.

graphrag and lightrag are placeholder stubs pending full implementations.
"""

from . import graphrag, lightrag, naive_rag, oragraphrag

REGISTRY = {
    "naive_rag": naive_rag,
    "graphrag": graphrag,
    "lightrag": lightrag,
    "oragraphrag": oragraphrag,
}
