"""Re-export shim for backward compat. Real logic lives in metrics.py."""

from oragraphrag.bench.metrics import _judge_call, score_correctness

__all__ = ["_judge_call", "score_correctness"]
