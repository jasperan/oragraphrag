import os
import pytest


@pytest.fixture
def env_no_oracle(monkeypatch):
    """Strip Oracle env vars so config defaults are tested."""
    for k in list(os.environ):
        if k.startswith("OGR__") or k.startswith("ORACLE_"):
            monkeypatch.delenv(k, raising=False)
    yield
