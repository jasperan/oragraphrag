"""Test fixtures.

`env_no_oracle` runs autouse for every test so a developer with `OGR__*` or
`ORACLE_*` exported in their shell never accidentally picks up real config
when tests construct `Config()` or `Settings()` with defaults.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def env_no_oracle(monkeypatch):
    """Strip OGR__ and ORACLE_ env vars so config defaults are tested."""
    for k in list(os.environ):
        if k.startswith("OGR__") or k.startswith("ORACLE_"):
            monkeypatch.delenv(k, raising=False)
    yield
