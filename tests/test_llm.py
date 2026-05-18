import json
from typing import Any

import pytest

from oragraphrag.config import Config
from oragraphrag.llm import LLM, OllamaBackend


class _StubResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


class _StubHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return _StubResponse(self.payload)


class _StubHttpClientCapturingKwargs:
    """Like _StubHttpClient but exposes the full kwargs of the last call."""

    def __init__(self, payload):
        self.payload = payload
        self.last_kwargs: dict[str, Any] = {}

    async def post(self, url, **kwargs):
        self.last_kwargs = {"url": url, **kwargs}
        return _StubResponse(self.payload)


@pytest.mark.asyncio
async def test_ollama_complete_returns_text():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": "hello"}})
    backend = OllamaBackend(cfg.llm.ollama, cfg.llm.request_timeout_s, http=client)
    out = await backend.complete("prompt")
    assert out == "hello"
    assert client.calls[0][0].endswith("/api/chat")


@pytest.mark.asyncio
async def test_ollama_complete_returns_json_when_schema_given():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": json.dumps({"x": 1})}})
    backend = OllamaBackend(cfg.llm.ollama, cfg.llm.request_timeout_s, http=client)
    out = await backend.complete("prompt", schema={"type": "object"})
    assert out == {"x": 1}
    _, payload = client.calls[0]
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_llm_routes_to_configured_provider():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": "ok"}})
    llm = LLM(cfg, http=client)
    assert await llm.complete("p") == "ok"


@pytest.mark.asyncio
async def test_ollama_passes_configured_timeout_to_post():
    cfg = Config()
    cfg.llm.provider = "ollama"
    cfg.llm.request_timeout_s = 12.5
    client = _StubHttpClientCapturingKwargs({"message": {"content": "x"}})
    backend = OllamaBackend(cfg.llm.ollama, cfg.llm.request_timeout_s, http=client)
    await backend.complete("p")
    assert client.last_kwargs.get("timeout") == 12.5


@pytest.mark.asyncio
async def test_llm_context_manager_closes_owned_client():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": "ok"}})
    async with LLM(cfg, http=client) as llm:
        await llm.complete("p")
    # The injected stub is NOT owned, so aclose should not have been invoked
    # against a missing aclose method — test passes by not raising.
