import json

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


@pytest.mark.asyncio
async def test_ollama_complete_returns_text():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": "hello"}})
    backend = OllamaBackend(cfg.llm.ollama, http=client)
    out = await backend.complete("prompt")
    assert out == "hello"
    assert client.calls[0][0].endswith("/api/chat")


@pytest.mark.asyncio
async def test_ollama_complete_returns_json_when_schema_given():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": json.dumps({"x": 1})}})
    backend = OllamaBackend(cfg.llm.ollama, http=client)
    out = await backend.complete("prompt", schema={"type": "object"})
    assert out == {"x": 1}


@pytest.mark.asyncio
async def test_llm_routes_to_configured_provider():
    cfg = Config()
    cfg.llm.provider = "ollama"
    client = _StubHttpClient({"message": {"content": "ok"}})
    llm = LLM(cfg, http=client)
    assert await llm.complete("p") == "ok"
