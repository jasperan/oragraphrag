"""LLM adapter: oci_grok | ollama backends. Used by extract and answer modules.

The adapter is async-first. Backends conform to a narrow protocol with one
method (`complete`) that returns either text or a JSON object depending on
whether a schema is provided. Retry is at the adapter level via tenacity so
each backend stays simple.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from oragraphrag.config import Config, OciGrokConfig, OllamaConfig


class LLMError(RuntimeError):
    """Raised when a backend cannot satisfy the request after retries."""


class _HttpClient(Protocol):
    async def post(self, url: str, json: Any = None, timeout: float | None = None): ...


class OllamaBackend:
    """Talks to /api/chat with format=json when a schema is requested."""

    def __init__(self, cfg: OllamaConfig, http: _HttpClient | None = None):
        self.cfg = cfg
        self._http = http or httpx.AsyncClient()

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str | dict:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature},
            "stream": False,
        }
        if schema is not None:
            payload["format"] = "json"
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        resp = await self._http.post(url, json=payload, timeout=None)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        if schema is not None:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMError(f"ollama returned non-JSON when schema requested: {e}") from e
        return content


class OciGrokBackend:
    """OCI Generative AI Grok 4.3 backend.

    Auth via ~/.oci/config profile (default), falling back to instance
    principals when running inside OCI. This mirrors the Oracle AI Developer
    Hub "Choose Your Path" pattern.

    Note: the OCI SDK is synchronous; we wrap the call in `asyncio.to_thread`
    so the async surface is uniform with OllamaBackend.
    """

    def __init__(self, cfg: OciGrokConfig):
        self.cfg = cfg
        self._client = None

    def _client_init(self):
        if self._client is None:
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient

            try:
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                self._client = GenerativeAiInferenceClient(config={}, signer=signer)
            except Exception:
                cfg_file = oci.config.from_file()
                self._client = GenerativeAiInferenceClient(config=cfg_file)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str | dict:
        import asyncio

        from oci.generative_ai_inference.models import (
            ChatDetails,
            GenericChatRequest,
            Message,
            OnDemandServingMode,
            TextContent,
        )

        client = self._client_init()
        msg = Message(role="USER", content=[TextContent(text=prompt)])
        req = GenericChatRequest(
            messages=[msg],
            temperature=temperature,
            max_tokens=4096,
        )
        details = ChatDetails(
            compartment_id=self.cfg.compartment_ocid,
            serving_mode=OnDemandServingMode(model_id=self.cfg.model),
            chat_request=req,
        )
        resp = await asyncio.to_thread(client.chat, details)
        text = resp.data.chat_response.choices[0].message.content[0].text
        if schema is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"oci_grok returned non-JSON when schema requested: {e}") from e
        return text


class LLM:
    """Adapter selecting a backend by config; retries on transient failures.

    If `fallback_on_outage` is set in config, a failure of the primary
    backend after exhausting retries dispatches to the alternate backend
    one time. This is the Choose-Your-Path-style fallback, not a recovery
    shim — it is an explicit configuration option the user opts into.
    """

    def __init__(self, cfg: Config, http: _HttpClient | None = None):
        self.cfg = cfg
        self._http = http
        self._primary = self._build(cfg.llm.provider)
        self._fallback = (
            self._build("ollama" if cfg.llm.provider != "ollama" else "oci_grok")
            if cfg.llm.fallback_on_outage
            else None
        )

    def _build(self, provider: str):
        if provider == "ollama":
            return OllamaBackend(self.cfg.llm.ollama, http=self._http)
        if provider == "oci_grok":
            return OciGrokBackend(self.cfg.llm.oci_grok)
        raise LLMError(f"unsupported provider: {provider}")

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str | dict:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.cfg.llm.max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=10.0),
                retry=retry_if_exception_type((httpx.HTTPError, LLMError)),
                reraise=True,
            ):
                with attempt:
                    return await self._primary.complete(
                        prompt, schema=schema, temperature=temperature
                    )
        except (httpx.HTTPError, LLMError) as e:
            if self._fallback is not None:
                return await self._fallback.complete(
                    prompt, schema=schema, temperature=temperature
                )
            raise LLMError(
                f"primary backend failed after {self.cfg.llm.max_retries} retries"
            ) from e
        raise LLMError("unreachable")
