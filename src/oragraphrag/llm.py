"""LLM adapter: oci_grok | ollama backends. Used by extract and answer modules.

The adapter is async-first. Backends conform to a narrow protocol with one
method (`complete`) that returns either text or a JSON object depending on
whether a schema is provided. Retry is at the adapter level via tenacity so
each backend stays simple.
"""

from __future__ import annotations

import asyncio
import json
import os
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

    def __init__(
        self,
        cfg: OllamaConfig,
        timeout_s: float = 60.0,
        http: _HttpClient | None = None,
    ):
        self.cfg = cfg
        self._timeout_s = timeout_s
        self._http_owned = http is None
        self._http = http if http is not None else httpx.AsyncClient(timeout=self._timeout_s)

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
        resp = await self._http.post(url, json=payload, timeout=self._timeout_s)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        if schema is not None:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMError(f"ollama returned non-JSON when schema requested: {e}") from e
        return content

    async def aclose(self) -> None:
        """Close the underlying http client if we created it ourselves."""
        if self._http_owned and self._http is not None:
            # Only call aclose on real httpx.AsyncClient instances; stubs may not have it.
            close = getattr(self._http, "aclose", None)
            if close is not None:
                await close()


class OciGrokBackend:
    """OCI Generative AI Grok 4.3 backend.

    Auth resolves in this order:
      1. If env var OCI_RESOURCE_PRINCIPAL_VERSION is set (i.e., running inside
         OCI), use instance-principal signer.
      2. Otherwise, read ~/.oci/config (the standard dev path).

    The OCI SDK is synchronous; calls are wrapped in asyncio.to_thread so the
    async surface is uniform with OllamaBackend. Client construction is
    guarded by asyncio.Lock so concurrent first-callers do not race.
    """

    def __init__(self, cfg: OciGrokConfig):
        self.cfg = cfg
        self._client = None
        self._init_lock = asyncio.Lock()

    async def _client_init(self):
        async with self._init_lock:
            if self._client is None:
                self._client = await asyncio.to_thread(self._build_client_sync)
        return self._client

    def _build_client_sync(self):
        """Synchronous OCI SDK client construction. Called via asyncio.to_thread()."""
        import oci
        from oci.generative_ai_inference import GenerativeAiInferenceClient

        # Prefer ~/.oci/config (the common dev case). Only attempt instance
        # principals when explicitly running inside OCI (signaled by env var).
        use_instance_principals = bool(os.environ.get("OCI_RESOURCE_PRINCIPAL_VERSION"))
        if use_instance_principals:
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            return GenerativeAiInferenceClient(config={}, signer=signer)
        cfg_file = oci.config.from_file()
        return GenerativeAiInferenceClient(config=cfg_file)

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str | dict:
        # OCI SDK imports are kept inline so non-OCI users can import OllamaBackend
        # without paying the SDK import cost or needing it installed.
        from oci.generative_ai_inference.models import (
            ChatDetails,
            GenericChatRequest,
            Message,
            OnDemandServingMode,
            TextContent,
        )

        client = await self._client_init()
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

    async def aclose(self) -> None:
        """No-op; the OCI SDK client does not need explicit cleanup."""
        return None


class LLM:
    """Adapter selecting a backend by config; retries on transient failures.

    Use as an async context manager so the underlying http client closes
    cleanly:

        async with LLM(cfg) as llm:
            answer = await llm.complete(prompt)

    If `fallback_on_outage` is set in config, a failure of the primary
    backend after exhausting retries dispatches to the alternate backend
    one time. The fallback is a documented user-opt-in config feature,
    defaulting to False.
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
            return OllamaBackend(
                self.cfg.llm.ollama,
                self.cfg.llm.request_timeout_s,
                http=self._http,
            )
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

    async def aclose(self) -> None:
        await self._primary.aclose()
        if self._fallback is not None:
            await self._fallback.aclose()

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
