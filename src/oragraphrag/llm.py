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

    Patterned after agent-harness's OCIGenAIProvider. Auth resolves in this
    order:
      1. Instance-principal signer (works on OCI compute).
      2. ~/.oci/config API-key profile (OCI_CONFIG_PROFILE env var).

    Compartment id comes from OCI_COMPARTMENT_ID env or
    cfg.llm.oci_grok.compartment_ocid (env wins if set). Region comes from
    OCI_REGION env or cfg.llm.oci_grok.region (env wins).

    The OCI SDK is synchronous; chat calls go through asyncio.to_thread so
    the async surface is uniform with OllamaBackend. Client construction is
    guarded by asyncio.Lock so concurrent first-callers don't race.

    Uses the role-specific message subclass UserMessage (not the generic
    Message) and api_format="GENERIC" — required for xAI Grok models on the
    OCI on-demand chat endpoint.
    """

    def __init__(self, cfg: OciGrokConfig):
        self.cfg = cfg
        self._client = None
        self._compartment_id: str | None = None
        self._region: str | None = None
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

        # Region: env wins over config.
        self._region = os.environ.get("OCI_REGION") or self.cfg.region or "us-chicago-1"
        endpoint = f"https://inference.generativeai.{self._region}.oci.oraclecloud.com"

        # Compartment: env wins over config.
        env_compartment = os.environ.get("OCI_COMPARTMENT_ID")
        self._compartment_id = env_compartment or self.cfg.compartment_ocid or None

        # Auth: instance principal first, then ~/.oci/config profile.
        try:
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            client = GenerativeAiInferenceClient(
                config={},
                signer=signer,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
            )
        except Exception:
            profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
            config_path = os.environ.get("OCI_CONFIG_FILE", "~/.oci/config")
            oci_config = oci.config.from_file(file_location=config_path, profile_name=profile)
            if not self._compartment_id:
                self._compartment_id = oci_config.get("compartment-id") or oci_config.get("tenancy")
            client = GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
            )

        if not self._compartment_id:
            raise LLMError(
                "OCI compartment id missing. Set OCI_COMPARTMENT_ID env var or "
                "configure cfg.llm.oci_grok.compartment_ocid."
            )
        return client

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str | dict:
        # OCI SDK imports are kept inline so non-OCI users can import
        # OllamaBackend without paying the SDK import cost.
        from oci.generative_ai_inference.models import (
            ChatDetails,
            GenericChatRequest,
            OnDemandServingMode,
            TextContent,
            UserMessage,
        )

        client = await self._client_init()
        user_msg = UserMessage(role="USER", content=[TextContent(text=prompt)])
        chat_request = GenericChatRequest(
            api_format="GENERIC",
            messages=[user_msg],
            temperature=temperature,
            max_tokens=4096,
            is_stream=False,
            top_p=1.0,
        )
        details = ChatDetails(
            compartment_id=self._compartment_id,
            serving_mode=OnDemandServingMode(model_id=self.cfg.model),
            chat_request=chat_request,
        )
        resp = await asyncio.to_thread(client.chat, details)
        # Response shape: resp.data.chat_response.choices[0].message.content[0].text
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
