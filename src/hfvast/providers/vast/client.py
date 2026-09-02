"""Typed thin wrapper around the official Vast.ai REST API.

Verified against https://docs.vast.ai/api-reference/ on 2026-09-02 (see
docs/research.md). Key behaviors encoded here:
  * Bearer auth; base https://console.vast.ai; paths under /api/v0/.
  * Per-endpoint minimum call intervals → serialized requests + 429 retry.
  * Error shapes: {"success": false, "error": "...", "msg": "..."}.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from hfvast.errors import ProviderAuthError, ProviderError, RateLimitError
from hfvast.utils.redact import redact

BASE_URL = "https://console.vast.ai"


def _is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


class VastClient:
    """Async HTTP client for the Vast.ai server API."""

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        min_interval: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(base_url=BASE_URL, follow_redirects=True, timeout=60.0)
        self._owns_client = client is None
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_request: float = 0.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _request(self, method: str, path: str, json_body: Any = None) -> Any:
        async with self._lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request = loop.time()
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            resp = await self._client.request(method, path, json=json_body, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Vast.ai is unreachable: {redact(exc)}") from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthError(
                f"Vast.ai rejected the API key (HTTP {resp.status_code}). "
                "Check VAST_API_KEY — create one at https://cloud.vast.ai/manage-keys/."
            )
        if resp.status_code == 429:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RateLimitError(f"Vast.ai rate limit exceeded after retries: {redact(_body(exc))}") from exc
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Vast.ai API error {resp.status_code} on {method} {path}: {redact(_body(exc))}"
            ) from exc

        if not resp.content:
            return {}
        data = resp.json()
        if isinstance(data, dict) and data.get("success") is False:
            msg = data.get("msg") or data.get("error") or "unknown error"
            raise ProviderError(f"Vast.ai rejected the request: {redact(msg)}")
        return data

    async def search_bundles(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        data = await self._request("POST", "/api/v0/bundles/", json_body=filters)
        offers = data.get("offers") if isinstance(data, dict) else None
        if not isinstance(offers, list):
            raise ProviderError("Vast.ai returned an unexpected offer-search response shape")
        return offers

    async def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        data = await self._request("GET", f"/api/v0/instances/{instance_id}/")
        instances = data.get("instances") if isinstance(data, dict) else None
        if isinstance(instances, dict):
            return dict(instances)
        if isinstance(instances, list) and instances:
            return dict(instances[0])
        return None

    async def destroy_instance(self, instance_id: int) -> None:
        await self._request("DELETE", f"/api/v0/instances/{instance_id}/")

    async def fetch_logs_url(self, instance_id: int, tail: int = 1000) -> str:
        data = await self._request(
            "PUT",
            f"/api/v0/instances/request_logs/{instance_id}/",
            json_body={"tail": str(tail)},
        )
        url = data.get("result_url") if isinstance(data, dict) else None
        if not url:
            raise ProviderError("Vast.ai did not return a log URL")
        return str(url)


def _body(exc: httpx.HTTPStatusError) -> str:
    text = exc.response.text[:300]
    return text if text else exc.response.reason_phrase
