"""Deployment health pollers (instance → endpoint → ready)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from hfvast.errors import HfvastError

ProgressCallback = Callable[[str], Awaitable[None]]


class BootstrapTimeoutError(HfvastError):
    pass


class BootstrapFailedError(HfvastError):
    pass


async def wait_endpoint_ready(
    endpoint: str,
    gateway_key: str,
    *,
    timeout_s: float = 5400.0,
    poll_interval_s: float = 10.0,
    client: httpx.AsyncClient | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Poll the gateway until the model reports ready (or bootstrap fails).

    Returns the final /health payload. Raises BootstrapFailedError when the
    in-instance bootstrap reports an error, BootstrapTimeoutError on timeout.
    """
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=15.0)
    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {gateway_key}"}

    #: If the endpoint has NEVER answered within this window, the host is
    #: almost certainly unroutable from this network (Vast geolocation data is
    #: unreliable — live-verified 2026-09-03) — fail fast to the next offer.
    never_reachable_deadline_s = 300.0
    ever_reachable = False
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_message = ""
        while True:
            try:
                resp = await http.get(f"{base}/health", headers=headers)
                if resp.status_code == 200:
                    data: dict[str, Any] = resp.json()
                    status = data.get("status")
                    bootstrap = data.get("bootstrap") or {}
                    message = str(bootstrap.get("message") or "")
                    bytes_done = int(bootstrap.get("bytes_done") or 0)
                    bytes_total = int(bootstrap.get("bytes_total") or 0)
                    if on_progress and (message != last_message or status == "ready"):
                        last_message = message
                        pct = f" ({bytes_done / bytes_total:.0%})" if bytes_total else ""
                        await on_progress(f"{status}: {message}{pct}")
                    ever_reachable = True
                    if status == "ok":
                        return data
                    if status == "error":
                        raise BootstrapFailedError(
                            f"bootstrap failed on the instance: {message or 'unknown error'} "
                            "(fetch `hfvast logs` for details)"
                        )
                else:
                    if on_progress:
                        await on_progress(f"gateway HTTP {resp.status_code}")
            except httpx.HTTPError:
                if on_progress:
                    await on_progress("endpoint unreachable, retrying…")
            if loop.time() >= deadline:
                raise BootstrapTimeoutError(f"deployment did not become ready within {timeout_s / 60:.0f} minutes")
            await asyncio.sleep(poll_interval_s)
    finally:
        if owns_client:
            await http.aclose()


async def fetch_instance_log(
    endpoint: str,
    gateway_key: str,
    name: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Fetch the tail of an in-instance log via the gateway (authed)."""
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=15.0)
    try:
        try:
            resp = await http.get(
                f"{endpoint.rstrip('/')}/internal/log?name={name}",
                headers={"Authorization": f"Bearer {gateway_key}"},
            )
            return resp.text[-4000:] if resp.status_code == 200 else ""
        except httpx.HTTPError:
            return ""
    finally:
        if owns_client:
            await http.aclose()


async def smoke_test(
    endpoint: str,
    gateway_key: str,
    *,
    timeout_s: float = 120.0,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Minimal chat completion to prove the OpenAI-compatible path works."""
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        resp = await http.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {gateway_key}"},
            json={
                "model": "model",
                "messages": [{"role": "user", "content": "Say 'ready'."}],
                "max_tokens": 16,
                "stream": False,
            },
        )
        if resp.status_code != 200:
            raise HfvastError(f"smoke test failed: HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise HfvastError(f"smoke test returned unexpected payload: {exc}") from exc
    finally:
        if owns_client:
            await http.aclose()
