"""The real packaged gateway binary, exercised end-to-end on localhost."""

import json

import httpx
from gateway_harness import FakeBackend, GatewayProcess


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-gateway-key-1234567890"}


async def test_gateway_requires_auth(tmp_path):
    with FakeBackend() as backend:
        gateway = GatewayProcess(tmp_path, backend.port).start()
        try:
            async with httpx.AsyncClient() as client:
                denied = await client.get(f"http://127.0.0.1:{gateway.port}/v1/models")
                assert denied.status_code == 401
                denied_post = await client.post(f"http://127.0.0.1:{gateway.port}/v1/chat/completions", json={})
                assert denied_post.status_code == 401
                state_denied = await client.get(f"http://127.0.0.1:{gateway.port}/internal/state")
                assert state_denied.status_code == 401
        finally:
            gateway.stop()


async def test_gateway_proxies_with_injected_backend_key(tmp_path):
    with FakeBackend() as backend:
        gateway = GatewayProcess(tmp_path, backend.port, state={"status": "ready"}).start()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
                    headers=_headers(),
                    json={"model": "model", "messages": []},
                )
                assert resp.status_code == 200
                # the gateway strips the user key and injects the backend key
                assert resp.json()["choices"][0]["message"]["content"].startswith("backend ok Bearer backend-key")
        finally:
            gateway.stop()


async def test_gateway_preserves_sse_streaming(tmp_path):
    with FakeBackend() as backend:
        gateway = GatewayProcess(tmp_path, backend.port, state={"status": "ready"}).start()
        try:
            arrivals: list[float] = []
            async with (
                httpx.AsyncClient(timeout=10) as client,
                client.stream(
                    "POST",
                    f"http://127.0.0.1:{gateway.port}/v1/chat/completions/stream",
                    headers=_headers(),
                    json={},
                ) as resp,
            ):
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        arrivals.append(line)
            # three SSE events arrived through the chunked proxy, unbuffered
            assert len(arrivals) == 3
            assert "DONE" in arrivals[-1]
        finally:
            gateway.stop()


async def test_gateway_health_reflects_bootstrap_state(tmp_path):
    with FakeBackend() as backend:
        gateway = GatewayProcess(
            tmp_path,
            backend.port,
            state={"status": "downloading", "message": "Q4_K_M", "bytes_done": 10, "bytes_total": 100},
        ).start()
        try:
            async with httpx.AsyncClient() as client:
                health = (await client.get(f"http://127.0.0.1:{gateway.port}/health")).json()
                assert health["status"] == "loading"
                assert health["bootstrap"]["bytes_total"] == 100

                gateway.set_state(status="ready", message="model ready")
                healthy = (await client.get(f"http://127.0.0.1:{gateway.port}/health")).json()
                assert healthy["status"] == "ok"
                assert healthy["backend"] == "ok"
        finally:
            gateway.stop()


async def test_gateway_tracks_activity(tmp_path):
    with FakeBackend() as backend:
        gateway = GatewayProcess(tmp_path, backend.port, state={"status": "ready", "ready_since": 1}).start()
        try:
            async with httpx.AsyncClient() as client:
                before = (
                    await client.get(f"http://127.0.0.1:{gateway.port}/internal/state", headers=_headers())
                ).json()
                assert before["last_activity"] == 0.0

                await client.post(
                    f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
                    headers=_headers(),
                    json={},
                )
                after = (await client.get(f"http://127.0.0.1:{gateway.port}/internal/state", headers=_headers())).json()
                assert after["last_activity"] > 0
                assert json.dumps(after)  # state payload is well-formed
        finally:
            gateway.stop()
