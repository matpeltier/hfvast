"""Full deployment orchestration against a mocked Vast API + real local gateway.

No cloud access, no spending — the Vast REST surface is a MockTransport and the
"instance endpoint" is the real packaged gateway subprocess on 127.0.0.1.
"""

import json
import threading
import time

import httpx
import pytest
from conftest import hf_mock_transport
from gateway_harness import FakeBackend, FakeBackendHandler, GatewayProcess

from hfvast.deploy.lifecycle import DeploymentOrchestrator, DeployOptions, DeploySecrets
from hfvast.errors import HfvastError
from hfvast.inspect.huggingface import HFInspector
from hfvast.planning.quote import QuoteBuilder, QuoteOptions
from hfvast.providers.vast.client import BASE_URL, VastClient
from hfvast.providers.vast.offers import SnapshotProvider, VastProvider
from hfvast.state import DeploymentStore
from hfvast.utils.hfref import parse_model_input

INSTANCE_ID = 4242
_gateway_port = [0]


def vast_mock(destroyed: list[int], created: list[int]):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "PUT" and "/asks/" in url:
            body = json.loads(request.content)
            assert body["runtype"] == "ssh"
            assert body["onstart"].startswith("#!/bin/sh")  # POSIX sh (dash-safe)
            assert "-p 8000:8000" in body["env"]
            assert "HF_TOKEN" not in body["env"] or body["env"]["HF_TOKEN"].startswith("hf_")
            created.append(1)
            return httpx.Response(200, json={"success": True, "new_contract": INSTANCE_ID})
        if request.method == "GET" and "/api/v0/instances/" in url:
            return httpx.Response(200, json={"instances": {"cur_state": "running", "status_msg": None}})
        if request.method == "GET" and "/api/v1/instances" in url:
            return httpx.Response(
                200,
                json={
                    "instances": [
                        {
                            "id": INSTANCE_ID,
                            "public_ipaddr": "127.0.0.1",
                            "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(_gateway_port[0])}]},
                        }
                    ]
                },
            )
        if request.method == "DELETE":
            destroyed.append(INSTANCE_ID)
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404, text="unexpected")

    return httpx.MockTransport(handler)


async def _build_quote():
    inspector = HFInspector(client=httpx.AsyncClient(transport=hf_mock_transport(), follow_redirects=True))
    try:
        builder = QuoteBuilder(inspector, SnapshotProvider())
        return await builder.build(
            parse_model_input("orcarouter/GLM-5.3-Flash-Uncensored-GGUF"),
            QuoteOptions(quant="Q2_K", context=8192, expected_session_hours=2.0),
        )
    finally:
        await inspector.aclose()


def _orchestrator(tmp_path, destroyed: list[int], created: list[int]):
    vast_key, hf = "test-vast-key-000000000", None
    client = VastClient(
        api_key=vast_key,
        client=httpx.AsyncClient(transport=vast_mock(destroyed, created), base_url=BASE_URL),
    )
    provider = VastProvider(client=client)
    store = DeploymentStore(path=tmp_path / "deployments.json")
    secrets = DeploySecrets(vast_api_key=vast_key, hf_token=hf)
    orch = DeploymentOrchestrator(provider, store, secrets, http=httpx.AsyncClient(timeout=10.0))
    return orch, store


async def test_deploy_end_to_end_ready_then_destroy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hfvast.deploy.lifecycle.secrets.token_urlsafe",
        lambda n=24: "test-gateway-key-1234567890",
    )
    quote = await _build_quote()
    assert quote.recommendation is not None

    destroyed: list[int] = []
    created: list[int] = []
    with FakeBackend() as backend:
        FakeBackendHandler.healthy = True
        gw = GatewayProcess(
            tmp_path,
            backend.port,
            state={"status": "downloading", "message": "q"},
            gateway_key="sk-hfvast-test-gateway-key-1234567890",
        ).start()
        _gateway_port[0] = gw.port
        # simulate the in-instance bootstrap finishing while deploy() polls
        timer = threading.Timer(
            0.5,
            lambda: gw.set_state(status="ready", message="model ready", ready_since=time.time()),
        )
        timer.start()
        orch, store = _orchestrator(tmp_path, destroyed, created)
        try:
            deployment = await orch.deploy(
                quote,
                DeployOptions(
                    poll_interval_s=0.05,
                    instance_timeout_s=5,
                    ready_timeout_s=20,
                    idle_timeout_s=1800,
                    max_runtime_s=21600,
                ),
            )
            assert deployment.status == "ready"
            assert deployment.instance_id == INSTANCE_ID
            assert deployment.endpoint == f"http://127.0.0.1:{gw.port}"
            assert deployment.api_key == "sk-hfvast-test-gateway-key-1234567890"
            assert deployment.hourly_total_usd > 0
            assert deployment.idle_timeout_s == 1800.0
            # persisted with the endpoint and ready status
            persisted = store.get(deployment.id)
            assert persisted is not None
            assert persisted.status == "ready"
            assert persisted.uptime_s >= 0
            assert created == [1]

            await orch.destroy(deployment, reason="test")
            assert deployment.status == "destroyed"
            assert deployment.destroyed_at is not None
            assert destroyed == [INSTANCE_ID]
            assert store.get(deployment.id).status == "destroyed"
        finally:
            timer.cancel()
            await orch.aclose()
            gw.stop()


async def test_deploy_failure_cleans_up_instance(tmp_path):
    quote = await _build_quote()

    destroyed: list[int] = []
    created: list[int] = []
    with FakeBackend() as backend:
        FakeBackendHandler.healthy = True  # backend healthy — bootstrap state says error
        gw = GatewayProcess(
            tmp_path, backend.port, state={"status": "error", "message": "model failed to load"}
        ).start()
        _gateway_port[0] = gw.port
        orch, store = _orchestrator(tmp_path, destroyed, created)
        try:
            with pytest.raises(HfvastError, match="failed on all candidate offers"):
                await orch.deploy(
                    quote,
                    DeployOptions(poll_interval_s=0.05, instance_timeout_s=5, ready_timeout_s=5),
                )
            # spec §30: each failed attempt destroyed its instance automatically
            assert destroyed == [INSTANCE_ID] * 4
            assert created == [1] * 4
            failed = [d for d in store.list_all() if d.status == "failed"]
            assert failed, "failed deployment is recorded for post-mortem"
            assert failed[0].last_error and "model failed to load" in failed[0].last_error
        finally:
            await orch.aclose()
            gw.stop()
