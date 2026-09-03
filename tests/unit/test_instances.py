import json

import httpx
import pytest

from hfvast.errors import ProviderError
from hfvast.providers.vast.client import BASE_URL, VastClient
from hfvast.providers.vast.instances import create_instance, discover_endpoint, wait_running


def _client(handler) -> VastClient:
    return VastClient(
        api_key="test-key-00000000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL),
        min_interval=0.01,
    )


async def test_create_instance_returns_contract():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert "/api/v0/asks/900101/" in str(request.url)
        body = json.loads(request.content)
        assert body["image"].startswith("ghcr.io/ggml-org/llama.cpp")
        assert body["disk"] == 232
        assert body["runtype"] == "ssh"
        assert "-p 8000:8000" in body["env"]
        return httpx.Response(200, json={"success": True, "new_contract": 4242})

    client = _client(handler)
    instance_id = await create_instance(
        client,
        900101,
        image="ghcr.io/ggml-org/llama.cpp:server-cuda",
        disk_gb=232,
        env={"-p 8000:8000": "1"},
        onstart="#!/bin/bash\ntrue",
        label="hfvast test",
    )
    await client.aclose()
    assert instance_id == 4242


async def test_wait_running_polls_until_ready():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        state = "loading" if calls["n"] < 3 else "running"
        return httpx.Response(200, json={"instances": {"cur_state": state, "status_msg": None}})

    client = _client(handler)
    record = await wait_running(client, 4242, timeout_s=5, poll_interval_s=0.01)
    await client.aclose()
    assert record["cur_state"] == "running"
    assert calls["n"] >= 3


async def test_wait_running_fails_on_terminal_state():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"instances": {"cur_state": "exited", "status_msg": "docker died"}})

    client = _client(handler)
    with pytest.raises(ProviderError, match="terminal state 'exited'"):
        await wait_running(client, 4242, timeout_s=5, poll_interval_s=0.01)
    await client.aclose()


async def test_discover_endpoint_parses_port_map():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "instances": [
                    {
                        "id": 4242,
                        "public_ipaddr": "65.130.162.74",
                        "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]},
                    }
                ]
            },
        )

    client = _client(handler)
    host, port = await discover_endpoint(client, 4242, 8000)
    await client.aclose()
    assert (host, port) == ("65.130.162.74", 33526)


async def test_destroy_instance():
    destroyed = []

    def handler(request: httpx.Request) -> httpx.Response:
        destroyed.append(request.method)
        return httpx.Response(200, json={"success": True})

    client = _client(handler)
    await client.destroy_instance(4242)
    await client.aclose()
    assert destroyed == ["DELETE"]
