"""Vast.ai instance lifecycle: create, await running, discover public endpoint.

Verified against docs.vast.ai on 2026-09-02:
  * create  = PUT /api/v0/asks/{offer_id}/ → {"new_contract": id, ...}
  * status  = GET /api/v0/instances/{id}/ (cur_state: loading→running; terminal
    failure states: exited/unknown/offline)
  * ports   = GET /api/v1/instances → {"8000/tcp":[{"HostIp","HostPort"}]}
  * destroy = DELETE /api/v0/instances/{id}/
"""

from __future__ import annotations

import asyncio
from typing import Any

from hfvast.errors import ProviderError
from hfvast.providers.vast.client import VastClient
from hfvast.utils.redact import redact

# "stopped" is terminal for us: hfvast never stops instances, so a stopped
# instance during provisioning means the container failed to start (e.g. OCI
# runtime / driver mismatch, live-verified 2026-09-03).
TERMINAL_FAILURE_STATES = {"exited", "unknown", "offline", "stopped"}


async def create_instance(
    client: VastClient,
    offer_id: int,
    *,
    image: str,
    disk_gb: int,
    env: dict[str, str],
    onstart: str,
    label: str,
) -> int:
    """Create an instance from a specific offer; returns the instance id."""
    body: dict[str, Any] = {
        "image": image,
        "disk": int(disk_gb),
        "env": env,
        "onstart": onstart,
        "label": label[:1024],
        "runtype": "ssh",
        "target_state": "running",
        "cancel_unavail": True,
    }
    data = await client._request("PUT", f"/api/v0/asks/{offer_id}/", json_body=body)
    contract = data.get("new_contract") if isinstance(data, dict) else None
    if contract is None:
        raise ProviderError(f"Vast.ai did not return a contract id: {redact(str(data)[:200])}")
    return int(contract)


async def wait_running(
    client: VastClient,
    instance_id: int,
    *,
    timeout_s: float = 900.0,
    poll_interval_s: float = 5.0,
    on_poll: Any = None,
) -> dict[str, Any]:
    """Poll until the instance reports running (or a terminal failure)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        raw = await client.get_instance(instance_id)
        record = raw or {}
        state = str(record.get("cur_state") or record.get("actual_status") or "").lower()
        if state == "running":
            return record
        if state in TERMINAL_FAILURE_STATES:
            msg = record.get("status_msg") or "no status message from host"
            raise ProviderError(f"Vast instance {instance_id} entered terminal state '{state}': {redact(str(msg))}")
        if loop.time() >= deadline:
            raise ProviderError(
                f"Vast instance {instance_id} did not reach 'running' within {timeout_s:.0f}s "
                f"(last state: {state or 'unknown'})"
            )
        if on_poll:
            on_poll(state)
        await asyncio.sleep(poll_interval_s)


async def discover_endpoint(
    client: VastClient,
    instance_id: int,
    container_port: int,
    *,
    timeout_s: float = 240.0,
    poll_interval_s: float = 6.0,
) -> tuple[str, int]:
    """Return (host, public_port) for a mapped container port.

    The Docker port map materializes on the Vast instance list shortly after the
    container starts — poll until it appears (live-verified 2026-09-03: it is
    NOT instantaneous after cur_state=running).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    last_error = ""
    while True:
        raw = await client._request("GET", "/api/v1/instances")
        instances = raw.get("instances") if isinstance(raw, dict) else None
        if isinstance(instances, list):
            record = next((i for i in instances if int(i.get("id", -1)) == instance_id), None)
            if record is not None:
                ip = record.get("public_ipaddr")
                ports = record.get("ports") or {}
                mapping = ports.get(f"{container_port}/tcp") or []
                host_port = int(mapping[0].get("HostPort", 0)) if mapping else 0
                if ip and host_port:
                    return str(ip), host_port
                last_error = f"no public mapping for port {container_port} yet (ports: {redact(str(ports)[:120])})"
        # guard against a dead instance while we wait
        state_record = await client.get_instance(instance_id) or {}
        state = str(state_record.get("cur_state") or "").lower()
        if state in TERMINAL_FAILURE_STATES:
            raise ProviderError(f"Instance {instance_id} entered terminal state '{state}' while waiting for ports")
        if loop.time() >= deadline:
            raise ProviderError(
                f"Instance {instance_id}: {last_error or 'never appeared in instance list'} within {timeout_s:.0f}s"
            )
        await asyncio.sleep(poll_interval_s)
