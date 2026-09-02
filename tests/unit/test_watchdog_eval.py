"""Watchdog decision logic (spec §28): idle, max-runtime, active requests, unreachable."""

from datetime import UTC, datetime, timedelta

import httpx

from hfvast.deploy.watchdog import UNREACHABLE_GRACE_S, evaluate
from hfvast.state import Deployment

NOW = datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)


def _deployment(**kwargs) -> Deployment:
    defaults = dict(
        id="dep-0001",
        model_repo="o/m",
        variant_id="Q4_K_M",
        backend="llama.cpp",
        status="ready",
        endpoint="http://127.0.0.1:9",
        api_key="sk-hfvast-test",
        created_at=NOW - timedelta(minutes=10),
        ready_at=NOW - timedelta(minutes=5),
        idle_timeout_s=1800.0,
        max_runtime_s=6 * 3600.0,
    )
    defaults.update(kwargs)
    return Deployment(**defaults)


def _state_handler(payload: dict, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


async def _evaluate(deployment: Deployment, payload: dict, **kwargs):
    client = httpx.AsyncClient(transport=_state_handler(payload))
    try:
        return await evaluate(None, deployment, now=NOW, client=client, **kwargs)
    finally:
        await client.aclose()


async def test_max_runtime_destroys_even_when_active():
    deployment = _deployment(created_at=NOW - timedelta(hours=7))
    decision, reason = await _evaluate(deployment, {"active_requests": 1, "last_activity": NOW.timestamp()})
    assert decision == "destroy"
    assert "max runtime" in reason


async def test_idle_timeout_destroys():
    payload = {"active_requests": 0, "last_activity": (NOW - timedelta(minutes=40)).timestamp(), "ready_since": 1}
    decision, reason = await _evaluate(_deployment(), payload)
    assert decision == "destroy"
    assert "idle" in reason


async def test_active_requests_pause_idle_timer():
    payload = {"active_requests": 2, "last_activity": (NOW - timedelta(hours=1)).timestamp(), "ready_since": 1}
    decision, reason = await _evaluate(_deployment(), payload)
    assert decision == "wait"
    assert "active" in reason


async def test_recent_activity_waits():
    payload = {"active_requests": 0, "last_activity": (NOW - timedelta(minutes=2)).timestamp(), "ready_since": 1}
    decision, _reason = await _evaluate(_deployment(), payload)
    assert decision == "wait"


async def test_health_checks_never_count_as_activity():
    """last_activity older than the idle timeout ⇒ destroy, even though the
    gateway answered our (health) poll just now."""
    payload = {"active_requests": 0, "last_activity": 0.0, "ready_since": (NOW - timedelta(minutes=60)).timestamp()}
    decision, _ = await _evaluate(_deployment(), payload)
    assert decision == "destroy"


async def test_not_ready_waits():
    decision, _reason = await _evaluate(_deployment(status="downloading", endpoint=None), {})
    assert decision == "wait"


def _failing_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return httpx.MockTransport(handler)


async def _evaluate_unreachable(deployment: Deployment, unreachable_for_s: float):
    client = httpx.AsyncClient(transport=_failing_transport())
    try:
        return await evaluate(None, deployment, now=NOW, client=client, unreachable_for_s=unreachable_for_s)
    finally:
        await client.aclose()


async def test_unreachable_beyond_grace_destroys():
    decision, reason = await _evaluate_unreachable(_deployment(), UNREACHABLE_GRACE_S + 1)
    assert decision == "destroy"
    assert "unreachable" in reason


async def test_unreachable_within_grace_waits():
    decision, _reason = await _evaluate_unreachable(_deployment(), 60)
    assert decision == "wait"
