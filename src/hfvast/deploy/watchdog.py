"""Local lifecycle daemon (primary auto-destroy path).

Spawned detached by `hfvast up` right after instance creation; survives the
terminal but NOT a machine reboot — that gap is covered by the in-container
watchdog (CONTAINER_API_KEY, self-destroy-only) plus the hard max runtime.
Limitations are documented, never hidden (spec §29).

Run: python -m hfvast.deploy.watchdog <deployment_id>
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from datetime import UTC, datetime

import httpx

from hfvast.config import resolve_credentials
from hfvast.state import Deployment, DeploymentStore
from hfvast.utils.redact import register_secrets

#: How long a READY deployment may be unreachable before we assume the host
#: died and destroy it (storage keeps billing while instances exist).
UNREACHABLE_GRACE_S = 1800.0


#: If a deployment is stuck before READY for this long, destroy it (failed
#: bootstrap, dead CLI, broken image pull...). Storage bills while it exists.
BOOTSTRAP_DEADLINE_S = 60 * 90


async def evaluate(
    provider: object,
    deployment: Deployment,
    *,
    now: datetime | None = None,
    unreachable_for_s: float = 0.0,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, str]:
    """Pure-ish decision: ("destroy", reason) | ("wait", detail).

    Rules (spec §28):
      * max runtime is unconditional;
      * idle only counts when READY, zero active requests, and the timer starts
        at ready/last-user-activity — health checks never count as activity;
      * long-running generations are never cut mid-flight (active_requests > 0
        blocks idle destroy).
    """
    now = now or datetime.now(UTC)
    age_s = (now - deployment.created_at).total_seconds()
    if age_s >= deployment.max_runtime_s:
        return "destroy", f"hard max runtime reached ({deployment.max_runtime_s / 3600:.0f}h)"

    if deployment.status != "ready" or not deployment.endpoint:
        return "wait", f"status={deployment.status}"

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=10.0)
    try:
        try:
            resp = await http.get(
                f"{deployment.endpoint.rstrip('/')}/internal/state",
                headers={"Authorization": f"Bearer {deployment.api_key}"},
            )
            resp.raise_for_status()
            state = resp.json()
        except httpx.HTTPError:
            if unreachable_for_s >= UNREACHABLE_GRACE_S:
                return "destroy", f"endpoint unreachable for {unreachable_for_s / 60:.0f} min"
            return "wait", "endpoint unreachable"

        active = int(state.get("active_requests") or 0)
        last_activity = float(state.get("last_activity") or 0)
        ready_since = float(state.get("ready_since") or 0)
        baseline = max(last_activity, ready_since)
        if baseline <= 0:
            baseline = deployment.ready_at.timestamp() if deployment.ready_at else now.timestamp()
        idle_s = now.timestamp() - baseline
        if active > 0:
            return "wait", f"active requests: {active} (idle timer paused)"
        if idle_s >= deployment.idle_timeout_s:
            return "destroy", f"idle for {idle_s / 60:.0f} min (timeout {deployment.idle_timeout_s / 60:.0f} min)"
        return "wait", f"idle for {idle_s / 60:.1f} min"
    finally:
        if owns_client:
            await http.aclose()


async def run_loop(deployment_id: str, interval_s: float = 30.0) -> None:
    vast_api_key, _ = resolve_credentials()
    if not vast_api_key:
        print("watchdog: VAST_API_KEY not set — cannot manage instance; exiting", file=sys.stderr)
        return
    from hfvast.providers.vast.offers import VastProvider

    register_secrets(vast_api_key)
    provider = VastProvider(api_key=vast_api_key)
    store = DeploymentStore()
    unreachable_s = 0.0
    print(f"watchdog: watching {deployment_id} every {interval_s:.0f}s", file=sys.stderr)

    while True:
        deployment = store.get(deployment_id)
        if deployment is None or not deployment.active:
            print("watchdog: deployment no longer active — exiting", file=sys.stderr)
            return
        try:
            decision, detail = await evaluate(provider, deployment, unreachable_for_s=unreachable_s)
            print(f"watchdog: {decision} ({detail})", file=sys.stderr)
            if decision == "destroy":
                orch_provider = provider
                await orch_provider.destroy_instance(deployment.instance_id)  # type: ignore[arg-type]
                deployment.status = "destroyed"
                deployment.destroyed_at = datetime.now(UTC)
                deployment.status_message = f"auto-destroyed by local watchdog ({detail})"
                store.upsert(deployment)
                return
            unreachable_s = 0.0 if detail != "endpoint unreachable" else unreachable_s + interval_s
        except Exception as exc:
            print(f"watchdog: cycle error: {exc!r}", file=sys.stderr)
        await asyncio.sleep(interval_s)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m hfvast.deploy.watchdog <deployment_id>", file=sys.stderr)
        raise SystemExit(2)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_loop(sys.argv[1]))


if __name__ == "__main__":
    main()
