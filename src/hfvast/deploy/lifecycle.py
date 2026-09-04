"""Deployment orchestration: provision → download → serve → verify (spec §23).

Failure policy (spec §30): any bootstrap failure destroys the instance and
reports the incurred estimate — unless --keep-on-failure. Nothing is ever left
running silently.
"""

from __future__ import annotations

import secrets
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from hfvast.deploy.bootstrap import BootstrapSpec, build_create_env, build_onstart
from hfvast.deploy.health import fetch_instance_log, smoke_test, wait_endpoint_ready
from hfvast.errors import HfvastError, ProviderError
from hfvast.models.model import ModelVariant
from hfvast.models.quote import DeploymentQuote
from hfvast.providers.vast.client import VastClient
from hfvast.providers.vast.instances import create_instance, discover_endpoint, wait_running
from hfvast.providers.vast.offers import VastProvider
from hfvast.runtimes import llama_cpp, sglang, vllm
from hfvast.runtimes.base import Backend, RuntimePlan
from hfvast.state import Deployment, DeploymentStore, new_deployment_id
from hfvast.utils.paths import state_dir
from hfvast.utils.redact import redact

ADAPTERS = {
    Backend.LLAMA_CPP: llama_cpp,
    Backend.VLLM: vllm,
    Backend.SGLANG: sglang,
}


class _OfferAttemptFailed(HfvastError):
    """One candidate offer failed (create/endpoint/bootstrap/smoke) — try the next."""


def base_variant(quote: DeploymentQuote) -> ModelVariant:
    """LoRA: the base model's weights variant."""
    assert quote.base is not None
    return quote.base.variants[0]


def runtime_image(backend: Backend) -> str:
    adapter = ADAPTERS[backend]
    return str(adapter.DEFAULT_IMAGE)


@dataclass
class DeploySecrets:
    """Process-local secrets (never persisted to state)."""

    vast_api_key: str | None
    hf_token: str | None


@dataclass
class DeployOptions:
    keep_on_failure: bool = False
    instance_timeout_s: float = 900.0
    #: Docker port maps materialize only after the image pull finishes, which
    #: can take many minutes on slow hosts (live-verified 2026-09-03).
    discover_timeout_s: float = 900.0
    ready_timeout_s: float = 5400.0
    poll_interval_s: float = 5.0
    idle_timeout_s: float = 1800.0
    max_runtime_s: float = 6 * 3600.0
    budget_usd: float | None = None
    on_progress: Any = None  # async callable(str)


class DeploymentOrchestrator:
    def __init__(
        self,
        provider: VastProvider,
        store: DeploymentStore,
        secrets: DeploySecrets,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._secrets = secrets
        self._http = http or httpx.AsyncClient(timeout=20.0)

    # ------------------------------------------------------------------ deploy

    async def deploy(self, quote: DeploymentQuote, options: DeployOptions) -> Deployment:
        if quote.recommendation is None:
            raise HfvastError("quote has no deployable recommendation")
        rec = quote.recommendation
        plan = next((p for p in quote.plans if p.variant.id == rec.variant_id), None)
        if plan is None or plan.requirements is None:
            raise HfvastError("selected variant plan is missing requirements")

        backend = Backend(plan.support.backend.value)
        adapter = ADAPTERS[backend]
        lora_modules = ["model=/opt/hfvast/adapters"] if quote.base is not None else None
        plan_model = quote.base if quote.base is not None else quote.model
        plan_variant = base_variant(quote) if quote.base is not None else plan.variant
        runtime_plan: RuntimePlan = adapter.build_plan(
            plan_model,
            plan_variant,
            plan.requirements,
            rec.offer.gpu_count,
            lora_modules=lora_modules,
        )
        if plan.support.level.value == "experimental":
            runtime_plan.notes.append("EXPERIMENTAL backend compatibility — see quote output")

        gateway_key = "sk-hfvast-" + secrets.token_urlsafe(24)
        backend_key = secrets.token_urlsafe(24)
        deployment = Deployment(
            id=new_deployment_id(quote.model.ref.repo_id),
            model_repo=quote.model.ref.repo_id,
            base_repo=quote.base.ref.repo_id if quote.base else None,
            revision=quote.model.ref.revision,
            variant_id=plan.variant.id,
            backend=backend.value,
            gpu_label=rec.offer.label,
            offer_id=rec.offer.offer_id,
            hourly_gpu_usd=rec.cost.hourly_gpu_usd,
            hourly_total_usd=rec.cost.hourly_total_usd,
            inet_down_usd_per_gb=rec.offer.inet_down_usd_per_gb,
            disk_gb=plan.requirements.disk_gb,
            context_length=quote.context_length,
            concurrency=quote.concurrency,
            cold_start_usd_estimate=rec.cost.cold_start_usd,
            bandwidth_usd_estimate=rec.cost.bandwidth_usd,
            idle_timeout_s=options.idle_timeout_s,
            max_runtime_s=options.max_runtime_s,
            budget_usd=options.budget_usd,
            status="creating",
            api_key=gateway_key,
        )
        self._store.upsert(deployment)

        spec = BootstrapSpec(
            deployment_id=deployment.id,
            model_repo=deployment.model_repo,
            revision=deployment.revision,
            files=list(plan.variant.files) + list(quote.model.mmproj_files),
            backend_cmd_args=runtime_plan.args,
            gateway_key=gateway_key,
            backend_key=backend_key,
            hf_token=self._secrets.hf_token,
            idle_timeout_s=deployment.idle_timeout_s,
            max_runtime_s=deployment.max_runtime_s,
        )
        if quote.base is not None:
            # LoRA serving: download the BASE weights, then the adapter files
            spec.model_repo = quote.base.ref.repo_id
            spec.revision = quote.base.ref.revision
            spec.files = list(base_variant(quote).files) + list(quote.base.mmproj_files)
            spec.adapter_repo = quote.model.ref.repo_id
            spec.adapter_files = list(plan.variant.files)

        # Offers vanish constantly on the marketplace and some hosts are
        # unreachable from certain networks — try the top candidates on ANY
        # per-offer failure (create, endpoint discovery, bootstrap, smoke test).
        candidates = [r.offer for r in plan.ranked_offers[:4]] or [rec.offer]
        failures: list[str] = []
        for idx, offer in enumerate(candidates, start=1):
            try:
                return await self._deploy_on_offer(quote, deployment, spec, offer, options)
            except (ProviderError, _OfferAttemptFailed) as exc:
                failures.append(f"offer {offer.offer_id} ({offer.label}): {redact(str(exc))}")
                deployment.status = "creating"
                deployment.status_message = f"candidate {idx}/{len(candidates)} failed — trying next"
                self._store.upsert(deployment)
        deployment.status = "failed"
        deployment.last_error = "; ".join(failures) or "no offers available"
        self._store.upsert(deployment)
        raise HfvastError(
            "deployment failed on all candidate offers:\n  - "
            + "\n  - ".join(failures)
            + f"\nIncurred estimate so far: ${deployment.estimated_spend_usd():.2f}"
        )

    async def _deploy_on_offer(
        self,
        quote: DeploymentQuote,
        deployment: Deployment,
        spec: BootstrapSpec,
        offer: Any,
        options: DeployOptions,
    ) -> Deployment:
        client: VastClient = self._provider._client
        env = build_create_env(spec, self._secrets.hf_token)
        onstart = build_onstart()
        backend = Backend(quote.plans[0].support.backend.value) if quote.plans else Backend.LLAMA_CPP

        async def progress(message: str) -> None:
            if options.on_progress:
                await options.on_progress(message)

        instance_id = await create_instance(
            client,
            offer.offer_id,
            image=runtime_image(backend),
            disk_gb=int(deployment.disk_gb),
            env=env,
            onstart=onstart,
            label=f"hfvast {deployment.id}",
        )
        deployment.instance_id = instance_id
        deployment.offer_id = offer.offer_id
        deployment.gpu_label = offer.label
        deployment.status = "provisioning"
        deployment.status_message = f"instance {instance_id} created"
        # Persist IMMEDIATELY after creation so the instance can always be
        # found and destroyed (spec §51).
        self._store.upsert(deployment)

        spawn_local_watchdog(deployment)

        try:
            await progress(f"waiting for instance {instance_id} to reach running state…")
            await wait_running(
                client,
                instance_id,
                timeout_s=options.instance_timeout_s,
                poll_interval_s=options.poll_interval_s,
            )
            host, port = await discover_endpoint(client, instance_id, 8000, timeout_s=options.discover_timeout_s)
            deployment.endpoint = f"http://{host}:{port}"
            deployment.status = "downloading"
            deployment.status_message = "endpoint discovered"
            self._store.upsert(deployment)

            await progress(f"endpoint {deployment.endpoint} — bootstrap running…")
            await wait_endpoint_ready(
                deployment.endpoint,
                deployment.api_key,
                timeout_s=options.ready_timeout_s,
                poll_interval_s=options.poll_interval_s,
                client=self._http,
                on_progress=options.on_progress,
            )
            deployment.status = "loading"
            self._store.upsert(deployment)

            await progress("running smoke test…")
            reply = await smoke_test(deployment.endpoint, deployment.api_key, client=self._http)
            deployment.status = "ready"
            deployment.ready_at = datetime.now(UTC)
            deployment.status_message = f"smoke test reply: {redact(reply[:40])!r}"
            self._store.upsert(deployment)
            return deployment
        except Exception as exc:
            deployment.status = "failed"
            deployment.last_error = redact(str(exc))
            self._store.upsert(deployment)
            log_tail = ""
            if deployment.endpoint and deployment.api_key:
                backend_log = await fetch_instance_log(
                    deployment.endpoint, deployment.api_key, "backend", client=self._http
                )
                boot_log = await fetch_instance_log(
                    deployment.endpoint, deployment.api_key, "bootstrap", client=self._http
                )
                log_tail = (backend_log or boot_log or "")[-1200:]
                if log_tail:
                    await progress("instance log tail:\n" + redact(log_tail))
            if options.keep_on_failure:
                raise HfvastError(
                    f"deployment failed: {redact(str(exc))}\n"
                    f"Incurred estimate: ${deployment.estimated_spend_usd():.2f}\n"
                    "Instance kept for debugging (--keep-on-failure)."
                ) from exc
            await progress("failure — destroying instance (spec §30), trying next offer…")
            try:
                await self.destroy(deployment, reason="failure cleanup")
            except Exception as cleanup_exc:
                await progress(
                    f"WARNING: cleanup failed ({redact(str(cleanup_exc))}) — "
                    f"destroy instance {deployment.instance_id} manually!"
                )
            raise _OfferAttemptFailed(
                f"{redact(str(exc))} [incurred ${deployment.estimated_spend_usd():.2f}; "
                "instance destroyed]" + (f"\nlog tail:\n{redact(log_tail)}" if log_tail else "")
            ) from exc

    async def aclose(self) -> None:
        await self._http.aclose()

    # ----------------------------------------------------------------- destroy

    async def destroy(self, deployment: Deployment, reason: str = "user request") -> Deployment:
        if deployment.instance_id is not None and deployment.status != "destroyed":
            client: VastClient = self._provider._client
            await client.destroy_instance(deployment.instance_id)
        stop_local_watchdog(deployment)
        if deployment.status != "failed":
            deployment.status = "destroyed"
        deployment.destroyed_at = datetime.now(UTC)
        deployment.status_message = f"destroyed ({reason})"
        self._store.upsert(deployment)
        return deployment


def spawn_local_watchdog(deployment: Deployment) -> int | None:
    """Start the detached local lifecycle daemon (primary auto-destroy path)."""
    try:
        state_dir().mkdir(parents=True, exist_ok=True)
        log_path = state_dir() / f"watch-{deployment.id}.log"
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "hfvast.deploy.watchdog", deployment.id],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        deployment.watchdog_pid = proc.pid
        return proc.pid
    except OSError:
        return None


def stop_local_watchdog(deployment: Deployment) -> None:
    pid = deployment.watchdog_pid
    if pid is None:
        return
    try:
        import os

        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    deployment.watchdog_pid = None
