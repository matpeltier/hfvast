"""Local deployment state (platformdirs), per spec §31.

Secrets policy: cloud credentials (VAST_API_KEY, HF_TOKEN) are NEVER persisted.
The per-deployment API key IS persisted because the endpoint is useless after a
CLI restart without it; it is deployment-scoped and dies with the instance, and
the state file is chmod 600 (documented in SECURITY.md).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from hfvast.errors import HfvastError
from hfvast.utils.paths import ensure_dirs, state_file


class Deployment(BaseModel):
    """Persisted deployment metadata + lifecycle state."""

    id: str = Field(..., description="friendly id, e.g. glm-5-3-flash-a8f2")
    model_repo: str
    revision: str | None = None
    variant_id: str
    backend: str
    provider: str = "vast"

    # provider resources
    instance_id: int | None = None
    offer_id: int | None = None
    gpu_label: str | None = None

    # pricing (all $/h except bandwidth which is $/GB)
    hourly_gpu_usd: float = 0.0
    hourly_total_usd: float = 0.0
    inet_down_usd_per_gb: float = 0.0
    disk_gb: float = 0.0
    context_length: int = 8192
    concurrency: int = 1

    # lifecycle
    status: str = "creating"
    status_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ready_at: datetime | None = None
    destroyed_at: datetime | None = None
    endpoint: str | None = None
    api_key: str = Field("", description="deployment-scoped gateway key (sk-hfvast-…)")
    cold_start_usd_estimate: float = 0.0
    bandwidth_usd_estimate: float = 0.0
    idle_timeout_s: float = 1800.0
    max_runtime_s: float = 6 * 3600.0
    budget_usd: float | None = Field(None, description="hard spend cap — watchdog destroys the instance when exceeded")
    watchdog_pid: int | None = None
    last_error: str | None = None

    @property
    def active(self) -> bool:
        return self.status not in ("destroyed", "failed")

    @property
    def uptime_s(self) -> float:
        end = self.destroyed_at or datetime.now(UTC)
        return max(0.0, (end - self.created_at).total_seconds())

    def estimated_spend_usd(self) -> float:
        """Estimated spend so far (compute+storage via dph_total, plus bandwidth)."""
        hours = self.uptime_s / 3600.0
        return self.hourly_total_usd * hours + self.bandwidth_usd_estimate


class DeploymentStore:
    """JSON-backed deployment registry with atomic 0600 writes."""

    def __init__(self, path: Any | None = None) -> None:
        self._path = path or state_file()

    def _load(self) -> list[Deployment]:
        path = self._path
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        result: list[Deployment] = []
        for item in data if isinstance(data, list) else []:
            try:
                result.append(Deployment.model_validate(item))
            except ValueError:
                continue  # corrupt entries are dropped, never crash the CLI
        return result

    def _save(self, deployments: list[Deployment]) -> None:
        ensure_dirs()
        payload = json.dumps([d.model_dump(mode="json") for d in deployments], indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=".deployments-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, str(self._path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def upsert(self, deployment: Deployment) -> None:
        deployments = self._load()
        for idx, existing in enumerate(deployments):
            if existing.id == deployment.id:
                deployments[idx] = deployment
                break
        else:
            deployments.append(deployment)
        self._save(deployments)

    def get(self, deployment_id: str) -> Deployment | None:
        for deployment in self._load():
            if deployment.id == deployment_id or deployment.id.startswith(deployment_id):
                return deployment
        return None

    def resolve(self, deployment_id: str | None) -> Deployment:
        """Resolve an id/prefix, defaulting to the single active deployment."""
        deployments = self._load()
        active = [d for d in deployments if d.active]
        if deployment_id is None:
            if len(active) == 1:
                return active[0]
            if not active:
                raise HfvastError("No active deployments. Start one with `hfvast up <model>`.")
            raise HfvastError("Multiple active deployments — specify one:\n  " + "\n  ".join(d.id for d in active))
        for deployment in active:
            if deployment.id == deployment_id or deployment.id.startswith(deployment_id):
                return deployment
        for deployment in deployments:
            if deployment.id == deployment_id or deployment.id.startswith(deployment_id):
                return deployment
        raise HfvastError(f"No deployment matching {deployment_id!r}.")

    def all_active(self) -> list[Deployment]:
        return [d for d in self._load() if d.active]

    def list_all(self) -> list[Deployment]:
        return self._load()

    def remove(self, deployment_id: str) -> bool:
        deployments = self._load()
        remaining = [d for d in deployments if d.id != deployment_id]
        if len(remaining) == len(deployments):
            return False
        self._save(remaining)
        return True


def new_deployment_id(model_repo: str) -> str:
    """glm-5.3-Flash/X_Y → glm-5-3-flash-a8f2 (stable-ish, collision-safe)."""
    import re
    import secrets

    name = model_repo.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    words = [w for w in name.split("-") if w][:3]
    return "-".join(words or ["model"])[:24].rstrip("-") + "-" + secrets.token_hex(2)
