"""Offer query construction and planning constraints."""

from __future__ import annotations

from pydantic import BaseModel

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.offers import OfferQuery


class PlanningConstraints(BaseModel):
    """User-controlled search constraints (CLI flags / config)."""

    min_reliability: float = 0.98
    min_download_mbps: float = 300.0
    max_gpus: int = 4
    gpu_filter: str | None = None
    allowed_geolocations: list[str] | None = None
    secure_cloud_only: bool = False
    max_hourly_usd: float | None = None
    max_startup_usd: float | None = None
    max_total_usd: float | None = None

    def capped(self) -> bool:
        return any(v is not None for v in (self.max_hourly_usd, self.max_startup_usd, self.max_total_usd))


def build_query(requirements: HardwareRequirements, constraints: PlanningConstraints) -> OfferQuery:
    # Keep the per-GPU floor low so the search stays wide (e.g. 200 GiB / 8 GPUs);
    # exact viability is checked per offer during ranking.
    min_per_gpu = max(1.0, requirements.recommended_vram_gib / constraints.max_gpus - 2.0)
    return OfferQuery(
        min_total_vram_gb=requirements.recommended_vram_gib,
        disk_gb=requirements.disk_gb,
        max_gpus=constraints.max_gpus,
        min_per_gpu_vram_gb=round(min_per_gpu, 1),
        min_download_mbps=constraints.min_download_mbps,
        min_reliability=constraints.min_reliability,
        gpu_filter=constraints.gpu_filter,
        allowed_geolocations=constraints.allowed_geolocations,
        secure_cloud_only=constraints.secure_cloud_only,
        max_hourly_usd=constraints.max_hourly_usd,
    )
