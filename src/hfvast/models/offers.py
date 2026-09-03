"""Provider-normalized offer and query types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OfferQuery(BaseModel):
    """Provider-agnostic offer search constraints built from HardwareRequirements."""

    min_total_vram_gb: float
    disk_gb: float
    max_gpus: int = 4
    min_per_gpu_vram_gb: float = 1.0
    min_download_mbps: float = 300.0
    min_reliability: float = 0.98
    gpu_filter: str | None = None
    allowed_geolocations: list[str] | None = Field(
        None, description="restrict to country codes (e.g. ['US', 'DE']) — some regions are unreachable"
    )
    secure_cloud_only: bool = False
    max_hourly_usd: float | None = None
    limit: int = 100


class GPUOffer(BaseModel):
    """A normalized rentable offer — units are normalized at the provider boundary."""

    model_config = ConfigDict(frozen=True)

    offer_id: int
    gpu_model: str
    gpu_count: int
    per_gpu_vram_gb: float
    total_vram_gb: float
    cpu_cores: int | None = None
    cpu_ram_gb: float | None = None
    disk_gb: float
    disk_bw_mbs: float | None = None
    inet_down_mbps: float = 0.0
    inet_up_mbps: float | None = None
    hourly_gpu_usd: float = Field(..., description="$/h GPU rental only")
    hourly_total_usd: float = Field(..., description="$/h incl. storage for the priced disk size")
    storage_per_gb_month_usd: float = 0.0
    inet_down_usd_per_gb: float = 0.0
    reliability: float = 0.0
    verified: bool = False
    dlperf: float | None = None
    gpu_mem_bw_gbs: float | None = None
    pcie_bw_gbs: float | None = None
    nvlink_gbs: float | None = None
    geolocation: str | None = None
    public_ipaddr: str | None = Field(None, description="host IP — Vast exposes it BEFORE renting")

    @property
    def label(self) -> str:
        return f"{self.gpu_count}× {self.gpu_model}"
