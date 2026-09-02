"""Cost domain objects."""

from __future__ import annotations

from pydantic import BaseModel


class CostBreakdown(BaseModel):
    """Estimated cost components for one offer and one session length.

    All values are *estimates*; Vast.ai billing is authoritative.
    """

    hourly_gpu_usd: float
    hourly_storage_usd: float
    hourly_total_usd: float
    download_seconds: float
    image_pull_seconds: float
    model_load_seconds: float
    cold_start_hours: float
    cold_start_usd: float
    bandwidth_usd: float
    runtime_usd: float
    total_session_usd: float
    session_hours: float
