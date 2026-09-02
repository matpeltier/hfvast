"""Hardware requirement domain objects."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VramBreakdown(BaseModel):
    """Visible inputs of a VRAM estimate (spec §49: never buried)."""

    weights_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    safety_gib: float
    total_gib: float
    assumptions: list[str] = Field(default_factory=list)


class HardwareRequirements(BaseModel):
    minimum_vram_gib: float = Field(..., description="absolute floor: weights + KV + overhead, no margin")
    recommended_vram_gib: float = Field(..., description="minimum + safety margin — what we filter for")
    disk_gb: float
    context_length: int
    concurrency: int
    breakdown: VramBreakdown
    reference_gpu_count: int = Field(1, description="GPU count assumed for the displayed breakdown")
