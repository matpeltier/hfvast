"""Build HardwareRequirements from a model + variant + serving options."""

from __future__ import annotations

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.planning.memory import estimate_vram
from hfvast.planning.storage import estimate_disk_gb
from hfvast.runtimes.base import Backend

#: Usable fraction of a reference GPU's VRAM (CUDA context, display, etc.).
REFERENCE_GPU_VRAM_GB = 24.0
REFERENCE_USABLE_FRACTION = 0.95


def reference_gpu_count(total_vram_gib: float) -> int:
    import math

    usable_per_gpu = REFERENCE_GPU_VRAM_GB * REFERENCE_USABLE_FRACTION
    return max(1, math.ceil(total_vram_gib / usable_per_gpu))


def build_requirements(
    model_info: ModelInfo,
    variant: ModelVariant,
    context_length: int,
    concurrency: int,
    backend: Backend,
) -> HardwareRequirements:
    first = estimate_vram(model_info, variant, context_length, concurrency, backend, gpu_count=1)
    ref_count = reference_gpu_count(first.total_gib)
    breakdown = estimate_vram(
        model_info,
        variant,
        context_length,
        concurrency,
        backend,
        gpu_count=ref_count,
        per_gpu_vram_gb=REFERENCE_GPU_VRAM_GB,
    )
    minimum = breakdown.total_gib - breakdown.safety_gib
    return HardwareRequirements(
        minimum_vram_gib=round(minimum, 2),
        recommended_vram_gib=breakdown.total_gib,
        disk_gb=estimate_disk_gb(model_info, variant),
        context_length=context_length,
        concurrency=concurrency,
        breakdown=breakdown,
        reference_gpu_count=ref_count,
    )
