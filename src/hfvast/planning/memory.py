"""VRAM estimation.

Model (spec §15):

    VRAM = weights + KV cache + runtime overhead + safety margin

Rules:
  * weights = actual stored weight bytes of the selected variant (GGUF aggregate /
    safetensors index size). MoE total weights — never just active parameters.
  * KV per token = 2 × block_count × head_count_kv × head_dim × kv_dtype_bytes,
    × context_length × concurrency. Inputs come from the parsed GGUF header when
    available; otherwise conservative fallbacks are recorded in `assumptions`.
  * runtime overhead mirrors each backend's documented memory model (llama.cpp
    reserves ~1 GiB margin per device via --fit; vLLM/SGLang keep ~8% of each GPU
    for activations/CUDA graphs plus a few GiB of runtime state).
  * safety = max(8 GiB, 5% of weights).

These are estimates and are always displayed with their inputs.
"""

from __future__ import annotations

from hfvast.models.hardware import VramBreakdown
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.runtimes.base import Backend

GIB = 1024**3

#: KV cache dtype bytes (F16 default across backends in V1).
KV_DTYPE_BYTES = 2

# Conservative fallbacks when the GGUF header is unavailable (gated repo etc.).
FALLBACK_LAYERS = 80
FALLBACK_KV_HEADS = 8
FALLBACK_HEAD_DIM = 128

# Per-backend overhead model (GiB).
_BASE_OVERHEAD: dict[Backend, float] = {Backend.LLAMA_CPP: 2.0, Backend.VLLM: 2.5, Backend.SGLANG: 2.5}
_PER_GPU_OVERHEAD: dict[Backend, float] = {Backend.LLAMA_CPP: 0.75, Backend.VLLM: 0.0, Backend.SGLANG: 0.0}
_FRACTION_OF_VRAM: dict[Backend, float] = {Backend.LLAMA_CPP: 0.0, Backend.VLLM: 0.08, Backend.SGLANG: 0.08}


def kv_bytes_per_token(model_info: ModelInfo) -> tuple[float, list[str]]:
    """Return (bytes/token, assumptions)."""
    header = model_info.gguf_header
    assumptions: list[str] = []
    if header is not None and header.block_count and header.head_count_kv:
        head_dim = (
            header.key_length
            or header.value_length
            or (header.embedding_length // header.head_count if header.embedding_length and header.head_count else None)
            or FALLBACK_HEAD_DIM
        )
        if head_dim == FALLBACK_HEAD_DIM and not header.key_length and not header.value_length:
            assumptions.append(f"head_dim not in header; fell back to {FALLBACK_HEAD_DIM}")
        bytes_per_token = 2 * header.block_count * header.head_count_kv * head_dim * KV_DTYPE_BYTES
        return float(bytes_per_token), assumptions

    fallback_bytes = 2 * FALLBACK_LAYERS * FALLBACK_KV_HEADS * FALLBACK_HEAD_DIM * KV_DTYPE_BYTES
    assumptions.append(
        "architecture shape unknown (no GGUF header): conservative fallback "
        f"{FALLBACK_LAYERS} layers × {FALLBACK_KV_HEADS} kv_heads × {FALLBACK_HEAD_DIM} head_dim"
    )
    return float(fallback_bytes), assumptions


def estimate_vram(
    model_info: ModelInfo,
    variant: ModelVariant,
    context_length: int,
    concurrency: int,
    backend: Backend,
    gpu_count: int = 1,
    per_gpu_vram_gb: float = 24.0,
) -> VramBreakdown:
    """Estimate required VRAM for one variant on a concrete GPU topology."""
    assumptions: list[str] = []

    # Weights: V1 serves text-only; the mmproj projector is not loaded into VRAM.
    weights_gib = variant.size_bytes / GIB
    if model_info.multimodal:
        assumptions.append("mmproj (vision) not loaded into VRAM — V1 serves text generation only")

    kv_per_token, kv_assumptions = kv_bytes_per_token(model_info)
    assumptions.extend(kv_assumptions)
    kv_gib = kv_per_token * context_length * concurrency / GIB

    overhead_gib = _BASE_OVERHEAD[backend] + _PER_GPU_OVERHEAD[backend] * gpu_count
    if _FRACTION_OF_VRAM[backend]:
        overhead_gib += _FRACTION_OF_VRAM[backend] * per_gpu_vram_gb * gpu_count

    safety_gib = max(8.0, 0.05 * weights_gib)
    total_gib = weights_gib + kv_gib + overhead_gib + safety_gib

    return VramBreakdown(
        weights_gib=round(weights_gib, 2),
        kv_cache_gib=round(kv_gib, 2),
        runtime_overhead_gib=round(overhead_gib, 2),
        safety_gib=round(safety_gib, 2),
        total_gib=round(total_gib, 2),
        assumptions=assumptions,
    )
