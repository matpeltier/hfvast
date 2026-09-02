from hfvast.models.model import GGUFHeaderInfo, ModelInfo, ModelVariant
from hfvast.planning.memory import estimate_vram, kv_bytes_per_token
from hfvast.runtimes.base import Backend
from hfvast.utils.hfref import parse_model_input

GIB = 1024**3


def _model(header: GGUFHeaderInfo | None = None, multimodal: bool = False) -> ModelInfo:
    return ModelInfo(
        ref=parse_model_input("org/model"),
        architecture="llama" if header else None,
        format="gguf",
        gguf_header=header,
        multimodal=multimodal,
    )


_HEADER = GGUFHeaderInfo(
    architecture="llama",
    context_length=131072,
    block_count=80,
    head_count=64,
    head_count_kv=8,
    key_length=128,
    embedding_length=8192,
)


def test_kv_math_with_header():
    bytes_token, assumptions = kv_bytes_per_token(_model(_HEADER))
    # 2 * 80 layers * 8 kv_heads * 128 head_dim * 2 bytes
    assert bytes_token == 2 * 80 * 8 * 128 * 2
    assert assumptions == []


def test_kv_fallback_without_header_is_conservative_and_documented():
    bytes_token, assumptions = kv_bytes_per_token(_model(None))
    assert bytes_token == 2 * 80 * 8 * 128 * 2  # conservative fallback
    assert assumptions  # fallback must be visible


def test_full_breakdown_sums():
    model = _model(_HEADER)
    variant = ModelVariant(id="Q4_K_M", quant="Q4_K_M", size_bytes=100 * GIB)
    breakdown = estimate_vram(model, variant, 8192, 1, Backend.LLAMA_CPP, gpu_count=1)
    kv_gib = (2 * 80 * 8 * 128 * 2) * 8192 / GIB
    assert breakdown.kv_cache_gib == round(kv_gib, 2)
    assert breakdown.runtime_overhead_gib == 2.75  # 2.0 base + 0.75 CUDA context
    assert breakdown.safety_gib == 8.0  # max(8, 5% of 100)
    assert abs(breakdown.total_gib - (100 + kv_gib + 2.75 + 8.0)) < 0.01


def test_concurrency_multiplies_kv():
    model = _model(_HEADER)
    variant = ModelVariant(id="Q4_K_M", quant="Q4_K_M", size_bytes=100 * GIB)
    one = estimate_vram(model, variant, 8192, 1, Backend.LLAMA_CPP)
    two = estimate_vram(model, variant, 8192, 2, Backend.LLAMA_CPP)
    assert abs(two.kv_cache_gib - 2 * one.kv_cache_gib) < 0.01


def test_vllm_overhead_uses_fraction_of_vram():
    model = _model(_HEADER)
    variant = ModelVariant(id="safetensors", size_bytes=100 * GIB)
    breakdown = estimate_vram(model, variant, 8192, 1, Backend.VLLM, gpu_count=4, per_gpu_vram_gb=80.0)
    assert breakdown.runtime_overhead_gib == round(2.5 + 0.08 * 80 * 4, 2)


def test_context_increases_vram():
    model = _model(_HEADER)
    variant = ModelVariant(id="Q4_K_M", size_bytes=100 * GIB)
    small = estimate_vram(model, variant, 2048, 1, Backend.LLAMA_CPP)
    big = estimate_vram(model, variant, 32768, 1, Backend.LLAMA_CPP)
    assert big.total_gib > small.total_gib
