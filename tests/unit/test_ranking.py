from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import GGUFHeaderInfo, ModelInfo, ModelVariant
from hfvast.models.offers import GPUOffer
from hfvast.planning.memory import estimate_vram
from hfvast.planning.ranking import OfferRanker
from hfvast.runtimes.base import Backend
from hfvast.utils.hfref import parse_model_input

GIB = 1024**3


def _model() -> ModelInfo:
    return ModelInfo(
        ref=parse_model_input("org/model-34b"),
        architecture="qwen2",
        format="gguf",
        gguf_header=GGUFHeaderInfo(architecture="qwen2", block_count=64, head_count_kv=8, key_length=128),
    )


def _variant(size_gb: float = 20.0) -> ModelVariant:
    return ModelVariant(id="Q4_K_M", quant="Q4_K_M", size_bytes=int(size_gb * 1e9))


def _requirements(ctx: int = 8192) -> HardwareRequirements:
    model, variant = _model(), _variant()
    breakdown = estimate_vram(model, variant, ctx, 1, Backend.LLAMA_CPP, gpu_count=1)
    return HardwareRequirements(
        minimum_vram_gib=breakdown.total_gib - breakdown.safety_gib,
        recommended_vram_gib=breakdown.total_gib,
        disk_gb=48.0,
        context_length=ctx,
        concurrency=1,
        breakdown=breakdown,
    )


def _offer(
    offer_id: int,
    gpu: str,
    count: int,
    per_gpu_gb: float,
    hourly: float,
    mbps: float,
    reliability: float = 0.995,
    nvlink: float = 0.0,
    mem_bw: float = 900.0,
    disk: float = 512.0,
    inet_cost: float = 0.05,
) -> GPUOffer:
    return GPUOffer(
        offer_id=offer_id,
        gpu_model=gpu,
        gpu_count=count,
        per_gpu_vram_gb=per_gpu_gb,
        total_vram_gb=per_gpu_gb * count,
        disk_gb=disk,
        inet_down_mbps=mbps,
        hourly_gpu_usd=hourly,
        hourly_total_usd=hourly,
        inet_down_usd_per_gb=inet_cost,
        reliability=reliability,
        verified=True,
        gpu_mem_bw_gbs=mem_bw,
        nvlink_gbs=nvlink,
    )


def _ranker(session_hours: float) -> OfferRanker:
    return OfferRanker(session_hours=session_hours, image_size_gb=8.0)


def test_insufficient_vram_rejected():
    offers = [_offer(1, "RTX 4090", 1, 24.0, 0.40, 900.0)]
    ranked = _ranker(2.0).rank(offers, _model(), _variant(), _requirements(), Backend.LLAMA_CPP)
    assert ranked == []


def test_low_reliability_rejected():
    offers = [_offer(1, "RTX A6000", 1, 48.0, 1.00, 900.0, reliability=0.85)]
    ranked = _ranker(2.0).rank(offers, _model(), _variant(), _requirements(), Backend.LLAMA_CPP)
    assert ranked == []


def test_fast_network_wins_short_session():
    cheap_slow = _offer(1, "RTX A6000", 1, 48.0, 1.50, 200.0, mem_bw=512.0)
    fast_pricy = _offer(2, "A100 PCIE", 1, 80.0, 1.80, 1500.0, mem_bw=1935.0)
    ranked = _ranker(1.0).rank([cheap_slow, fast_pricy], _model(), _variant(), _requirements(), Backend.LLAMA_CPP)
    assert next(r.offer.offer_id for r in ranked) == 2


def test_cheap_machine_wins_long_session():
    cheap_slow = _offer(1, "RTX A6000", 1, 48.0, 1.50, 200.0, mem_bw=512.0)
    fast_pricy = _offer(2, "A100 PCIE", 1, 80.0, 1.80, 1500.0, mem_bw=1935.0)
    ranked = _ranker(12.0).rank([cheap_slow, fast_pricy], _model(), _variant(), _requirements(), Backend.LLAMA_CPP)
    assert next(r.offer.offer_id for r in ranked) == 1


def test_nvlink_and_gpu_count_penalties():
    two_gpu_nvlink = _offer(1, "A100 SXM4", 2, 80.0, 2.20, 900.0, nvlink=600.0)
    eight_gpu_pcie = _offer(2, "RTX 3090", 8, 24.0, 2.10, 900.0, mem_bw=936.0, disk=1024.0)
    ranked = _ranker(2.0).rank(
        [eight_gpu_pcie, two_gpu_nvlink], _model(), _variant(), _requirements(), Backend.LLAMA_CPP
    )
    assert ranked[0].offer.offer_id == 1
    assert any("NVLink" in p for p in ranked[0].pros)
    assert any("8 GPUs" in c or "GPUs where" in c for c in ranked[1].cons)


def test_reasons_are_generated_for_recommendation():
    offer = _offer(1, "RTX A6000", 1, 48.0, 1.20, 900.0)
    ranked = _ranker(2.0).rank([offer], _model(), _variant(), _requirements(), Backend.LLAMA_CPP)
    assert ranked[0].pros  # explainability: at least reliability/VRAM reasons present
