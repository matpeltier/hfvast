"""LoRA support: end-to-end quote flow with a mocked Hub (adapter + base)."""

import httpx
import pytest

from hfvast.errors import ModelNotSupportedError, PlanError
from hfvast.inspect.huggingface import HFInspector
from hfvast.models.model import ModelFormat
from hfvast.planning.quote import QuoteBuilder, QuoteOptions
from hfvast.providers.vast.offers import SnapshotProvider
from hfvast.runtimes.base import SupportLevel
from hfvast.runtimes.llama_cpp import build_plan as llama_build_plan
from hfvast.utils.hfref import parse_model_input


def _transport(adapter_config: dict | None, adapter_files: list[dict], base_config: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/api/models/my-org/my-lora" in url:
            if "tree/" in url:
                return httpx.Response(200, json=adapter_files)
            return httpx.Response(
                200,
                json={
                    "id": "my-org/my-lora",
                    "gated": False,
                    "tags": ["lora"],
                    "library_name": "peft",
                    "config": {},
                    "sha": "abc",
                },
            )
        if "/api/models/my-org/base-model" in url:
            if "tree/" in url:
                return httpx.Response(
                    200,
                    json=[
                        {"type": "file", "path": "config.json", "size": 700},
                        {"type": "file", "path": "model-00001-of-00002.safetensors", "size": 9_000_000_000},
                        {"type": "file", "path": "model-00002-of-00002.safetensors", "size": 9_000_000_000},
                    ],
                )
            return httpx.Response(
                200,
                json={
                    "id": "my-org/base-model",
                    "gated": False,
                    "pipeline_tag": "text-generation",
                    "tags": ["text-generation"],
                    "config": {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768},
                    "safetensors": {"parameters": {"BF16": 7615616512}, "total": 7615616512},
                    "sha": "def",
                },
            )
        if url.endswith("/adapter_config.json"):
            if adapter_config is None:
                return httpx.Response(404, text="missing")
            return httpx.Response(200, json=adapter_config)
        if url.endswith("/config.json"):
            return httpx.Response(200, json=base_config)
        if "adapter_config" in url or "config.json" in url:
            return httpx.Response(404, text="missing")
        return httpx.Response(404, text=url)

    return httpx.MockTransport(handler)


async def _quote_lora():
    inspector = HFInspector(
        client=httpx.AsyncClient(
            transport=_transport(
                adapter_config={"base_model_name_or_path": "my-org/base-model", "r": 64},
                adapter_files=[
                    {"type": "file", "path": "adapter_config.json", "size": 5000},
                    {"type": "file", "path": "adapter_model.safetensors", "size": 1_200_000_000},
                ],
                base_config={"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768},
            )
        )
    )
    try:
        return await QuoteBuilder(inspector, SnapshotProvider()).build(
            parse_model_input("my-org/my-lora"), QuoteOptions(context=8192)
        )
    finally:
        await inspector.aclose()


async def test_lora_end_to_end_plans_on_base_and_enables_lora():
    quote = await _quote_lora()
    assert quote.model.format is ModelFormat.LORA_ADAPTER
    assert quote.model.peft_layout is True
    assert quote.base is not None
    assert quote.base.ref.repo_id == "my-org/base-model"
    assert quote.base.architecture == "Qwen2ForCausalLM"

    # VRAM planned on the BASE weights, disk includes the adapter
    reqs = quote.plans[0].requirements
    assert reqs.breakdown.weights_gib > 10  # ~18 GiB of base weights, not 1.2 GB
    assert reqs.disk_gb >= 18 + 1.2

    # vLLM serving with the adapter hot-loaded under the canonical name "model"
    support = quote.plans[0].support
    assert support.backend.value == "vllm"
    assert support.level is SupportLevel.SUPPORTED
    assert "LoRA serving" in support.reason

    assert quote.recommendation is not None
    assert quote.recommendation.cost.total_session_usd > 0


async def test_lora_without_peft_layout_is_refused():
    inspector = HFInspector(
        client=httpx.AsyncClient(
            transport=_transport(
                adapter_config=None,
                # this repo layout (custom names, no adapter_model.safetensors)
                adapter_files=[{"type": "file", "path": "custom-lora-v1.safetensors", "size": 1_200_000_000}],
                base_config={"architectures": ["Qwen2ForCausalLM"]},
            )
        )
    )
    with pytest.raises(ModelNotSupportedError, match="PEFT serving layout"):
        await QuoteBuilder(inspector, SnapshotProvider()).build(
            parse_model_input("my-org/my-lora"), QuoteOptions(context=8192)
        )
    await inspector.aclose()


async def test_lora_without_declared_base_needs_base_model_flag():
    inspector = HFInspector(
        client=httpx.AsyncClient(
            transport=_transport(
                adapter_config={"r": 64},  # PEFT layout, but no base_model_name_or_path
                adapter_files=[
                    {"type": "file", "path": "adapter_config.json", "size": 5000},
                    {"type": "file", "path": "adapter_model.safetensors", "size": 1_200_000_000},
                ],
                base_config={},
            )
        )
    )
    builder = QuoteBuilder(inspector, SnapshotProvider())
    with pytest.raises(PlanError, match="--base-model"):
        await builder.build(parse_model_input("my-org/my-lora"), QuoteOptions(context=8192))
    await inspector.aclose()


def test_llama_cpp_plan_ignores_lora_modules():
    from hfvast.models.hardware import HardwareRequirements
    from hfvast.models.model import ModelInfo, ModelVariant
    from hfvast.planning.memory import estimate_vram
    from hfvast.runtimes.base import Backend

    model = ModelInfo(ref=parse_model_input("o/m"), architecture="qwen2", format="gguf")
    variant = ModelVariant(id="Q4_K_M", size_bytes=10**9)
    breakdown = estimate_vram(model, variant, 8192, 1, Backend.LLAMA_CPP)
    reqs = HardwareRequirements(
        minimum_vram_gib=1,
        recommended_vram_gib=2,
        disk_gb=24,
        context_length=8192,
        concurrency=1,
        breakdown=breakdown,
    )
    plan = llama_build_plan(model, variant, reqs, gpu_count=1, lora_modules=["model=/x"])
    assert plan.backend.value == "llama.cpp"  # accepted, ignored
