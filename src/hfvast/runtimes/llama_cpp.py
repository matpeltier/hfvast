"""llama.cpp runtime adapter (GGUF)."""

from __future__ import annotations

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.runtimes.base import Backend, RuntimePlan

#: Upstream CUDA server image (pin the version in the packaged runtime image in M2).
DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"


def build_plan(
    model: ModelInfo,
    variant: ModelVariant,
    requirements: HardwareRequirements,
    gpu_count: int,
) -> RuntimePlan:
    """Planned ``llama-server`` invocation for a GGUF variant."""
    first_shard = variant.files[0].path if variant.files else ""
    ctx = requirements.context_length
    parallel = max(1, requirements.concurrency)
    args = [
        "llama-server",
        "--host",
        "0.0.0.0",
        "--port",
        "8001",
        "--model",
        f"/models/{first_shard.rsplit('/', 1)[-1]}",
        "--ctx-size",
        str(ctx),
        "--parallel",
        str(parallel),
        "--gpu-layers",
        "99",
        "--split-mode",
        "layer",
        "--api-key-file",
        "/run/hfvast/backend_api_key",
    ]
    if gpu_count > 1:
        args += ["--tensor-split", ",".join(["1"] * gpu_count)]
    notes = [
        "GGUF splits are auto-detected from the first shard (-00001-of-000NN).",
        "llama.cpp --fit reserves ~1 GiB margin per device (default).",
    ]
    if model.multimodal:
        notes.append("mmproj (vision projector) detected; V1 serves text-generation only — vision disabled.")
    return RuntimePlan(
        backend=Backend.LLAMA_CPP,
        image=DEFAULT_IMAGE,
        args=args,
        health_path="/health",
        api_prefix="/v1",
        notes=notes,
    )
