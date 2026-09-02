"""SGLang runtime adapter (safetensors)."""

from __future__ import annotations

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.runtimes.base import Backend, RuntimePlan

DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.18-cu12"  # pinned; bump via registry/releases


def build_plan(
    model: ModelInfo,
    variant: ModelVariant,
    requirements: HardwareRequirements,
    gpu_count: int,
) -> RuntimePlan:
    args = [
        "sglang.launch_server",
        "--model-path",
        f"/models/{model.ref.repo_id.split('/')[-1]}",
        "--served-model-name",
        "model",
        "--context-length",
        str(requirements.context_length),
        "--max-running-requests",
        str(max(1, requirements.concurrency)),
        "--tp-size",
        str(gpu_count),
        "--api-key",
        "/run/hfvast/backend_api_key",
        "--host",
        "0.0.0.0",
        "--port",
        "8001",
    ]
    if model.requires_trust_remote_code:
        args.append("--trust-remote-code")
    return RuntimePlan(
        backend=Backend.SGLANG,
        image=DEFAULT_IMAGE,
        args=args,
        health_path="/health",
        api_prefix="/v1",
        notes=[
            "SGLang exempts /health and /metrics from --api-key auth — gateway fronts it publicly.",
        ],
    )
