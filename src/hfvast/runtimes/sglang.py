"""SGLang runtime adapter (safetensors)."""

from __future__ import annotations

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.runtimes.base import Backend, RuntimePlan

DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.18-cu12"  # pinned; bump via registry/releases

MODELS_DIR = "/opt/hfvast/models"
ROOT = "/opt/hfvast"


def build_plan(
    model: ModelInfo,
    variant: ModelVariant,
    requirements: HardwareRequirements,
    gpu_count: int,
    lora_modules: list[str] | None = None,
) -> RuntimePlan:
    model_dir = f"{MODELS_DIR}/{model.ref.repo_id.split('/')[-1]}"
    args = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_dir,
        "--served-model-name",
        "model",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--context-length",
        str(requirements.context_length),
        "--max-running-requests",
        str(max(1, requirements.concurrency)),
        "--tp-size",
        str(gpu_count),
        "--api-key",
        "@BACKEND_KEY@",
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
            "SGLang exempts /health and /metrics from --api-key auth — backend binds to "
            "127.0.0.1 only; the gateway is the public surface.",
        ],
    )
