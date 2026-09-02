"""vLLM runtime adapter (safetensors)."""

from __future__ import annotations

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.runtimes.base import Backend, RuntimePlan

DEFAULT_IMAGE = "vllm/vllm-openai:v0.11.0"  # pinned; bump via registry/releases

MODELS_DIR = "/opt/hfvast/models"
ROOT = "/opt/hfvast"


def build_plan(
    model: ModelInfo,
    variant: ModelVariant,
    requirements: HardwareRequirements,
    gpu_count: int,
) -> RuntimePlan:
    """Planned ``vllm serve`` invocation (backend bound to localhost only)."""
    model_dir = f"{MODELS_DIR}/{model.ref.repo_id.split('/')[-1]}"
    args = [
        "vllm",
        "serve",
        model_dir,
        "--served-model-name",
        "model",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--max-model-len",
        str(requirements.context_length),
        "--max-num-seqs",
        str(max(1, requirements.concurrency)),
        "--tensor-parallel-size",
        str(gpu_count),
        "--api-key",
        "@BACKEND_KEY@",
    ]
    if model.requires_trust_remote_code:
        args.append("--trust-remote-code")
    notes = [
        "vLLM reserves ~8% of each GPU (gpu-memory-utilization 0.92 default) for activations/graphs.",
        "Documented caveat: --api-key only protects /v1, /v2, /inference — the backend binds to "
        "127.0.0.1 only and the hfvast gateway is the public surface.",
    ]
    return RuntimePlan(
        backend=Backend.VLLM,
        image=DEFAULT_IMAGE,
        args=args,
        health_path="/health",
        api_prefix="/v1",
        notes=notes,
    )
