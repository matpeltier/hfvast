"""Runtime (inference backend) abstraction."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Backend(StrEnum):
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    SGLANG = "sglang"


class SupportLevel(StrEnum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class RuntimeSupport(BaseModel):
    """Explicit, explainable backend compatibility statement (spec §13/§50)."""

    backend: Backend
    supported: bool
    level: SupportLevel
    confidence: str = Field("unknown", description="e.g. verified|reported|heuristic")
    reason: str = ""

    @property
    def deployable(self) -> bool:
        """True only when provisioning money for this backend is acceptable."""
        return self.level is not SupportLevel.UNSUPPORTED


class RuntimePlan(BaseModel):
    """How a selected backend will be launched on the instance [M2 provisioner]."""

    backend: Backend
    image: str
    args: list[str] = Field(default_factory=list)
    health_path: str = "/health"
    api_prefix: str = "/v1"
    notes: list[str] = Field(default_factory=list)
