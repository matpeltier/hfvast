"""Core domain types describing a Hugging Face model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ModelFormat(StrEnum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    UNKNOWN = "unknown"


class ModelTask(StrEnum):
    TEXT_GENERATION = "text-generation"
    VISION_LANGUAGE = "vision-language"
    OTHER = "other"
    UNKNOWN = "unknown"


class QuantTier(StrEnum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    QUALITY = "quality"


class HFModelRef(BaseModel):
    """A normalized Hugging Face model reference."""

    model_config = ConfigDict(frozen=True)

    repo_id: str = Field(..., description="owner/name, e.g. orcarouter/GLM-5.3-Flash-Uncensored-GGUF")
    revision: str | None = Field(None, description="git revision (branch/tag/commit), None = default")

    def to_url(self) -> str:
        base = f"https://huggingface.co/{self.repo_id}"
        return f"{base}/tree/{self.revision}" if self.revision else base


class ModelFile(BaseModel):
    path: str
    size_bytes: int = 0


class ModelVariant(BaseModel):
    """One deployable variant of a model.

    For GGUF repositories, every shard of a split file group forms exactly ONE
    variant (``model-Q4_K_M-00001-of-00005.gguf`` … ``-00005-of-00005.gguf``).
    """

    id: str = Field(..., description="variant id, usually the quantization label (e.g. Q4_K_M)")
    quant: str | None = None
    size_bytes: int
    files: list[ModelFile] = Field(default_factory=list)
    tier: QuantTier | None = None
    is_split: bool = False

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024**3)


class GGUFHeaderInfo(BaseModel):
    """Authoritative metadata parsed remotely from a GGUF file header."""

    architecture: str | None = None
    file_type: int | None = None
    file_type_name: str | None = None
    name: str | None = None
    context_length: int | None = None
    block_count: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    key_length: int | None = None
    value_length: int | None = None
    embedding_length: int | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None


class ModelInfo(BaseModel):
    """Normalized result of inspecting a Hugging Face repository."""

    ref: HFModelRef
    task: ModelTask = ModelTask.UNKNOWN
    architecture: str | None = None
    format: ModelFormat = ModelFormat.UNKNOWN
    dtype: str | None = None
    parameter_count: int | None = None
    weight_bytes: int | None = None
    context_length: int | None = None
    quantization: str | None = None
    variants: list[ModelVariant] = Field(default_factory=list)
    requires_trust_remote_code: bool = False
    gated: bool = False
    multimodal: bool = False
    mmproj_files: list[ModelFile] = Field(default_factory=list)
    gguf_header: GGUFHeaderInfo | None = None
    safetensors_params: dict[str, int] | None = None
    quantization_config: dict[str, object] | None = None
    notes: list[str] = Field(default_factory=list)

    def variant_by_id(self, variant_id: str) -> ModelVariant | None:
        lowered = variant_id.lower()
        for variant in self.variants:
            if variant.id.lower() == lowered:
                return variant
        return None

    def total_variant_bytes(self) -> int:
        return sum(v.size_bytes for v in self.variants)
