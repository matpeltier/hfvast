"""Backend compatibility registry.

This module is DATA, deliberately isolated: compatibility changes quickly and this
registry must be updatable independently of all other code (spec §13/§42).

Confidence levels and support levels:
  SUPPORTED     — verified against upstream docs/source at the date below.
  EXPERIMENTAL  — plausible but unverified, or requires explicit user confirmation
                  before spending money (spec §50).
  UNSUPPORTED   — never spend money on it.

Checked 2026-09-02 against:
  * llama.cpp src/llama-arch.cpp (ggml-org/llama.cpp master) — LLM_ARCH_* names and
    their GGUF arch strings (e.g. GLM-5 family == "glm-dsa").
  * vLLM docs/models/supported_models.md (generative architectures) and
    docs/features/quantization/README.md.
  * SGLang docs/docs/supported-models/generative_models.mdx and
    python/sglang/srt/models/ (EntryClass registrations).
"""

from __future__ import annotations

from hfvast.models.model import ModelFormat, ModelTask
from hfvast.runtimes.base import (
    Backend,
    SupportLevel,  # re-exported for convenience
)

#: GGUF architecture strings llama.cpp master dispatches on (subset relevant to
#: causal LMs; verified by direct inspection of src/llama-arch.cpp, 2026-09-02).
LLAMA_CPP_GGUF_ARCHS = frozenset(
    {
        "llama",
        "llama4",
        "qwen2",
        "qwen2moe",
        "qwen3",
        "qwen3moe",
        "qwen3next",
        "qwen35",
        "qwen35moe",
        "qwen4exp",
        "glm4",
        "glm4moe",
        "glm-dsa",
        "deepseek2",
        "deepseek32",
        "deepseek4",
        "gemma2",
        "gemma3",
        "gemma3n",
        "gemma4",
        "mistral3",
        "mistral4",
        "phi2",
        "phi3",
        "phimoe",
        "chatglm",
        "gpt-oss",
    }
)

#: Known GGUF arch strings from *newer* model families whose llama.cpp support
#: lives under a different internal arch name. Genuine architecture-specific
#: compatibility rules (the only kind allowed outside this module).
GGUF_ARCH_FAMILY_ALIASES: dict[str, tuple[str, str]] = {
    # GGUF header says        (llama.cpp arch, reason)
    "glm5next": (
        "glm-dsa",
        "GLM-5.x family: llama.cpp serves GLM-5 under arch 'glm-dsa' as of 2026-09-02; "
        "header declares 'glm5next' — verify the pinned runtime image loads it before spending",
    ),
}

#: Transformers architectures vLLM documents as natively supported for generative
#: text models (subset; checked 2026-09-02). Matched against config.json
#: `architectures[0]`.
VLLM_SAFETENSORS_ARCHS = frozenset(
    {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "GlmForCausalLM",
        "Glm4ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForCausalLM",
        "Gemma4ForCausalLM",
        "Phi3ForCausalLM",
        "PhiMoEForCausalLM",
        "GptOssForCausalLM",
    }
)

#: Architectures SGLang source-registers (subset; 2026-09-02).
SGLANG_SAFETENSORS_ARCHS = frozenset(
    {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Glm4ForCausalLM",
        "Glm4MoeForCausalLM",
        "GlmMoeDsaForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForCausalLM",
        "Phi3ForCausalLM",
        "GptOssForCausalLM",
    }
)

#: Tasks hfvast V1 will never provision for (spec §9).
UNSUPPORTED_TASKS: dict[ModelTask, str] = {
    ModelTask.OTHER: "task is not causal text generation",
    ModelTask.UNKNOWN: "task could not be determined from repository metadata",
}

__all__ = [
    "GGUF_ARCH_FAMILY_ALIASES",
    "LLAMA_CPP_GGUF_ARCHS",
    "SGLANG_SAFETENSORS_ARCHS",
    "UNSUPPORTED_TASKS",
    "VLLM_SAFETENSORS_ARCHS",
    "Backend",
    "ModelFormat",
    "ModelTask",
    "SupportLevel",
]
