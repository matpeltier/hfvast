"""Backend selection from the runtime compatibility registry."""

from __future__ import annotations

from hfvast.errors import ModelNotSupportedError
from hfvast.models.model import ModelFormat, ModelInfo
from hfvast.runtimes import registry
from hfvast.runtimes.base import Backend, RuntimeSupport, SupportLevel
from hfvast.runtimes.upstream import LiveRegistry


def _unsupported(backend: Backend, reason: str) -> RuntimeSupport:
    return RuntimeSupport(
        backend=backend, supported=False, level=SupportLevel.UNSUPPORTED, confidence="registry", reason=reason
    )


def evaluate_support(model_info: ModelInfo, live: LiveRegistry | None = None) -> list[RuntimeSupport]:
    """Evaluate every backend against the registry. Ordered best-first.

    ``live`` carries fresh upstream lists (llama.cpp master / vLLM main); when
    absent, the baked snapshot is used. Confidence labels state which was used.
    """
    live = live or LiveRegistry()
    llama_archs = live.llama_gguf_archs or registry.LLAMA_CPP_GGUF_ARCHS
    vllm_archs = live.vllm_archs or registry.VLLM_SAFETENSORS_ARCHS
    llama_source = "live upstream" if live.llama_gguf_archs else f"snapshot {registry.CHECK_DATE}"
    vllm_source = "live upstream" if live.vllm_archs else f"snapshot {registry.CHECK_DATE}"

    if model_info.format is ModelFormat.LORA_ADAPTER:
        base = model_info.base_model_ref or "unknown base (pass --base-model <org/name>)"
        reason = f"{registry.LORA_UNSUPPORTED_REASON} Base: {base}. {registry.LORA_MANUAL_PATHS}"
        return [_unsupported(backend, reason) for backend in (Backend.LLAMA_CPP, Backend.VLLM, Backend.SGLANG)]

    if model_info.task in registry.UNSUPPORTED_TASKS:
        return [
            _unsupported(backend, registry.UNSUPPORTED_TASKS[model_info.task])
            for backend in (Backend.LLAMA_CPP, Backend.VLLM, Backend.SGLANG)
        ]
    # Note: VISION_LANGUAGE is NOT auto-rejected — GGUF repos with an mmproj
    # projector are ordinary causal LMs plus an optional vision head; V1 serves
    # the text-generation path (the arch registry below still gates everything,
    # and vision-language safetensors archs are not in any backend registry).
    if model_info.requires_trust_remote_code:
        return [
            _unsupported(
                backend,
                "repository requires trust_remote_code (custom code) — hfvast V1 never executes arbitrary remote code",
            )
            for backend in (Backend.LLAMA_CPP, Backend.VLLM, Backend.SGLANG)
        ]

    arch = model_info.architecture
    results: list[RuntimeSupport] = []

    if model_info.format is ModelFormat.GGUF:
        if arch and arch in llama_archs:
            results.append(
                RuntimeSupport(
                    backend=Backend.LLAMA_CPP,
                    supported=True,
                    level=SupportLevel.SUPPORTED,
                    confidence="verified",
                    reason=f"GGUF architecture '{arch}' is in llama.cpp's arch registry ({llama_source})",
                )
            )
        elif arch and arch in registry.GGUF_ARCH_FAMILY_ALIASES:
            _, alias_reason = registry.GGUF_ARCH_FAMILY_ALIASES[arch]
            results.append(
                RuntimeSupport(
                    backend=Backend.LLAMA_CPP,
                    supported=True,
                    level=SupportLevel.EXPERIMENTAL,
                    confidence="reported",
                    reason=alias_reason,
                )
            )
        elif arch is None:
            results.append(
                RuntimeSupport(
                    backend=Backend.LLAMA_CPP,
                    supported=True,
                    level=SupportLevel.EXPERIMENTAL,
                    confidence="heuristic",
                    reason="GGUF architecture unknown (header not parsed) — verify before spending",
                )
            )
        else:
            results.append(
                _unsupported(
                    Backend.LLAMA_CPP,
                    f"GGUF architecture '{arch}' is not in llama.cpp's arch registry ({llama_source})",
                )
            )
        results.append(
            RuntimeSupport(
                backend=Backend.VLLM,
                supported=arch == "llama",
                level=SupportLevel.EXPERIMENTAL if arch == "llama" else SupportLevel.UNSUPPORTED,
                confidence="reported",
                reason="vLLM GGUF support is limited to a subset of architectures (llama-family)",
            )
        )
        results.append(_unsupported(Backend.SGLANG, "GGUF serving is not a supported SGLang path in V1"))

    elif model_info.format is ModelFormat.SAFETENSORS:
        if arch and arch in vllm_archs:
            results.append(
                RuntimeSupport(
                    backend=Backend.VLLM,
                    supported=True,
                    level=SupportLevel.SUPPORTED,
                    confidence="verified",
                    reason=f"architecture '{arch}' is documented as natively supported by vLLM ({vllm_source})",
                )
            )
        else:
            results.append(
                _unsupported(
                    Backend.VLLM,
                    f"architecture {arch or 'unknown'} is not in vLLM's supported registry "
                    f"({vllm_source}); the Transformers fallback backend is not enabled in V1",
                )
            )
        if arch and arch in registry.SGLANG_SAFETENSORS_ARCHS:
            results.append(
                RuntimeSupport(
                    backend=Backend.SGLANG,
                    supported=True,
                    level=SupportLevel.EXPERIMENTAL,
                    confidence="reported",
                    reason=f"architecture '{arch}' is source-registered in SGLang (checked 2026-09-02)",
                )
            )
        else:
            results.append(_unsupported(Backend.SGLANG, f"architecture {arch or 'unknown'} not in SGLang registry"))
        results.append(_unsupported(Backend.LLAMA_CPP, "llama.cpp serves GGUF only — safetensors require conversion"))
    else:
        results.append(_unsupported(Backend.LLAMA_CPP, f"unsupported weight format: {model_info.format}"))
        results.append(_unsupported(Backend.VLLM, f"unsupported weight format: {model_info.format}"))
        results.append(_unsupported(Backend.SGLANG, f"unsupported weight format: {model_info.format}"))

    level_rank = {SupportLevel.SUPPORTED: 0, SupportLevel.EXPERIMENTAL: 1, SupportLevel.UNSUPPORTED: 2}
    results.sort(key=lambda r: level_rank[r.level])
    return results


def evaluate_lora_support(adapter: ModelInfo, base_info: ModelInfo, live: LiveRegistry | None = None) -> RuntimeSupport:
    """Can this adapter be SERVED (not just merged) on top of its base model?

    Serving path: vLLM `--enable-lora --lora-modules`, which requires:
      * the adapter in PEFT layout (adapter_config.json + adapter_model.safetensors);
      * the base model to be a vLLM-supported causal LM in safetensors format.
    Video/image bases (diffusion runtimes) and llama.cpp (no PEFT hot-load) are
    refused with the precise reason.
    """
    if not adapter.peft_layout:
        return _unsupported(
            Backend.VLLM,
            "adapter is not in the PEFT serving layout (adapter_config.json + "
            "adapter_model.safetensors required by vLLM --enable-lora); merge it into the "
            "base model and deploy the merged checkpoint instead",
        )
    if base_info.format is not ModelFormat.SAFETENSORS:
        return _unsupported(
            Backend.VLLM,
            f"the base model is {base_info.format.value}, not safetensors — vLLM LoRA serving "
            "requires a safetensors base",
        )
    if base_info.task in registry.UNSUPPORTED_TASKS:
        return _unsupported(
            Backend.VLLM,
            f"the base model is not a causal LM: {registry.UNSUPPORTED_TASKS[base_info.task]}",
        )
    arch = base_info.architecture
    if arch and arch in registry.VLLM_SAFETENSORS_ARCHS:
        return RuntimeSupport(
            backend=Backend.VLLM,
            supported=True,
            level=SupportLevel.SUPPORTED,
            confidence="verified",
            reason=f"LoRA serving on vLLM: base architecture '{arch}' is supported (checked "
            f"{live.label() if live else 'snapshot ' + registry.CHECK_DATE})",
        )
    if arch and arch in registry.SGLANG_SAFETENSORS_ARCHS:
        return RuntimeSupport(
            backend=Backend.SGLANG,
            supported=True,
            level=SupportLevel.EXPERIMENTAL,
            confidence="reported",
            reason=f"base architecture '{arch}' is SGLang-registered; SGLang --lora-paths serving "
            "is less battle-tested than vLLM",
        )
    return _unsupported(
        Backend.VLLM,
        f"the base architecture {arch or 'unknown'} is not in vLLM's supported registry — "
        "LoRA serving requires a supported base",
    )


def select_backend(
    model_info: ModelInfo, override: Backend | None = None, live: LiveRegistry | None = None
) -> RuntimeSupport:
    """Pick the best backend, honoring an explicit override (still validated)."""
    results = evaluate_support(model_info, live)
    by_backend = {r.backend: r for r in results}

    if override is not None:
        chosen = by_backend.get(override)
        if chosen is None:
            raise ModelNotSupportedError(f"unknown backend override {override}")
        if chosen.level is SupportLevel.UNSUPPORTED:
            raise ModelNotSupportedError(
                reason=f"backend {override.value} cannot serve this model: {chosen.reason}",
                detected=f"format={model_info.format.value} architecture={model_info.architecture}",
                possible_backend=", ".join(f"{r.backend.value} ({r.level.value})" for r in results if r.deployable)
                or "none",
            )
        return chosen

    for candidate in results:
        if candidate.level is SupportLevel.SUPPORTED:
            return candidate
    for candidate in results:
        if candidate.level is SupportLevel.EXPERIMENTAL:
            return candidate

    best_reason = results[0].reason if results else "no compatible backend"
    raise ModelNotSupportedError(
        reason=best_reason,
        detected=(
            f"task={model_info.task.value} format={model_info.format.value} architecture={model_info.architecture}"
        ),
        possible_backend="none in the hfvast runtime registry",
    )
