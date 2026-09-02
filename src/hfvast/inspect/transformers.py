"""Transformers/safetensors config analysis helpers (pure functions)."""

from __future__ import annotations

from typing import Any

QUANT_METHOD_LABELS = {
    "awq": "AWQ",
    "gptq": "GPTQ",
    "fp8": "FP8",
    "bitsandbytes": "bitsandbytes",
    "compressed-tensors": "compressed-tensors",
    "quark": "Quark",
    "modelopt": "ModelOpt",
    "modelopt_fp8": "ModelOpt FP8",
    "torchao": "TorchAO",
    "hqq": "HQQ",
    "marlin": "Marlin",
    "mxfp4": "MXFP4",
}


def architecture_from_config(config: dict[str, Any]) -> str | None:
    archs = config.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        return archs[0]
    return None


def context_length_from_config(config: dict[str, Any]) -> int | None:
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def quantization_label_from_config(config: dict[str, Any]) -> str | None:
    quant_cfg = config.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        return None
    method = quant_cfg.get("quant_method")
    if isinstance(method, str):
        return QUANT_METHOD_LABELS.get(method.lower(), method)
    if quant_cfg.get("load_in_4bit") or quant_cfg.get("bits") == 4:
        return "bitsandbytes-4bit"
    if quant_cfg.get("load_in_8bit") or quant_cfg.get("bits") == 8:
        return "bitsandbytes-8bit"
    return None


def requires_trust_remote_code(config: dict[str, Any]) -> bool:
    return bool(config.get("auto_map")) or bool(config.get("trust_remote_code"))


def dtype_label(config: dict[str, Any], param_counts: dict[str, int] | None) -> str | None:
    if param_counts:
        # e.g. {"BF16": 8030261248, "F32": 123} — report the dominant dtype.
        dominant = max(param_counts, key=lambda k: param_counts[k])
        return dominant
    dtype = config.get("torch_dtype") or config.get("dtype")
    return dtype if isinstance(dtype, str) else None
