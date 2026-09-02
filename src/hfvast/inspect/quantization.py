"""Quantization detection and tier heuristics from GGUF/file names."""

from __future__ import annotations

import re

from hfvast.models.model import QuantTier

# Matches common quant labels: Q4_K_M, Q3_K_S, Q5_0, IQ4_XS, IQ2_XXS, F16, BF16, MXFP4_MOE, TQ1_0 ...
QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(IQ[1-4]_(?:XXS|XS|S|M|NL|BLD|BNL)|Q[2-8]_K_(?:S|M|L|XL|XXL|XXS|TQ)|Q[2-8]_[01]|Q[2-8]_K|"
    r"F16|F32|FP16|FP32|BF16|MXFP4(?:_MOE)?|TQ[12]_[01])"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_EXTREME_QUANTS = {"Q2_K", "Q2_0", "Q2_1", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "TQ1_0", "TQ2_0"}

_BALANCED = {"Q4_K_S", "Q4_K_M", "Q4_0", "Q4_1", "IQ4_XS", "IQ4_NL", "Q5_K_S", "MXFP4", "MXFP4_MOE"}
_QUALITY = {"Q5_K_M", "Q5_K_L", "Q6_K", "Q8_0", "F16", "FP16", "BF16", "F32", "FP32"}


def detect_quant(name: str) -> str | None:
    """Extract the most specific quant label from a file or variant name."""
    matches = QUANT_RE.findall(name)
    return matches[-1].upper() if matches else None


def tier_for_quant(quant: str | None) -> QuantTier | None:
    """Map a quant label to its conceptual tier (ECONOMY / BALANCED / QUALITY)."""
    if quant is None:
        return None
    q = quant.upper()
    if q in _EXTREME_QUANTS:
        return QuantTier.ECONOMY
    if q in _BALANCED:
        return QuantTier.BALANCED
    if q in _QUALITY:
        return QuantTier.QUALITY
    # Q3 family and remaining IQ3 are the economy edge; anything unknown is unclassified.
    if q.startswith(("Q3", "IQ3")):
        return QuantTier.ECONOMY
    return None


def extreme_warning(quant: str | None) -> str | None:
    if quant and quant.upper() in _EXTREME_QUANTS:
        return (
            f"{quant} is an extremely aggressive quantization; expected quality loss is material. "
            "hfvast recommends Q4-family or better unless VRAM forces otherwise."
        )
    return None
