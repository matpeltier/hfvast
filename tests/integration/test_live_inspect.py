"""Integration tests hitting REAL external services (HF Hub).

Opt-in only: HFVAST_LIVE=1 pytest -m integration
These never create resources or spend money — inspection is read-only.
"""

import os

import pytest

pytestmark = pytest.mark.integration

requires_live = pytest.mark.skipif(
    os.environ.get("HFVAST_LIVE") != "1", reason="set HFVAST_LIVE=1 to run live inspection"
)


@requires_live
async def test_live_inspect_glm_gguf():
    from hfvast.config import resolve_credentials
    from hfvast.inspect.huggingface import HFInspector
    from hfvast.models.model import ModelFormat, QuantTier
    from hfvast.utils.hfref import parse_model_input

    _, hf_token = resolve_credentials()
    ref = parse_model_input("https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-GGUF")
    inspector = HFInspector(token=hf_token)
    try:
        info = await inspector.inspect(ref)
    finally:
        await inspector.aclose()

    assert info.format is ModelFormat.GGUF
    assert info.architecture == "glm5next"
    assert info.context_length == 1_048_576
    by_id = {v.id: v for v in info.variants}
    assert abs(by_id["Q4_K_M"].size_bytes / 1e9 - 193.0) < 1.0
    assert by_id["Q2_K"].tier is QuantTier.ECONOMY


@requires_live
async def test_live_inspect_safetensors():
    from hfvast.config import resolve_credentials
    from hfvast.inspect.huggingface import HFInspector
    from hfvast.models.model import ModelFormat
    from hfvast.utils.hfref import parse_model_input

    _, hf_token = resolve_credentials()
    ref = parse_model_input("Qwen/Qwen2.5-7B-Instruct")
    inspector = HFInspector(token=hf_token)
    try:
        info = await inspector.inspect(ref)
    finally:
        await inspector.aclose()
    assert info.format is ModelFormat.SAFETENSORS
    assert info.architecture == "Qwen2ForCausalLM"
