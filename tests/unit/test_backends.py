import pytest

from hfvast.errors import ModelNotSupportedError
from hfvast.models.model import ModelFormat, ModelInfo, ModelTask
from hfvast.planning.backends import evaluate_support, select_backend
from hfvast.runtimes.base import Backend, SupportLevel
from hfvast.utils.hfref import parse_model_input


def _model(**kwargs) -> ModelInfo:
    defaults: dict = {
        "ref": parse_model_input("org/model"),
        "task": ModelTask.TEXT_GENERATION,
        "format": ModelFormat.GGUF,
        "architecture": "llama",
    }
    defaults.update(kwargs)
    return ModelInfo(**defaults)


def test_gguf_known_arch_supported_by_llama_cpp():
    results = evaluate_support(_model(architecture="qwen3"))
    llama = next(r for r in results if r.backend is Backend.LLAMA_CPP)
    assert llama.level is SupportLevel.SUPPORTED
    assert llama.confidence == "verified"


def test_glm5next_is_experimental_not_supported():
    """GLM-5.3-Flash declares GGUF arch 'glm5next'; llama.cpp serves GLM-5 as 'glm-dsa'
    (checked 2026-09-02). It must be EXPERIMENTAL — never silently 'supported'."""
    results = evaluate_support(_model(architecture="glm5next"))
    llama = next(r for r in results if r.backend is Backend.LLAMA_CPP)
    assert llama.level is SupportLevel.EXPERIMENTAL
    assert "glm-dsa" in llama.reason


def test_unknown_gguf_arch_unsupported():
    results = evaluate_support(_model(architecture="totally-new-arch"))
    llama = next(r for r in results if r.backend is Backend.LLAMA_CPP)
    assert llama.level is SupportLevel.UNSUPPORTED
    with pytest.raises(ModelNotSupportedError):
        select_backend(_model(architecture="totally-new-arch"))


def test_safetensors_routes_to_vllm():
    results = evaluate_support(_model(format=ModelFormat.SAFETENSORS, architecture="Qwen2ForCausalLM"))
    vllm = next(r for r in results if r.backend is Backend.VLLM)
    assert vllm.level is SupportLevel.SUPPORTED
    llama = next(r for r in results if r.backend is Backend.LLAMA_CPP)
    assert llama.level is SupportLevel.UNSUPPORTED


def test_diffusion_like_task_never_provisions():
    with pytest.raises(ModelNotSupportedError):
        select_backend(_model(task=ModelTask.OTHER))


def test_trust_remote_code_rejected():
    with pytest.raises(ModelNotSupportedError):
        select_backend(_model(requires_trust_remote_code=True))


def test_backend_override_still_validates():
    model = _model(format=ModelFormat.SAFETENSORS, architecture="LlamaForCausalLM")
    chosen = select_backend(model, override=Backend.VLLM)
    assert chosen.backend is Backend.VLLM
    with pytest.raises(ModelNotSupportedError):
        select_backend(model, override=Backend.LLAMA_CPP)  # llama.cpp cannot serve safetensors
