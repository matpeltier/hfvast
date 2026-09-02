import pytest
from conftest import REPO, hf_mock_transport

from hfvast.errors import GatedAccessError, ModelNotFoundError
from hfvast.inspect.huggingface import HFInspector
from hfvast.models.model import ModelFormat, ModelTask, QuantTier


async def test_inspect_real_gguf_payload(gguf_inspector):
    from hfvast.utils.hfref import parse_model_input

    info = await gguf_inspector.inspect(parse_model_input(REPO))
    await gguf_inspector.aclose()

    assert info.ref.repo_id == REPO
    assert info.format is ModelFormat.GGUF
    assert info.task is ModelTask.VISION_LANGUAGE  # image-text-to-text pipeline tag
    assert info.multimodal
    assert info.gated is True
    assert info.architecture == "glm5next"  # from the Hub's gguf object
    assert info.context_length == 1_048_576
    assert info.parameter_count == 320_759_404_382

    # Variants from the real tree (multi-shard grouping)
    by_id = {v.id: v for v in info.variants}
    assert set(by_id) == {"Q2_K", "Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0"}
    assert abs(by_id["Q4_K_M"].size_bytes / 1e9 - 192.97) < 0.1
    assert by_id["Q4_K_M"].is_split and len(by_id["Q4_K_M"].files) == 5
    assert by_id["Q2_K"].tier is QuantTier.ECONOMY
    assert by_id["Q4_K_M"].tier is QuantTier.BALANCED
    assert by_id["Q6_K"].tier is QuantTier.QUALITY
    assert by_id["Q8_0"].tier is QuantTier.QUALITY

    # mmproj detected
    assert any("mmproj" in f.path for f in info.mmproj_files)

    # Gated: header read failed → conservative fallback note present
    assert info.gguf_header is None
    assert any("header not readable" in n for n in info.notes)


async def test_gated_repo_friendly_error():
    import httpx

    transport = hf_mock_transport(status_overrides={"/api/models/": (401, {"x-error-code": "GatedRepo"})})
    inspector = HFInspector(client=httpx.AsyncClient(transport=transport))
    from hfvast.utils.hfref import parse_model_input

    with pytest.raises(GatedAccessError) as excinfo:
        await inspector.inspect(parse_model_input(REPO))
    assert "HF_TOKEN" in str(excinfo.value)
    await inspector.aclose()


async def test_missing_repo_not_found():
    import httpx

    def handler(request):
        return httpx.Response(401, text="Invalid username or password.")

    inspector = HFInspector(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    from hfvast.utils.hfref import parse_model_input

    with pytest.raises(ModelNotFoundError):
        await inspector.inspect(parse_model_input("ghost/model"))
    await inspector.aclose()


async def test_safetensors_inspection():
    import httpx

    modelinfo = {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "gated": False,
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "tags": ["text-generation"],
        "config": {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768},
        "safetensors": {"parameters": {"BF16": 7615616512}, "total": 7615616512},
        "usedStorage": 15464628224,
        "sha": "abc123",
    }
    tree = [
        {"type": "file", "path": "config.json", "size": 700},
        {"type": "file", "path": "model-00001-of-00004.safetensors", "size": 3864726569},
        {"type": "file", "path": "model-00002-of-00004.safetensors", "size": 3864726570},
        {"type": "file", "path": "model-00003-of-00004.safetensors", "size": 3864726571},
        {"type": "file", "path": "model-00004-of-00004.safetensors", "size": 3864726568},
    ]
    config_json = {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768, "torch_dtype": "bfloat16"}
    transport = httpx.MockTransport(hf_mock_transport(modelinfo=modelinfo, tree=tree, config_json=config_json).handler)
    inspector = HFInspector(client=httpx.AsyncClient(transport=transport))
    from hfvast.utils.hfref import parse_model_input

    info = await inspector.inspect(parse_model_input("Qwen/Qwen2.5-7B-Instruct"))
    await inspector.aclose()

    assert info.format is ModelFormat.SAFETENSORS
    assert info.task is ModelTask.TEXT_GENERATION
    assert info.architecture == "Qwen2ForCausalLM"
    assert info.parameter_count == 7615616512
    assert info.context_length == 32768
    assert info.dtype == "BF16"
    assert len(info.variants) == 1
    assert info.variants[0].size_bytes == sum(f["size"] for f in tree if "safetensors" in f["path"])
