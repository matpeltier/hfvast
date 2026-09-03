import base64
import gzip

from hfvast.deploy.bootstrap import (
    BootstrapSpec,
    build_create_env,
    build_onstart,
    build_payload,
    decode_payload,
    describe_env,
    encode_payload,
)
from hfvast.models.model import ModelFile


def _spec(**kwargs) -> BootstrapSpec:
    defaults = dict(
        deployment_id="glm-test-aa11",
        model_repo="org/model-GGUF",
        revision=None,
        files=[
            ModelFile(path="Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", size_bytes=1000),
            ModelFile(path="Q4_K_M/m-Q4_K_M-00002-of-00002.gguf", size_bytes=2000),
        ],
        backend_cmd_args=["llama-server", "--port", "8001", "--api-key-file", "/opt/hfvast/backend_api_key"],
        gateway_key="gw-key-1234567890",
        backend_key="bk-key-1234567890",
        hf_token="hf_tokenvalue123456789",
    )
    defaults.update(kwargs)
    return BootstrapSpec(**defaults)


def test_payload_roundtrip():
    payload = build_payload(_spec())
    env = encode_payload(payload)
    assert decode_payload(env) == payload
    # chunks are modest in size
    assert all(len(v) <= 4096 for v in env.values())


def test_payload_contains_components_and_config():
    payload = build_payload(_spec())
    assert "gateway.py" in payload
    assert "watchdog.py" in payload
    assert "HFVAST_GATEWAY_EOF" in payload
    # the token VALUE is never baked in; the embedded downloader reads HF_TOKEN from env
    assert 'os.environ.get("HF_TOKEN"' in payload
    assert "hf_tokenvalue123456789" not in payload.split("HFVAST_FILES_EOF")[0]
    assert "https://huggingface.co/org/model-GGUF/resolve/main" in payload
    assert "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf\t1000" in payload
    # state transitions in order
    assert payload.index("status downloading") < payload.index("status loading") < payload.index("status ready")


def test_backend_cmd_key_substitution():
    args = ["vllm", "serve", "m", "--api-key", "@BACKEND_KEY@"]
    payload = build_payload(_spec(backend_cmd_args=args))
    assert "bk-key-1234567890" in payload
    assert "@BACKEND_KEY@" not in payload


def test_real_llama_cpp_plan_survives_generation():
    from hfvast.models.hardware import HardwareRequirements
    from hfvast.models.model import GGUFHeaderInfo, ModelInfo, ModelVariant
    from hfvast.planning.memory import estimate_vram
    from hfvast.runtimes.base import Backend
    from hfvast.runtimes.llama_cpp import build_plan
    from hfvast.utils.hfref import parse_model_input

    model = ModelInfo(
        ref=parse_model_input("org/m"),
        architecture="qwen2",
        format="gguf",
        gguf_header=GGUFHeaderInfo(architecture="qwen2", block_count=32, head_count_kv=8, key_length=128),
    )
    variant = ModelVariant(
        id="Q4_K_M",
        quant="Q4_K_M",
        size_bytes=10**9,
        files=[ModelFile(path="Q4_K_M/m-00001-of-00002.gguf", size_bytes=500)],
    )
    breakdown = estimate_vram(model, variant, 8192, 1, Backend.LLAMA_CPP)
    reqs = HardwareRequirements(
        minimum_vram_gib=1,
        recommended_vram_gib=2,
        disk_gb=24,
        context_length=8192,
        concurrency=1,
        breakdown=breakdown,
    )
    plan = build_plan(model, variant, reqs, gpu_count=2)
    payload = build_payload(_spec(backend_cmd_args=plan.args))
    assert "--tensor-split 1,1" in payload
    assert "/opt/hfvast/models/m-00001-of-00002.gguf" in payload


def test_onstart_stub_small_and_functional():
    stub = build_onstart()
    assert len(stub) < 4048  # Vast onstart limit
    assert "HFVAST_PAYLOAD_" in stub and "base64 -d" in stub


def test_create_env_secrets_and_chunks():
    env = build_create_env(_spec(), hf_token="hf_tokenvalue123456789")
    assert env["HF_TOKEN"] == "hf_tokenvalue123456789"
    assert env["HFVAST_GATEWAY_KEY"] == "gw-key-1234567890"
    assert "-p 8000:8000" in env
    decoded = gzip.decompress(
        base64.b64decode("".join(v for k, v in sorted(env.items()) if k.startswith("HFVAST_PAYLOAD_")))
    ).decode()
    assert "hfvast bootstrap payload" in decoded


def test_describe_env_never_leaks_secrets():
    env = build_create_env(_spec(), hf_token="hf_tokenvalue123456789")
    described = describe_env(env)
    assert "hf_tokenvalue123456789" not in str(described)
    assert described["HF_TOKEN"] == "<set>"
