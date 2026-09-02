import pytest

from hfvast.config import (
    AppConfig,
    alias_add,
    alias_lookup,
    alias_remove,
    load_config,
    resolve_credentials,
    save_config,
)


def test_defaults_when_missing(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config.defaults.idle_timeout == "30m"
    assert config.defaults.max_runtime == "6h"
    assert config.defaults.expected_session == "2h"
    assert config.defaults.min_reliability == 0.98
    assert config.cost.max_hourly is None
    assert config.aliases == {}


def test_roundtrip_aliases(tmp_path):
    path = tmp_path / "config.toml"
    alias_add("glm-uncensored", "https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-GGUF", path=path)
    alias_add("qwen", "Qwen/Qwen2.5-7B-Instruct", preferences={"quant": "Q4_K_M"}, path=path)
    config = load_config(path)
    assert set(config.aliases) == {"glm-uncensored", "qwen"}
    assert config.aliases["qwen"].preferences == {"quant": "Q4_K_M"}
    assert alias_lookup("glm-uncensored", path=path).endswith("GLM-5.3-Flash-Uncensored-GGUF")
    assert alias_lookup("not-an-alias", path=path) == "not-an-alias"

    assert alias_remove("qwen", path=path) is True
    assert alias_remove("qwen", path=path) is False
    assert set(load_config(path).aliases) == {"glm-uncensored"}


def test_sections_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    config = AppConfig()
    config.cost.max_hourly = 3.0
    config.defaults.context = 4096
    config.vast.secure_cloud_only = True
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.cost.max_hourly == 3.0
    assert loaded.defaults.context == 4096
    assert loaded.vast.secure_cloud_only is True


def test_invalid_toml_raises_friendly_error(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("not [ valid toml", encoding="utf-8")
    from hfvast.config import ConfigError

    with pytest.raises(ConfigError):
        load_config(path)


def test_env_preferred_over_flag(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "vast-env-key-1234567890")
    monkeypatch.setenv("HF_TOKEN", "hf_environmenttoken123456")
    vast, hf = resolve_credentials("vast-flag-key-0987654321", "hf_flagtoken098765432")
    assert vast == "vast-env-key-1234567890"
    assert hf == "hf_environmenttoken123456"


def test_flag_used_when_env_missing(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast, hf = resolve_credentials("vast-flag-key-0987654321", None)
    assert vast == "vast-flag-key-0987654321"
    assert hf is None
