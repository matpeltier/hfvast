"""Configuration file + credential resolution.

Precedence (spec §36): CLI args > environment > config file > defaults.
Credentials (spec §4): environment is PREFERRED over CLI flags because shell
arguments leak into history/process listings; a flag is used only when the env
var is unset. Secrets are registered with the central redactor on resolution
and are never persisted to config or state.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, Field

from hfvast.utils.paths import config_file, ensure_dirs
from hfvast.utils.redact import register_secrets


class DefaultsSection(BaseModel):
    idle_timeout: str = "30m"
    max_runtime: str = "6h"
    expected_session: str = "2h"
    min_reliability: float = 0.98
    min_download_mbps: float = 300.0
    context: int = 8192
    concurrency: int = 1


class CostSection(BaseModel):
    max_hourly: float | None = None
    max_startup: float | None = None
    max_total: float | None = None


class VastSection(BaseModel):
    secure_cloud_only: bool = False


class AliasEntry(BaseModel):
    url: str
    preferences: dict[str, str] = Field(default_factory=dict)


class AppConfig(BaseModel):
    defaults: DefaultsSection = Field(default_factory=DefaultsSection)
    cost: CostSection = Field(default_factory=CostSection)
    vast: VastSection = Field(default_factory=VastSection)
    aliases: dict[str, AliasEntry] = Field(default_factory=dict)


class ConfigError(Exception):
    pass


def load_config(path: Path | None = None) -> AppConfig:
    file = path or config_file()
    if not file.exists():
        return AppConfig()
    try:
        raw = tomllib.loads(file.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"Could not read config file {file}: {exc}") from exc
    try:
        return AppConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid config file {file}: {exc}") from exc


def save_config(config: AppConfig, path: Path | None = None) -> None:
    file = path or config_file()
    ensure_dirs()
    data: dict[str, Any] = {
        "defaults": _drop_none(config.defaults.model_dump()),
        "cost": _drop_none(config.cost.model_dump()),
        "vast": config.vast.model_dump(),
        "aliases": {name: _drop_none(entry.model_dump()) for name, entry in config.aliases.items()},
    }
    file.write_text(tomli_w.dumps(data), encoding="utf-8")


def _drop_none(section: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in section.items() if v is not None}


# ------------------------------------------------------------------- aliases


def alias_add(name: str, url: str, preferences: dict[str, str] | None = None, path: Path | None = None) -> None:
    config = load_config(path)
    config.aliases[name] = AliasEntry(url=url, preferences=preferences or {})
    save_config(config, path)


def alias_remove(name: str, path: Path | None = None) -> bool:
    config = load_config(path)
    if name not in config.aliases:
        return False
    del config.aliases[name]
    save_config(config, path)
    return True


def alias_lookup(name_or_url: str, path: Path | None = None) -> str:
    """Resolve an alias name to a model URL; pass through anything else."""
    config = load_config(path)
    entry = config.aliases.get(name_or_url)
    return entry.url if entry else name_or_url


# -------------------------------------------------------------- credentials


def resolve_credentials(
    vast_api_key: str | None = None,
    hf_token: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve (vast_api_key, hf_token). Environment wins over CLI flags.

    Resolved secrets are registered with the central redactor immediately.
    """
    import os

    resolved_vast = os.environ.get("VAST_API_KEY") or vast_api_key
    resolved_hf = os.environ.get("HF_TOKEN") or hf_token
    register_secrets(resolved_vast, resolved_hf)
    return resolved_vast, resolved_hf
