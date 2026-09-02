"""Filesystem locations (config/state) via platformdirs."""

from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

_dirs = PlatformDirs(appname="hfvast", appauthor=False)


def config_dir() -> Path:
    return Path(_dirs.user_config_dir)


def config_file() -> Path:
    return config_dir() / "config.toml"


def state_dir() -> Path:
    return Path(_dirs.user_state_dir)


def state_file() -> Path:
    return state_dir() / "deployments.json"


def ensure_dirs() -> None:
    config_dir().mkdir(parents=True, exist_ok=True)
    state_dir().mkdir(parents=True, exist_ok=True)
