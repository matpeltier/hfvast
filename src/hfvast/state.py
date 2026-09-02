"""Local deployment state (platformdirs), per spec §31.

Secrets are NEVER persisted here — only deployment metadata.
"""

from __future__ import annotations

import json
from typing import Any

from hfvast.utils.paths import ensure_dirs, state_file


def load_deployments() -> list[dict[str, Any]]:
    file = state_file()
    if not file.exists():
        return []
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def save_deployments(deployments: list[dict[str, Any]]) -> None:
    ensure_dirs()
    state_file().write_text(json.dumps(deployments, indent=2), encoding="utf-8")
