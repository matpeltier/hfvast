"""Live upstream compatibility registry.

The baked lists in ``registry.py`` are snapshots — they age. The authoritative
sources are public and stable:
  * llama.cpp: ``src/llama-arch.cpp`` — the LLM_ARCH_NAMES table maps GGUF
    architecture strings; if the string is there, llama-server can load it;
  * vLLM: ``docs/models/supported_models.md`` — generative architectures are
    listed as ``XxxForCausalLM``.

Fetching these at quote/inspect time means support detection adapts to upstream
releases with ZERO manual maintenance and ZERO per-architecture testing: the
backend itself declares what it can load. Cache TTL 24 h in the state dir; on
network failure we fall back to the baked snapshot.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from hfvast.utils.paths import state_dir

LLAMA_CPP_ARCH_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp"
)
VLLM_MODELS_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/main/docs/models/supported_models.md"
)

CACHE_TTL_S = 24 * 3600
FETCH_TIMEOUT_S = 20.0

_ARCH_STRING_RE = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}')
_VLLM_ARCH_RE = re.compile(r"`([A-Z][A-Za-z0-9]*(?:ForCausalLM|ForConditionalGeneration))`")


@dataclass
class LiveRegistry:
    """Fresh upstream support data (or a stale/baked fallback)."""

    llama_gguf_archs: frozenset[str] = field(default_factory=frozenset)
    vllm_archs: frozenset[str] = field(default_factory=frozenset)
    fetched_at: float = 0.0
    source: str = "baked"  # "live" | "cache" | "baked"

    @property
    def age_hours(self) -> float:
        if not self.fetched_at:
            return float("inf")
        return (time.time() - self.fetched_at) / 3600.0

    def label(self) -> str:
        if self.source == "live" and self.fetched_at:
            stamp = datetime.fromtimestamp(self.fetched_at, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
            return f"live upstream lists (llama.cpp master + vLLM main, fetched {stamp})"
        if self.source == "cache":
            return f"cached upstream lists ({self.age_hours:.0f} h old)"
        return "baked snapshot (upstream fetch failed)"


def _cache_path() -> Path:
    return state_dir() / "upstream_registry.json"


def _parse_llama_archs(source: str) -> frozenset[str]:
    """Extract GGUF arch strings from LLM_ARCH_NAMES table entries."""
    return frozenset(m.strip().lower() for m in _ARCH_STRING_RE.findall(source))


def _parse_vllm_archs(source: str) -> frozenset[str]:
    return frozenset(_VLLM_ARCH_RE.findall(source))


def _load_cache() -> dict[str, Any] | None:
    path = _cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("fetched_at"):
            return data
    except (OSError, ValueError):
        pass
    return None


def _save_cache(registry: LiveRegistry) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fetched_at": registry.fetched_at,
                    "llama": sorted(registry.llama_gguf_archs),
                    "vllm": sorted(registry.vllm_archs),
                    "source": registry.source,
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # cache is best-effort


async def load_live_registry(client: httpx.AsyncClient | None = None, refresh: bool = False) -> LiveRegistry:
    """Fetch upstream support lists (24 h cache); fall back to cache, then baked."""
    cache = _load_cache()
    if not refresh and cache and time.time() - float(cache["fetched_at"]) < CACHE_TTL_S:
        return LiveRegistry(
            llama_gguf_archs=frozenset(cache["llama"]),
            vllm_archs=frozenset(cache["vllm"]),
            fetched_at=float(cache["fetched_at"]),
            source="cache",
        )

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=FETCH_TIMEOUT_S, follow_redirects=True)
    try:
        llama_resp, vllm_resp = await asyncio.gather(
            http.get(LLAMA_CPP_ARCH_URL), http.get(VLLM_MODELS_URL)
        )
        if llama_resp.status_code == 200 and vllm_resp.status_code == 200:
            registry = LiveRegistry(
                llama_gguf_archs=_parse_llama_archs(llama_resp.text),
                vllm_archs=_parse_vllm_archs(vllm_resp.text),
                fetched_at=time.time(),
                source="live",
            )
            if registry.llama_gguf_archs:  # sanity: parse produced something
                _save_cache(registry)
                return registry
    except httpx.HTTPError:
        pass
    finally:
        if owns_client:
            await http.aclose()

    if cache:
        return LiveRegistry(
            llama_gguf_archs=frozenset(cache["llama"]),
            vllm_archs=frozenset(cache["vllm"]),
            fetched_at=float(cache["fetched_at"]),
            source="cache",
        )
    return LiveRegistry()  # fully baked fallback
