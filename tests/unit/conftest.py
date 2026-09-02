"""Shared fixtures: a MockTransport serving real captured HF API payloads."""

import json
from pathlib import Path

import httpx
import pytest

from hfvast.inspect.huggingface import HFInspector
from hfvast.utils.hfref import parse_model_input

FIXTURES = Path(__file__).parent.parent / "fixtures"

REPO = "orcarouter/GLM-5.3-Flash-Uncensored-GGUF"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def hf_mock_transport(
    modelinfo: dict | None = None,
    tree: list | None = None,
    config_json: dict | None = None,
    status_overrides: dict[str, tuple[int, dict[str, str]]] | None = None,
) -> httpx.MockTransport:
    modelinfo = modelinfo or _load("hf_modelinfo_gguf.json")
    tree = tree if tree is not None else _load("hf_tree_gguf.json")
    overrides = status_overrides or {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for fragment, (status, headers) in overrides.items():
            if fragment in url:
                return httpx.Response(status, text="denied", headers=headers)
        if "/tree/" in url:
            return httpx.Response(200, json=tree)
        if url.endswith("/config.json"):
            if config_json is None:
                return httpx.Response(404, text="not found")
            return httpx.Response(200, json=config_json)
        if "/api/models/" in url:
            return httpx.Response(200, json=modelinfo)
        return httpx.Response(404, text="unknown")

    return httpx.MockTransport(handler)


@pytest.fixture
def gguf_inspector() -> HFInspector:
    """Inspector over the real captured GLM-5.3-Flash GGUF payloads (header 401s: gated)."""
    return HFInspector(
        client=httpx.AsyncClient(transport=hf_mock_transport(), follow_redirects=True),
        token=None,
    )


__all__ = ["FIXTURES", "REPO", "gguf_inspector", "hf_mock_transport", "parse_model_input"]
