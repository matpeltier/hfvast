"""Parse user-supplied model references into :class:`HFModelRef`."""

from __future__ import annotations

from urllib.parse import urlparse

from hfvast.errors import ModelInputError
from hfvast.models.model import HFModelRef

_HF_HOSTS = {"huggingface.co", "hf.co", "www.huggingface.co", "www.hf.co"}


def parse_model_input(raw: str) -> HFModelRef:
    """Accept ``org/model``, ``org/model@rev``, or a huggingface.co URL."""
    text = raw.strip()
    if not text:
        raise ModelInputError("Model reference is empty. Use e.g. https://huggingface.co/org/model")

    if "://" in text:
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in _HF_HOSTS:
            raise ModelInputError(
                f"Not a Hugging Face URL: {text!r}\nExpected https://huggingface.co/<org>/<model> or <org>/<model>"
            )
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ModelInputError(
                f"Incomplete Hugging Face URL: {text!r}\nExpected https://huggingface.co/<org>/<model>"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        revision: str | None = None
        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            revision = "/".join(parts[3:])
    else:
        repo_id = text
        revision = None

    if "@" in repo_id:
        repo_id, _, rev = repo_id.partition("@")
        if not rev:
            raise ModelInputError(f"Empty revision in {text!r}")
        revision = revision or rev

    repo_id = repo_id.strip("/")
    if repo_id.count("/") != 1 or " " in repo_id or any(ch in repo_id for ch in "?#"):
        raise ModelInputError(
            f"Invalid model reference: {text!r}\n"
            "Expected <org>/<model> (exactly one '/'), e.g. orcarouter/GLM-5.3-Flash-Uncensored-GGUF"
        )
    if revision and not revision.strip("/"):
        raise ModelInputError(f"Invalid revision in {text!r}")
    return HFModelRef(repo_id=repo_id, revision=revision or None)
