"""Inspector protocol."""

from __future__ import annotations

from typing import Protocol

from hfvast.models.model import HFModelRef, ModelInfo


class Inspector(Protocol):
    """Something that can inspect a model repository without downloading weights."""

    async def inspect(self, ref: HFModelRef) -> ModelInfo: ...
