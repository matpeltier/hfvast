"""Cloud provider abstraction (spec §17/§42).

V1 ships only :class:`hfvast.providers.vast.VastProvider`, but the planner depends
only on this protocol, so future RunPod/Lambda/Modal providers slot in without
redesigning the planner.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hfvast.models.offers import GPUOffer, OfferQuery


@runtime_checkable
class ComputeProvider(Protocol):
    """A rentable-GPU cloud provider."""

    #: "vast" | "sample" | ... — "sample" data must always be labeled as such in UI.
    name: str
    data_source: str  # "live" | "sample"

    async def search_offers(self, query: OfferQuery) -> list[GPUOffer]: ...

    async def get_instance(self, instance_id: int) -> dict[str, Any] | None: ...

    async def destroy_instance(self, instance_id: int) -> None: ...

    async def logs(self, instance_id: int, tail: int = 1000) -> str: ...


class InstanceSpec:
    """Provisioning payload for `create_instance` [M2]."""

    def __init__(
        self,
        image: str,
        disk_gb: int,
        env: dict[str, str],
        onstart: str,
        label: str,
        ports: list[tuple[int, int]] | None = None,
    ) -> None:
        self.image = image
        self.disk_gb = disk_gb
        self.env = env
        self.onstart = onstart
        self.label = label
        self.ports = ports or []
