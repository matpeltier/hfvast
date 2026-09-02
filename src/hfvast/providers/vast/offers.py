"""Vast.ai offer search: query building, normalization, snapshot fallback."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from hfvast.errors import ProviderError
from hfvast.models.offers import GPUOffer, OfferQuery
from hfvast.providers.vast.client import VastClient
from hfvast.utils.redact import redact

_GB = 1e9
_MIB = 1024 * 1024


def build_bundle_filters(query: OfferQuery) -> dict[str, Any]:
    """Translate an OfferQuery into the Vast /api/v0/bundles/ filter body."""
    filters: dict[str, Any] = {
        "rentable": {"eq": True},
        "num_gpus": {"gte": 1, "lte": query.max_gpus},
        "gpu_ram": {"gte": int(query.min_per_gpu_vram_gb * 1024)},  # MB per GPU (REST units)
        "disk_space": {"gte": int(query.disk_gb)},
        "reliability": {"gte": query.min_reliability},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
        "limit": query.limit,
        # Disk size used for pricing storage within dph_total:
        "allocated_storage": int(query.disk_gb),
    }
    if query.min_per_gpu_vram_gb > 1.0:
        pass  # gpu_ram filter already set above
    if query.min_download_mbps > 0:
        filters["inet_down"] = {"gte": int(query.min_download_mbps)}
    if query.gpu_filter:
        filters["gpu_name"] = {"in": _gpu_name_variants(query.gpu_filter)}
    if query.secure_cloud_only:
        filters["verified"] = {"eq": True}
        filters["external"] = {"eq": False}
    # NOTE: max_hourly_usd is intentionally NOT sent as a dph_total filter —
    # caps are enforced after ranking so we can report the cheapest over-cap
    # offer to the user (spec scenario E) instead of an opaque empty result.
    return filters


def _gpu_name_variants(name: str) -> list[str]:
    """Vast writes GPU names like `RTX_3090`, `A100_SXM4` — try common variants."""
    cleaned = name.strip().replace(" ", "_").replace("-", "_")
    return [cleaned, cleaned.upper(), cleaned.replace("RTX_", "RTX_")] if cleaned else []


def normalize_offer(raw: dict[str, Any]) -> GPUOffer:
    """Convert a raw Vast offer record into the provider-agnostic GPUOffer."""
    try:
        offer_id = int(raw["id"])
        gpu_name = str(raw.get("gpu_name", "unknown"))
        num_gpus = int(raw.get("num_gpus") or 1)
        gpu_ram_mb = float(raw.get("gpu_ram") or 0)
        disk_gb = float(raw.get("disk_space") or 0)
        hourly_total = float(raw.get("dph_total") or 0)
        hourly_base = float(raw.get("dph_base") or hourly_total)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"Malformed offer record from Vast.ai: {redact(exc)}") from exc

    per_gpu_vram_gb = gpu_ram_mb / 1024.0  # REST gpu_ram is MB (MiB) per GPU
    return GPUOffer(
        offer_id=offer_id,
        gpu_model=gpu_name,
        gpu_count=num_gpus,
        per_gpu_vram_gb=round(per_gpu_vram_gb, 2),
        total_vram_gb=round(per_gpu_vram_gb * num_gpus, 2),
        cpu_cores=_opt_int(raw.get("cpu_cores")),
        cpu_ram_gb=round(float(raw["cpu_ram"]) / 1024.0, 1) if raw.get("cpu_ram") else None,
        disk_gb=disk_gb,
        disk_bw_mbs=_opt_float(raw.get("disk_bw")),
        inet_down_mbps=float(raw.get("inet_down") or 0),
        inet_up_mbps=_opt_float(raw.get("inet_up")),
        hourly_gpu_usd=hourly_base,
        hourly_total_usd=hourly_total,
        storage_per_gb_month_usd=float(raw.get("storage_cost") or 0),
        inet_down_usd_per_gb=float(raw.get("inet_down_cost") or 0),
        reliability=float(raw.get("reliability") or 0),
        verified=bool(raw.get("verified") or raw.get("verification") == "verified"),
        dlperf=_opt_float(raw.get("dlperf")),
        gpu_mem_bw_gbs=_opt_float(raw.get("gpu_mem_bw")),
        pcie_bw_gbs=_opt_float(raw.get("pcie_bw")),
        nvlink_gbs=_opt_float(raw.get("bw_nvlink")),
        geolocation=raw.get("geolocation") if isinstance(raw.get("geolocation"), str) else None,
    )


def _opt_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def load_snapshot() -> dict[str, Any]:
    """Load the bundled sample-offers snapshot (clearly labeled, never live data)."""
    resource = resources.files("hfvast").joinpath("data/vast_offers_sample.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class VastProvider:
    """Live Vast.ai provider."""

    name = "vast"
    data_source = "live"

    def __init__(self, client: VastClient | None = None, api_key: str | None = None) -> None:
        if client is None:
            if not api_key:
                raise ValueError("VastProvider requires api_key or a VastClient")
            client = VastClient(api_key)
        self._client = client

    async def search_offers(self, query: OfferQuery) -> list[GPUOffer]:
        raw = await self._client.search_bundles(build_bundle_filters(query))
        return [normalize_offer(r) for r in raw]

    async def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        return await self._client.get_instance(instance_id)

    async def destroy_instance(self, instance_id: int) -> None:
        await self._client.destroy_instance(instance_id)

    async def logs(self, instance_id: int, tail: int = 1000) -> str:
        url = await self._client.fetch_logs_url(instance_id, tail)
        import httpx

        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text


class SnapshotProvider:
    """Bundled sample offers — used only when no VAST_API_KEY is configured.

    Output is always labeled `sample` so the UI can mark it clearly as NOT live
    data (spec §56: never fake live results).
    """

    name = "sample"
    data_source = "sample"

    def __init__(self) -> None:
        self._snapshot = load_snapshot()

    async def search_offers(self, query: OfferQuery) -> list[GPUOffer]:
        offers = [normalize_offer(r) for r in self._snapshot.get("offers", [])]
        return [o for o in offers if self._matches(o, query)]

    async def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        raise NotImplementedError("instance commands arrive in Milestone 2")

    async def destroy_instance(self, instance_id: int) -> None:
        raise NotImplementedError("instance commands arrive in Milestone 2")

    async def logs(self, instance_id: int, tail: int = 1000) -> str:
        raise NotImplementedError("instance commands arrive in Milestone 2")

    @staticmethod
    def _matches(offer: GPUOffer, query: OfferQuery) -> bool:
        if offer.gpu_count > query.max_gpus:
            return False
        if offer.per_gpu_vram_gb < query.min_per_gpu_vram_gb:
            return False
        if offer.disk_gb < query.disk_gb:
            return False
        if offer.inet_down_mbps < query.min_download_mbps:
            return False
        if offer.reliability < query.min_reliability:
            return False
        if query.gpu_filter:
            needle = query.gpu_filter.lower().replace(" ", "_")
            if needle not in offer.gpu_model.lower().replace(" ", "_"):
                return False
        return not (query.secure_cloud_only and not offer.verified)
