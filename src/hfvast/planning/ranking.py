"""Explainable offer ranking (spec §19–§21).

Viability first (usable VRAM, disk, reliability), then
score = expected session cost + topology penalties. Every penalty and pro is
surfaced as a reason string — no opaque score.
"""

from __future__ import annotations

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.models.offers import GPUOffer
from hfvast.models.quote import RankedOffer
from hfvast.planning.memory import estimate_vram
from hfvast.providers.vast.pricing import DEFAULT_NETWORK_EFFICIENCY, cost_breakdown
from hfvast.runtimes.base import Backend

GB = 1e9

#: Penalty weights, as a fraction of the session cost per triggered con.
PENALTY_NO_NVLINK = 0.03
PENALTY_EXTRA_GPUS = 0.01
PENALTY_SLOW_PCIE = 0.02
PENALTY_TIGHT_DISK = 0.01


class OfferRanker:
    """Ranks offers for one (model, variant, requirements) combination."""

    def __init__(
        self,
        session_hours: float,
        network_efficiency: float = DEFAULT_NETWORK_EFFICIENCY,
        image_size_gb: float = 8.0,
    ) -> None:
        self._session = session_hours
        self._efficiency = network_efficiency
        self._image_gb = image_size_gb

    def rank(
        self,
        offers: list[GPUOffer],
        model: ModelInfo,
        variant: ModelVariant,
        requirements: HardwareRequirements,
        backend: Backend,
    ) -> list[RankedOffer]:
        download_gb = (variant.size_bytes + sum(f.size_bytes for f in model.mmproj_files)) / GB

        viable: list[RankedOffer] = []
        rejections: dict[str, str] = {}
        for offer in offers:
            reason = self._rejection_reason(offer, model, variant, requirements, backend)
            if reason is not None:
                rejections[str(offer.offer_id)] = reason
                continue
            cost = cost_breakdown(
                offer,
                download_gb=download_gb,
                session_hours=self._session,
                efficiency=self._efficiency,
                image_size_gb=self._image_gb,
            )
            ranked = RankedOffer(offer=offer, cost=cost, viable=True)
            self._attach_reasons(ranked, model, variant, requirements, backend, download_gb)
            viable.append(ranked)

        if not viable:
            return []

        min_gpus = min(o.offer.gpu_count for o in viable)
        for ranked in viable:
            self._attach_penalties(ranked, requirements, min_gpus)

        viable.sort(key=lambda r: (r.effective_cost_usd, -r.offer.reliability))
        return viable

    # ---------------------------------------------------------------- helpers

    def _rejection_reason(
        self,
        offer: GPUOffer,
        model: ModelInfo,
        variant: ModelVariant,
        requirements: HardwareRequirements,
        backend: Backend,
    ) -> str | None:
        breakdown = estimate_vram(
            model,
            variant,
            requirements.context_length,
            requirements.concurrency,
            backend,
            gpu_count=offer.gpu_count,
            per_gpu_vram_gb=offer.per_gpu_vram_gb,
        )
        if breakdown.total_gib > offer.total_vram_gb * 1.001:
            return f"needs {breakdown.total_gib:.0f} GiB usable VRAM, offer provides {offer.total_vram_gb:.0f} GB"
        if offer.disk_gb < requirements.disk_gb:
            return f"needs {requirements.disk_gb:.0f} GB disk, offer provides {offer.disk_gb:.0f} GB"
        if offer.reliability < 0.90:
            return f"reliability {offer.reliability:.2%} too low"
        return None

    def _attach_reasons(
        self,
        ranked: RankedOffer,
        model: ModelInfo,
        variant: ModelVariant,
        requirements: HardwareRequirements,
        backend: Backend,
        download_gb: float,
    ) -> None:
        offer = ranked.offer
        breakdown = estimate_vram(
            model,
            variant,
            requirements.context_length,
            requirements.concurrency,
            backend,
            gpu_count=offer.gpu_count,
            per_gpu_vram_gb=offer.per_gpu_vram_gb,
        )
        headroom = offer.total_vram_gb - breakdown.total_gib
        pros: list[str] = []
        cons: list[str] = []

        if headroom >= 0.1 * offer.total_vram_gb:
            pros.append(f"sufficient usable VRAM (+{headroom:.0f} GB headroom)")
        elif headroom > 0:
            cons.append(f"only {headroom:.0f} GB VRAM headroom")
        pros.append(f"reliability {offer.reliability:.2%}")
        dl_minutes = ranked.cost.download_seconds / 60.0
        if offer.inet_down_mbps >= 1000:
            pros.append(f"fast model download {offer.inet_down_mbps:.0f} Mb/s ({dl_minutes:.0f} min)")
        elif dl_minutes > 45:
            cons.append(f"slow model download {offer.inet_down_mbps:.0f} Mb/s (~{dl_minutes:.0f} min)")
        if offer.nvlink_gbs and offer.gpu_count > 1:
            pros.append(f"NVLink interconnect ({offer.nvlink_gbs:.0f} GB/s)")
        if offer.gpu_mem_bw_gbs:
            pros.append(f"GPU memory bandwidth {offer.gpu_mem_bw_gbs:.0f} GB/s/GPU")
        if offer.gpu_count == 1:
            pros.append("single GPU — no multi-GPU topology risk")
        if offer.pcie_bw_gbs and offer.pcie_bw_gbs < 8.0 and offer.gpu_count > 1:
            cons.append(f"low PCIe bandwidth {offer.pcie_bw_gbs:.1f} GB/s")
        disk_headroom = offer.disk_gb - requirements.disk_gb
        if requirements.disk_gb > 0 and disk_headroom < 0.1 * requirements.disk_gb:
            cons.append(f"disk headroom only {disk_headroom:.0f} GB")

        ranked.pros = pros
        ranked.cons = cons

    def _attach_penalties(self, ranked: RankedOffer, requirements: HardwareRequirements, min_gpus: int) -> None:
        offer = ranked.offer
        session = ranked.cost.total_session_usd
        penalty = 0.0
        if offer.gpu_count > 1 and not (offer.nvlink_gbs and offer.nvlink_gbs > 0):
            penalty += PENALTY_NO_NVLINK * session
        if offer.gpu_count > min_gpus:
            penalty += PENALTY_EXTRA_GPUS * session
            if not any("NVLink" in p for p in ranked.pros):
                ranked.cons.append(f"uses {offer.gpu_count} GPUs where {min_gpus} suffice (no NVLink penalty)")
        if offer.pcie_bw_gbs and offer.pcie_bw_gbs < 8.0 and offer.gpu_count > 1:
            penalty += PENALTY_SLOW_PCIE * session
        if offer.disk_gb - requirements.disk_gb < 0.1 * requirements.disk_gb:
            penalty += PENALTY_TIGHT_DISK * session
        ranked.penalty_usd = round(penalty, 4)
