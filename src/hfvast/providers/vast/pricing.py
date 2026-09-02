"""Vast.ai cost math (Vast billing semantics).

Billing model (docs.vast.ai, verified 2026-09-02):
  * compute: $/h while running (`dph_total` already includes storage priced for
    the `allocated_storage` size sent with the offer search);
  * storage: $/GB/month billed every second the instance exists;
  * bandwidth: $/GB charged per byte both directions (on top of dph_total).

All outputs are estimates; Vast billing is authoritative.
"""

from __future__ import annotations

from hfvast.models.offers import GPUOffer
from hfvast.models.pricing import CostBreakdown

STORAGE_HOURS_PER_MONTH = 730.0
GB = 1e9

#: Efficiency factor on advertised download bandwidth (spec §20: 60–80%).
DEFAULT_NETWORK_EFFICIENCY = 0.7

#: Model-load throughput multiplier on aggregate GPU memory bandwidth.
LOAD_BW_FRACTION = 0.5
MIN_LOAD_SECONDS = 120.0
MAX_LOAD_SECONDS = 2400.0

BASE_IMAGE_PULL_SECONDS = 60.0


def download_seconds(size_gb: float, inet_down_mbps: float, efficiency: float) -> float:
    if inet_down_mbps <= 0:
        return 0.0
    megabits = size_gb * 8 * 1000
    return megabits / (inet_down_mbps * max(0.1, efficiency))


def model_load_seconds(offer: GPUOffer, weights_gb: float) -> float:
    if not offer.gpu_mem_bw_gbs or offer.gpu_mem_bw_gbs <= 0:
        return 600.0
    aggregate_gbs = offer.gpu_mem_bw_gbs * offer.gpu_count * LOAD_BW_FRACTION
    seconds = weights_gb / aggregate_gbs
    return max(MIN_LOAD_SECONDS, min(MAX_LOAD_SECONDS, seconds))


def cost_breakdown(
    offer: GPUOffer,
    download_gb: float,
    session_hours: float,
    efficiency: float = DEFAULT_NETWORK_EFFICIENCY,
    image_size_gb: float = 8.0,
) -> CostBreakdown:
    hourly_gpu = offer.hourly_gpu_usd
    # storage component embedded in dph_total for the priced disk size; derive the
    # split defensively (never negative).
    hourly_storage = max(0.0, offer.hourly_total_usd - hourly_gpu)
    hourly_total = offer.hourly_total_usd

    dl_seconds = download_seconds(download_gb, offer.inet_down_mbps, efficiency)
    pull_seconds = BASE_IMAGE_PULL_SECONDS + download_seconds(image_size_gb, max(offer.inet_down_mbps, 1.0), efficiency)
    load_seconds = model_load_seconds(offer, download_gb)
    cold_hours = (dl_seconds + pull_seconds + load_seconds) / 3600.0

    bandwidth_usd = offer.inet_down_usd_per_gb * (download_gb + image_size_gb)
    cold_start_usd = cold_hours * hourly_total + bandwidth_usd
    runtime_usd = hourly_total * session_hours

    return CostBreakdown(
        hourly_gpu_usd=hourly_gpu,
        hourly_storage_usd=hourly_storage,
        hourly_total_usd=hourly_total,
        download_seconds=dl_seconds,
        image_pull_seconds=pull_seconds,
        model_load_seconds=load_seconds,
        cold_start_hours=cold_hours,
        cold_start_usd=cold_start_usd,
        bandwidth_usd=bandwidth_usd,
        runtime_usd=runtime_usd,
        total_session_usd=cold_start_usd + runtime_usd,
        session_hours=session_hours,
    )
