"""Pre-rent host reachability probing (spec §19–21: rank by *real* quality).

Vast's `geolocation` data is unreliable (live-verified 2026-09-03: hosts geo-
tagged "Washington, US" on China Unicom IP space, unreachable from many
networks), but the offer response exposes `public_ipaddr` BEFORE renting.

Probe: a TCP connect that gets REFUSED (or opens) means the host is routable
from this network; a TIMEOUT means the IP is blackholed — an instance there
could never serve us. Unreachable offers are re-ranked last with a visible
reason; they are still deployable if nothing else exists.
"""

from __future__ import annotations

import asyncio

from hfvast.models.quote import RankedOffer

PROBE_PORT = 80
PROBE_TIMEOUT_S = 4.0
PROBE_TOP_N = 10


async def probe_reachable(ip: str, port: int = PROBE_PORT, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """True if the host answers TCP at all (open or refused) — i.e. routable."""

    async def _one(p: int) -> bool:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(ip, p), timeout=timeout_s)
            writer.close()
            return True
        except ConnectionRefusedError:
            return True  # host answered: port closed but the host is routable
        except OSError:
            return False  # timeout / unreachable / filtered

    if await _one(port):
        return True
    return await _one(443)  # some hosts filter :80 but are otherwise routable


async def probe_and_rerank(ranked: list[RankedOffer], top_n: int = PROBE_TOP_N) -> None:
    """Probe the top N offers' host IPs concurrently; sink unreachable ones."""
    targets = [r for r in ranked[:top_n] if r.offer.public_ipaddr]
    if not targets:
        return
    results = await asyncio.gather(*(probe_reachable(r.offer.public_ipaddr or "") for r in targets))
    for ranked_offer, reachable in zip(targets, results, strict=True):
        ranked_offer.reachable = reachable
        if not reachable:
            ip = ranked_offer.offer.public_ipaddr
            ranked_offer.cons.append(f"host {ip} is unreachable from your network (pre-rent probe timed out)")
            ranked_offer.penalty_usd += ranked_offer.cost.total_session_usd * 0.5
    ranked.sort(key=lambda r: (r.reachable is False, r.effective_cost_usd, -r.offer.reliability))
