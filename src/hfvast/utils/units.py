"""Unit formatting and parsing helpers."""

from __future__ import annotations

GIB = 1024**3
MIB = 1024**2

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def human_bytes(n: float | int) -> str:
    """Format a byte count as a human string (GB for storage-ish sizes)."""
    value = float(n)
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}" if value / scale < 100 else f"{value / scale:.0f} {unit}"
    return f"{value:.0f} B"


def human_gib(gib: float) -> str:
    return f"{gib:.1f} GiB" if gib < 100 else f"{gib:.0f} GiB"


def money(amount: float) -> str:
    return f"${amount:,.2f}"


def money_rate(amount_per_hour: float) -> str:
    return f"${amount_per_hour:.2f}/h"


def parse_duration(text: str) -> float:
    """Parse ``30m``/``2h``/``1d``/``45s`` (or a bare number of minutes) into seconds."""
    raw = text.strip().lower()
    if not raw:
        raise ValueError("empty duration")
    if raw[-1].isdigit():
        return float(raw) * 60.0
    unit = raw[-1]
    if unit not in _DURATION_UNITS:
        raise ValueError(f"unknown duration unit in {text!r} (use s/m/h/d)")
    return float(raw[:-1]) * _DURATION_UNITS[unit]


def human_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s" if sec else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def minutes(seconds: float) -> str:
    mins = seconds / 60.0
    return f"~{mins:.0f} min" if mins >= 2 else f"~{max(1, int(mins * 60))}s"


def percent(fraction: float) -> str:
    return f"{fraction * 100:.2f}%"


def ceil_to(value: float, step: float) -> float:
    """Round ``value`` up to the next multiple of ``step``."""
    import math

    return math.ceil(value / step) * step
