"""Independent day-session anchor + 3-source day-close resolver.

The TWSE spot index (^TWII) has NO night session, so it can never be a night
value. That makes it the tie-breaker that stops a mislabeled (night-as-day)
futures bar from being trusted — regardless of what any single parser/env
produced. We never trust a parse's session label; we anchor on spot.

``resolve_day_close`` combines up to three sources — ^TWII spot, TAIFEX 一般,
and the Shioaji day-session oracle — and returns the value that the evidence
supports as the true day close, or ``None`` (fail-loud) when nothing is
consistent with the night-proof spot. Spot closes are cached to disk so repeated
runs / scans don't re-hit yfinance.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "spot_cache.json"
_TIMEOUT = 8  # seconds — keep the daemon path from hanging on a slow yfinance
# |MXF day close − spot| plausible band. Day closes sat 26–330pt from spot across
# 2026; night values diverged 193–2,043. Beyond this a value is treated as "not a
# day value" (night/anomaly). Kept generous so a large legit basis never fails.
_BASIS_BAND = 500.0
# TAIFEX 一般 and the Shioaji day oracle measure the same MXF day close; within
# this they are a consensus.
_AGREE_TOL = 40.0


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.debug("spot_ref: cache write failed: %s", exc)


def _download(start: date, end: date, timeout: int) -> dict:
    """yfinance ^TWII closes for [start, end] inclusive → {iso date: close}."""
    import yfinance as yf

    sp = yf.download(
        "^TWII", start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        progress=False, auto_adjust=True, timeout=timeout,
    )
    if sp is None or sp.empty:
        return {}
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = [c[0] for c in sp.columns]
    idx = pd.to_datetime(sp.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return {ts.date().isoformat(): float(v) for ts, v in zip(idx, sp["Close"])}


def fetch_spot_range(start: date, end: date, *, timeout: int = _TIMEOUT) -> dict:
    """{date: ^TWII close} over [start, end]; cache-first, one network call."""
    cache = _load_cache()
    out: dict[date, float] = {}
    missing = False
    d = start
    while d <= end:
        iso = d.isoformat()
        if iso in cache:
            out[d] = cache[iso]
        elif d.weekday() < 5:
            missing = True
        d += timedelta(days=1)
    if missing:
        try:
            fresh = _download(start, end, timeout)
        except Exception as exc:
            logger.warning("spot_ref: ^TWII download failed (%s)", exc)
            fresh = {}
        if fresh:
            cache.update(fresh)
            _save_cache(cache)
            for iso, v in fresh.items():
                out[date.fromisoformat(iso)] = v
    return out


def fetch_spot_close(day: date, *, timeout: int = _TIMEOUT) -> float | None:
    """^TWII day close for *day*, or None. Cache-first; short timeout."""
    cache = _load_cache()
    iso = day.isoformat()
    if iso in cache:
        return cache[iso]
    try:
        fresh = _download(day, day + timedelta(days=3), timeout)
    except Exception as exc:
        logger.warning("spot_ref: ^TWII fetch failed for %s (%s)", day, exc)
        return None
    if fresh:
        cache.update(fresh)
        _save_cache(cache)
    return fresh.get(iso)


def resolve_day_close(
    day: date,
    *,
    taifex: float | None = None,
    shioaji: float | None = None,
    spot: float | None = None,
    basis_band: float = _BASIS_BAND,
) -> tuple[float | None, str]:
    """Return (day_close, detail) or (None, reason) for *day*.

    Spot (^TWII) is night-proof truth. Among the MXF candidates (TAIFEX 一般,
    Shioaji day oracle), the one within ``basis_band`` of spot AND closest to it
    is the day close. If none is within band (all look like night) → fail-loud.
    When spot is unavailable, degrade to TAIFEX/Shioaji agreement rather than a
    blanket reject (a lone unverified source is accepted but flagged).
    """
    if spot is None:
        spot = fetch_spot_close(day)

    mxf: dict[str, float] = {}
    if taifex is not None:
        mxf["taifex"] = float(taifex)
    if shioaji is not None:
        mxf["shioaji"] = float(shioaji)
    if not mxf:
        return None, "no MXF candidate"

    if spot is None:
        if len(mxf) == 2:
            a, b = mxf["taifex"], mxf["shioaji"]
            if abs(a - b) <= _AGREE_TOL:
                return (a + b) / 2, f"no-spot; taifex/shioaji agree ({a:.0f}/{b:.0f})"
            return None, f"no-spot; taifex {a:.0f} vs shioaji {b:.0f} disagree — fail-loud"
        k, v = next(iter(mxf.items()))
        return v, f"no-spot; single source {k}={v:.0f} (unverified)"

    within = {k: v for k, v in mxf.items() if abs(v - spot) <= basis_band}
    if not within:
        cand = ", ".join(f"{k}={v:.0f}(Δ{v - spot:+.0f})" for k, v in mxf.items())
        return None, f"all MXF far from spot {spot:.0f} [{cand}] — night/anomaly, fail-loud"
    k = min(within, key=lambda kk: abs(within[kk] - spot))
    return within[k], f"spot={spot:.0f}; {k}={within[k]:.0f} (|Δspot|={abs(within[k] - spot):.0f})"
