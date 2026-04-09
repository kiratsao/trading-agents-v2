"""Donchian Channel Breakout Swing Strategy V2 — with three entry filters.

Builds on :class:`~tw_futures.strategies.swing.breakout_swing.BreakoutSwingStrategy`
(V1) by adding three entry-quality filters designed to reduce false breakouts.
Exit logic (ATR trailing stop, max holding days, reverse signal) is unchanged.

Additional filters (applied only on new entries, not during position management)
----------------------------------------------------------------------------------
1. **Volume Confirmation** — breakout bar volume must exceed 1.2 × 20-day average
   volume.  A low-volume breakout is treated as a false breakout.

2. **ATR Volatility Regime** — if today's ATR(14) falls in the *top 20 %* of
   the rolling 60-day ATR distribution, the environment is too chaotic and stop
   runs are frequent; skip entry.  Low-volatility breakouts (bottom 20 %) are
   *not* blocked — they often represent genuine range expansions.

3. **Breakout Strength** — close must exceed the Donchian band by more than
   ``0.3 × ATR(14)``.  A tick-above-the-band close is noise, not momentum.

Minimum data requirement: 75 rows (ATR needs 14 bars; ATR-rank needs 60 ATR
values → 73 bars total; rounded up to 75 for stability).

Parameters (new, in addition to all V1 parameters)
---------------------------------------------------
vol_mult :
    Minimum ratio of today's volume to the 20-day average (default 1.2).
atr_rank_period :
    Look-back for the rolling ATR percentile rank (default 60).
atr_high_pct :
    ATR percentile threshold above which entries are blocked (default 0.80,
    i.e., top 20 %).
breakout_strength_mult :
    Minimum excess beyond the Donchian band, expressed as a multiple of
    ATR(14) (default 0.3).

Usage
-----
>>> from tw_futures.strategies.swing.breakout_swing_v2 import BreakoutSwingV2Strategy
>>> strategy = BreakoutSwingV2Strategy(product="TX")
>>> sig = strategy.generate_signal(data=df, current_position=0,
...                                entry_price=None, entry_date=None,
...                                equity=2_000_000)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tw_futures.strategies.swing.breakout_swing import (
    _ATR_PERIOD,
    _ATR_STOP_MULT,
    _DONCHIAN_PERIOD,
    _MAX_HOLD_DAYS,
    _NEUTRAL_ZONE_PCT,
    _RISK_PER_TRADE,
    _TREND_PERIOD,
    BreakoutSwingStrategy,
    Signal,
)

logger = logging.getLogger(__name__)

# V2 requires more warm-up: ATR(14) + 60-bar ATR rank window
_MIN_DATA_ROWS_V2: int = 75

# New filter defaults
_VOL_MULT: float = 1.2
_ATR_RANK_PERIOD: int = 60
_ATR_HIGH_PCT: float = 0.80  # top 20 % → skip
_BREAKOUT_STRENGTH_MULT: float = 0.30


class BreakoutSwingV2Strategy(BreakoutSwingStrategy):
    """V1 Donchian Breakout with three additional entry-quality filters.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"`` (same as V1).
    vol_mult :
        Volume threshold multiplier (default 1.2 = 120 % of 20-day avg vol).
    atr_rank_period :
        Rolling window for ATR percentile rank (default 60 bars).
    atr_high_pct :
        Upper ATR percentile cutoff; entries blocked above this level
        (default 0.80 = top 20 %).
    breakout_strength_mult :
        Minimum breakout excess beyond the Donchian band as a fraction of
        ATR(14) (default 0.30).
    All other keyword arguments are forwarded to :class:`BreakoutSwingStrategy`.
    """

    def __init__(
        self,
        product: str = "TX",
        donchian_period: int = _DONCHIAN_PERIOD,
        trend_period: int = _TREND_PERIOD,
        atr_period: int = _ATR_PERIOD,
        atr_stop_mult: float = _ATR_STOP_MULT,
        risk_per_trade: float = _RISK_PER_TRADE,
        max_hold_days: int = _MAX_HOLD_DAYS,
        neutral_zone_pct: float = _NEUTRAL_ZONE_PCT,
        vol_mult: float = _VOL_MULT,
        atr_rank_period: int = _ATR_RANK_PERIOD,
        atr_high_pct: float = _ATR_HIGH_PCT,
        breakout_strength_mult: float = _BREAKOUT_STRENGTH_MULT,
    ) -> None:
        super().__init__(
            product=product,
            donchian_period=donchian_period,
            trend_period=trend_period,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            risk_per_trade=risk_per_trade,
            max_hold_days=max_hold_days,
            neutral_zone_pct=neutral_zone_pct,
        )
        self.vol_mult = vol_mult
        self.atr_rank_period = atr_rank_period
        self.atr_high_pct = atr_high_pct
        self.breakout_strength_mult = breakout_strength_mult

    # ------------------------------------------------------------------
    # Indicator computation — extends V1 with vol MA and ATR rank
    # ------------------------------------------------------------------

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return V1 indicators plus ``vol_ma20`` and ``atr_rank``."""
        df = super()._compute_indicators(data)

        # Volume 20-day moving average (shifted by 1 — no look-ahead on volume)
        if "volume" in data.columns:
            df["vol_ma20"] = data["volume"].astype(float).rolling(20).mean().shift(1)
        else:
            df["vol_ma20"] = float("nan")

        # ATR percentile rank within a rolling window
        # rank(pct=True) → 1.0 = highest ATR in window (extreme volatility)
        df["atr_rank"] = df["atr"].rolling(self.atr_rank_period).rank(pct=True)

        return df

    # ------------------------------------------------------------------
    # Primary interface — same flow as V1 but extra filters on entry
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        entry_date: pd.Timestamp | None,
        equity: float,
    ) -> Signal:
        """Generate a signal with V2 entry filters applied when flat.

        When in a position the exit logic is identical to V1.  The three
        new filters are evaluated only on potential new entries.
        """
        # ── Guard: minimum data (stricter than V1) ─────────────────────
        if len(data) < _MIN_DATA_ROWS_V2:
            return Signal(
                action="hold",
                reason=(f"insufficient data: {len(data)} rows < {_MIN_DATA_ROWS_V2} required (V2)"),
            )

        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            return Signal(action="hold", reason=f"missing columns: {missing}")

        # ── Indicators (V1 + vol_ma20 + atr_rank) ─────────────────────
        indic = self._compute_indicators(data)
        latest = indic.iloc[-1]

        current_close = float(latest["close"])
        atr = float(latest["atr"])
        upper = float(latest["upper"])
        lower = float(latest["lower"])
        sma50 = float(latest["sma50"])
        vol_ma20 = float(latest["vol_ma20"])
        atr_rank = float(latest["atr_rank"])
        current_vol = float(data.iloc[-1]["volume"]) if "volume" in data.columns else float("nan")
        current_date = data.index[-1]

        # ── Guard: ATR ─────────────────────────────────────────────────
        if atr == 0 or np.isnan(atr):
            return Signal(
                action="hold",
                reason="ATR is zero — skipping to avoid division error",
            )

        # ── Raw Donchian signals ───────────────────────────────────────
        long_signal = current_close > upper and not np.isnan(upper)
        short_signal = current_close < lower and not np.isnan(lower)

        # ── Trend filter (SMA50 + neutral zone) ───────────────────────
        long_signal, short_signal = self._apply_trend_filter(
            current_close, sma50, long_signal, short_signal
        )

        # ── In-position → V1 exit logic (unchanged) ───────────────────
        if current_position != 0:
            return self._generate_exit_signal(
                data=data,
                indic=indic,
                current_position=current_position,
                entry_price=entry_price,
                entry_date=entry_date,
                current_close=current_close,
                current_date=current_date,
                atr=atr,
                long_signal=long_signal,
                short_signal=short_signal,
                equity=equity,
            )

        # ── Flat → apply V2 entry filters before attempting entry ──────
        if long_signal or short_signal:
            blocked = self._apply_entry_filters(
                current_close=current_close,
                current_vol=current_vol,
                vol_ma20=vol_ma20,
                atr_rank=atr_rank,
                atr=atr,
                upper=upper,
                lower=lower,
                long_signal=long_signal,
                short_signal=short_signal,
            )
            if blocked is not None:
                return blocked

        # ── Entry signal (V1 logic) ────────────────────────────────────
        return self._generate_entry_signal(
            current_close=current_close,
            atr=atr,
            long_signal=long_signal,
            short_signal=short_signal,
            equity=equity,
            upper=upper,
            lower=lower,
            sma50=sma50,
        )

    # ------------------------------------------------------------------
    # V2 entry filter chain
    # ------------------------------------------------------------------

    def _apply_entry_filters(
        self,
        current_close: float,
        current_vol: float,
        vol_ma20: float,
        atr_rank: float,
        atr: float,
        upper: float,
        lower: float,
        long_signal: bool,
        short_signal: bool,
    ) -> Signal | None:
        """Evaluate the three V2 entry filters in order.

        Returns a ``Signal(action="hold")`` if *any* filter blocks entry, or
        ``None`` if all filters pass (caller proceeds with V1 entry logic).
        """
        direction_str = "long" if long_signal else "short"

        # ── Filter 1: Volume Confirmation ─────────────────────────────
        if not np.isnan(vol_ma20) and vol_ma20 > 0:
            if np.isnan(current_vol) or current_vol < self.vol_mult * vol_ma20:
                ratio = (
                    current_vol / vol_ma20 if not np.isnan(current_vol) and vol_ma20 > 0 else 0.0
                )
                logger.debug(
                    "V2 filter 1 blocked %s: vol=%.0f < %.1f×MA20(%.0f)  ratio=%.2f",
                    direction_str,
                    current_vol,
                    self.vol_mult,
                    vol_ma20,
                    ratio,
                )
                return Signal(
                    action="hold",
                    reason=(
                        f"V2 filter: volume {current_vol:.0f} < "
                        f"{self.vol_mult}×MA20 ({vol_ma20:.0f})  "
                        f"ratio={ratio:.2f}"
                    ),
                )

        # ── Filter 2: ATR Volatility Regime ───────────────────────────
        if not np.isnan(atr_rank) and atr_rank > self.atr_high_pct:
            logger.debug(
                "V2 filter 2 blocked %s: atr_rank=%.2f > %.2f (extreme vol)",
                direction_str,
                atr_rank,
                self.atr_high_pct,
            )
            return Signal(
                action="hold",
                reason=(
                    f"V2 filter: ATR rank {atr_rank:.2f} > {self.atr_high_pct} "
                    f"(extreme volatility regime)"
                ),
            )

        # ── Filter 3: Breakout Strength ────────────────────────────────
        min_excess = self.breakout_strength_mult * atr
        if long_signal and not np.isnan(upper):
            excess = current_close - upper
            if excess < min_excess:
                logger.debug(
                    "V2 filter 3 blocked long: excess=%.0f < min=%.0f (0.3×ATR)",
                    excess,
                    min_excess,
                )
                return Signal(
                    action="hold",
                    reason=(
                        f"V2 filter: breakout excess {excess:.0f} < "
                        f"0.3×ATR ({min_excess:.0f})  "
                        f"close={current_close:.0f} upper={upper:.0f}"
                    ),
                )

        if short_signal and not np.isnan(lower):
            excess = lower - current_close
            if excess < min_excess:
                logger.debug(
                    "V2 filter 3 blocked short: excess=%.0f < min=%.0f (0.3×ATR)",
                    excess,
                    min_excess,
                )
                return Signal(
                    action="hold",
                    reason=(
                        f"V2 filter: breakout excess {excess:.0f} < "
                        f"0.3×ATR ({min_excess:.0f})  "
                        f"close={current_close:.0f} lower={lower:.0f}"
                    ),
                )

        # All filters passed
        return None
