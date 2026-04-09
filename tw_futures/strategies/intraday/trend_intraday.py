"""EMA-Crossover Trend-Following Intraday Strategy for TAIFEX futures.

Time frame
----------
15-minute bars.  Requires at least 200 bars of warm-up so the slow EMA(200)
is fully initialised.

Entry logic — EMA crossover + trend confirmation
-------------------------------------------------
* Fast EMA  : EMA(10)  ≈ 2.5 hours of 15min bars
* Slow EMA  : EMA(40)  ≈ 10 hours  ≈ 2 trading days
* Trend EMA : EMA(200) ≈ 50 hours  ≈ 10 trading days

Golden cross (long entry)
  EMA10 crosses above EMA40 **and** close > EMA200.
  Confirmed by 2 consecutive bars with EMA10 > EMA40 after the cross.

Death cross (short entry)
  EMA10 crosses below EMA40 **and** close < EMA200.
  Confirmed by 2 consecutive bars with EMA10 < EMA40 after the cross.

Crossover confirmation rule
  At bar *i* a confirmed signal requires:
    bars[-3]: EMA10 on the same side as the new position (original cross)
    bars[-2]: EMA10 maintains direction (1st confirmation)
    bars[-1]: EMA10 maintains direction (2nd confirmation — current bar)
  This catches the cross at i-2 and verifies it held for 2 subsequent bars,
  filtering most whipsaw false crossovers.

Exit logic — first triggered wins
----------------------------------
a. Reverse cross: EMA10 crosses back through EMA40 in the opposite direction.
b. ATR trailing stop: 1.5 × ATR(14, 15min).
   Anchored to the running highest high (long) / lowest low (short) since entry.
c. Fixed hard stop: entry_price ± 150 points.
   For TX: 150 pt × 200 TWD/pt = TWD 30,000 maximum loss per contract.
d. Intraday EOD: TimeGuards forces flat at 13:30 — no overnight positions.

Position sizing
---------------
contracts = max(1, floor((equity × risk_per_trade) / (stop_distance × tick_value)))

stop_distance = min(1.5×ATR, hard_stop_points)  — conservative sizing

Minimum data
------------
200 bars of 15min OHLCV (≈ 10 trading days at ~20 bars per day).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from tw_futures.strategies.common.time_guards import TimeGuards

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_EMA_FAST: Final[int] = 10
_EMA_SLOW: Final[int] = 40
_EMA_TREND: Final[int] = 200
_ATR_PERIOD: Final[int] = 14
_ATR_STOP_MULT: Final[float] = 1.5
_HARD_STOP_PTS: Final[float] = 150.0  # index points
_RISK_PER_TRADE: Final[float] = 0.02  # 2 % of equity
_CONFIRM_BARS: Final[int] = 2  # bars EMA10 must hold direction after cross
_MIN_DATA_ROWS: Final[int] = _EMA_TREND

_TICK_VALUE: Final[dict[str, float]] = {
    "TX": 200.0,
    "MTX": 50.0,
}


# ── Signal ─────────────────────────────────────────────────────────────────────


@dataclass
class Signal:
    """Trade instruction returned by TrendIntradayStrategy.generate_signal.

    Attributes
    ----------
    action :
        ``"buy"`` | ``"sell"`` | ``"close"`` | ``"hold"``
    contracts :
        Number of contracts (0 for hold / close when not sizing).
    reason :
        Human-readable explanation.
    stop_loss :
        Initial ATR trailing stop level (None for hold).
    hard_stop :
        Fixed hard stop level (None for hold).
    """

    action: str
    contracts: int = 0
    reason: str = ""
    stop_loss: float | None = None
    hard_stop: float | None = None

    def __str__(self) -> str:
        sl = f"  stop={self.stop_loss:.0f}" if self.stop_loss is not None else ""
        hs = f"  hard={self.hard_stop:.0f}" if self.hard_stop is not None else ""
        return (
            f"Signal(action={self.action!r}  contracts={self.contracts}"
            f"  reason={self.reason!r}{sl}{hs})"
        )


# ── Strategy ───────────────────────────────────────────────────────────────────


class TrendIntradayStrategy:
    """EMA-crossover trend-following intraday strategy for TAIFEX TX / MTX.

    Parameters
    ----------
    product :
        ``"TX"`` (大台, 200 TWD/pt) or ``"MTX"`` (小台, 50 TWD/pt).
    ema_fast :
        Fast EMA period in 15min bars (default 10).
    ema_slow :
        Slow EMA period in 15min bars (default 40).
    ema_trend :
        Trend filter EMA period in 15min bars (default 200).
    atr_period :
        ATR look-back in 15min bars (default 14).
    atr_stop_mult :
        ATR multiplier for trailing stop distance (default 1.5).
    hard_stop_points :
        Fixed hard stop distance in index points (default 150).
    risk_per_trade :
        Fraction of equity risked per trade for sizing (default 0.02 = 2%).
    confirm_bars :
        Number of consecutive bars EMA10 must hold above/below EMA40 after the
        initial cross before entry is allowed (default 2).
    """

    def __init__(
        self,
        product: str = "TX",
        ema_fast: int = _EMA_FAST,
        ema_slow: int = _EMA_SLOW,
        ema_trend: int = _EMA_TREND,
        atr_period: int = _ATR_PERIOD,
        atr_stop_mult: float = _ATR_STOP_MULT,
        hard_stop_points: float = _HARD_STOP_PTS,
        risk_per_trade: float = _RISK_PER_TRADE,
        confirm_bars: int = _CONFIRM_BARS,
    ) -> None:
        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be 'TX' or 'MTX', got {product!r}")

        self.product = product
        self.tick_value = _TICK_VALUE[product]
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.hard_stop_points = hard_stop_points
        self.risk_per_trade = risk_per_trade
        self.confirm_bars = confirm_bars
        self._time_guards = TimeGuards()

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        data_15min: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        equity: float,
        current_time: pd.Timestamp,
        entry_time: pd.Timestamp | None = None,
    ) -> Signal:
        """Generate a trade signal from the latest 15min bar in *data_15min*.

        Parameters
        ----------
        data_15min :
            15-minute OHLCV DataFrame with columns ``open, high, low, close,
            volume`` and a tz-aware ``pd.DatetimeIndex`` (Asia/Taipei).
            Must contain at least 200 rows.
        current_position :
            Net contracts: +N = long N, −N = short N, 0 = flat.
        entry_price :
            Average entry price of current position.  ``None`` if flat.
        equity :
            Current account equity in TWD (used for position sizing).
        current_time :
            Timestamp of the current (latest) bar — used for time guards.
        entry_time :
            Timestamp when the current position was opened.  Used to compute
            "highest high / lowest low since entry" for the ATR trailing stop.
            ``None`` falls back to using entry_price as the anchor.

        Returns
        -------
        Signal
        """
        # ── Guard: minimum data ────────────────────────────────────────
        if len(data_15min) < _MIN_DATA_ROWS:
            return Signal(
                action="hold",
                reason=(f"Insufficient data: {len(data_15min)} bars < {_MIN_DATA_ROWS} required"),
            )

        required = {"open", "high", "low", "close"}
        missing = required - set(data_15min.columns)
        if missing:
            return Signal(action="hold", reason=f"Missing columns: {missing}")

        # ── Compute indicators ─────────────────────────────────────────
        indic = self._compute_indicators(data_15min)
        latest = indic.iloc[-1]

        close = float(latest["close"])
        ema_f = float(latest["ema_fast"])
        ema_s = float(latest["ema_slow"])
        ema_t = float(latest["ema_trend"])
        atr = float(latest["atr"])

        if np.isnan(atr) or atr <= 0:
            return Signal(action="hold", reason="ATR is zero or NaN — skipping")

        # ── Time guards — force-close check ───────────────────────────
        if current_position != 0:
            force, reason = self._time_guards.should_force_close(current_time)
            if force:
                logger.debug("TrendIntraday force-close at %s: %s", current_time, reason)
                return Signal(action="close", reason=reason)

        # ── In-position: check exit conditions ────────────────────────
        if current_position != 0:
            return self._check_exit(
                indic=indic,
                current_position=current_position,
                entry_price=entry_price,
                close=close,
                atr=atr,
                current_time=current_time,
                entry_time=entry_time,
            )

        # ── Flat: check time-based entry block ─────────────────────────
        blocked, block_reason = self._time_guards.should_block_entry(current_time)
        if blocked:
            return Signal(action="hold", reason=block_reason)

        # ── Flat: check for entry signals ─────────────────────────────
        long_sig, short_sig = self._check_crossover(indic, close, ema_t)
        if not long_sig and not short_sig:
            return Signal(action="hold", reason="no EMA crossover signal")

        direction = 1 if long_sig else -1
        return self._build_entry_signal(close, atr, direction, equity)

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return input data with EMA + ATR columns appended."""
        df = data[["open", "high", "low", "close"]].copy()
        if "volume" in data.columns:
            df["volume"] = data["volume"]

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        df["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()
        df["ema_trend"] = close.ewm(span=self.ema_trend, adjust=False).mean()

        # ATR(14) on 15min bars
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        return df

    # ------------------------------------------------------------------
    # Crossover detection
    # ------------------------------------------------------------------

    def _check_crossover(
        self,
        indic: pd.DataFrame,
        close: float,
        ema_trend: float,
    ) -> tuple[bool, bool]:
        """Return (long_signal, short_signal) with 2-bar confirmation.

        Confirmation rule: at current bar [-1], we require:
          bars[-3]: EMA_fast on the *opposite* side of EMA_slow (pre-cross)
          bars[-2]: EMA_fast crossed to the *new* side (cross bar = 1st bar)
          bars[-1]: EMA_fast maintains new direction (2nd bar = confirmation)

        This checks for a cross that happened 2 bars ago and has held since.
        """
        n_confirm = self.confirm_bars
        needed = n_confirm + 2  # need cross bar + confirm bars + 1 pre-cross bar

        if len(indic) < needed:
            return False, False

        ema_f = indic["ema_fast"].values
        ema_s = indic["ema_slow"].values

        # Relative position: +1 if fast > slow, -1 if fast < slow
        def _side(i: int) -> int:
            return 1 if ema_f[i] > ema_s[i] else -1

        pre_cross = _side(-needed)  # must be OPPOSITE of new side
        post_bars = [_side(-(n_confirm - k)) for k in range(n_confirm)]

        # Long: pre_cross == -1, all post_bars == +1, close > ema_trend
        long_signal = pre_cross == -1 and all(s == 1 for s in post_bars) and close > ema_trend

        # Short: pre_cross == +1, all post_bars == -1, close < ema_trend
        short_signal = pre_cross == 1 and all(s == -1 for s in post_bars) and close < ema_trend

        return long_signal, short_signal

    # ------------------------------------------------------------------
    # Exit check (while in position)
    # ------------------------------------------------------------------

    def _check_exit(
        self,
        indic: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        close: float,
        atr: float,
        current_time: pd.Timestamp,
        entry_time: pd.Timestamp | None = None,
    ) -> Signal:
        """Check all exit conditions for an open position."""
        direction = 1 if current_position > 0 else -1
        ep = entry_price if entry_price is not None else close

        ema_f = indic["ema_fast"].values
        ema_s = indic["ema_slow"].values

        # ── a. Reverse EMA crossover ───────────────────────────────────
        if len(indic) >= 2:
            prev_diff = ema_f[-2] - ema_s[-2]
            curr_diff = ema_f[-1] - ema_s[-1]
            if direction == 1 and prev_diff >= 0 and curr_diff < 0:
                return Signal(
                    action="close",
                    reason="EMA death-cross — close long",
                )
            if direction == -1 and prev_diff <= 0 and curr_diff > 0:
                return Signal(
                    action="close",
                    reason="EMA golden-cross — close short",
                )

        # ── b. ATR trailing stop ───────────────────────────────────────
        # Anchor to highest high (long) / lowest low (short) since entry.
        # When entry_time is known, slice indic to bars on/after entry so
        # we don't use the all-time extreme as anchor (which would produce
        # a stop that's unreachable from current price).
        atr_stop_dist = self.atr_stop_mult * atr
        since_entry = _slice_since_entry(indic, entry_time, ep, direction)

        if direction == 1:
            anchor = max(float(since_entry["high"].max()), ep)
            atr_stop = anchor - atr_stop_dist
            if close <= atr_stop:
                return Signal(action="close", reason=f"ATR trailing stop hit ({atr_stop:.0f})")
        else:
            anchor = min(float(since_entry["low"].min()), ep)
            atr_stop = anchor + atr_stop_dist
            if close >= atr_stop:
                return Signal(action="close", reason=f"ATR trailing stop hit ({atr_stop:.0f})")

        # ── c. Fixed hard stop ─────────────────────────────────────────
        hard_dist = self.hard_stop_points
        if direction == 1 and close <= ep - hard_dist:
            return Signal(
                action="close",
                reason=f"Hard stop hit (long entry={ep:.0f}  stop={ep - hard_dist:.0f})",
            )
        if direction == -1 and close >= ep + hard_dist:
            return Signal(
                action="close",
                reason=f"Hard stop hit (short entry={ep:.0f}  stop={ep + hard_dist:.0f})",
            )

        # ── Hold — compute updated trailing stop for the backtester ───
        atr_stop_level = (
            atr_stop
            if "atr_stop" in dir()
            else (ep - atr_stop_dist if direction == 1 else ep + atr_stop_dist)
        )
        hard_stop_level = ep - direction * hard_dist

        return Signal(
            action="hold",
            reason="in position — no exit triggered",
            stop_loss=atr_stop_level,
            hard_stop=hard_stop_level,
        )

    # ------------------------------------------------------------------
    # Entry signal construction
    # ------------------------------------------------------------------

    def _build_entry_signal(
        self,
        close: float,
        atr: float,
        direction: int,
        equity: float,
    ) -> Signal:
        """Build a buy/sell signal with position sizing."""
        stop_dist = min(self.atr_stop_mult * atr, self.hard_stop_points)
        stop_dist = max(stop_dist, 1.0)  # guard against zero

        n_contracts = max(
            1, math.floor((equity * self.risk_per_trade) / (stop_dist * self.tick_value))
        )

        atr_stop_level = close - direction * self.atr_stop_mult * atr
        hard_stop_level = close - direction * self.hard_stop_points

        action = "buy" if direction == 1 else "sell"
        dir_str = "long" if direction == 1 else "short"
        logger.debug(
            "TrendIntraday %s entry: close=%.0f  atr=%.0f  stop=%.0f  n=%d",
            dir_str,
            close,
            atr,
            atr_stop_level,
            n_contracts,
        )
        return Signal(
            action=action,
            contracts=n_contracts,
            reason=(
                f"EMA{self.ema_fast}/EMA{self.ema_slow} confirmed {dir_str} crossover  "
                f"close={close:.0f}  atr={atr:.0f}"
            ),
            stop_loss=atr_stop_level,
            hard_stop=hard_stop_level,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _slice_since_entry(
    indic: pd.DataFrame,
    entry_time: pd.Timestamp | None,
    entry_price: float,
    direction: int,
) -> pd.DataFrame:
    """Return the sub-DataFrame from the entry bar onwards.

    Falls back to the last bar only when entry_time is unknown, anchoring the
    trailing stop to the current bar's high/low — the safest possible stop.
    """
    if entry_time is not None:
        mask = indic.index >= entry_time
        sliced = indic[mask]
        if not sliced.empty:
            return sliced

    # Fallback: just use current (last) bar — stop anchors to current price
    return indic.iloc[[-1]]
