"""EMA Crossover Trend Following Strategy.

Philosophy
----------
「持倉數週到數月，讓趨勢跑完」— Hold positions for weeks to months; don't cut
trends short.  Only two reasons to exit: trend reversal or extreme adverse move.

Core Logic
----------
1. **Direction**: EMA(20) vs EMA(60)
   - EMA20 > EMA60 → bull trend → look for long entry
   - EMA20 < EMA60 → bear trend → look for short entry

2. **3-bar Confirmation**: after a crossover, wait 3 consecutive bars in the
   new direction before entering.  If the signal reverses during the 3-bar
   window → discard, wait for the next clean crossover.

3. **Exit — two conditions only**:
   a. Reverse EMA crossover → close + start 3-bar confirm for opposite side
   b. Extreme hard stop: entry ± 5 × ATR(20) → exit immediately

4. **Position sizing**: always 1 contract (TX for 2M, MTX for 1M).  No pyramid.

5. **TimeGuards**: settlement-day rollover only (close at 13:15 on 3rd Wednesday,
   re-enter on the next bar in the same direction).  No Friday close — long-term
   holds must span weekends.

6. **Design target**: ~10–20 round-trips per year; near-zero flat time.

Signals returned
----------------
  action = "buy"   → open long  (1 contract)
  action = "sell"  → open short (1 contract)
  action = "close" → flatten (reverse or hard stop triggered)
  action = "hold"  → no action
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from tw_futures.strategies.common.time_guards import TimeGuards

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_EMA_FAST: Final[int] = 20
_EMA_SLOW: Final[int] = 60
_ATR_PERIOD: Final[int] = 20
_HARD_STOP_ATR_MULT: Final[float] = 5.0  # 5× ATR hard stop
_CONFIRM_BARS: Final[int] = 3  # bars to wait after crossover

_TICK_VALUE: Final[dict[str, float]] = {"TX": 200.0, "MTX": 50.0}
_MIN_DATA_ROWS: Final[int] = _EMA_SLOW + _ATR_PERIOD  # need slow EMA + ATR


# ── Signal (re-exported for convenience) ─────────────────────────────────────


@dataclass
class Signal:
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


# ── Strategy ──────────────────────────────────────────────────────────────────


class TrendFollowLongStrategy:
    """EMA crossover trend-following strategy for TAIFEX TX / MTX.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"``.
    ema_fast, ema_slow :
        EMA look-backs (default 20, 60).
    atr_period :
        ATR look-back for hard-stop calculation (default 20).
    hard_stop_atr_mult :
        Hard-stop distance in ATR multiples (default 5.0).
    confirm_bars :
        Number of bars the new trend must persist before entry (default 3).
    """

    def __init__(
        self,
        product: str = "TX",
        ema_fast: int = _EMA_FAST,
        ema_slow: int = _EMA_SLOW,
        atr_period: int = _ATR_PERIOD,
        hard_stop_atr_mult: float = _HARD_STOP_ATR_MULT,
        confirm_bars: int = _CONFIRM_BARS,
    ) -> None:
        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be 'TX' or 'MTX', got {product!r}")

        self.product = product
        self.tick_value = _TICK_VALUE[product]
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.hard_stop_atr_mult = hard_stop_atr_mult
        self.confirm_bars = confirm_bars

        self._time_guards = TimeGuards()

        # ── Crossover confirmation state ──────────────────────────────────
        # _pending_dir: +1 = waiting to confirm long, -1 = waiting to confirm short, 0 = none
        self._pending_dir: int = 0
        self._confirm_count: int = 0  # how many bars we've seen in the new direction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        entry_date: pd.Timestamp | None,
        equity: float,
        current_time: pd.Timestamp | None = None,
    ) -> Signal:
        """Generate a trend-following signal."""
        if len(data) < _MIN_DATA_ROWS:
            return Signal(
                action="hold",
                reason=f"insufficient data: {len(data)} < {_MIN_DATA_ROWS}",
            )

        # ── Compute indicators ────────────────────────────────────────────
        indic = self._compute_indicators(data)
        latest = indic.iloc[-1]
        prev = indic.iloc[-2]

        ema20 = latest["ema20"]
        ema60 = latest["ema60"]
        atr = latest["atr"]
        close = latest["close"]

        prev_ema20 = prev["ema20"]
        prev_ema60 = prev["ema60"]

        if np.isnan(ema20) or np.isnan(ema60) or np.isnan(atr) or atr == 0:
            return Signal(action="hold", reason="indicators not ready")

        # ── Resolve current_time ──────────────────────────────────────────
        if current_time is None:
            import zoneinfo

            tz = zoneinfo.ZoneInfo("Asia/Taipei")
            bd = data.index[-1]
            if hasattr(bd, "tz_convert"):
                bd = bd.tz_convert(tz)
            current_time = pd.Timestamp(bd.year, bd.month, bd.day, 14, 0, 0, tzinfo=tz)

        # ── TimeGuard: settlement-day rollover only ───────────────────────
        # We skip Friday close intentionally (long-term holds span weekends).
        if current_position != 0:
            if self._time_guards.is_settlement_close(current_time):
                return Signal(
                    action="close",
                    contracts=1,
                    reason="settlement-day rollover",
                )

        # ── Current trend direction ───────────────────────────────────────
        cur_dir = 1 if ema20 > ema60 else -1

        # ── Detect crossover (sign change from previous bar) ─────────────
        prev_dir = 1 if prev_ema20 > prev_ema60 else -1
        crossed = cur_dir != prev_dir

        if crossed:
            # New pending direction; reset confirmation counter
            self._pending_dir = cur_dir
            self._confirm_count = 1
        elif self._pending_dir != 0:
            if cur_dir == self._pending_dir:
                self._confirm_count += 1
            else:
                # Direction flipped again before confirm → discard
                self._pending_dir = 0
                self._confirm_count = 0

        # ── Hard stop check (in-position only) ───────────────────────────
        if current_position != 0 and entry_price is not None:
            direction = 1 if current_position > 0 else -1
            hard_stop = entry_price - direction * self.hard_stop_atr_mult * atr
            stop_hit = (direction == 1 and close < hard_stop) or (
                direction == -1 and close > hard_stop
            )
            if stop_hit:
                # Reset pending state so we re-enter cleanly after stop
                self._pending_dir = 0
                self._confirm_count = 0
                return Signal(
                    action="close",
                    contracts=1,
                    reason=(
                        f"hard stop: close={close:.0f}  "
                        f"entry={entry_price:.0f}  "
                        f"stop={hard_stop:.0f}  ({self.hard_stop_atr_mult}×ATR)"
                    ),
                    hard_stop=hard_stop,
                )

        # ── Reverse crossover — close existing position ───────────────────
        if current_position != 0:
            direction = 1 if current_position > 0 else -1
            if cur_dir != direction:
                # Opposite trend confirmed → close immediately; 3-bar confirm
                # for entry in new direction is already tracked in pending state
                return Signal(
                    action="close",
                    contracts=1,
                    reason=(
                        f"EMA crossover reversed: EMA20={ema20:.0f} "
                        f"{'>' if cur_dir == 1 else '<'} EMA60={ema60:.0f}"
                    ),
                )
            # Still in same direction → hold
            return Signal(
                action="hold",
                contracts=1,
                reason=(
                    f"holding {direction:+d}: EMA20={ema20:.0f} EMA60={ema60:.0f}  ATR={atr:.0f}"
                ),
            )

        # ── Entry: flat → need 3-bar confirmation ────────────────────────
        if self._pending_dir != 0 and self._confirm_count >= self.confirm_bars:
            hard_stop_px = close - self._pending_dir * self.hard_stop_atr_mult * atr
            if self._pending_dir == 1:
                self._pending_dir = 0
                self._confirm_count = 0
                return Signal(
                    action="buy",
                    contracts=1,
                    reason=(
                        f"EMA bull confirm ({self.confirm_bars} bars): "
                        f"EMA20={ema20:.0f} > EMA60={ema60:.0f}  ATR={atr:.0f}"
                    ),
                    hard_stop=hard_stop_px,
                )
            else:
                self._pending_dir = 0
                self._confirm_count = 0
                return Signal(
                    action="sell",
                    contracts=1,
                    reason=(
                        f"EMA bear confirm ({self.confirm_bars} bars): "
                        f"EMA20={ema20:.0f} < EMA60={ema60:.0f}  ATR={atr:.0f}"
                    ),
                    hard_stop=hard_stop_px,
                )

        return Signal(
            action="hold",
            reason=(
                f"waiting: pending={self._pending_dir:+d}  "
                f"confirm={self._confirm_count}/{self.confirm_bars}"
                if self._pending_dir != 0
                else (f"flat: EMA20={ema20:.0f} {'>' if cur_dir == 1 else '<'} EMA60={ema60:.0f}")
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset crossover-confirmation state (call when position is externally closed)."""
        self._pending_dir = 0
        self._confirm_count = 0

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["open", "high", "low", "close"]].copy().astype(float)
        df["ema20"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema60"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        df["prev_close"] = df["close"].shift(1)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - df["prev_close"]).abs(),
                (df["low"] - df["prev_close"]).abs(),
            ),
        )
        df["atr"] = df["tr"].rolling(self.atr_period).mean()
        return df.drop(columns=["prev_close", "tr"])
