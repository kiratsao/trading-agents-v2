"""Opening Range Breakout (ORB) Intraday Strategy for TAIFEX futures.

Time frame
----------
15-minute bars.

Opening Range definition
------------------------
First 60 minutes of trading: 08:45 → 09:30 (bars timestamped 08:45, 09:00, 09:15, 09:30).
ORB High = max(high) of those 4 bars.
ORB Low  = min(low)  of those 4 bars.

Entry logic (after 09:45 bar appears)
--------------------------------------
* Long  : close > ORB High  (breakout above)
* Short : close < ORB Low   (breakdown below)
* One trade per day (traded_today flag tracked in instance state).

Entry filters
--------------
* ORB width (High − Low) must be in [30, 200] points.
  - Too narrow (< 30 pt): high false-breakout risk.
  - Too wide   (> 200 pt): stop too large, risk sizing unfavourable.
* No entry during TimeGuards no-trade zones or force-close windows.

Position sizing
---------------
risk         = equity × risk_per_trade  (default 2%)
stop_distance = ORB width  (full opening range)
contracts    = max(1, floor(risk / (stop_distance × tick_value)))

Stop loss
---------
* Long  : stop at ORB Low  (the other side of the range)
* Short : stop at ORB High (the other side of the range)
* Stop checked at each bar's close.

Force close
-----------
* 13:15 — Friday & settlement-day swing close (TimeGuards)
* 13:15 — ORB EOD cutoff (one bar before TAIFEX close) ← intraday only
  Using 13:15 ensures execution before the 13:30–13:45 thin-liquidity window.
* 13:30 — hard EOD exit for any remaining intraday position (TimeGuards)

Products supported
------------------
TX (大台): tick_value = 200 TWD/pt
MTX (小台): tick_value = 50 TWD/pt

Notes
-----
* Daily state (traded_today, orb_high, orb_low, last_date) is tracked as
  instance state, so reuse the same instance across an entire day's bar stream.
* The backtester creates one instance per backtest run, so state resets
  naturally at the start of each run.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import time
from typing import Final

import pandas as pd

from tw_futures.strategies.common.time_guards import TimeGuards

logger = logging.getLogger(__name__)

# ── Tick values ───────────────────────────────────────────────────────────────
_TICK_VALUE: Final[dict[str, float]] = {"TX": 200.0, "MTX": 50.0}

# ── ORB session constants ─────────────────────────────────────────────────────
_ORB_START_TIME: Final[time] = time(8, 45)
_ORB_END_TIME: Final[time] = time(9, 30)  # last bar in the opening range
_ENTRY_AFTER: Final[time] = time(9, 45)  # first bar eligible for entry
_ORB_EOD_CUTOFF: Final[time] = time(13, 15)  # force-close cutoff for ORB


# ---------------------------------------------------------------------------
# Signal dataclass (shared with trend_intraday; redefined here for self-containment)
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """Intraday trading signal.

    Parameters
    ----------
    action :
        ``"buy"`` / ``"sell"`` (open) · ``"close"`` (flatten) · ``"hold"``
    contracts :
        Number of contracts.  0 for hold/no-action signals.
    reason :
        Human-readable explanation.
    stop_loss :
        Stop-loss price level (for state persistence), or ``None``.
    hard_stop :
        Hard stop price level (fixed, non-trailing), or ``None``.
    """

    action: str
    contracts: int = 0
    reason: str = ""
    stop_loss: float | None = None
    hard_stop: float | None = None


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------


class ORBIntradayStrategy:
    """Opening Range Breakout strategy on 15-minute TAIFEX futures bars.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"``.
    min_orb_pts :
        Minimum ORB width in points.  Narrower ranges → no trade.
    max_orb_pts :
        Maximum ORB width in points.  Wider ranges → no trade.
    risk_per_trade :
        Fraction of equity risked per trade (default 2%).
    """

    def __init__(
        self,
        product: str = "TX",
        min_orb_pts: float = 30.0,
        max_orb_pts: float = 200.0,
        risk_per_trade: float = 0.02,
    ) -> None:
        self.product = product.upper()
        self.min_orb_pts = min_orb_pts
        self.max_orb_pts = max_orb_pts
        self.risk_per_trade = risk_per_trade
        self.tick_value = _TICK_VALUE.get(self.product, 200.0)

        self._time_guards = TimeGuards()

        # ── Daily state ────────────────────────────────────────────────────
        # Reset at the start of each new calendar day.
        self._last_date: pd.Timestamp | None = None
        self._orb_high: float | None = None
        self._orb_low: float | None = None
        self._traded_today: bool = False

    # ------------------------------------------------------------------
    # Public interface
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
        """Generate a trading signal for the current 15-min bar.

        Parameters
        ----------
        data_15min :
            All available 15-min bars up to and including the current bar.
            Index: ``DatetimeIndex`` (tz-aware, Asia/Taipei).
        current_position :
            Current signed position: ``+n`` = long, ``-n`` = short, ``0`` = flat.
        entry_price :
            Price at which the current position was opened, or ``None`` if flat.
        equity :
            Current account equity in TWD.
        current_time :
            Timestamp of the current bar (tz-aware).
        entry_time :
            Timestamp when the current position was opened, or ``None``.

        Returns
        -------
        Signal
        """
        if data_15min.empty:
            return Signal(action="hold", reason="no data")

        # ── Normalise time to Taipei ───────────────────────────────────────
        from tw_futures.strategies.common.time_guards import _to_taipei

        local_ts = _to_taipei(current_time)
        bar_date = local_ts.date()
        bar_time = local_ts.time()

        # ── Day rollover — reset daily state ──────────────────────────────
        if self._last_date is None or bar_date != self._last_date.date():
            self._last_date = local_ts
            self._orb_high = None
            self._orb_low = None
            self._traded_today = False
            logger.debug("ORB: new day %s — daily state reset.", bar_date)

        # ── Build / update the opening range ──────────────────────────────
        self._update_orb(data_15min, bar_date)

        # ── Force-close: check TimeGuards + ORB EOD cutoff ────────────────
        if current_position != 0:
            # ORB EOD cutoff at 13:15
            if bar_time >= _ORB_EOD_CUTOFF:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"ORB EOD force-close ({bar_time.strftime('%H:%M')})",
                )
            # TimeGuards: Friday/settlement/intraday EOD
            force, reason = self._time_guards.should_force_close(local_ts)
            if force:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"TimeGuard force-close: {reason}",
                )

            # Stop-loss check
            sl_signal = self._check_stop(current_position, entry_price, data_15min)
            if sl_signal is not None:
                return sl_signal

            # Compute and return trailing stop level for state persistence
            stop_level = self._stop_level(current_position, entry_price)
            return Signal(action="hold", reason="in position", stop_loss=stop_level)

        # ── Entry logic ───────────────────────────────────────────────────
        if self._traded_today:
            return Signal(action="hold", reason="already traded today")

        if self._orb_high is None or self._orb_low is None:
            return Signal(action="hold", reason="opening range not yet established")

        # Not yet in entry window
        if bar_time < _ENTRY_AFTER:
            return Signal(
                action="hold",
                reason=f"waiting for ORB window (entry after {_ENTRY_AFTER.strftime('%H:%M')})",
            )

        # Block entry during no-trade zones / force-close windows
        blocked, reason = self._time_guards.should_block_entry(local_ts)
        if blocked:
            return Signal(action="hold", reason=f"entry blocked: {reason}")

        # ORB width filter
        orb_width = self._orb_high - self._orb_low
        if orb_width < self.min_orb_pts:
            return Signal(
                action="hold",
                reason=f"ORB too narrow ({orb_width:.0f} pts < {self.min_orb_pts:.0f} min)",
            )
        if orb_width > self.max_orb_pts:
            return Signal(
                action="hold",
                reason=f"ORB too wide ({orb_width:.0f} pts > {self.max_orb_pts:.0f} max)",
            )

        current_close = float(data_15min["close"].iloc[-1])
        contracts = self._size_contracts(equity, orb_width)

        # Long breakout
        if current_close > self._orb_high:
            self._traded_today = True
            stop = self._orb_low  # stop at other side of ORB
            return Signal(
                action="buy",
                contracts=contracts,
                reason=f"ORB long breakout  close={current_close:.0f} > ORB_H={self._orb_high:.0f}",
                stop_loss=stop,
                hard_stop=stop,
            )

        # Short breakdown
        if current_close < self._orb_low:
            self._traded_today = True
            stop = self._orb_high  # stop at other side of ORB
            return Signal(
                action="sell",
                contracts=contracts,
                reason=f"ORB short breakdown  close={current_close:.0f} < ORB_L={self._orb_low:.0f}",
                stop_loss=stop,
                hard_stop=stop,
            )

        return Signal(action="hold", reason="no ORB breakout")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_orb(self, data_15min: pd.DataFrame, bar_date) -> None:
        """Compute (or confirm) ORB high/low from today's opening range bars."""
        from tw_futures.strategies.common.time_guards import _to_taipei

        today_mask = pd.Series(data_15min.index, index=data_15min.index).apply(
            lambda ts: _to_taipei(ts).date() == bar_date
        )
        today_bars = data_15min[today_mask]

        orb_mask = today_bars.index.map(
            lambda ts: _ORB_START_TIME <= _to_taipei(ts).time() <= _ORB_END_TIME
        )
        orb_bars = today_bars[orb_mask]

        if orb_bars.empty:
            return

        self._orb_high = float(orb_bars["high"].max())
        self._orb_low = float(orb_bars["low"].min())

    def _check_stop(
        self,
        current_position: int,
        entry_price: float | None,
        data_15min: pd.DataFrame,
    ) -> Signal | None:
        """Return a close Signal if stop-loss is breached, else None."""
        if entry_price is None or self._orb_high is None or self._orb_low is None:
            return None

        current_close = float(data_15min["close"].iloc[-1])

        if current_position > 0:  # long → stop at ORB low
            if current_close <= self._orb_low:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"ORB stop-loss hit (long): close={current_close:.0f} ≤ ORB_L={self._orb_low:.0f}",
                )
        elif current_position < 0:  # short → stop at ORB high
            if current_close >= self._orb_high:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"ORB stop-loss hit (short): close={current_close:.0f} ≥ ORB_H={self._orb_high:.0f}",
                )
        return None

    def _stop_level(self, current_position: int, entry_price: float | None) -> float | None:
        """Return the current stop-loss price level."""
        if self._orb_high is None or self._orb_low is None:
            return None
        if current_position > 0:
            return self._orb_low
        if current_position < 0:
            return self._orb_high
        return None

    def _size_contracts(self, equity: float, orb_width: float) -> int:
        """Compute contract count based on fixed-risk sizing."""
        if equity <= 0 or orb_width <= 0 or self.tick_value <= 0:
            return 1
        risk_amount = equity * self.risk_per_trade
        stop_twd = orb_width * self.tick_value
        raw = risk_amount / stop_twd
        return max(1, math.floor(raw))
