"""Donchian Channel Breakout Swing Strategy V3 — Chandelier Exit + Re-entry.

Two targeted improvements over V1 (no additional entry filters):

1. **Chandelier Exit** (wider, anchored to price extremes)
   - V1 trailing stop = highest *close* − 2×ATR
   - V3 trailing stop = highest *high* − 3×ATR  (long)
                        lowest  *low*  + 3×ATR  (short)
   - Using true intraday extremes as the anchor gives the trade more room to
     breathe through typical intraday noise; the wider 3×ATR multiplier reduces
     whipsaw stop-outs in Taiwan's volatile futures market.

2. **One-Time Re-entry** (recapture lost trend)
   - After ANY position closes (stop, max-hold, signal), a re-entry window of
     *reentry_window* trading days opens.
   - If a valid Donchian breakout signal (same direction + trend filter) appears
     within that window, enter once more.
   - Only one re-entry is allowed per original trade cycle.  Once the re-entry
     trade exits for any reason, the state resets fully.
   - Opposite-direction entries within the window are allowed normally and
     do not consume the re-entry allowance.

All other logic identical to V1:
- Donchian Channel 20-day (shift-1, no look-ahead)
- Trend Filter SMA50 ±1%
- Max Holding 10 trading days
- Position Sizing: 2% fixed-risk, min 1 contract

Internal state
--------------
The strategy keeps five mutable variables that persist across ``generate_signal``
calls (the backtester creates one instance and calls it repeatedly):

``_prev_position``      int              position from the previous call
``_exit_date``          Timestamp|None   date the last position was detected as closed
``_exit_direction``     int              ±1, direction of the last closed position
``_reentry_used``       bool             True once we have issued the re-entry signal
``_in_reentry_trade``   bool             True while the current position is a re-entry

Call ``reset_state()`` if you reuse the same instance across multiple backtests.
"""

from __future__ import annotations

import logging

import pandas as pd

from tw_futures.strategies.swing.breakout_swing import (
    _ATR_PERIOD,
    _DONCHIAN_PERIOD,
    _MAX_HOLD_DAYS,
    _NEUTRAL_ZONE_PCT,
    _RISK_PER_TRADE,
    _TREND_PERIOD,
    BreakoutSwingStrategy,
    Signal,
)

logger = logging.getLogger(__name__)

_CHANDELIER_MULT: float = 3.0  # wider than V1's 2×ATR
_REENTRY_WINDOW: int = 3  # trading days after exit


class BreakoutSwingV3Strategy(BreakoutSwingStrategy):
    """Donchian Breakout V3 — Chandelier Exit + one-time re-entry.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"``.
    atr_stop_mult :
        ATR multiplier for the Chandelier stop (default 3.0).
    reentry_window :
        Trading days after an exit during which a re-entry is permitted
        (default 3).
    All remaining keyword arguments are forwarded to
    :class:`BreakoutSwingStrategy`.
    """

    def __init__(
        self,
        product: str = "TX",
        donchian_period: int = _DONCHIAN_PERIOD,
        trend_period: int = _TREND_PERIOD,
        atr_period: int = _ATR_PERIOD,
        atr_stop_mult: float = _CHANDELIER_MULT,  # 3× by default
        risk_per_trade: float = _RISK_PER_TRADE,
        max_hold_days: int = _MAX_HOLD_DAYS,
        neutral_zone_pct: float = _NEUTRAL_ZONE_PCT,
        reentry_window: int = _REENTRY_WINDOW,
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
        self.reentry_window = reentry_window
        self.reset_state()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset mutable state — call before reusing this instance for a new backtest."""
        self._prev_position: int = 0
        self._exit_date: pd.Timestamp | None = None
        self._exit_direction: int = 0
        self._reentry_used: bool = False
        self._in_reentry_trade: bool = False

    # ------------------------------------------------------------------
    # Chandelier Exit — override trailing stop computation
    # ------------------------------------------------------------------

    def _compute_trailing_stop(
        self,
        data: pd.DataFrame,
        direction: int,
        entry_date: pd.Timestamp | None,
        atr: float,
    ) -> float:
        """Chandelier Exit: anchor to highest high (long) or lowest low (short).

        stop_long  = max(high  since entry) − atr_stop_mult × ATR
        stop_short = min(low   since entry) + atr_stop_mult × ATR
        """
        if entry_date is not None:
            mask = data.index >= entry_date
            highs = data.loc[mask, "high"].astype(float)
            lows = data.loc[mask, "low"].astype(float)
        else:
            highs = data["high"].astype(float)
            lows = data["low"].astype(float)

        if highs.empty:
            # Degenerate case: fall back to close-based (V1 behaviour)
            return super()._compute_trailing_stop(data, direction, entry_date, atr)

        offset = self.atr_stop_mult * atr
        if direction == 1:
            return float(highs.max()) - offset
        else:
            return float(lows.min()) + offset

    # ------------------------------------------------------------------
    # Primary interface — with re-entry state tracking
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        entry_date: pd.Timestamp | None,
        equity: float,
    ) -> Signal:
        """Generate a signal; manages Chandelier stop and one-time re-entry."""
        current_date = data.index[-1]

        # ── 1. Track position transitions ────────────────────────────────
        self._track_transitions(current_position, current_date, data)

        # ── 2. When in a position: delegate entirely to V1 exit logic ────
        #   (which calls our overridden _compute_trailing_stop → Chandelier)
        if current_position != 0:
            sig = super().generate_signal(data, current_position, entry_price, entry_date, equity)
            # Update prev_position AFTER computing signal
            self._prev_position = current_position
            return sig

        # ── 3. Flat: run V1 signal generation logic ───────────────────────
        #   (guards, indicators, trend filter, entry)
        sig = super().generate_signal(data, current_position, entry_price, entry_date, equity)

        # ── 4. Apply re-entry rules to entry signals ──────────────────────
        if sig.action in ("buy", "sell"):
            sig = self._apply_reentry_rules(sig, current_date, data)

        self._prev_position = current_position
        return sig

    # ------------------------------------------------------------------
    # Re-entry helpers
    # ------------------------------------------------------------------

    def _track_transitions(
        self,
        current_position: int,
        current_date: pd.Timestamp,
        data: pd.DataFrame,
    ) -> None:
        """Update internal state based on observed position changes."""
        prev = self._prev_position

        if prev != 0 and current_position == 0:
            # A position just closed (detected one bar after the actual close)
            if self._in_reentry_trade:
                # Re-entry trade ended → full reset (no more re-entry for this cycle)
                self.reset_state()
            else:
                # Original trade ended → open re-entry window
                self._exit_date = current_date
                self._exit_direction = 1 if prev > 0 else -1
                self._reentry_used = False
                self._in_reentry_trade = False

        elif prev == 0 and current_position != 0:
            # A position was just opened (backtester executed our last signal)
            new_dir = 1 if current_position > 0 else -1
            if (
                self._exit_date is not None
                and self._reentry_used
                and new_dir == self._exit_direction
            ):
                # Backtester confirmed our re-entry trade
                self._in_reentry_trade = True
            else:
                # Normal fresh entry or opposite-direction entry — clear re-entry state
                self._in_reentry_trade = False
                self._exit_date = None
                self._reentry_used = False

        # Expire the re-entry window when time runs out
        if self._exit_date is not None and not self._reentry_used and current_position == 0:
            bars_since_exit = int(
                data.loc[(data.index > self._exit_date) & (data.index <= current_date)].shape[0]
            )
            if bars_since_exit > self.reentry_window:
                self._exit_date = None
                self._exit_direction = 0

    def _apply_reentry_rules(
        self,
        sig: Signal,
        current_date: pd.Timestamp,
        data: pd.DataFrame,
    ) -> Signal:
        """Enforce the one-time re-entry constraint for same-direction entries.

        Rules
        -----
        * If no re-entry window is active → pass signal through unchanged.
        * If re-entry window active and signal is **opposite** direction
          → pass through (not counted against re-entry allowance).
        * If re-entry window active and signal is **same** direction:
          - Not yet used → allow, mark ``_reentry_used = True``.
          - Already used  → block with a hold signal.
        """
        if self._exit_date is None:
            # No active re-entry window → normal entry
            return sig

        signal_dir = 1 if sig.action == "buy" else -1

        if signal_dir != self._exit_direction:
            # Opposite-direction signal within window → treat as fresh entry
            # (Clear the re-entry state so this new trade starts clean)
            self._exit_date = None
            self._exit_direction = 0
            self._reentry_used = False
            return sig

        # Same-direction signal within the re-entry window
        if not self._reentry_used:
            self._reentry_used = True
            logger.debug(
                "V3 re-entry: %s on %s  (exit_dir=%+d  window=%d bars)",
                sig.action,
                current_date.date(),
                self._exit_direction,
                self.reentry_window,
            )
            return Signal(
                action=sig.action,
                contracts=sig.contracts,
                reason=f"V3 re-entry ({sig.reason})",
                stop_loss=sig.stop_loss,
            )
        else:
            # Re-entry already consumed → block
            logger.debug(
                "V3 re-entry blocked on %s: already used for this cycle",
                current_date.date(),
            )
            return Signal(
                action="hold",
                reason=(
                    "V3 re-entry blocked: one re-entry already used "
                    f"(exit_date={self._exit_date.date() if self._exit_date else '?'})"
                ),
            )
