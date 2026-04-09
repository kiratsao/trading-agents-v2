"""Donchian Breakout Swing Strategy with 3-Stage Pyramid Scaling.

Philosophy
----------
「順勢加碼，鎖利出場」— Only scale up when winning; lock profits at peak.

Three-stage rocket
------------------
Stage 1 — Initial entry (50 % of max):
  Donchian breakout + SMA50 trend filter triggers entry.
  Enters at 50 % of max-sized contracts (fixed 2 % risk).

Stage 2 — Confirm and double (50 % → 100 %):
  When unrealised profit ≥ 1 × entry-bar ATR, add the remaining 50 %.
  Only scale up in a confirmed trend — never chase.

Stage 3 — Lock profits (100 % → 50 %):
  When unrealised profit ≥ 2 × entry-bar ATR, reduce back to 50 %.
  Half the position is cashed out at a high-profit moment; the other half
  continues running with the ATR trailing stop.

Exit (any remaining position)
-------------------------------
a. Hard stop: entry ± 100 index points (max fixed loss per lot).
b. ATR trailing stop (2 × ATR14, trailing from entry).
c. Reverse Donchian signal → close + open opposite.
d. Max hold: 10 trading days.
e. TimeGuards: Friday 13:15+, settlement day 13:15+.

Position sizing
---------------
  max_contracts = floor( (equity × 0.02) / (2 × ATR × tick_value) )
  stage1_size   = max(1, floor(max_contracts × 0.50))
  stage2_add    = max_contracts − stage1_size
  stage3_target = stage1_size   (reduce back)

No ML filter. No fixed contract cap (risk rule is the natural limit).

Signals returned
----------------
  action = "buy"    → open long (stage 1)
  action = "sell"   → open short (stage 1)
  action = "add"    → add contracts to existing position (stage 2)
  action = "reduce" → reduce position by n contracts (stage 3)
  action = "close"  → flatten (exit trigger)
  action = "hold"   → no action

The backtester must handle "add" and "reduce" — see FuturesBacktester.run().
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

_time_guards = TimeGuards()

# ── Constants ─────────────────────────────────────────────────────────────────

_DONCHIAN_PERIOD: Final[int] = 20
_TREND_PERIOD: Final[int] = 50
_ATR_PERIOD: Final[int] = 14
_ATR_STOP_MULT: Final[float] = 2.0
_RISK_PER_TRADE: Final[float] = 0.02
_MAX_HOLD_DAYS: Final[int] = 10
_NEUTRAL_ZONE: Final[float] = 0.01
_HARD_STOP_PTS: Final[float] = 100.0  # fixed hard stop in index points
_L2_ATR_MULT: Final[float] = 1.0  # unrealised >= 1× ATR → add to 100 %
_L3_ATR_MULT: Final[float] = 2.0  # unrealised >= 2× ATR → reduce to 50 %

_TICK_VALUE: Final[dict[str, float]] = {"TX": 200.0, "MTX": 50.0}
_MIN_DATA_ROWS: Final[int] = _TREND_PERIOD


# ── Signal ────────────────────────────────────────────────────────────────────


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


class BreakoutSwingScaledStrategy:
    """3-stage pyramid Donchian Breakout strategy for TAIFEX TX / MTX.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"``.
    donchian_period, trend_period, atr_period :
        Indicator look-backs (defaults: 20, 50, 14).
    atr_stop_mult :
        ATR multiplier for trailing stop (default 2.0).
    risk_per_trade :
        Fraction of equity risked on full position (default 0.02 = 2 %).
    max_hold_days :
        Maximum trading days before forced exit (default 10).
    hard_stop_points :
        Fixed hard stop distance in index points (default 100).
    l2_atr_mult, l3_atr_mult :
        Unrealised-profit thresholds (in entry-bar ATR units) that trigger
        stage-2 add and stage-3 reduction (defaults: 1.0, 2.0).
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
        hard_stop_points: float = _HARD_STOP_PTS,
        l2_atr_mult: float = _L2_ATR_MULT,
        l3_atr_mult: float = _L3_ATR_MULT,
    ) -> None:
        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be 'TX' or 'MTX', got {product!r}")

        self.product = product
        self.tick_value = _TICK_VALUE[product]
        self.donchian_period = donchian_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.risk_per_trade = risk_per_trade
        self.max_hold_days = max_hold_days
        self.hard_stop_points = hard_stop_points
        self.l2_atr_mult = l2_atr_mult
        self.l3_atr_mult = l3_atr_mult

        # ── Per-trade pyramid state (instance vars) ───────────────────────
        self._pyramid_level: int = 0  # 0=flat,1=L1,2=L2,3=L3
        self._entry_atr: float = 0.0  # ATR snapshot at entry
        self._max_contracts: int = 0  # max contracts for current trade
        self._stage1_size: int = 0  # L1 contract count

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
        """Generate a pyramid-aware trade signal."""
        if len(data) < _MIN_DATA_ROWS:
            return Signal(
                action="hold",
                reason=f"insufficient data: {len(data)} < {_MIN_DATA_ROWS}",
            )

        # ── Detect position reset (went flat externally) ──────────────────
        if current_position == 0 and self._pyramid_level != 0:
            self._reset_pyramid()

        # ── Compute indicators ────────────────────────────────────────────
        indic = self._compute_indicators(data)
        latest = indic.iloc[-1]

        close = latest["close"]
        atr = latest["atr"]
        upper = latest["upper"]
        lower = latest["lower"]
        sma50 = latest["sma50"]
        bar_ts = data.index[-1]

        # ── Time guards ───────────────────────────────────────────────────
        if current_time is None:
            import zoneinfo

            tz = zoneinfo.ZoneInfo("Asia/Taipei")
            bd = bar_ts
            if hasattr(bd, "tz_convert"):
                bd = bd.tz_convert(tz)
            current_time = pd.Timestamp(bd.year, bd.month, bd.day, 14, 0, 0, tzinfo=tz)

        if current_position != 0:
            force, reason = _time_guards.should_force_close(current_time)
            if force:
                self._reset_pyramid()
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"TimeGuard: {reason}",
                )

        # ── Guard ─────────────────────────────────────────────────────────
        if atr == 0 or np.isnan(atr):
            return Signal(action="hold", reason="ATR is zero")

        # ── Trend filter ──────────────────────────────────────────────────
        long_signal, short_signal = self._apply_trend_filter(close, sma50, upper, lower)

        # ── Branch: in position ───────────────────────────────────────────
        if current_position != 0:
            return self._handle_in_position(
                data=data,
                close=close,
                atr=atr,
                current_position=current_position,
                entry_price=entry_price,
                entry_date=entry_date,
                long_signal=long_signal,
                short_signal=short_signal,
                equity=equity,
                indic=indic,
            )

        # ── Branch: flat → entry ──────────────────────────────────────────
        return self._handle_entry(
            close, atr, upper, lower, sma50, long_signal, short_signal, equity
        )

    # ------------------------------------------------------------------
    # In-position handler
    # ------------------------------------------------------------------

    def _handle_in_position(
        self,
        data: pd.DataFrame,
        close: float,
        atr: float,
        current_position: int,
        entry_price: float | None,
        entry_date: pd.Timestamp | None,
        long_signal: bool,
        short_signal: bool,
        equity: float,
        indic: pd.DataFrame,
    ) -> Signal:
        direction = 1 if current_position > 0 else -1
        ep = entry_price if entry_price is not None else close
        unrealised = direction * (close - ep)

        # ── a. Max holding period ─────────────────────────────────────────
        if entry_date is not None:
            days_held = len(data[data.index >= entry_date])
            if days_held > self.max_hold_days:
                self._reset_pyramid()
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"max hold {days_held} > {self.max_hold_days} days",
                )

        # ── b. ATR trailing stop ──────────────────────────────────────────
        trailing_stop = self._compute_trailing_stop(data, direction, entry_date, atr)
        stop_hit = (direction == 1 and close < trailing_stop) or (
            direction == -1 and close > trailing_stop
        )
        if stop_hit:
            self._reset_pyramid()
            return Signal(
                action="close",
                contracts=abs(current_position),
                reason=f"ATR trailing stop  close={close:.0f}  stop={trailing_stop:.0f}",
                stop_loss=trailing_stop,
            )

        # ── c. Reverse signal ─────────────────────────────────────────────
        if direction == 1 and short_signal:
            contracts = self._size_max(atr, equity)
            self._reset_pyramid()
            if contracts == 0:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason="reverse short — new short too small",
                )
            stop = close + self.atr_stop_mult * atr
            return Signal(
                action="sell",
                contracts=contracts,
                reason=f"reverse to short  close={close:.0f}",
                stop_loss=stop,
                hard_stop=close + self.hard_stop_points,
            )
        if direction == -1 and long_signal:
            contracts = self._size_max(atr, equity)
            self._reset_pyramid()
            if contracts == 0:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason="reverse long — new long too small",
                )
            stop = close - self.atr_stop_mult * atr
            return Signal(
                action="buy",
                contracts=contracts,
                reason=f"reverse to long  close={close:.0f}",
                stop_loss=stop,
                hard_stop=close - self.hard_stop_points,
            )

        # ── d. Pyramid stage transitions ──────────────────────────────────

        # L1 → L2: add up to max when unrealised >= 1× entry ATR
        if self._pyramid_level == 1 and unrealised >= self._entry_atr * self.l2_atr_mult:
            n_add = self._max_contracts - abs(current_position)
            if n_add > 0:
                self._pyramid_level = 2
                new_stop = close - direction * self.atr_stop_mult * atr
                logger.debug(
                    "SCALED L1→L2: add %d contracts  unrealised=%.0f pts  atr=%.0f",
                    n_add,
                    unrealised,
                    self._entry_atr,
                )
                return Signal(
                    action="add",
                    contracts=n_add,
                    reason=f"Stage 2 add: unrealised={unrealised:.0f} >= {self._entry_atr * self.l2_atr_mult:.0f} (1×ATR)",
                    stop_loss=trailing_stop,
                )
            else:
                self._pyramid_level = 2  # already at max

        # L2 → L3: reduce to 50 % when unrealised >= 2× entry ATR
        elif self._pyramid_level == 2 and unrealised >= self._entry_atr * self.l3_atr_mult:
            # Reduce to stage1_size (lock half profits)
            n_current = abs(current_position)
            target = self._stage1_size
            n_reduce = max(0, n_current - target)
            if n_reduce > 0:
                self._pyramid_level = 3
                logger.debug(
                    "SCALED L2→L3: reduce %d contracts  unrealised=%.0f pts",
                    n_reduce,
                    unrealised,
                )
                return Signal(
                    action="reduce",
                    contracts=n_reduce,
                    reason=f"Stage 3 reduce: unrealised={unrealised:.0f} >= {self._entry_atr * self.l3_atr_mult:.0f} (2×ATR) — lock profits",
                    stop_loss=trailing_stop,
                )
            else:
                self._pyramid_level = 3

        # No stage transition → hold with updated stop
        return Signal(
            action="hold",
            contracts=abs(current_position),
            reason=f"holding (level={self._pyramid_level})",
            stop_loss=trailing_stop,
        )

    # ------------------------------------------------------------------
    # Entry handler
    # ------------------------------------------------------------------

    def _handle_entry(
        self,
        close: float,
        atr: float,
        upper: float,
        lower: float,
        sma50: float,
        long_signal: bool,
        short_signal: bool,
        equity: float,
    ) -> Signal:
        if long_signal:
            n_max = self._size_max(atr, equity)
            if n_max == 0:
                return Signal(action="hold", reason="long signal: size=0")
            n1 = max(1, math.floor(n_max * 0.50))
            # Initialise pyramid state
            self._pyramid_level = 1
            self._entry_atr = max(atr, 1.0)
            self._max_contracts = n_max
            self._stage1_size = n1
            stop = close - self.atr_stop_mult * atr
            hstop = close - self.hard_stop_points
            return Signal(
                action="buy",
                contracts=n1,
                reason=(
                    f"L1 long: {n1}/{n_max} contracts  "
                    f"close={close:.0f} > upper={upper:.0f}  sma50={sma50:.0f}"
                ),
                stop_loss=stop,
                hard_stop=hstop,
            )

        if short_signal:
            n_max = self._size_max(atr, equity)
            if n_max == 0:
                return Signal(action="hold", reason="short signal: size=0")
            n1 = max(1, math.floor(n_max * 0.50))
            self._pyramid_level = 1
            self._entry_atr = max(atr, 1.0)
            self._max_contracts = n_max
            self._stage1_size = n1
            stop = close + self.atr_stop_mult * atr
            hstop = close + self.hard_stop_points
            return Signal(
                action="sell",
                contracts=n1,
                reason=(
                    f"L1 short: {n1}/{n_max} contracts  "
                    f"close={close:.0f} < lower={lower:.0f}  sma50={sma50:.0f}"
                ),
                stop_loss=stop,
                hard_stop=hstop,
            )

        return Signal(action="hold", reason="no breakout signal")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_pyramid(self) -> None:
        self._pyramid_level = 0
        self._entry_atr = 0.0
        self._max_contracts = 0
        self._stage1_size = 0

    def _size_max(self, atr: float, equity: float) -> int:
        """Max contracts for a full-sized 2%-risk trade."""
        if atr <= 0 or equity <= 0:
            return 0
        risk_amount = equity * self.risk_per_trade
        cost_per_contract = self.atr_stop_mult * atr * self.tick_value
        if cost_per_contract <= 0:
            return 0
        return max(1, math.floor(risk_amount / cost_per_contract))

    def _apply_trend_filter(
        self,
        close: float,
        sma50: float,
        upper: float,
        lower: float,
    ) -> tuple[bool, bool]:
        if np.isnan(sma50) or sma50 == 0:
            return False, False
        deviation = abs(close - sma50) / sma50
        if deviation < _NEUTRAL_ZONE:
            return False, False
        long_sig = (not np.isnan(upper)) and (close > upper)
        short_sig = (not np.isnan(lower)) and (close < lower)
        if close > sma50:
            return long_sig, False
        else:
            return False, short_sig

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["open", "high", "low", "close"]].copy().astype(float)
        df["upper"] = df["high"].rolling(self.donchian_period).max().shift(1)
        df["lower"] = df["low"].rolling(self.donchian_period).min().shift(1)
        df["sma50"] = df["close"].rolling(self.trend_period).mean()
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

    def _compute_trailing_stop(
        self,
        data: pd.DataFrame,
        direction: int,
        entry_date: pd.Timestamp | None,
        atr: float,
    ) -> float:
        if entry_date is not None:
            closes = data.loc[data.index >= entry_date, "close"]
        else:
            closes = data["close"]
        if closes.empty:
            closes = data["close"]
        offset = self.atr_stop_mult * atr
        if direction == 1:
            return float(closes.max()) - offset
        else:
            return float(closes.min()) + offset
