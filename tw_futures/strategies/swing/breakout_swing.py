"""Donchian Channel Breakout Swing Strategy for TAIFEX futures (台指期波段策略).

Strategy logic
--------------
1. **Donchian Channel** (20-day):
   - Upper Band = rolling 20-day max of *high* (shifted 1 to avoid look-ahead)
   - Lower Band = rolling 20-day min of *low*  (shifted 1)
   - close > Upper Band → long signal
   - close < Lower Band → short signal

2. **Trend Filter** (50-day SMA):
   - close > SMA50      → accept long signals only
   - close < SMA50      → accept short signals only
   - |close − SMA50| / SMA50 < 1%  → no-trade neutral zone

3. **Entry**: at the signal bar's close price (simulation).

4. **Exit** — first triggered wins:
   a. Reverse signal → close current + open opposite
   b. ATR Trailing Stop (2×ATR14): floor for longs, ceiling for shorts
      - Stop trails the running extremum (highest close for longs, lowest for shorts)
      - since entry; triggered when current close crosses the stop level.
   c. Max holding period 10 trading days → forced flat.

5. **Position Sizing**:
   - Fixed-risk: contracts = ⌊(equity × 0.02) / (2 × ATR × tick_value)⌋, min 1.
   - TX  = 200 TWD / point;  MTX = 50 TWD / point.

Defensive rules
---------------
- < 50 data rows → Signal(action="hold", reason="insufficient data")
- ATR == 0       → Signal(action="hold", reason="ATR is zero")
- Sized contracts == 0 → Signal(action="hold", reason="position too small")

Usage
-----
>>> strategy = BreakoutSwingStrategy(product="TX")
>>> sig = strategy.generate_signal(
...     data=df,                 # pd.DataFrame with OHLCV, ≥50 rows
...     current_position=0,
...     entry_price=None,
...     entry_date=None,
...     equity=5_000_000,
... )
>>> print(sig)
Signal(action='buy', contracts=2, reason='Donchian breakout long ...', stop_loss=32100.0)
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

# ── Strategy parameters (all overridable via constructor) ─────────────────────

_DONCHIAN_PERIOD: Final[int] = 20
_TREND_PERIOD: Final[int] = 50
_ATR_PERIOD: Final[int] = 14
_ATR_STOP_MULT: Final[float] = 2.0
_RISK_PER_TRADE: Final[float] = 0.02  # 2 % of equity per trade
_MAX_HOLD_DAYS: Final[int] = 10  # trading days
_NEUTRAL_ZONE_PCT: Final[float] = 0.01  # ±1 % around SMA50

_TICK_VALUE: Final[dict[str, float]] = {
    "TX": 200.0,  # TWD per index point  大台
    "MTX": 50.0,  # TWD per index point  小台
}

_MIN_DATA_ROWS: Final[int] = _TREND_PERIOD  # need at least 50 rows for SMA50


# ── Signal dataclass ──────────────────────────────────────────────────────────


@dataclass
class Signal:
    """Trade instruction returned by :meth:`BreakoutSwingStrategy.generate_signal`.

    Attributes
    ----------
    action :
        ``"buy"``   — open or add long position
        ``"sell"``  — open or add short position
        ``"close"`` — exit current position (no reversal)
        ``"hold"``  — do nothing
    contracts :
        Number of contracts to trade (0 for hold/close when not sizing).
    reason :
        Human-readable explanation of the decision.
    stop_loss :
        Initial or current trailing stop price level.
        ``None`` when action is "hold".
    """

    action: str
    contracts: int = 0
    reason: str = ""
    stop_loss: float | None = None

    def __str__(self) -> str:
        sl = f"  stop={self.stop_loss:.0f}" if self.stop_loss is not None else ""
        return (
            f"Signal(action={self.action!r}  contracts={self.contracts}"
            f"  reason={self.reason!r}{sl})"
        )


# ── Strategy ──────────────────────────────────────────────────────────────────


class BreakoutSwingStrategy:
    """Donchian Channel Breakout swing strategy for TAIFEX TX / MTX futures.

    Parameters
    ----------
    product :
        ``"TX"`` (大台指, 200 TWD/pt) or ``"MTX"`` (小台指, 50 TWD/pt).
    donchian_period :
        Look-back for Donchian channel (default 20).
    trend_period :
        Look-back for SMA trend filter (default 50).
    atr_period :
        Look-back for ATR stop calculation (default 14).
    atr_stop_mult :
        ATR multiplier for trailing stop distance (default 2.0).
    risk_per_trade :
        Fraction of equity risked per trade (default 0.02 = 2 %).
    max_hold_days :
        Maximum trading days to hold before forced exit (default 10).
    neutral_zone_pct :
        SMA50 neutral-zone half-width as a fraction (default 0.01 = 1 %).
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
        self.neutral_zone_pct = neutral_zone_pct

    # ------------------------------------------------------------------
    # Primary interface
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
        """Generate a trade signal from the latest bar in *data*.

        Parameters
        ----------
        data :
            OHLCV DataFrame with columns ``open, high, low, close, volume``
            and a ``pd.DatetimeIndex``.  Must have at least 50 rows.
        current_position :
            Net contracts currently held: positive = long, negative = short,
            0 = flat.
        entry_price :
            Average entry price of the current position.  ``None`` if flat.
        entry_date :
            Bar timestamp when the current position was opened.  ``None`` if flat.
        equity :
            Current account equity in TWD (used for position sizing).
        current_time :
            Optional explicit timestamp for time-guard checks.  Defaults to
            ``data.index[-1]`` (the latest bar's timestamp).  For daily bars the
            time component is set to 14:00 Asia/Taipei so Friday and settlement-
            day checks fire correctly.

        Returns
        -------
        Signal
        """
        # ── Guard: minimum data ────────────────────────────────────────
        if len(data) < _MIN_DATA_ROWS:
            return Signal(
                action="hold",
                reason=f"insufficient data: {len(data)} rows < {_MIN_DATA_ROWS} required",
            )

        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            return Signal(action="hold", reason=f"missing columns: {missing}")

        # ── Compute indicators ─────────────────────────────────────────
        indic = self._compute_indicators(data)
        latest = indic.iloc[-1]

        current_close = latest["close"]
        atr = latest["atr"]
        upper = latest["upper"]
        lower = latest["lower"]
        sma50 = latest["sma50"]
        current_date = data.index[-1]

        # ── Time guards — derive a post-close timestamp for daily bars ─
        # Daily bars have no meaningful time component; we synthesise 14:00 so
        # is_friday_close / is_settlement_close checks work correctly.
        if current_time is None:
            import zoneinfo

            tz = zoneinfo.ZoneInfo("Asia/Taipei")
            bar_date = current_date
            if hasattr(bar_date, "tz_convert"):
                bar_date = bar_date.tz_convert(tz)
            current_time = pd.Timestamp(
                bar_date.year,
                bar_date.month,
                bar_date.day,
                14,
                0,
                0,
                tzinfo=tz,
            )

        # Force-close on Fridays and settlement days (swing strategy — NOT intraday EOD)
        if current_position != 0:
            if _time_guards.is_friday_close(current_time):
                reason = "Friday close — flattening before weekend gap"
                force = True
            elif _time_guards.is_settlement_close(current_time):
                reason = "Settlement day — closing before TAIFEX final settlement"
                force = True
            else:
                force = False
                reason = ""
            if force:
                logger.debug(
                    "BreakoutSwing time-guard close on %s: %s", current_date.date(), reason
                )
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=f"TimeGuard: {reason}",
                )

        # ── Guard: ATR ─────────────────────────────────────────────────
        if atr == 0 or np.isnan(atr):
            return Signal(action="hold", reason="ATR is zero — skipping to avoid division error")

        # ── Determine raw signals ──────────────────────────────────────
        long_signal = current_close > upper and not np.isnan(upper)
        short_signal = current_close < lower and not np.isnan(lower)

        # Apply trend filter (neutral zone kills both signals)
        long_signal, short_signal = self._apply_trend_filter(
            current_close, sma50, long_signal, short_signal
        )

        # ── Branch: in a position ──────────────────────────────────────
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

        # ── Branch: flat — look for entry ──────────────────────────────
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
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with all strategy indicators aligned to *data*."""
        df = data[["open", "high", "low", "close"]].copy().astype(float)

        # Donchian channel — shift(1) prevents look-ahead
        df["upper"] = df["high"].rolling(self.donchian_period).max().shift(1)
        df["lower"] = df["low"].rolling(self.donchian_period).min().shift(1)

        # Trend SMA
        df["sma50"] = df["close"].rolling(self.trend_period).mean()

        # ATR (Wilder's)
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

    # ------------------------------------------------------------------
    # Trend filter
    # ------------------------------------------------------------------

    def _apply_trend_filter(
        self,
        close: float,
        sma50: float,
        long_signal: bool,
        short_signal: bool,
    ) -> tuple[bool, bool]:
        """Apply SMA50 trend filter; kill signals in neutral zone."""
        if np.isnan(sma50) or sma50 == 0:
            return False, False

        deviation = abs(close - sma50) / sma50

        # Neutral zone: price too close to SMA50 — avoid choppy conditions
        if deviation < self.neutral_zone_pct:
            logger.debug(
                "Trend filter: neutral zone (close=%.0f  sma50=%.0f  dev=%.2f%%)",
                close,
                sma50,
                deviation * 100,
            )
            return False, False

        if close > sma50:
            # Uptrend — only longs
            return long_signal, False
        else:
            # Downtrend — only shorts
            return False, short_signal

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        indic: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        entry_date: pd.Timestamp | None,
        current_close: float,
        current_date: pd.Timestamp,
        atr: float,
        long_signal: bool,
        short_signal: bool,
        equity: float,
    ) -> Signal:
        """Evaluate three exit conditions; return first that triggers."""
        direction = 1 if current_position > 0 else -1

        # ── Exit condition c: max holding period ──────────────────────
        if entry_date is not None:
            # Count trading bars since entry (inclusive of entry bar)
            days_held = len(data[data.index >= entry_date])
            if days_held > self.max_hold_days:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason=(
                        f"max holding period reached ({days_held} trading days > "
                        f"{self.max_hold_days})"
                    ),
                )

        # ── Exit condition b: ATR trailing stop ────────────────────────
        trailing_stop = self._compute_trailing_stop(
            data=data,
            direction=direction,
            entry_date=entry_date,
            atr=atr,
        )
        stop_triggered = (direction == 1 and current_close < trailing_stop) or (
            direction == -1 and current_close > trailing_stop
        )
        if stop_triggered:
            return Signal(
                action="close",
                contracts=abs(current_position),
                reason=(
                    f"ATR trailing stop triggered  "
                    f"close={current_close:.0f}  stop={trailing_stop:.0f}"
                ),
                stop_loss=trailing_stop,
            )

        # ── Exit condition a: reverse signal → close + open opposite ──
        if direction == 1 and short_signal:
            contracts = self._size_contracts(atr, equity)
            if contracts == 0:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason="reverse short signal — close long (new short too small to open)",
                )
            return Signal(
                action="sell",
                contracts=contracts,
                reason=(
                    f"reverse signal: close long + open short  "
                    f"(Donchian lower={indic.iloc[-1]['lower']:.0f}  "
                    f"close={current_close:.0f})"
                ),
                stop_loss=current_close + self.atr_stop_mult * atr,
            )

        if direction == -1 and long_signal:
            contracts = self._size_contracts(atr, equity)
            if contracts == 0:
                return Signal(
                    action="close",
                    contracts=abs(current_position),
                    reason="reverse long signal — close short (new long too small to open)",
                )
            return Signal(
                action="buy",
                contracts=contracts,
                reason=(
                    f"reverse signal: close short + open long  "
                    f"(Donchian upper={indic.iloc[-1]['upper']:.0f}  "
                    f"close={current_close:.0f})"
                ),
                stop_loss=current_close - self.atr_stop_mult * atr,
            )

        # ── No exit triggered ─────────────────────────────────────────
        return Signal(
            action="hold",
            contracts=abs(current_position),
            reason="holding position — no exit condition met",
            stop_loss=trailing_stop,
        )

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _generate_entry_signal(
        self,
        current_close: float,
        atr: float,
        long_signal: bool,
        short_signal: bool,
        equity: float,
        upper: float,
        lower: float,
        sma50: float,
    ) -> Signal:
        """Generate an entry signal when flat."""
        if long_signal:
            contracts = self._size_contracts(atr, equity)
            if contracts == 0:
                return Signal(
                    action="hold",
                    reason=(
                        f"long signal filtered: position size = 0  "
                        f"(equity={equity:,.0f}  ATR={atr:.0f})"
                    ),
                )
            stop = current_close - self.atr_stop_mult * atr
            return Signal(
                action="buy",
                contracts=contracts,
                reason=(
                    f"Donchian breakout long  "
                    f"close={current_close:.0f} > upper={upper:.0f}  "
                    f"sma50={sma50:.0f}  ATR={atr:.0f}"
                ),
                stop_loss=stop,
            )

        if short_signal:
            contracts = self._size_contracts(atr, equity)
            if contracts == 0:
                return Signal(
                    action="hold",
                    reason=(
                        f"short signal filtered: position size = 0  "
                        f"(equity={equity:,.0f}  ATR={atr:.0f})"
                    ),
                )
            stop = current_close + self.atr_stop_mult * atr
            return Signal(
                action="sell",
                contracts=contracts,
                reason=(
                    f"Donchian breakout short  "
                    f"close={current_close:.0f} < lower={lower:.0f}  "
                    f"sma50={sma50:.0f}  ATR={atr:.0f}"
                ),
                stop_loss=stop,
            )

        return Signal(action="hold", reason="no breakout signal")

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _size_contracts(self, atr: float, equity: float) -> int:
        """Compute contract size from fixed-risk formula.

        contracts = max(1, ⌊ (equity × risk_pct) / (atr_mult × ATR × tick_value) ⌋)

        Returns at least 1 when ATR and equity are positive (「最少 1 口」rule).
        Returns 0 only for degenerate inputs (ATR ≤ 0 or equity ≤ 0); those
        cases are already guarded in ``generate_signal`` before this is called.
        """
        if atr <= 0 or equity <= 0:
            return 0
        risk_amount = equity * self.risk_per_trade
        stop_distance = self.atr_stop_mult * atr  # index points
        cost_per_contract = stop_distance * self.tick_value  # TWD
        if cost_per_contract <= 0:
            return 0
        return max(1, math.floor(risk_amount / cost_per_contract))

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def _compute_trailing_stop(
        self,
        data: pd.DataFrame,
        direction: int,
        entry_date: pd.Timestamp | None,
        atr: float,
    ) -> float:
        """Return the current ATR trailing stop level.

        For longs  : trailing_stop = highest_close_since_entry − atr_mult × ATR
        For shorts : trailing_stop = lowest_close_since_entry  + atr_mult × ATR

        If *entry_date* is unknown, uses the entire *data* series.
        """
        if entry_date is not None:
            since_entry = data.loc[data.index >= entry_date, "close"]
        else:
            since_entry = data["close"]

        if since_entry.empty:
            since_entry = data["close"]

        stop_offset = self.atr_stop_mult * atr

        if direction == 1:  # long
            return float(since_entry.max()) - stop_offset
        else:  # short
            return float(since_entry.min()) + stop_offset
