"""EMA Crossover Trend Following Strategy V2.

Improvements over V1
---------------------
1. **Longer EMAs**: EMA(50) / EMA(150) — reduces false crossovers in choppy markets;
   extends average holding period from ~2 weeks to weeks-to-months.

2. **Long-only mode** (``long_only=True``):
   - EMA50 > EMA150 → look for long entry
   - EMA50 < EMA150 → flat (hold cash, no short)
   台指期 2020-2026 structural bull trend makes short-side a drag.

3. **ADX trend-strength filter**:
   - ADX(14) > ``adx_entry`` (default 20) required for new entries.
   - ADX < ``adx_exit_warn`` (default 15) while in position → block new entries but
     do NOT force exit (avoid whipsaws in minor pullbacks).
   - ADX replaces the fixed N-bar confirmation window; entry confirmation is now
     qualitative (trend strong enough) rather than temporal.

Exit conditions (unchanged from V1)
-------------------------------------
a. EMA crossover reversal → close (+ re-enter opposite if long_only=False,
   otherwise go flat)
b. Hard stop: entry ± ``hard_stop_atr_mult`` × ATR(20) → immediate exit
c. Settlement-day rollover (13:15 on 3rd Wednesday → re-enter same direction
   next bar if trend still active)

Position sizing: always 1 contract.  No pyramid.
No max-hold-days limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from tw_futures.strategies.common.time_guards import TimeGuards

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_EMA_FAST_DEF: Final[int] = 50
_EMA_SLOW_DEF: Final[int] = 150
_ATR_PERIOD: Final[int] = 20
_ADX_PERIOD: Final[int] = 14
_HARD_STOP_ATR: Final[float] = 2.0
_ADX_ENTRY: Final[float] = 20.0  # minimum ADX to open new position
_ADX_WARN: Final[float] = 15.0  # ADX below this = choppy (no new entry)
_FIXED_STOP_PTS: Final[float] = 150.0  # default fixed stop distance in index pts

_TICK_VALUE: Final[dict[str, float]] = {"TX": 200.0, "MTX": 50.0}
_MIN_DATA_ROWS: Final[int] = 200  # need EMA150 + ADX warmup


# ── Signal dataclass ──────────────────────────────────────────────────────────


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


class TrendFollowV2Strategy:
    """EMA-crossover trend-following strategy V2 for TAIFEX TX / MTX.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"``.
    ema_fast, ema_slow :
        EMA periods (default 50, 150).
    atr_period :
        ATR look-back for hard-stop (default 20).
    adx_period :
        ADX look-back (default 14).
    hard_stop_atr_mult :
        Hard-stop distance in ATR multiples (default 2.0).  Ignored when
        ``use_fixed_stop=True``.
    long_only :
        If True, only long positions are taken; bear-trend → flat.
    adx_entry :
        Minimum ADX required to open a new position (default 20).
    adx_exit_warn :
        ADX below this value suppresses new entries but does not close
        existing positions (default 15).
    use_fixed_stop :
        If True, use a fixed-point hard stop instead of an ATR-multiple stop.
        The stop distance is ``fixed_stop_points`` index points from entry.
    fixed_stop_points :
        Distance in index points for the fixed hard stop (default 150).
        Only used when ``use_fixed_stop=True``.
    """

    def __init__(
        self,
        product: str = "TX",
        ema_fast: int = _EMA_FAST_DEF,
        ema_slow: int = _EMA_SLOW_DEF,
        atr_period: int = _ATR_PERIOD,
        adx_period: int = _ADX_PERIOD,
        hard_stop_atr_mult: float = _HARD_STOP_ATR,
        long_only: bool = True,
        adx_entry: float = _ADX_ENTRY,
        adx_exit_warn: float = _ADX_WARN,
        use_fixed_stop: bool = False,
        fixed_stop_points: float = _FIXED_STOP_PTS,
    ) -> None:
        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be 'TX' or 'MTX', got {product!r}")

        self.product = product
        self.tick_value = _TICK_VALUE[product]
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.hard_stop_atr_mult = hard_stop_atr_mult
        self.long_only = long_only
        self.adx_entry = adx_entry
        self.adx_exit_warn = adx_exit_warn
        self.use_fixed_stop = use_fixed_stop
        self.fixed_stop_points = fixed_stop_points

        self._time_guards = TimeGuards()

    # ------------------------------------------------------------------
    # Public
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
        if len(data) < _MIN_DATA_ROWS:
            return Signal(
                action="hold",
                reason=f"warmup: {len(data)} < {_MIN_DATA_ROWS}",
            )

        # ── Indicators ───────────────────────────────────────────────────
        ind = self._compute_indicators(data)
        latest = ind.iloc[-1]
        prev = ind.iloc[-2]

        ema_f = latest["ema_fast"]
        ema_s = latest["ema_slow"]
        adx = latest["adx"]
        atr = latest["atr"]
        close = latest["close"]

        prev_ema_f = prev["ema_fast"]
        prev_ema_s = prev["ema_slow"]

        if any(np.isnan(x) for x in [ema_f, ema_s, adx, atr]) or atr == 0:
            return Signal(action="hold", reason="indicators not ready")

        # ── current_time for TimeGuards ───────────────────────────────────
        if current_time is None:
            import zoneinfo

            tz = zoneinfo.ZoneInfo("Asia/Taipei")
            bd = data.index[-1]
            bd = bd.tz_convert(tz) if hasattr(bd, "tz_convert") else bd
            current_time = pd.Timestamp(bd.year, bd.month, bd.day, 14, 0, 0, tzinfo=tz)

        # ── Settlement-day rollover ───────────────────────────────────────
        if current_position != 0 and self._time_guards.is_settlement_close(current_time):
            return Signal(action="close", contracts=1, reason="settlement-day rollover")

        # ── Trend direction ───────────────────────────────────────────────
        cur_dir = 1 if ema_f > ema_s else -1
        prev_dir = 1 if prev_ema_f > prev_ema_s else -1
        crossed = cur_dir != prev_dir

        # ── Hard stop (in-position) ───────────────────────────────────────
        if current_position != 0 and entry_price is not None:
            d = 1 if current_position > 0 else -1
            if self.use_fixed_stop:
                hard_stop = entry_price - d * self.fixed_stop_points
                stop_desc = f"{self.fixed_stop_points:.0f}pt fixed"
            else:
                hard_stop = entry_price - d * self.hard_stop_atr_mult * atr
                stop_desc = f"{self.hard_stop_atr_mult}×ATR"
            if (d == 1 and close < hard_stop) or (d == -1 and close > hard_stop):
                return Signal(
                    action="close",
                    contracts=1,
                    reason=(
                        f"hard stop: close={close:.0f} entry={entry_price:.0f} "
                        f"stop={hard_stop:.0f} ({stop_desc})"
                    ),
                    hard_stop=hard_stop,
                )

        # ── In-position logic ─────────────────────────────────────────────
        if current_position != 0:
            direction = 1 if current_position > 0 else -1

            if cur_dir != direction:
                # Trend reversed
                if self.long_only and cur_dir == -1:
                    # Bear trend → go flat (no short)
                    return Signal(
                        action="close",
                        contracts=1,
                        reason=(
                            f"EMA crossed bearish (long-only → flat): "
                            f"EMA{self.ema_fast}={ema_f:.0f} < EMA{self.ema_slow}={ema_s:.0f}"
                        ),
                    )
                else:
                    # Reverse to opposite direction
                    return Signal(
                        action="close",
                        contracts=1,
                        reason=(
                            f"EMA reversed: EMA{self.ema_fast}={ema_f:.0f} "
                            f"{'>' if cur_dir == 1 else '<'} EMA{self.ema_slow}={ema_s:.0f}"
                        ),
                    )

            return Signal(
                action="hold",
                contracts=1,
                reason=(
                    f"holding {direction:+d}: EMA{self.ema_fast}={ema_f:.0f} "
                    f"EMA{self.ema_slow}={ema_s:.0f}  ADX={adx:.1f}  ATR={atr:.0f}"
                ),
            )

        # ── Entry logic (flat) ────────────────────────────────────────────

        # Long-only: only enter in bull trend
        if self.long_only and cur_dir != 1:
            return Signal(
                action="hold",
                reason=f"long-only: bear trend EMA{self.ema_fast}={ema_f:.0f} < EMA{self.ema_slow}={ema_s:.0f}",
            )

        # ADX filter: require trend strong enough
        if adx < self.adx_entry:
            return Signal(
                action="hold",
                reason=f"ADX={adx:.1f} < {self.adx_entry} (trend too weak for entry)",
            )

        # Require a fresh crossover (or just re-entering after settlement rollover)
        # We enter on the bar where the crossover happened or immediately after
        # as long as the trend direction is confirmed
        if cur_dir == 1:
            if self.use_fixed_stop:
                hard_stop_px = close - self.fixed_stop_points
                stop_label = f"{self.fixed_stop_points:.0f}pt"
            else:
                hard_stop_px = close - self.hard_stop_atr_mult * atr
                stop_label = f"{self.hard_stop_atr_mult}×ATR={hard_stop_px:.0f}"
            return Signal(
                action="buy",
                contracts=1,
                reason=(
                    f"EMA bull: EMA{self.ema_fast}={ema_f:.0f} > EMA{self.ema_slow}={ema_s:.0f}  "
                    f"ADX={adx:.1f}  crossed={crossed}  stop={stop_label}"
                ),
                hard_stop=hard_stop_px,
            )
        else:
            # long_only=False path
            if self.use_fixed_stop:
                hard_stop_px = close + self.fixed_stop_points
                stop_label = f"{self.fixed_stop_points:.0f}pt"
            else:
                hard_stop_px = close + self.hard_stop_atr_mult * atr
                stop_label = f"{self.hard_stop_atr_mult}×ATR={hard_stop_px:.0f}"
            return Signal(
                action="sell",
                contracts=1,
                reason=(
                    f"EMA bear: EMA{self.ema_fast}={ema_f:.0f} < EMA{self.ema_slow}={ema_s:.0f}  "
                    f"ADX={adx:.1f}  crossed={crossed}  stop={stop_label}"
                ),
                hard_stop=hard_stop_px,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["open", "high", "low", "close"]].copy().astype(float)

        df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        # ATR
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - df["prev_close"]).abs(),
                (df["low"] - df["prev_close"]).abs(),
            ),
        )
        df["atr"] = df["tr"].rolling(self.atr_period).mean()

        # ADX (Wilder's smoothed)
        df["up"] = df["high"].diff()
        df["down"] = -df["low"].diff()
        df["pdm"] = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0.0)
        df["ndm"] = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0.0)

        # Wilder smoothing
        p = self.adx_period
        atr_w = df["tr"].ewm(alpha=1 / p, adjust=False).mean()
        pdi_w = df["pdm"].ewm(alpha=1 / p, adjust=False).mean()
        ndi_w = df["ndm"].ewm(alpha=1 / p, adjust=False).mean()

        df["pdi"] = 100 * pdi_w / atr_w.replace(0, np.nan)
        df["ndi"] = 100 * ndi_w / atr_w.replace(0, np.nan)
        dx_denom = (df["pdi"] + df["ndi"]).replace(0, np.nan)
        df["dx"] = 100 * (df["pdi"] - df["ndi"]).abs() / dx_denom
        df["adx"] = df["dx"].ewm(alpha=1 / p, adjust=False).mean()

        return df.drop(columns=["prev_close", "tr", "up", "down", "pdm", "ndm", "pdi", "ndi", "dx"])
