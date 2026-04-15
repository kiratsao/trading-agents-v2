"""V2b 策略引擎 — EMA 金叉 + N日確認 + 移動止損 + 加碼 + 反馬丁格爾。

特性
----
* 進場：EMA(fast) > EMA(slow) 且連續 N 日確認 + ADX(14) > threshold，空手時才建倉
* 止損：追蹤高點止損 close < highest_high − trail_atr_mult×ATR
* 離場：EMA 死叉 或 結算日 或 追蹤止損（ADX 不影響出場）
* 加碼：浮盈 ≥ pyramid_atr_mult×ATR 時追加口數（pyramid_size_fraction 規模）
* 口數：反馬丁格爾 scale ladder
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.strategy.common.indicators import adx as calc_adx
from src.strategy.common.indicators import atr as calc_atr
from src.strategy.common.indicators import ema as calc_ema

logger = logging.getLogger(__name__)

_TICK_VALUE: dict[str, float] = {"MTX": 50.0, "MXF": 50.0, "TX": 200.0, "TXF": 200.0}


@dataclass
class Signal:
    action: str  # "buy" | "close" | "hold" | "add"
    contracts: int
    reason: str
    stop_loss: float | None = None

    def __str__(self) -> str:
        return f"Signal(action={self.action!r}, contracts={self.contracts}, reason={self.reason!r})"


_MTX_MARGIN: float = 119_250.0  # NTD original margin per MTX/MXF contract


def _anti_martingale_contracts(
    equity: float,
    ladder: list[dict] | None = None,
) -> int:
    """Anti-martingale scale ladder with margin safety cap.

    Default: 35萬→2口, 48萬→3口, 60萬→4口

    Final contracts = min(ladder_contracts, floor(equity / MTX_MARGIN))
    so that position never exceeds what the account margin can support.
    """
    if not ladder:
        ladder = [
            {"equity": 350_000, "contracts": 2},
            {"equity": 480_000, "contracts": 3},
            {"equity": 600_000, "contracts": 4},
        ]
    n = 1
    for entry in sorted(ladder, key=lambda x: x["equity"]):
        if equity >= entry["equity"]:
            n = entry["contracts"]

    # Margin safety cap: never exceed floor(equity / margin_per_contract)
    import math
    max_by_margin = math.floor(equity / _MTX_MARGIN)
    return min(n, max(1, max_by_margin))


def _third_wednesday(dt_date: date) -> date:
    """Return the third Wednesday of the given month."""
    import calendar

    y, m = dt_date.year, dt_date.month
    c = calendar.monthcalendar(y, m)
    wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
    return date(y, m, wednesdays[2])


def _is_settlement_day(ts: pd.Timestamp) -> bool:
    """True if *ts* falls on the 3rd Wednesday of its month (TAIFEX monthly expiry)."""
    return ts.date() == _third_wednesday(ts.date())


class V2bEngine:
    """EMA crossover with N-day confirmation, trailing stop, pyramid, anti-martingale.

    Parameters
    ----------
    product :
        ``"MTX"`` (default) or ``"TX"`` or ``"MXF"``.
    ema_fast, ema_slow :
        EMA periods (default 30, 100).
    atr_period :
        ATR period (default 14).
    confirm_days :
        Consecutive bull bars needed before entry (default 3).
    trail_atr_mult :
        Trailing stop = highest_high - trail_atr_mult × ATR (default 2.0).
    pyramid_atr_mult :
        Scale-in threshold: float_profit ≥ pyramid_atr_mult × ATR (default 1.0).
    pyramid_size_fraction :
        Scale-in size as fraction of initial position (default 0.5, rounds to ≥1).
    adx_threshold :
        Minimum ADX(14) for entry (default 25). Set to 0 to disable.
    ladder :
        Anti-martingale equity thresholds [{equity, contracts}, ...].
    exit_mode :
        ``"trailing"`` (default) or ``"chandelier"``.
    """

    def __init__(
        self,
        product: str = "MTX",
        ema_fast: int = 30,
        ema_slow: int = 100,
        atr_period: int = 14,
        confirm_days: int = 3,
        trail_atr_mult: float = 2.0,
        pyramid_atr_mult: float = 1.0,
        pyramid_size_fraction: float = 0.5,
        adx_threshold: float = 25.0,
        ladder: list[dict] | None = None,
        exit_mode: str = "trailing",
        chandelier_lookback: int = 20,
        chandelier_mult: float = 3.0,
    ) -> None:
        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be TX/MTX/MXF, got {product!r}")
        if exit_mode not in ("trailing", "chandelier"):
            raise ValueError(f"exit_mode must be 'trailing' or 'chandelier', got {exit_mode!r}")

        self.product = product
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.confirm_days = confirm_days
        self.trail_atr_mult = trail_atr_mult
        self.pyramid_atr_mult = pyramid_atr_mult
        self.pyramid_size_fraction = pyramid_size_fraction
        self.adx_threshold = adx_threshold
        self.ladder = ladder
        self.exit_mode = exit_mode
        self.chandelier_lookback = chandelier_lookback
        self.chandelier_mult = chandelier_mult

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int = 0,
        entry_price: float | None = None,
        equity: float = 350_000.0,
        highest_high: float | None = None,
        contracts: int = 0,
        tsmc_signal=None,
    ) -> Signal:
        """Compute trading signal for the current bar.

        Parameters
        ----------
        data :
            OHLCV DataFrame up to and including the *current* bar.
            Must have columns ``open, high, low, close, volume``
            and a DatetimeIndex.
        current_position :
            Signed position (>0 = long, 0 = flat).
        entry_price :
            Average entry price (None if flat).
        equity :
            Current account equity in NTD.
        highest_high :
            Running highest close since entry (for trailing stop).
        contracts :
            Current number of open contracts.
        tsmc_signal :
            Optional TsmcSignal for early-entry logic.
        """
        min_bars = max(self.ema_slow, self.atr_period) + self.confirm_days + 5
        if len(data) < min_bars:
            logger.debug("warmup: %d bars (need ≥%d)", len(data), min_bars)
            return Signal("hold", 0, f"warmup: {len(data)} bars")

        ind = self._compute_indicators(data)
        latest = ind.iloc[-1]
        close = float(latest["close"])
        ema_f = float(latest["ema_fast"])
        ema_s = float(latest["ema_slow"])
        atr_v = float(latest["atr"])
        adx_v = float(latest["adx"]) if "adx" in latest.index else 0.0
        bull_streak = int(latest["bull_streak"])
        cur_ts = data.index[-1]

        if not (ema_f > 0 and ema_s > 0 and atr_v > 0):
            return Signal("hold", 0, "indicators not ready")

        n_contracts = _anti_martingale_contracts(equity, self.ladder)

        # ── Settlement day force-close (only affects existing positions) ──
        if current_position > 0 and _is_settlement_day(cur_ts):
            return Signal("close", contracts or current_position, "settlement-day force close")
        # position == 0 on settlement day → fall through to normal entry logic

        # ── In position: check exits ────────────────────────────────
        if current_position > 0 and entry_price is not None:
            hh = highest_high if highest_high is not None else close
            hh = max(hh, close)

            # Trailing / chandelier stop
            if self.exit_mode == "chandelier":
                lookback_high = float(data["high"].iloc[-self.chandelier_lookback :].max())
                stop = lookback_high - self.chandelier_mult * atr_v
                if close < stop:
                    tsmc_note = (
                        " [tsmc bearish tighten]"
                        if (tsmc_signal and getattr(tsmc_signal, "direction_bias", "") == "bearish")
                        else ""
                    )
                    return Signal(
                        "close",
                        contracts or current_position,
                        f"chandelier stop: close={close:.0f}{tsmc_note}",
                        stop_loss=stop,
                    )
            else:
                trail_stop = hh - self.trail_atr_mult * atr_v
                if close < trail_stop:
                    tsmc_note = (
                        " [tsmc bearish tighten]"
                        if (tsmc_signal and getattr(tsmc_signal, "direction_bias", "") == "bearish")
                        else ""
                    )
                    return Signal(
                        "close",
                        contracts or current_position,
                        f"trailing stop: close={close:.0f} < highest={hh:.0f}{tsmc_note}",
                        stop_loss=trail_stop,
                    )

            # Death cross
            if ema_f < ema_s:
                return Signal(
                    "close",
                    contracts or current_position,
                    f"death cross: EMA{self.ema_fast}={ema_f:.0f} < EMA{self.ema_slow}={ema_s:.0f}",
                )

            # Pyramid scale-in (per-contract points, same unit as ATR)
            float_profit = close - entry_price
            add_threshold = self.pyramid_atr_mult * atr_v
            if float_profit >= add_threshold:
                import math as _math
                add_n = max(1, round((contracts or current_position) * self.pyramid_size_fraction))
                # Margin safety cap: total position must not exceed floor(equity / margin)
                cur_contracts = contracts or current_position
                max_total = max(1, _math.floor(equity / _MTX_MARGIN))
                add_n = min(add_n, max(0, max_total - cur_contracts))
                if add_n <= 0:
                    return Signal(
                        "hold",
                        cur_contracts,
                        f"pyramid skipped: margin cap {max_total}口 already reached",
                        stop_loss=hh - self.trail_atr_mult * atr_v,
                    )
                return Signal(
                    "add",
                    add_n,
                    f"pyramid: float_profit={float_profit:.0f}, add={add_n}",
                    stop_loss=hh - self.trail_atr_mult * atr_v,
                )

            # Hold
            trail_stop = hh - self.trail_atr_mult * atr_v
            return Signal(
                "hold",
                contracts or current_position,
                "holding position",
                stop_loss=trail_stop,
            )

        # ── Flat: check entry ───────────────────────────────────────
        golden = ema_f > ema_s
        if golden and bull_streak >= self.confirm_days:
            # ADX regime filter: only enter when trend is strong enough
            if self.adx_threshold > 0 and adx_v < self.adx_threshold:
                return Signal(
                    "hold", 0,
                    f"ADX too low — no trend (ADX={adx_v:.1f} < {self.adx_threshold})",
                )
            trail_stop = close - self.trail_atr_mult * atr_v
            return Signal(
                "buy",
                n_contracts,
                (
                    f"golden cross + {bull_streak}-day confirmation: "
                    f"EMA{self.ema_fast}={ema_f:.0f} > EMA{self.ema_slow}={ema_s:.0f}"
                    f" ADX={adx_v:.0f}"
                ),
                stop_loss=trail_stop,
            )

        # TSMC early entry (1 fewer confirm day)
        if (
            tsmc_signal is not None
            and getattr(tsmc_signal, "direction_bias", "") == "bullish"
            and golden
            and bull_streak >= max(1, self.confirm_days - 1)
        ):
            trail_stop = close - self.trail_atr_mult * atr_v
            return Signal(
                "buy",
                n_contracts,
                (
                    f"tsmc early entry: bull_streak={bull_streak} "
                    f"(needed {self.confirm_days}-1={self.confirm_days - 1}-day confirm), "
                    f"tsmc={getattr(tsmc_signal, 'direction_bias', '')} "
                    f"conf={getattr(tsmc_signal, 'confidence', 0):.2f}"
                ),
                stop_loss=trail_stop,
            )

        cross_desc = "bullish" if golden else "bearish"
        return Signal("hold", 0, f"no entry signal ({cross_desc})")

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = df.copy()
        ind["ema_fast"] = calc_ema(df["close"], self.ema_fast)
        ind["ema_slow"] = calc_ema(df["close"], self.ema_slow)
        ind["atr"] = calc_atr(df, self.atr_period)
        ind["adx"] = calc_adx(df, 14)

        # bull_streak: consecutive bars with ema_fast > ema_slow
        bull = (ind["ema_fast"] > ind["ema_slow"]).astype(int)
        streaks = []
        s = 0
        for v in bull:
            s = s + 1 if v else 0
            streaks.append(s)
        ind["bull_streak"] = streaks
        return ind
