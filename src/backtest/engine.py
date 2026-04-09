"""V2b 回測引擎 — MXF / 小台指。

設計重點
--------
* Look-ahead 防護：signal 使用 bar[t-1] 資料切片，bar[t] open 執行
* Anti-Martingale：equity 梯度放大口數（由 V2bEngine ladder 決定）
* 成本：單邊 160 NTD（手續費 + 交易稅 + slippage）
* Pyramid：處理 V2bEngine 回傳的 "add" action（舊版 bug：僅處理 buy/close）
* MDD：使用 mark-to-market equity（含未實現損益）；舊版 bug 為僅記錄已實現 equity

Bugs fixed vs old engine (trading-agents-v2/src/backtest/engine.py)
--------------------------------------------------------------------
1. MDD understated: equity_curve now records realized_equity + unrealized_pnl
   at every close → MDD correctly captures floating losses during open positions.
2. Pyramid never fired: "add" action was not handled (fell through if/elif).
3. Cost: COST_PER_SIDE updated to 160 NTD (was 105 in old engine).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.v2b_engine import V2bEngine

logger = logging.getLogger(__name__)

# ── Cost constants ────────────────────────────────────────────────────────────
TICK_VALUE: float = 50.0      # NTD per index point, MXF / MTX
COST_PER_SIDE: float = 160.0   # NTD commission per side (手續費 + 交易稅 + slippage)
ROUND_TRIP: float = COST_PER_SIDE * 2
MTX_MARGIN: float = 119_250.0  # NTD original margin per MTX/MXF contract


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    contracts: int
    direction: int   # +1 long
    pnl_pts: float   # exit − entry per contract
    pnl_twd: float   # NTD after costs
    reason: str
    entry_reason: str = ""


@dataclass
class BacktestResult:
    equity_curve: pd.Series   # mark-to-market equity at each bar
    trades: list[TradeRecord]
    metrics: dict


# ── Engine ────────────────────────────────────────────────────────────────────


class BacktestEngine:
    """Bar-by-bar backtester for V2b MXF strategy.

    Parameters
    ----------
    strategy :
        V2bEngine instance.  Defaults to V2bEngine(product="MXF").
    initial_capital :
        Starting equity in NTD (default 350_000).
    exec_timing :
        ``"next_day_open"`` — execute at bar[t].open, signal from bar[t-1].
        ``"same_day_close"`` — execute at bar[t].close, signal from bar[t].
    ladder :
        Anti-martingale equity thresholds; forwarded to V2bEngine if strategy
        is not pre-configured.
    """

    def __init__(
        self,
        strategy: V2bEngine | None = None,
        initial_capital: float = 350_000.0,
        exec_timing: str = "next_day_open",
        ladder: list[dict] | None = None,
    ) -> None:
        self.strategy = strategy or V2bEngine(product="MXF")
        self.initial_capital = initial_capital
        self.exec_timing = exec_timing
        self.ladder = ladder

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        tsmc_signals: dict | None = None,
    ) -> BacktestResult:
        """Run backtest on daily OHLCV data.

        Parameters
        ----------
        data :
            Daily DataFrame with DatetimeIndex and columns
            ``open, high, low, close, volume``.
        tsmc_signals :
            Optional ``{date: TsmcSignal}`` mapping.

        Returns
        -------
        BacktestResult with mark-to-market equity_curve.
        """
        df = data.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        realized_equity: float = self.initial_capital
        position: int = 0
        entry_price: float | None = None
        entry_date: pd.Timestamp | None = None
        entry_reason: str = ""
        highest_high: float | None = None
        pyramided: bool = False

        equity_points: list[tuple] = []
        trades: list[TradeRecord] = []

        same_day = self.exec_timing == "same_day_close"

        for i in range(1, len(df)):
            today = df.index[i]
            row = df.iloc[i]
            close = float(row["close"])

            # Signal slice: same-day includes bar[i]; next-day excludes it
            data_slice = df.iloc[: i + 1] if same_day else df.iloc[:i]

            tsmc_sig = tsmc_signals.get(today.date()) if tsmc_signals else None

            sig = self.strategy.generate_signal(
                data=data_slice,
                current_position=position,
                entry_price=entry_price,
                equity=realized_equity,
                highest_high=highest_high,
                contracts=position,
                tsmc_signal=tsmc_sig,
            )

            exec_price = float(row["close"] if same_day else row["open"])

            # ── Close ──────────────────────────────────────────────
            if sig.action == "close" and position > 0 and entry_price is not None:
                pnl_pts = exec_price - entry_price
                pnl_twd = pnl_pts * position * TICK_VALUE - ROUND_TRIP * position
                realized_equity += pnl_twd
                trades.append(TradeRecord(
                    entry_date=str(entry_date.date()),
                    exit_date=str(today.date()),
                    entry_price=entry_price,
                    exit_price=exec_price,
                    contracts=position,
                    direction=1,
                    pnl_pts=pnl_pts,
                    pnl_twd=pnl_twd,
                    reason=sig.reason,
                    entry_reason=entry_reason,
                ))
                logger.debug(
                    "CLOSE %s px=%.0f pnl=%.0f eq=%.0f",
                    today.date(), exec_price, pnl_twd, realized_equity,
                )
                position = 0
                entry_price = None
                entry_date = None
                entry_reason = ""
                highest_high = None
                pyramided = False

            # ── Buy (initial entry) ────────────────────────────────
            elif sig.action == "buy" and position == 0:
                n = sig.contracts if sig.contracts > 0 else 1
                margin_required = n * MTX_MARGIN
                if margin_required > realized_equity:
                    logger.warning(
                        "⚠️ margin exceeded: %d口需要 %,.0f 但 equity=%,.0f  [%s]",
                        n, margin_required, realized_equity, today.date(),
                    )
                realized_equity -= COST_PER_SIDE * n
                position = n
                entry_price = exec_price
                entry_date = today
                entry_reason = sig.reason
                highest_high = exec_price
                pyramided = False
                logger.debug(
                    "BUY   %s px=%.0f n=%d eq=%.0f",
                    today.date(), exec_price, n, realized_equity,
                )

            # ── Add (pyramid scale-in) ─────────────────────────────
            elif sig.action == "add" and position > 0 and not pyramided and entry_price is not None:
                add_n = sig.contracts if sig.contracts > 0 else 1
                total = position + add_n
                margin_required = total * MTX_MARGIN
                if margin_required > realized_equity:
                    logger.warning(
                        "⚠️ margin exceeded: %d口需要 %,.0f 但 equity=%,.0f  [%s]",
                        total, margin_required, realized_equity, today.date(),
                    )
                entry_price = (entry_price * position + exec_price * add_n) / total
                realized_equity -= COST_PER_SIDE * add_n
                position = total
                pyramided = True
                logger.debug(
                    "ADD   %s px=%.0f +%d→%d eq=%.0f",
                    today.date(), exec_price, add_n, total, realized_equity,
                )

            # ── Update trailing high ───────────────────────────────
            if position > 0:
                if highest_high is None or close > highest_high:
                    highest_high = close

            # ── Mark-to-market equity (key fix: includes unrealized PnL) ──
            if position > 0 and entry_price is not None:
                unrealized = (close - entry_price) * position * TICK_VALUE
                mtm_equity = realized_equity + unrealized
            else:
                mtm_equity = realized_equity

            equity_points.append((today, mtm_equity))

        # ── Force-close remaining position at last close ──────────────
        if position > 0 and entry_price is not None:
            last_row = df.iloc[-1]
            last_date = df.index[-1]
            exit_price = float(last_row["close"])
            pnl_pts = exit_price - entry_price
            pnl_twd = pnl_pts * position * TICK_VALUE - ROUND_TRIP * position
            realized_equity += pnl_twd
            trades.append(TradeRecord(
                entry_date=str(entry_date.date()),
                exit_date=str(last_date.date()),
                entry_price=entry_price,
                exit_price=exit_price,
                contracts=position,
                direction=1,
                pnl_pts=pnl_pts,
                pnl_twd=pnl_twd,
                reason="end-of-backtest liquidation",
                entry_reason=entry_reason,
            ))
            if equity_points and equity_points[-1][0] == last_date:
                equity_points[-1] = (last_date, realized_equity)
            else:
                equity_points.append((last_date, realized_equity))

        ec = pd.Series(
            [e for _, e in equity_points],
            index=pd.DatetimeIndex([d for d, _ in equity_points]),
            name="equity_twd",
        )
        metrics = _compute_metrics(ec, trades, self.initial_capital)
        return BacktestResult(equity_curve=ec, trades=trades, metrics=metrics)


# ── Metrics ───────────────────────────────────────────────────────────────────


def _compute_metrics(
    ec: pd.Series,
    trades: list[TradeRecord],
    initial_capital: float,
) -> dict:
    if ec.empty or not trades:
        return {}

    years = (ec.index[-1] - ec.index[0]).days / 365.25
    final = float(ec.iloc[-1])
    cagr = ((final / initial_capital) ** (1 / max(years, 1e-9)) - 1) * 100

    roll_max = ec.cummax()
    drawdown = (ec - roll_max) / roll_max
    mdd = float(drawdown.min()) * 100

    daily_ret = ec.pct_change().dropna()
    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        if daily_ret.std() > 0
        else np.nan
    )
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    pnls = [t.pnl_twd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "CAGR_%": round(cagr, 2),
        "MDD_%": round(mdd, 2),
        "Sharpe": round(float(sharpe), 3) if not np.isnan(sharpe) else "n/a",
        "Calmar": round(float(calmar), 3) if not np.isnan(calmar) else "n/a",
        "Win_Rate_%": round(win_rate, 2),
        "Profit_Factor": round(float(profit_factor), 3) if not np.isnan(profit_factor) else "n/a",
        "Total_Trades": len(trades),
        "Final_Equity": round(final, 0),
    }
