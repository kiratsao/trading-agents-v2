"""TW Futures backtester — margin-aware event-driven simulation for TAIFEX futures.

Key differences from the US equity backtester
----------------------------------------------
* **Margin accounting** — tracks initial margin (開倉保證金) and maintenance
  margin (維持保證金) per contract; forces close on margin call.
* **Daily mark-to-market** — unrealised P&L credited/debited at each day's
  close (TAIFEX daily settlement).
* **Transaction costs** — commission (手續費, fixed per contract) + transaction
  tax (交易稅, rate on notional) + slippage (assumed adverse fill in ticks).
* **Intraday stop check** — ATR trailing stop validated against today's
  high/low *before* the end-of-day signal; exit price = stop level.
* **P&L unit** — TWD; tick_value = 200 (TX) or 50 (MTX).

Simulation flow per trading day
---------------------------------
1. Check intraday trailing stop against today's high/low → force exit at stop
   price if triggered (MTM from prev_close to stop price, then deduct costs).
2. Mark-to-market remaining position to today's close.
3. Check maintenance margin adequacy → force close if margin call.
4. Call ``strategy.generate_signal(data_slice, ...)`` with data up to today.
5. Execute returned signal at today's close price.
6. Update trailing stop from signal's ``stop_loss`` field.
7. Append today's equity to the curve.

Example
-------
>>> from tw_futures.strategies.swing.breakout_swing import BreakoutSwingStrategy
>>> strategy = BreakoutSwingStrategy(product="TX")
>>> bt = FuturesBacktester()
>>> result = bt.run(strategy, df, initial_capital=2_000_000, product="TX")
>>> print(result.metrics)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from tw_futures.backtester.bias_check import check_signal_date

logger = logging.getLogger(__name__)


# ── Product constants ─────────────────────────────────────────────────────────

_TICK_VALUE: dict[str, float] = {
    "TX": 200.0,  # 大台指  TWD per index point
    "MTX": 50.0,  # 小台指  TWD per index point
}

# TAIFEX-published margins (TWD per contract, effective 2026-04-01)
_DEFAULT_INITIAL_MARGIN: dict[str, float] = {
    "TX": 477_000.0,
    "MTX": 119_250.0,
}
_DEFAULT_MAINTENANCE_MARGIN: dict[str, float] = {
    "TX": 366_000.0,
    "MTX": 91_500.0,
}


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class FuturesBacktestResult:
    """All outputs from a completed futures backtest.

    Attributes
    ----------
    equity_curve :
        Daily account equity in TWD.
        Index: ``pd.DatetimeIndex`` (Asia/Taipei), sorted ascending.
        Name: ``"equity_twd"``.
    trades :
        One dict per *closed* trade.  Keys:
        ``entry_date, exit_date, direction, contracts, entry_price,
        exit_price, gross_pnl, entry_cost, exit_cost, net_pnl,
        holding_days, exit_reason``.
    metrics :
        Performance summary dict (see :meth:`FuturesBacktester.run`).
    margin_calls :
        Timestamps where a margin call forced an involuntary position close.
    """

    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    margin_calls: list[pd.Timestamp] = field(default_factory=list)


# ── Main class ────────────────────────────────────────────────────────────────


class FuturesBacktester:
    """Margin-aware event-driven backtester for TAIFEX TX / MTX futures.

    Parameters
    ----------
    initial_margin_per_contract :
        Override the per-product default initial margin (TWD per contract).
        ``None`` → use TAIFEX 2026-04-01 rates (TX: 477,000; MTX: 119,250).
    maintenance_margin_per_contract :
        Override the per-product default maintenance margin (TWD per contract).
        ``None`` → use TAIFEX 2026-04-01 rates (TX: 366,000; MTX: 91,500).
    commission_per_contract :
        One-way commission in TWD per contract (default 100).
    tax_rate :
        Transaction tax rate on notional value (default 0.00002 = 0.002 %).
        Applied once per side per trade.
    slippage_ticks :
        Adverse fill in index points per contract per side (default 2).
        Represents assumed price slippage away from the theoretical fill price.
    """

    def __init__(
        self,
        initial_margin_per_contract: float | None = None,
        maintenance_margin_per_contract: float | None = None,
        commission_per_contract: float = 100.0,
        tax_rate: float = 0.00002,
        slippage_ticks: int = 2,
    ) -> None:
        self._override_initial = initial_margin_per_contract
        self._override_maintenance = maintenance_margin_per_contract
        self.commission_per_contract = commission_per_contract
        self.tax_rate = tax_rate
        self.slippage_ticks = slippage_ticks

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        strategy,
        data: pd.DataFrame,
        initial_capital: float = 2_000_000.0,
        product: str = "TX",
    ) -> FuturesBacktestResult:
        """Run a full futures backtest.

        Parameters
        ----------
        strategy :
            Object implementing
            ``generate_signal(data, current_position, entry_price,
            entry_date, equity) -> Signal``.
        data :
            Daily OHLCV DataFrame with columns ``open, high, low, close,
            volume``.  Index: ``pd.DatetimeIndex`` sorted ascending.
            Include at least 50 rows of warmup before the target trading
            period so strategy indicators (SMA50, Donchian, ATR) are
            fully initialised before the first potential signal.
        initial_capital :
            Starting account equity in TWD (default 2,000,000).
        product :
            ``"TX"`` (大台, 200 TWD/pt) or ``"MTX"`` (小台, 50 TWD/pt).

        Returns
        -------
        FuturesBacktestResult
            Contains ``equity_curve``, ``trades``, ``metrics``,
            ``margin_calls``.

        Raises
        ------
        ValueError
            If *product* is not ``"TX"`` or ``"MTX"``.
        """
        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be 'TX' or 'MTX', got {product!r}")

        tick_value = _TICK_VALUE[product]
        init_margin = self._override_initial or _DEFAULT_INITIAL_MARGIN[product]
        maint_margin = self._override_maintenance or _DEFAULT_MAINTENANCE_MARGIN[product]

        logger.info(
            "FuturesBacktester.run  product=%s  rows=%d  [%s → %s]  "
            "capital=%.0f  init_margin=%.0f  maint_margin=%.0f",
            product,
            len(data),
            data.index[0].date() if len(data) else "?",
            data.index[-1].date() if len(data) else "?",
            initial_capital,
            init_margin,
            maint_margin,
        )

        # ── Simulation state ──────────────────────────────────────────────
        account_value: float = float(initial_capital)
        position: int = 0  # net contracts (+long, −short)
        entry_price: float | None = None
        entry_date: pd.Timestamp | None = None
        entry_cost: float = 0.0  # cost paid to open current position
        current_stop: float | None = None
        current_hard: float | None = None  # fixed hard stop for scaled strategy
        prev_close: float | None = None
        peak_leverage: float = 0.0  # max(abs(pos)×init_margin / equity)

        equity_history: list[tuple[pd.Timestamp, float]] = []
        trades: list[dict] = []
        margin_calls: list[pd.Timestamp] = []

        # ── Helpers (closures over simulation state) ──────────────────────

        def _cost(price: float, n: int) -> float:
            """One-way transaction cost for *n* contracts at *price*."""
            notional = price * tick_value * n
            return (
                self.commission_per_contract * n
                + notional * self.tax_rate
                + self.slippage_ticks * tick_value * n
            )

        def _close(
            exit_price: float,
            exit_date: pd.Timestamp,
            reason: str,
        ) -> None:
            nonlocal position, entry_price, entry_date, current_stop
            nonlocal account_value, entry_cost

            if position == 0:
                return

            direction = 1 if position > 0 else -1
            n = abs(position)
            ep = entry_price if entry_price is not None else exit_price
            gross_pnl = direction * (exit_price - ep) * n * tick_value
            ec = _cost(exit_price, n)
            net_pnl = gross_pnl - entry_cost - ec

            if entry_date is not None:
                mask = (data.index >= entry_date) & (data.index <= exit_date)
                holding_days = int(mask.sum())
            else:
                holding_days = 0

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "direction": direction,
                    "contracts": n,
                    "entry_price": ep,
                    "exit_price": exit_price,
                    "gross_pnl": round(gross_pnl, 0),
                    "entry_cost": round(entry_cost, 0),
                    "exit_cost": round(ec, 0),
                    "net_pnl": round(net_pnl, 0),
                    "holding_days": holding_days,
                    "exit_reason": reason,
                }
            )

            account_value -= ec
            position = 0
            entry_price = None
            entry_date = None
            current_stop = None
            entry_cost = 0.0

            logger.debug(
                "CLOSE %s %+d @ %.0f  gross=%.0f  net=%.0f  %s",
                exit_date.date(),
                direction * n,
                exit_price,
                gross_pnl,
                net_pnl,
                reason,
            )

        def _open(
            price: float,
            n: int,
            direction: int,
            open_date: pd.Timestamp,
        ) -> bool:
            """Open *n* contracts in *direction* (+1 long, −1 short).

            Returns ``False`` and logs a warning if insufficient margin.
            """
            nonlocal position, entry_price, entry_date, current_stop
            nonlocal account_value, entry_cost, current_hard

            if n <= 0:
                return False

            ec = _cost(price, n)
            margin_needed = n * init_margin

            if account_value - ec < margin_needed:
                logger.warning(
                    "Insufficient margin: skip %d %s on %s  (account=%.0f  need=%.0f)",
                    n,
                    product,
                    open_date.date(),
                    account_value,
                    margin_needed + ec,
                )
                return False

            account_value -= ec
            position = direction * n
            entry_price = price
            entry_date = open_date
            current_stop = None
            current_hard = None
            entry_cost = ec

            logger.debug(
                "OPEN %s %+d @ %.0f  cost=%.0f",
                open_date.date(),
                position,
                price,
                ec,
            )
            return True

        def _add(
            price: float,
            n_add: int,
            open_date: pd.Timestamp,
        ) -> bool:
            """Add *n_add* contracts to the existing position (same direction).

            Updates ``entry_price`` to the weighted average and accumulates
            ``entry_cost``.  Returns ``False`` if insufficient margin.
            """
            nonlocal position, entry_price, account_value, entry_cost

            if n_add <= 0 or position == 0:
                return False

            direction = 1 if position > 0 else -1
            ec = _cost(price, n_add)
            total_n = abs(position) + n_add
            margin_needed = total_n * init_margin

            if account_value - ec < margin_needed:
                logger.warning(
                    "Insufficient margin for add: skip %d %s on %s  (account=%.0f  need=%.0f)",
                    n_add,
                    product,
                    open_date.date(),
                    account_value,
                    margin_needed + ec,
                )
                return False

            # Weighted average entry price
            old_ep = entry_price if entry_price is not None else price
            old_n = abs(position)
            entry_price = (old_ep * old_n + price * n_add) / total_n
            account_value -= ec
            position += direction * n_add
            entry_cost += ec

            logger.debug(
                "ADD %s +%d @ %.0f  new_pos=%+d  avg_ep=%.0f  cost=%.0f",
                open_date.date(),
                n_add,
                price,
                position,
                entry_price,
                ec,
            )
            return True

        def _reduce_partial(
            price: float,
            n_reduce: int,
            close_date: pd.Timestamp,
            reason: str,
        ) -> None:
            """Partially close *n_reduce* contracts (lock partial profits).

            Books a trade record for the reduced portion. Remaining position
            continues with the same ``entry_price`` and ``entry_date``.
            """
            nonlocal position, account_value, entry_cost

            if position == 0:
                return
            direction = 1 if position > 0 else -1
            n_close = min(n_reduce, abs(position))
            if n_close <= 0:
                return

            ep = entry_price if entry_price is not None else price
            gross_pnl = direction * (price - ep) * n_close * tick_value
            ec = _cost(price, n_close)

            # Proportional share of accumulated entry cost
            prop_entry_cost = entry_cost * (n_close / abs(position)) if abs(position) > 0 else 0.0
            net_pnl = gross_pnl - prop_entry_cost - ec

            if entry_date is not None:
                mask = (data.index >= entry_date) & (data.index <= close_date)
                holding_days = int(mask.sum())
            else:
                holding_days = 0

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": close_date,
                    "direction": direction,
                    "contracts": n_close,
                    "entry_price": ep,
                    "exit_price": price,
                    "gross_pnl": round(gross_pnl, 0),
                    "entry_cost": round(prop_entry_cost, 0),
                    "exit_cost": round(ec, 0),
                    "net_pnl": round(net_pnl, 0),
                    "holding_days": holding_days,
                    "exit_reason": reason + " [partial]",
                }
            )

            account_value -= ec
            position -= direction * n_close
            entry_cost -= prop_entry_cost

            logger.debug(
                "REDUCE %s -%d @ %.0f  net=%.0f  remaining=%+d  %s",
                close_date.date(),
                n_close,
                price,
                net_pnl,
                position,
                reason,
            )

        # ── Day loop ──────────────────────────────────────────────────────

        for i in range(len(data)):
            row = data.iloc[i]
            today = data.index[i]
            today_high = float(row["high"])
            today_low = float(row["low"])
            today_close = float(row["close"])

            # ── 1. Intraday stop checks (trailing + hard) ──────────────
            if position != 0:
                direction = 1 if position > 0 else -1

                # ATR trailing stop
                if current_stop is not None:
                    stop_hit = (direction == 1 and today_low <= current_stop) or (
                        direction == -1 and today_high >= current_stop
                    )
                    if stop_hit:
                        stop_price = current_stop
                        if prev_close is not None:
                            account_value += (
                                direction * (stop_price - prev_close) * abs(position) * tick_value
                            )
                        _close(stop_price, today, "intraday trailing stop")
                        equity_history.append((today, account_value))
                        prev_close = today_close
                        continue

                # Fixed hard stop (used by scaled strategy)
                if current_hard is not None:
                    hard_hit = (direction == 1 and today_low <= current_hard) or (
                        direction == -1 and today_high >= current_hard
                    )
                    if hard_hit:
                        hard_price = current_hard
                        if prev_close is not None:
                            account_value += (
                                direction * (hard_price - prev_close) * abs(position) * tick_value
                            )
                        _close(hard_price, today, "hard stop")
                        equity_history.append((today, account_value))
                        prev_close = today_close
                        continue

            # ── 2. Daily MTM to close ─────────────────────────────────
            if position != 0 and prev_close is not None:
                direction = 1 if position > 0 else -1
                account_value += direction * (today_close - prev_close) * abs(position) * tick_value

            # ── 3. Margin call check ───────────────────────────────────
            if position != 0:
                maint_req = abs(position) * maint_margin
                if account_value < maint_req:
                    logger.warning(
                        "Margin call on %s  account=%.0f < req=%.0f",
                        today.date(),
                        account_value,
                        maint_req,
                    )
                    margin_calls.append(today)
                    _close(today_close, today, "margin call")
                    equity_history.append((today, account_value))
                    prev_close = today_close
                    continue

            # ── 4. Strategy signal ────────────────────────────────────
            data_slice = data.iloc[: i + 1]
            check_signal_date(data_slice, today)  # invariant — should never raise

            try:
                sig = strategy.generate_signal(
                    data=data_slice,
                    current_position=position,
                    entry_price=entry_price,
                    entry_date=entry_date,
                    equity=account_value,
                )
            except NotImplementedError:
                equity_history.append((today, account_value))
                prev_close = today_close
                continue
            except Exception as exc:
                logger.error(
                    "generate_signal raised on %s: %s — skipping bar",
                    today.date(),
                    exc,
                )
                equity_history.append((today, account_value))
                prev_close = today_close
                continue

            # ── 5. Execute signal ─────────────────────────────────────
            action = sig.action
            n_contracts = getattr(sig, "contracts", 0)

            if action == "close" and position != 0:
                _close(today_close, today, sig.reason or "strategy close")

            elif action == "buy":
                if position < 0:
                    _close(today_close, today, f"reverse to long — {sig.reason}")
                if position == 0:
                    _open(today_close, n_contracts, direction=1, open_date=today)

            elif action == "sell":
                if position > 0:
                    _close(today_close, today, f"reverse to short — {sig.reason}")
                if position == 0:
                    _open(today_close, n_contracts, direction=-1, open_date=today)

            elif action == "add" and position != 0:
                # Add to existing position (pyramid scale-in)
                _add(today_close, n_contracts, today)

            elif action == "reduce" and position != 0:
                # Partial close (pyramid profit-taking)
                _reduce_partial(today_close, n_contracts, today, sig.reason or "pyramid reduce")

            # "hold" → fall through to stop update

            # ── 6. Update trailing stop from signal ───────────────────
            # Stops can only tighten: long stop moves up, short stop moves down.
            if sig.stop_loss is not None and position != 0:
                direction = 1 if position > 0 else -1
                if current_stop is None:
                    current_stop = sig.stop_loss
                elif direction == 1:
                    current_stop = max(current_stop, sig.stop_loss)
                else:
                    current_stop = min(current_stop, sig.stop_loss)

            # Hard stop (only update once at open, never move)
            if getattr(sig, "hard_stop", None) is not None and current_hard is None:
                current_hard = sig.hard_stop

            # ── Peak leverage tracking ─────────────────────────────────
            if position != 0 and account_value > 0:
                lev = abs(position) * init_margin / account_value
                if lev > peak_leverage:
                    peak_leverage = lev

            equity_history.append((today, account_value))
            prev_close = today_close

        # ── Force-close open position at end of data ──────────────────────
        if position != 0:
            last_close = float(data.iloc[-1]["close"])
            last_date = data.index[-1]
            _close(last_close, last_date, "end of backtest")
            if equity_history and equity_history[-1][0] == last_date:
                equity_history[-1] = (last_date, account_value)
            else:
                equity_history.append((last_date, account_value))

        # ── Build result ───────────────────────────────────────────────────
        equity_curve = pd.Series(
            {ts: val for ts, val in equity_history},
            name="equity_twd",
            dtype=float,
        )
        equity_curve.index.name = "date"

        metrics = _compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            data=data,
            initial_capital=initial_capital,
            margin_call_count=len(margin_calls),
            peak_leverage=peak_leverage,
        )

        logger.info(
            "Backtest complete  product=%s  trades=%d  "
            "total_return=%.2f%%  CAGR=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%",
            product,
            len(trades),
            metrics.get("Total Return (%)", float("nan")),
            metrics.get("CAGR (%)", float("nan")),
            metrics.get("Sharpe Ratio", float("nan")),
            metrics.get("Max Drawdown (%)", float("nan")),
        )

        return FuturesBacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            margin_calls=margin_calls,
        )

    # ── Intraday (15-min bar) backtester ─────────────────────────────────────

    def run_intraday(
        self,
        strategy,
        data_15min: pd.DataFrame,
        initial_capital: float = 2_000_000.0,
        product: str = "TX",
    ) -> FuturesBacktestResult:
        """Run a bar-by-bar intraday backtest on 15-minute OHLCV data.

        Key differences from :meth:`run`
        ---------------------------------
        * Iterates every 15-min bar, not every day.
        * Forces all positions flat at 13:30 each day (no overnight holds).
        * Applies :class:`~tw_futures.strategies.common.time_guards.TimeGuards`
          (no-trade zones, Friday close, settlement-day close).
        * Calls ``strategy.generate_signal(data_slice, position, entry_price,
          equity, current_time)`` — note the 5th positional arg is a timestamp,
          not entry_date.
        * Equity curve is recorded once per bar (not once per day).

        Parameters
        ----------
        strategy :
            Object implementing
            ``generate_signal(data_15min, current_position, entry_price,
            equity, current_time) -> Signal``.
        data_15min :
            15-minute OHLCV DataFrame.  Index must be tz-aware
            (Asia/Taipei).  Must contain ``open, high, low, close``.
        initial_capital :
            Starting equity in TWD.
        product :
            ``"TX"`` or ``"MTX"``.

        Returns
        -------
        FuturesBacktestResult
            ``equity_curve`` is indexed by bar timestamp;
            ``trades[*]["holding_days"]`` records holding in fractional days.
        """
        from tw_futures.strategies.common.time_guards import TimeGuards

        product = product.upper()
        if product not in _TICK_VALUE:
            raise ValueError(f"product must be 'TX' or 'MTX', got {product!r}")

        tick_value = _TICK_VALUE[product]
        init_margin = self._override_initial or _DEFAULT_INITIAL_MARGIN[product]
        maint_margin = self._override_maintenance or _DEFAULT_MAINTENANCE_MARGIN[product]
        tg = TimeGuards()

        n_bars = len(data_15min)
        logger.info(
            "FuturesBacktester.run_intraday  product=%s  bars=%d  [%s → %s]  capital=%.0f",
            product,
            n_bars,
            data_15min.index[0] if n_bars else "?",
            data_15min.index[-1] if n_bars else "?",
            initial_capital,
        )

        account_value: float = float(initial_capital)
        position: int = 0
        entry_price: float | None = None
        entry_time: pd.Timestamp | None = None
        entry_cost: float = 0.0
        current_stop: float | None = None
        current_hard: float | None = None

        equity_history: list[tuple[pd.Timestamp, float]] = []
        trades: list[dict] = []
        margin_calls: list[pd.Timestamp] = []

        def _cost(price: float, n: int) -> float:
            notional = price * tick_value * n
            return (
                self.commission_per_contract * n
                + notional * self.tax_rate
                + self.slippage_ticks * tick_value * n
            )

        def _close(exit_price: float, exit_time: pd.Timestamp, reason: str) -> None:
            nonlocal position, entry_price, entry_time, current_stop
            nonlocal account_value, entry_cost, current_hard
            if position == 0:
                return
            direction = 1 if position > 0 else -1
            n = abs(position)
            ep = entry_price if entry_price is not None else exit_price
            gross_pnl = direction * (exit_price - ep) * n * tick_value
            ec = _cost(exit_price, n)
            net_pnl = gross_pnl - entry_cost - ec

            # Holding in fractional trading days (390 min = ~1 day)
            if entry_time is not None:
                holding_mins = (exit_time - entry_time).total_seconds() / 60
                holding_days = holding_mins / 390.0
            else:
                holding_days = 0.0

            trades.append(
                {
                    "entry_date": entry_time,
                    "exit_date": exit_time,
                    "direction": direction,
                    "contracts": n,
                    "entry_price": ep,
                    "exit_price": exit_price,
                    "gross_pnl": round(gross_pnl, 0),
                    "entry_cost": round(entry_cost, 0),
                    "exit_cost": round(ec, 0),
                    "net_pnl": round(net_pnl, 0),
                    "holding_days": round(holding_days, 3),
                    "exit_reason": reason,
                }
            )
            account_value -= ec
            position = 0
            entry_price = None
            entry_time = None
            current_stop = None
            current_hard = None
            entry_cost = 0.0

        def _open(price: float, n: int, direction: int, ts: pd.Timestamp) -> bool:
            nonlocal position, entry_price, entry_time, current_stop
            nonlocal account_value, entry_cost, current_hard
            if n <= 0:
                return False
            ec = _cost(price, n)
            margin_needed = n * init_margin
            if account_value - ec < margin_needed:
                logger.warning(
                    "Intraday: insufficient margin on %s (account=%.0f need=%.0f)",
                    ts,
                    account_value,
                    margin_needed + ec,
                )
                return False
            account_value -= ec
            position = direction * n
            entry_price = price
            entry_time = ts
            current_stop = None
            current_hard = None
            entry_cost = ec
            return True

        # ── Bar loop ──────────────────────────────────────────────────────
        for i in range(n_bars):
            row = data_15min.iloc[i]
            bar_ts = data_15min.index[i]
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            # ── 1. Intraday stop checks (hard stop + ATR stop) ─────────
            if position != 0:
                direction = 1 if position > 0 else -1

                # ATR trailing stop
                if current_stop is not None:
                    stop_hit = (direction == 1 and bar_low <= current_stop) or (
                        direction == -1 and bar_high >= current_stop
                    )
                    if stop_hit:
                        _close(current_stop, bar_ts, "ATR trailing stop (intraday)")
                        equity_history.append((bar_ts, account_value))
                        continue

                # Fixed hard stop
                if current_hard is not None:
                    hard_hit = (direction == 1 and bar_low <= current_hard) or (
                        direction == -1 and bar_high >= current_hard
                    )
                    if hard_hit:
                        _close(current_hard, bar_ts, "Hard stop (intraday)")
                        equity_history.append((bar_ts, account_value))
                        continue

            # ── 2. Margin check ────────────────────────────────────────
            if position != 0:
                maint_req = abs(position) * maint_margin
                if account_value < maint_req:
                    margin_calls.append(bar_ts)
                    _close(bar_close, bar_ts, "margin call")
                    equity_history.append((bar_ts, account_value))
                    continue

            # ── 3. TimeGuards force-close ──────────────────────────────
            if position != 0:
                force, fc_reason = tg.should_force_close(bar_ts)
                if force:
                    _close(bar_close, bar_ts, f"TimeGuard: {fc_reason}")
                    equity_history.append((bar_ts, account_value))
                    continue

            # ── 4. Strategy signal ─────────────────────────────────────
            data_slice = data_15min.iloc[: i + 1]
            try:
                sig = strategy.generate_signal(
                    data_15min=data_slice,
                    current_position=position,
                    entry_price=entry_price,
                    equity=account_value,
                    current_time=bar_ts,
                    entry_time=entry_time,
                )
            except NotImplementedError:
                equity_history.append((bar_ts, account_value))
                continue
            except Exception as exc:
                logger.error(
                    "run_intraday: generate_signal raised on %s: %s — skipping bar",
                    bar_ts,
                    exc,
                )
                equity_history.append((bar_ts, account_value))
                continue

            # ── 5. Execute signal ──────────────────────────────────────
            action = sig.action
            n_contracts = getattr(sig, "contracts", 0)

            if action == "close" and position != 0:
                _close(bar_close, bar_ts, sig.reason or "strategy close")

            elif action == "buy":
                if position < 0:
                    _close(bar_close, bar_ts, f"reverse to long — {sig.reason}")
                if position == 0:
                    _open(bar_close, n_contracts, direction=1, ts=bar_ts)

            elif action == "sell":
                if position > 0:
                    _close(bar_close, bar_ts, f"reverse to short — {sig.reason}")
                if position == 0:
                    _open(bar_close, n_contracts, direction=-1, ts=bar_ts)

            # ── 6. Update trailing stop ────────────────────────────────
            if getattr(sig, "stop_loss", None) is not None and position != 0:
                direction = 1 if position > 0 else -1
                if current_stop is None:
                    current_stop = sig.stop_loss
                elif direction == 1:
                    current_stop = max(current_stop, sig.stop_loss)
                else:
                    current_stop = min(current_stop, sig.stop_loss)

            # Update hard stop (always use the entry-based fixed level)
            if getattr(sig, "hard_stop", None) is not None and position != 0:
                current_hard = sig.hard_stop

            equity_history.append((bar_ts, account_value))

        # ── Force-close any open position at end of data ──────────────────
        if position != 0:
            last_close = float(data_15min.iloc[-1]["close"])
            last_ts = data_15min.index[-1]
            _close(last_close, last_ts, "end of backtest")
            if equity_history and equity_history[-1][0] == last_ts:
                equity_history[-1] = (last_ts, account_value)
            else:
                equity_history.append((last_ts, account_value))

        # ── Build result ───────────────────────────────────────────────────
        equity_curve = pd.Series(
            {ts: val for ts, val in equity_history},
            name="equity_twd",
            dtype=float,
        )
        equity_curve.index.name = "date"

        # For metrics, resample equity curve to daily for Sharpe calculation
        daily_equity = equity_curve.resample("D").last().dropna()
        metrics = _compute_metrics(
            equity_curve=daily_equity if len(daily_equity) >= 2 else equity_curve,
            trades=trades,
            data=data_15min,
            initial_capital=initial_capital,
            margin_call_count=len(margin_calls),
        )
        metrics["Data Type"] = "intraday_15min"

        logger.info(
            "run_intraday complete  product=%s  trades=%d  "
            "total_return=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%",
            product,
            len(trades),
            metrics.get("Total Return (%)", float("nan")),
            metrics.get("Sharpe Ratio", float("nan")),
            metrics.get("Max Drawdown (%)", float("nan")),
        )

        return FuturesBacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            margin_calls=margin_calls,
        )


# ── Metrics ───────────────────────────────────────────────────────────────────


def _compute_metrics(
    equity_curve: pd.Series,
    trades: list[dict],
    data: pd.DataFrame,
    initial_capital: float,
    margin_call_count: int,
    peak_leverage: float = 0.0,
) -> dict:
    if len(equity_curve) < 2:
        return {"error": "insufficient equity curve data"}

    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = max((end - start).days / 365.25, 1e-9)
    final = float(equity_curve.iloc[-1])

    # Returns
    total_return = (final / initial_capital - 1.0) * 100.0
    cagr = ((final / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    # Sharpe (annualised, rf = 0)
    daily_rets = equity_curve.pct_change().dropna()
    std = float(daily_rets.std())
    sharpe = (float(daily_rets.mean()) / std * math.sqrt(252)) if std > 0 else 0.0

    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdowns = equity_curve / rolling_max - 1.0
    max_dd = abs(float(drawdowns.min())) * 100.0

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else float("inf")

    # Trade stats
    if trades:
        n_trades = len(trades)
        pnls = [t["net_pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / n_trades * 100.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        )
        avg_hold = sum(t["holding_days"] for t in trades) / n_trades

        # Max consecutive losses
        max_consec = cur_consec = 0
        for p in pnls:
            if p <= 0:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        # Total transaction costs (entry + exit costs across all trades)
        total_costs = sum(t.get("entry_cost", 0) + t.get("exit_cost", 0) for t in trades)
        # Worst single trade loss
        max_loss = min(losses) if losses else 0.0
    else:
        n_trades = max_consec = 0
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0.0
        total_costs = 0.0
        max_loss = 0.0

    # Benchmark: TX buy-&-hold over the same data range
    close_s = data["close"].dropna()
    benchmark = (
        (float(close_s.iloc[-1]) / float(close_s.iloc[0]) - 1.0) * 100.0
        if len(close_s) >= 2
        else float("nan")
    )

    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": round(max_dd, 2),
        "Calmar Ratio": round(calmar, 3),
        "Total Trades": n_trades,
        "Win Rate (%)": round(win_rate, 1),
        "Profit Factor": round(profit_factor, 3),
        "Avg Win (TWD)": round(avg_win, 0),
        "Avg Loss (TWD)": round(avg_loss, 0),
        "Max Single Loss (TWD)": round(max_loss, 0),
        "Total Cost (TWD)": round(total_costs, 0),
        "Avg Holding Days": round(avg_hold, 1),
        "Max Consecutive Losses": max_consec,
        "Peak Leverage (×)": round(peak_leverage, 2),
        "Margin Call Count": margin_call_count,
        "Benchmark (%)": round(benchmark, 2),
    }
