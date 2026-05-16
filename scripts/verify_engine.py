"""Ground-truth backtest for V2b strategy (MXF / 小台指).

This script is the *authoritative reference implementation*.  It is written as
a plain for-loop with zero dependency on src/backtest/engine.py so it can be
used to audit that module.

Key design decisions (vs old engine.py bugs)
--------------------------------------------
1. **Mark-to-market equity**: at every bar close the equity curve is recorded
   as ``realized_equity + unrealized_pnl``.  This ensures MDD captures
   floating losses during open positions.  (Old engine recorded only realized
   equity → MDD was ~3× too small.)

2. **"add" action handled**: V2bEngine returns Signal("add") for pyramid
   scale-in.  Old engine only handled "buy" and "close"; "add" silently fell
   through and pyramid never executed.

3. **Execution price**: configurable via --exec_timing:
   * ``next_day_open``  (default) — execute at bar[t].open, signal from
     bar[t-1] slice.  Simulates "signal at 14:30, order placed for next open".
   * ``same_day_close`` — execute at bar[t].close, signal from bar[t] slice.
     Simulates night-session entry at 15:05 close price.

4. **Cost**: 160 NTD per side (matching src/backtest/engine.py constants).

Usage
-----
    python scripts/verify_engine.py
    python scripts/verify_engine.py --exec_timing same_day_close
    python scripts/verify_engine.py --compare          # compare vs old engine
    python scripts/verify_engine.py --save             # write trades/equity CSV
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import src.*
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy.v2b_engine import V2bEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_VALUE: float = 50.0        # NTD per index point, MXF / MTX
COST_PER_SIDE: float = 160.0    # NTD commission per side
ROUND_TRIP: float = COST_PER_SIDE * 2
MTX_MARGIN: float = 131_500.0   # NTD original margin per MTX/MXF contract (TAIFEX 2026-04-27)

DATA_CANDIDATES = [
    ROOT / "data" / "MXF_Daily_Clean_2020_to_now.parquet",
    ROOT / "data" / "TXF_Daily_Real.parquet",
    Path.home() / "trading-agents-v2" / "data" / "TXF_Daily_Real.parquet",
    Path("../trading-agents-v2/data/TXF_Daily_Real.parquet"),
]

# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    contracts: int
    pnl_pts: float   # exit − entry (per contract, long)
    pnl_twd: float   # NTD after costs
    reason: str
    entry_reason: str = ""


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------


def run_backtest(
    data: pd.DataFrame,
    initial_capital: float = 350_000.0,
    exec_timing: str = "next_day_open",  # "next_day_open" | "same_day_close"
    ema_fast: int = 30,
    ema_slow: int = 100,
    atr_stop_mult: float = 2.0,
    confirm_days: int = 2,
    ladder: list[dict] | None = None,
    verbose: bool = False,
    use_hma: bool = False,
    dynamic_stop: bool = False,
) -> tuple[pd.Series, list[Trade], dict]:
    """Run V2b ground-truth backtest.

    Returns
    -------
    equity_curve : pd.Series
        Mark-to-market equity at each bar (including unrealized PnL).
    trades : list[Trade]
    metrics : dict
    """
    _ladder = ladder or [
        {"equity": 350_000, "contracts": 2},
        {"equity": 480_000, "contracts": 3},
        {"equity": 600_000, "contracts": 4},
    ]
    strategy = V2bEngine(
        product="MXF",
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        trail_atr_mult=atr_stop_mult,
        confirm_days=confirm_days,
        ladder=_ladder,
        use_hma=use_hma,
        dynamic_stop=dynamic_stop,
    )

    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── State ──────────────────────────────────────────────────────────
    realized_equity: float = initial_capital
    position: int = 0          # open contracts (long)
    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None
    entry_reason: str = ""
    highest_high: float | None = None
    pyramided: bool = False

    equity_points: list[tuple[pd.Timestamp, float]] = []
    trades: list[Trade] = []

    # Determine loop range based on execution timing
    if exec_timing == "same_day_close":
        # Signal on bar[i] data slice (inclusive), execute at bar[i].close
        loop_start = 1
        def get_slice(i):
            return df.iloc[: i + 1]
        def get_exec_price(row):
            return float(row["close"])
    else:
        # Signal on bar[i-1] slice, execute at bar[i].open (default)
        loop_start = 1
        def get_slice(i):
            return df.iloc[:i]
        def get_exec_price(row):
            return float(row["open"])

    for i in range(loop_start, len(df)):
        today = df.index[i]
        row = df.iloc[i]
        close = float(row["close"])
        data_slice = get_slice(i)

        sig = strategy.generate_signal(
            data=data_slice,
            current_position=position,
            entry_price=entry_price,
            equity=realized_equity,
            highest_high=highest_high,
            contracts=position,
        )

        exec_price = get_exec_price(row)

        # ── Process signal ──────────────────────────────────────────
        if sig.action == "close" and position > 0 and entry_price is not None:
            pnl_pts = exec_price - entry_price
            pnl_twd = pnl_pts * position * TICK_VALUE - ROUND_TRIP * position
            realized_equity += pnl_twd
            trades.append(Trade(
                entry_date=str(entry_date.date()),
                exit_date=str(today.date()),
                entry_price=entry_price,
                exit_price=exec_price,
                contracts=position,
                pnl_pts=pnl_pts,
                pnl_twd=pnl_twd,
                reason=sig.reason,
                entry_reason=entry_reason,
            ))
            if verbose:
                print(f"  CLOSE {today.date()}  px={exec_price:.0f}  "
                      f"pnl={pnl_twd:+,.0f}  eq={realized_equity:,.0f}  [{sig.reason[:60]}]")
            position = 0
            entry_price = None
            entry_date = None
            entry_reason = ""
            highest_high = None
            pyramided = False

        elif sig.action == "buy" and position == 0:
            n = sig.contracts if sig.contracts > 0 else 1
            margin_required = n * MTX_MARGIN
            if margin_required > realized_equity:
                print(
                    f"  ⚠️ margin exceeded: {n}口需要 {margin_required:,.0f} "
                    f"但 equity={realized_equity:,.0f}  [{today.date()}]"
                )
            realized_equity -= COST_PER_SIDE * n  # entry commission
            position = n
            entry_price = exec_price
            entry_date = today
            entry_reason = sig.reason
            highest_high = exec_price
            pyramided = False
            if verbose:
                print(f"  BUY   {today.date()}  px={exec_price:.0f}  "
                      f"n={n}  eq={realized_equity:,.0f}  [{sig.reason[:60]}]")

        elif sig.action == "add" and position > 0 and not pyramided and entry_price is not None:
            add_n = sig.contracts if sig.contracts > 0 else 1
            total = position + add_n
            margin_required = total * MTX_MARGIN
            if margin_required > realized_equity:
                print(
                    f"  ⚠️ margin exceeded: {total}口需要 {margin_required:,.0f} "
                    f"但 equity={realized_equity:,.0f}  [{today.date()}]"
                )
            # Weighted average entry price
            new_entry = (entry_price * position + exec_price * add_n) / total
            realized_equity -= COST_PER_SIDE * add_n
            if verbose:
                print(f"  ADD   {today.date()}  px={exec_price:.0f}  "
                      f"+{add_n}→{total}  eq={realized_equity:,.0f}")
            entry_price = new_entry
            position = total
            pyramided = True

        # ── Update trailing high ────────────────────────────────────
        if position > 0:
            if highest_high is None or close > highest_high:
                highest_high = close

        # ── Mark-to-market equity for curve ────────────────────────
        if position > 0 and entry_price is not None:
            unrealized = (close - entry_price) * position * TICK_VALUE
            mtm_equity = realized_equity + unrealized
        else:
            mtm_equity = realized_equity

        equity_points.append((today, mtm_equity))

    # ── Force-close remaining position at last close ──────────────────
    if position > 0 and entry_price is not None:
        last_row = df.iloc[-1]
        last_date = df.index[-1]
        exit_price = float(last_row["close"])
        pnl_pts = exit_price - entry_price
        pnl_twd = pnl_pts * position * TICK_VALUE - ROUND_TRIP * position
        realized_equity += pnl_twd
        trades.append(Trade(
            entry_date=str(entry_date.date()),
            exit_date=str(last_date.date()),
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=position,
            pnl_pts=pnl_pts,
            pnl_twd=pnl_twd,
            reason="end-of-backtest liquidation",
            entry_reason=entry_reason,
        ))
        # Update last equity point to realized
        if equity_points and equity_points[-1][0] == last_date:
            equity_points[-1] = (last_date, realized_equity)
        else:
            equity_points.append((last_date, realized_equity))

    ec = pd.Series(
        [e for _, e in equity_points],
        index=pd.DatetimeIndex([d for d, _ in equity_points]),
        name="equity_twd",
    )
    metrics = _compute_metrics(ec, trades, initial_capital)
    return ec, trades, metrics


# ---------------------------------------------------------------------------
# Metrics (same formula for both engines)
# ---------------------------------------------------------------------------


def _compute_metrics(
    ec: pd.Series,
    trades: list[Trade],
    initial_capital: float,
) -> dict:
    if ec.empty or not trades:
        return {}

    years = (ec.index[-1] - ec.index[0]).days / 365.25
    final = float(ec.iloc[-1])

    cagr = ((final / initial_capital) ** (1 / max(years, 1e-9)) - 1) * 100

    # MDD on MTM equity curve (includes unrealized losses)
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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def _load_data() -> pd.DataFrame:
    for p in DATA_CANDIDATES:
        if Path(p).exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            print(f"Data: {p}  ({len(df)} bars, {df.index[0].date()} → {df.index[-1].date()})")
            return df
    raise FileNotFoundError(
        "TXF_Daily_Real.parquet not found. Tried:\n"
        + "\n".join(f"  {p}" for p in DATA_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Compare vs old engine
# ---------------------------------------------------------------------------


def _run_old_engine(df: pd.DataFrame, initial_capital: float) -> dict | None:
    """Run the old BacktestEngine from trading-agents-v2 for comparison."""
    old_root = Path.home() / "trading-agents-v2"
    if not old_root.exists():
        print("[SKIP] trading-agents-v2 not found, skipping old engine comparison.")
        return None

    import sys as _sys

    # Temporarily add old repo to sys.path
    _sys.path.insert(0, str(old_root))
    try:
        from src.backtest.engine import BacktestEngine as OldEngine
        from src.strategy.v2b_engine import V2bEngine as OldV2b

        old_eng = OldEngine(
            strategy=OldV2b(product="MTX"),
            initial_capital=initial_capital,
        )
        result = old_eng.run(df)
        return result.metrics
    except Exception as exc:
        print(f"[WARN] Old engine failed: {exc}")
        return None
    finally:
        _sys.path.remove(str(old_root))


# ---------------------------------------------------------------------------
# Trade comparison
# ---------------------------------------------------------------------------


def _compare_trades(
    new_trades: list[Trade],
    df: pd.DataFrame,
    initial_capital: float,
) -> None:
    """Compare new vs old engine trade-by-trade to find first divergence."""
    old_root = Path.home() / "trading-agents-v2"
    if not old_root.exists():
        return

    sys.path.insert(0, str(old_root))
    try:
        from src.backtest.engine import BacktestEngine as OldEngine
        from src.strategy.v2b_engine import V2bEngine as OldV2b

        old_eng = OldEngine(
            strategy=OldV2b(product="MTX"),
            initial_capital=initial_capital,
        )
        result = old_eng.run(df)
        old_trades = result.trades
    except Exception as exc:
        print(f"[WARN] Trade comparison failed: {exc}")
        sys.path.remove(str(old_root))
        return
    finally:
        if str(old_root) in sys.path:
            sys.path.remove(str(old_root))

    print(f"\n{'─'*70}")
    print(f"Trade comparison: verify={len(new_trades)}  old={len(old_trades)}")
    print(f"{'─'*70}")
    header = f"{'#':>3}  {'Entry':>10}  {'Exit':>10}  {'Px_in':>7}  {'Px_out':>7}  "
    header += f"{'N':>3}  {'PnL_new':>10}  {'PnL_old':>10}  {'Δ':>8}"
    print(header)
    print("─" * 70)
    first_div = None
    for idx, (nt, ot) in enumerate(zip(new_trades, old_trades), 1):
        delta = nt.pnl_twd - ot.pnl_twd
        marker = "  ← FIRST DIVERGENCE" if (first_div is None and abs(delta) > 1) else ""
        if first_div is None and abs(delta) > 1:
            first_div = idx
        same_entry = nt.entry_date == ot.entry_date
        same_exit = nt.exit_date == ot.exit_date
        e_mark = " !" if not same_entry else ""
        x_mark = " !" if not same_exit else ""
        print(
            f"{idx:>3}  {nt.entry_date:>10}{e_mark}  {nt.exit_date:>10}{x_mark}  "
            f"{nt.entry_price:>7.0f}  {nt.exit_price:>7.0f}  "
            f"{nt.contracts:>3}  {nt.pnl_twd:>10,.0f}  {ot.pnl_twd:>10,.0f}  "
            f"{delta:>+8,.0f}{marker}"
        )
        if idx > 30 and first_div is None:
            print("  ... (showing first 30; no divergence yet)")
            break
        if idx > 30:
            break

    if new_trades[len(old_trades):]:
        extra = len(new_trades) - len(old_trades)
        print(f"\n  +{extra} extra trades in new engine (pyramid/exits)")

    if old_trades[len(new_trades):]:
        print(f"\n  +{len(old_trades) - len(new_trades)} extra trades in old engine")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_metrics(label: str, m: dict) -> None:
    pad = 14
    print(f"\n{'━'*40}")
    print(f"  {label}")
    print(f"{'━'*40}")
    for k, v in m.items():
        print(f"  {k:<{pad}} {v}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="V2b ground-truth backtest")
    p.add_argument(
        "--exec_timing",
        choices=["next_day_open", "same_day_close"],
        default="next_day_open",
        help="Execution price timing (default: next_day_open)",
    )
    p.add_argument(
        "--initial_capital",
        type=float,
        default=350_000.0,
        help="Starting equity NTD (default 350000)",
    )
    p.add_argument("--compare", action="store_true", help="Compare vs old engine metrics")
    p.add_argument("--trade_diff", action="store_true",
                   help="Show trade-by-trade diff vs old engine")
    p.add_argument("--save", action="store_true", help="Save trades + equity curve to results/")
    p.add_argument("--verbose", action="store_true", help="Print each trade as it executes")
    p.add_argument("--use_hma", action="store_true", help="Use Hull Moving Average instead of EMA")
    p.add_argument("--dynamic_stop", action="store_true", help="Use dynamic ATR stop based on ADX")
    p.add_argument("--ema_fast", type=int, default=30, help="Fast EMA period")
    p.add_argument("--ema_slow", type=int, default=100, help="Slow EMA period")
    args = p.parse_args(argv)

    df = _load_data()

    print(f"\nRunning verify_engine  exec_timing={args.exec_timing}  "
          f"capital={args.initial_capital:,.0f}")

    ec, trades, metrics = run_backtest(
        df,
        initial_capital=args.initial_capital,
        exec_timing=args.exec_timing,
        verbose=args.verbose,
        use_hma=args.use_hma,
        dynamic_stop=args.dynamic_stop,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
    )

    _print_metrics(f"verify_engine  [{args.exec_timing}]", metrics)

    if args.compare or args.trade_diff:
        print("\nRunning old BacktestEngine for comparison...")
        old_m = _run_old_engine(df, args.initial_capital)
        if old_m:
            _print_metrics("old BacktestEngine (MTX, cost=105)", old_m)

            # Side-by-side diff
            keys = list(metrics.keys())
            print(f"\n{'─'*55}")
            print(f"  {'Metric':<14}  {'verify_engine':>14}  {'old_engine':>14}  {'Δ':>8}")
            print(f"{'─'*55}")
            for k in keys:
                nv = metrics.get(k, "—")
                ov = old_m.get(k, "—")
                try:
                    delta = f"{float(nv) - float(ov):+.2f}"
                except (TypeError, ValueError):
                    delta = "—"
                print(f"  {k:<14}  {str(nv):>14}  {str(ov):>14}  {delta:>8}")

    if args.trade_diff:
        _compare_trades(trades, df, args.initial_capital)

    if args.save:
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        # Equity curve
        ec_path = out_dir / f"verify_equity_{args.exec_timing}.csv"
        ec.to_csv(ec_path, header=True)
        print(f"\nEquity curve saved → {ec_path}")
        # Trades
        tr_path = out_dir / f"verify_trades_{args.exec_timing}.csv"
        rows = [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "contracts": t.contracts,
                "pnl_pts": t.pnl_pts,
                "pnl_twd": t.pnl_twd,
                "reason": t.reason,
                "entry_reason": t.entry_reason,
            }
            for t in trades
        ]
        pd.DataFrame(rows).to_csv(tr_path, index=False)
        print(f"Trades saved       → {tr_path}")


if __name__ == "__main__":
    main()
