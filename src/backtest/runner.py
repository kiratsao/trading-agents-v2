"""Backtest runner CLI — V2b MTX strategy.

Usage
-----
    python -m src.backtest.runner
    python -m src.backtest.runner --timing same_day_close
    python -m src.backtest.runner --timing next_day_open --cost 250
    python -m src.backtest.runner --yearly
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_CANDIDATES = [
    ROOT / "data" / "MXF_Daily_Clean_2020_to_now.parquet",
    ROOT / "data" / "TXF_Daily_Real.parquet",
    Path.home() / "trading-agents-v2" / "data" / "TXF_Daily_Real.parquet",
]


def _load_data() -> "pd.DataFrame":
    import pandas as pd

    for p in DATA_CANDIDATES:
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            print(f"Data: {p.name}  ({len(df)} bars, {df.index[0].date()} → {df.index[-1].date()})")
            return df
    raise FileNotFoundError(
        "No data file found. Run: python tw_futures/data/fetch_and_save.py\n"
        "Or check candidates:\n" + "\n".join(f"  {p}" for p in DATA_CANDIDATES)
    )


def _run(timing: str, cost: float, initial_capital: float) -> "BacktestResult":
    import src.backtest.engine as _eng

    orig_cps = _eng.COST_PER_SIDE
    orig_rt = _eng.ROUND_TRIP
    _eng.COST_PER_SIDE = cost
    _eng.ROUND_TRIP = cost * 2

    from src.backtest.engine import BacktestEngine

    eng = BacktestEngine(initial_capital=initial_capital, exec_timing=timing)
    df = _load_data()
    result = eng.run(df)

    _eng.COST_PER_SIDE = orig_cps
    _eng.ROUND_TRIP = orig_rt
    return result, df


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n{'━' * 45}")
    print(f"  {label}")
    print(f"{'━' * 45}")
    for k, v in m.items():
        print(f"  {k:<18} {v}")


def _print_yearly(ec: "pd.Series") -> None:
    print(f"\n  {'Year':<6}  {'Start':>10}  {'End':>10}  {'Ret%':>8}  {'MDD%':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
    for yr in range(ec.index[0].year, ec.index[-1].year + 1):
        yr_ec = ec[ec.index.year == yr]
        if len(yr_ec) == 0:
            continue
        start = float(yr_ec.iloc[0])
        end = float(yr_ec.iloc[-1])
        ret = (end / start - 1) * 100
        roll_max = yr_ec.cummax()
        mdd = float(((yr_ec - roll_max) / roll_max).min()) * 100
        print(f"  {yr:<6}  {start:>10,.0f}  {end:>10,.0f}  {ret:>8.1f}%  {mdd:>8.1f}%")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="V2b backtest runner")
    p.add_argument(
        "--timing",
        choices=["next_day_open", "same_day_close"],
        default="same_day_close",
        help="Execution timing (default: same_day_close = night_open)",
    )
    p.add_argument("--cost", type=float, default=160.0, help="Cost per side NTD (default 160)")
    p.add_argument("--capital", type=float, default=350_000.0, help="Initial capital NTD")
    p.add_argument("--yearly", action="store_true", help="Print yearly breakdown")
    p.add_argument(
        "--ab",
        action="store_true",
        help="Compare next_day_open vs same_day_close side-by-side",
    )
    args = p.parse_args(argv)

    if args.ab:
        results = {}
        for timing in ["next_day_open", "same_day_close"]:
            r, _ = _run(timing, args.cost, args.capital)
            results[timing] = r
        keys = list(results["next_day_open"].metrics.keys())
        print(f"\n{'━'*60}")
        print(f"  A/B comparison  cost={args.cost:.0f}/side  capital={args.capital:,.0f}")
        print(f"{'━'*60}")
        print(f"  {'Metric':<18}  {'next_day_open':>16}  {'night_open':>16}  {'Δ(B-A)':>8}")
        print(f"  {'─'*18}  {'─'*16}  {'─'*16}  {'─'*8}")
        for k in keys:
            va = results["next_day_open"].metrics[k]
            vb = results["same_day_close"].metrics[k]
            try:
                delta = f"{float(vb) - float(va):+.2f}"
            except (TypeError, ValueError):
                delta = "—"
            print(f"  {k:<18}  {str(va):>16}  {str(vb):>16}  {delta:>8}")
        return

    result, df = _run(args.timing, args.cost, args.capital)
    _print_metrics(
        f"V2b  {args.timing}  cost={args.cost:.0f}/side  capital={args.capital:,.0f}",
        result.metrics,
    )
    if args.yearly:
        print(f"\nYearly breakdown:")
        _print_yearly(result.equity_curve)


if __name__ == "__main__":
    main()
