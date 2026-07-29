"""Restart-gate design matrix: risk-cap gate + re-entry filters A/B backtest.

Compares the V2b baseline against the proposed restart gates over 2020→now:
  * risk-cap  — stop risk (2×ATR×口數×50) ≤ risk_cap_pct × equity, plus a
                margin buffer of 1×ATR×口數×50 (both sizing-only)
  * filter A  — cooldown N trading days after a trailing-stop exit
  * filter B  — entry requires close back above EMA30
Report-only: prints CAGR/MDD/Sharpe/win-rate/whipsaw per variant, the 2026-07
trade detail, and the would-be restart sizing at a given equity. No defaults
are changed anywhere — enabling anything is a config decision.

Usage:
    python scripts/backtest_gate_matrix.py [--data PATH] [--equity 650708]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.strategy.v2b_engine import V2bEngine

_WHIPSAW_MAX_HOLD = 5  # trading days — losing trade held ≤ this = whipsaw


def _mk_engine(**kw) -> V2bEngine:
    return V2bEngine(
        product="MXF", ema_fast=30, ema_slow=100, confirm_days=2,
        adx_threshold=25, trail_atr_mult=2.0, **kw,
    )


VARIANTS: dict[str, dict] = {
    "baseline": {},
    "cap15": {"risk_cap_pct": 0.15, "margin_buffer_atr": 1.0},
    "cap20": {"risk_cap_pct": 0.20, "margin_buffer_atr": 1.0},
    "A3": {"cooldown_days": 3},
    "A5": {"cooldown_days": 5},
    "A10": {"cooldown_days": 10},
    "B": {"reentry_require_above_ema_fast": True},
    "cap15+A3": {"risk_cap_pct": 0.15, "margin_buffer_atr": 1.0, "cooldown_days": 3},
    "cap15+A5": {"risk_cap_pct": 0.15, "margin_buffer_atr": 1.0, "cooldown_days": 5},
    "cap15+A10": {"risk_cap_pct": 0.15, "margin_buffer_atr": 1.0, "cooldown_days": 10},
    "cap15+B": {"risk_cap_pct": 0.15, "margin_buffer_atr": 1.0,
                "reentry_require_above_ema_fast": True},
    "cap20+A5": {"risk_cap_pct": 0.20, "margin_buffer_atr": 1.0, "cooldown_days": 5},
    "cap20+B": {"risk_cap_pct": 0.20, "margin_buffer_atr": 1.0,
                "reentry_require_above_ema_fast": True},
}


def _hold_days(df: pd.DataFrame, t) -> int:
    i0 = df.index.searchsorted(pd.Timestamp(t.entry_date))
    i1 = df.index.searchsorted(pd.Timestamp(t.exit_date))
    return int(i1 - i0)


def _whipsaws(df: pd.DataFrame, res: BacktestResult) -> int:
    return sum(
        1 for t in res.trades
        if t.pnl_twd < 0 and _hold_days(df, t) <= _WHIPSAW_MAX_HOLD
    )


def _july_2026(res: BacktestResult) -> list:
    return [
        t for t in res.trades
        if t.exit_date >= "2026-07-01" or t.entry_date >= "2026-07-01"
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/MXF_Daily_Clean_2020_to_now.parquet")
    ap.add_argument("--equity", type=float, default=650_708.0,
                    help="restart equity for the would-be sizing table")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"data: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}"
          f"  last close {df['close'].iloc[-1]:,.0f}")

    # Current ATR (for the restart-sizing table)
    probe = _mk_engine()
    ind = probe._compute_indicators(df)
    atr_now = float(ind["atr"].iloc[-1])
    ema30_now = float(ind["ema_fast"].iloc[-1])
    close_now = float(ind["close"].iloc[-1])
    print(f"now: ATR(14)={atr_now:,.0f}  EMA30={ema30_now:,.0f}  close={close_now:,.0f}\n")

    rows = []
    details: dict[str, list] = {}
    for name, kw in VARIANTS.items():
        res = BacktestEngine(
            strategy=_mk_engine(**kw), initial_capital=350_000,
            exec_timing="same_day_close",
        ).run(df)
        m = res.metrics
        rows.append({
            "variant": name, "CAGR%": m["CAGR_%"], "MDD%": m["MDD_%"],
            "Sharpe": m["Sharpe"], "Win%": m["Win_Rate_%"],
            "PF": m["Profit_Factor"], "Trades": m["Total_Trades"],
            "Whipsaw": _whipsaws(df, res),
            "FinalEq": f"{m['Final_Equity']:,.0f}",
        })
        details[name] = _july_2026(res)

    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── 2026-07 trades per variant ──")
    for name, trades in details.items():
        if not trades:
            print(f"{name:12} (no 2026-07 trades)")
        for t in trades:
            print(f"{name:12} {t.entry_date}→{t.exit_date} {t.contracts}口 "
                  f"@{t.entry_price:,.0f}→{t.exit_price:,.0f} "
                  f"pnl={t.pnl_twd:+,.0f}  [{t.reason[:44]}]")

    print(f"\n── would-be restart sizing @ equity {args.equity:,.0f} ──")
    import yaml

    from src.strategy.v2b_engine import _anti_martingale_contracts
    with open("config/accounts.yaml", encoding="utf-8") as f:
        acc = yaml.safe_load(f)["accounts"]["mxf_aggressive"]
    ladder = acc["scale_ladder"]
    margin = float(acc["margin_per_contract"])
    max_c = acc.get("max_contracts")
    n0 = _anti_martingale_contracts(args.equity, ladder, max_c, margin)
    print(f"ladder sizing (accounts.yaml): {n0}口  "
          f"(margin {margin:,.0f}/口 → 佔用 {n0 * margin:,.0f}, "
          f"stop風險 2×ATR×50×{n0} = {2 * atr_now * 50 * n0:,.0f} "
          f"= {2 * atr_now * 50 * n0 / args.equity:.0%} equity)")
    for name, kw in VARIANTS.items():
        eng = _mk_engine(**kw, margin_per_contract=margin, max_contracts=max_c)
        n, note = eng._risk_capped_contracts(n0, args.equity, atr_now)
        blockb = (eng.reentry_require_above_ema_fast and close_now < ema30_now)
        label = "封鎖(濾網B: close<EMA30)" if blockb else f"{n}口"
        print(f"{name:12} {n0}口 → {label}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
