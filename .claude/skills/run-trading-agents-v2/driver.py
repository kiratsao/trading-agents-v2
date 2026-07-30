#!/usr/bin/env python
"""Credential-free driver for trading-agents-v2.

Drives the app's real decision surface WITHOUT any broker/LINE side effects:

    signal    what the daemon's 14:30 decision would be (pure generate_signal,
              built from accounts.yaml exactly like src/scheduler/main.py)
    backtest  full BacktestEngine baseline run on the repo parquet
    verify    scripts/verify_data.py parquet health check
    smoke     all three; non-zero exit on any failure

NEVER use `python -m src.scheduler.main --run-once` to "check the signal" —
it is NOT a dry run (run_signal/run_execution submit real orders when a
broker builds). This driver imports no broker code and sends no orders.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

ACCOUNT = "mxf_aggressive"


def _load_parquet(path: Path):
    import pandas as pd

    if not path.exists():
        sys.exit(f"FAIL: {path} not found — run scripts/init_data.py (needs Shioaji creds)")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _build_engine():
    """Mirror src/scheduler/main.py::_build_orchestrators for one account."""
    import yaml

    from src.strategy.v2b_engine import V2bEngine

    cfg = yaml.safe_load((ROOT / "config/accounts.yaml").read_text(encoding="utf-8"))
    acc = cfg["accounts"][ACCOUNT]
    params = acc.get("strategy_params", {})
    engine = V2bEngine(
        product=acc.get("product", "MXF"),
        ema_fast=params.get("ema_fast", 30),
        ema_slow=params.get("ema_slow", 100),
        trail_atr_mult=params.get("atr_stop_mult", 2.0),
        confirm_days=params.get("confirm_days", 2),
        adx_threshold=params.get("adx_threshold", 25),
        ladder=[{"equity": e["equity"], "contracts": e["contracts"]}
                for e in acc.get("scale_ladder", [])],
        max_contracts=acc.get("max_contracts"),
        margin_per_contract=acc.get("margin_per_contract"),
        risk_cap_pct=params.get("risk_cap_pct"),
        margin_buffer_atr=params.get("margin_buffer_atr"),
        cooldown_days=params.get("cooldown_days", 0),
        reentry_require_above_ema_fast=params.get("reentry_above_ema_fast", False),
    )
    return engine, acc


def cmd_signal(args) -> int:
    engine, acc = _build_engine()
    product = acc.get("product", "MXF")
    df = _load_parquet(Path(args.parquet) if args.parquet
                       else ROOT / f"data/{product}_Daily_Clean_2020_to_now.parquet")
    equity = args.equity if args.equity is not None else float(acc.get("equity", 350_000))

    ind = engine._compute_indicators(df).iloc[-1]
    print(f"data: {len(df)} bars {df.index[0].date()} → {df.index[-1].date()} "
          f"(signal is based on the LAST bar date — check it is what you expect)")
    print(f"indicators@{df.index[-1].date()}: close={ind['close']:,.0f} "
          f"EMA{engine.ema_fast}={ind['ema_fast']:,.1f} "
          f"EMA{engine.ema_slow}={ind['ema_slow']:,.1f} "
          f"ATR={ind['atr']:,.1f} ADX={ind['adx']:.2f}")

    sig = engine.generate_signal(data=df, current_position=args.position, equity=equity)
    print(f"signal(position={args.position}, equity={equity:,.0f}): "
          f"{sig.action} {sig.contracts}口 — {sig.reason}")
    return 0


def cmd_backtest(args) -> int:
    from src.backtest.engine import BacktestEngine
    from src.strategy.v2b_engine import V2bEngine

    df = _load_parquet(Path(args.parquet) if args.parquet
                       else ROOT / "data/MXF_Daily_Clean_2020_to_now.parquet")
    r = BacktestEngine(
        strategy=V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                           confirm_days=2, adx_threshold=25),
        initial_capital=350_000,
        exec_timing="same_day_close",
    ).run(df)
    print(f"data: {len(df)} bars → {df.index[-1].date()}")
    for k, v in r.metrics.items():
        print(f"  {k}: {v}")
    trades = r.metrics.get("Total_Trades", 0)
    if not trades or trades < 10:
        print("FAIL: suspiciously few trades — engine or data broken")
        return 1
    return 0


def cmd_verify(args) -> int:
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_data.py"), "--product", "MXF"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(p.stdout, end="", flush=True)
    if p.returncode == 0:
        return 0
    # On a dev clone the parquet is normally stale (updates run on the GCP
    # deployment at 14:25). STALE alone is a warning; anything else is real.
    issues = [ln.strip() for ln in p.stdout.splitlines() if ln.strip().startswith("- ")]
    if not args.strict and issues and all(i.startswith("- STALE") for i in issues):
        print("WARN: STALE only — expected on a dev clone; use --strict to fail on it")
        return 0
    return p.returncode


def cmd_smoke(args) -> int:
    fails = 0
    for name, fn in [("signal", cmd_signal), ("backtest", cmd_backtest),
                     ("verify", cmd_verify)]:
        print(f"\n=== {name} ===", flush=True)
        rc = fn(args)
        print(f"=== {name}: {'OK' if rc == 0 else f'FAIL (rc={rc})'} ===", flush=True)
        fails += rc != 0
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("signal", cmd_signal), ("backtest", cmd_backtest),
                     ("verify", cmd_verify), ("smoke", cmd_smoke)]:
        sp = sub.add_parser(name)
        sp.add_argument("--parquet", help="override parquet path (e.g. a scratch copy)")
        sp.add_argument("--equity", type=float, default=None,
                        help="equity for signal sizing (default: accounts.yaml)")
        sp.add_argument("--position", type=int, default=0,
                        help="current position in contracts (default: 0 = flat)")
        sp.add_argument("--strict", action="store_true",
                        help="verify: fail on STALE too (default: warn on dev clones)")
        sp.set_defaults(fn=fn)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
