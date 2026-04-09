"""Live trading entry point.

Modes
-----
    --mode rebalance   Manually trigger a full portfolio rebalance.
    --mode daily       Run the daily health check (no orders placed unless
                       a circuit breaker fires).
    --mode auto        Inspect today's date and run rebalance on quarter-end
                       trading days, daily check otherwise.

All execution results are appended to ``data/execution_log.json``.

Usage examples
--------------
    python scripts/run_live.py --mode daily
    python scripts/run_live.py --mode rebalance          # prompts for confirm
    python scripts/run_live.py --mode rebalance --force  # skips confirm (cron)
    python scripts/run_live.py --mode auto
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_live")

from us_equity.orchestrator import TradingOrchestrator  # noqa: E402 — after sys.path patch

# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------


def _print_rebalance_summary(report: dict) -> None:
    W = 72
    ok = report.get("ok", False)
    cancelled = report.get("cancelled", False)
    status_str = "CANCELLED" if cancelled else ("OK" if ok else "FAILED")
    bar = "=" * W
    print()
    print(bar)
    print(f"  {'REBALANCE SUMMARY  —  ' + status_str:^{W - 4}}")
    print(bar)

    if report.get("equity"):
        print(f"  {'Equity':<26}  ${report['equity']:>14,.2f}")
    if report.get("post_equity"):
        print(f"  {'Post-fill equity':<26}  ${report['post_equity']:>14,.2f}")

    # Risk status
    rs = report.get("risk_status", "—")
    print(f"  {'Risk status':<26}  {rs:>15}")

    # Orders
    orders = report.get("orders", [])
    if orders:
        sells = [o for o in orders if o.get("side") == "sell"]
        buys = [o for o in orders if o.get("side") == "buy"]
        errors = [o for o in orders if o.get("status") == "error"]
        print(f"  {'Orders submitted':<26}  {len(orders):>15d}")
        print(f"    {'SELL':<24}  {len(sells):>15d}")
        print(f"    {'BUY':<24}  {len(buys):>15d}")
        if errors:
            print(f"    {'Errors':<24}  {len(errors):>15d}")

        print()
        print(
            f"  {'Symbol':<8}  {'Side':<5}  {'Status':<12}  {'Filled Qty':>12}  {'Avg Price':>10}"
        )
        print("  " + "-" * 54)
        for o in orders:
            sym = o.get("symbol", "?")
            side = str(o.get("side", "?")).upper()[:4]
            status = str(o.get("status", "?"))[:11]
            filled = o.get("filled_qty", 0.0)
            price = o.get("filled_avg_price", 0.0)
            print(f"  {sym:<8}  {side:<5}  {status:<12}  {filled:>12.4f}  {price:>10.4f}")

    # Drift report
    drift = report.get("drift_report", {})
    if drift:
        warnings = {s: v for s, v in drift.items() if v.get("status") != "ok"}
        if warnings:
            print()
            print(f"  DRIFT WARNINGS ({len(warnings)})")
            print(f"  {'Symbol':<8}  {'Expected':>9}  {'Actual':>9}  {'Drift pp':>10}  Status")
            print("  " + "-" * 50)
            for sym, v in sorted(warnings.items()):
                print(
                    f"  {sym:<8}  {v['expected_weight']:>8.2%}  "
                    f"{v['actual_weight']:>8.2%}  {v['drift_pct']:>+9.2f}  "
                    f"{v['status']}"
                )

    if report.get("error"):
        print()
        print(f"  ERROR: {report['error']}")

    print(bar)
    print()


def _print_daily_summary(report: dict) -> None:
    W = 72
    ok = report.get("ok", False)
    bar = "=" * W
    print()
    print(bar)
    print(f"  {'DAILY CHECK SUMMARY  —  ' + ('OK' if ok else 'FAILED'):^{W - 4}}")
    print(bar)

    if report.get("equity"):
        print(f"  {'Equity':<30}  ${report['equity']:>12,.2f}")

    dr = report.get("daily_return")
    if dr is not None:
        print(f"  {'Daily return':<30}  {dr:>+11.3%}")

    ks = report.get("kill_switch_status", "—")
    ks_icon = "✓" if ks == "active" else "⚠" if ks == "triggered" else "✗"
    print(f"  {'KillSwitch':<30}  {ks_icon} {ks:>9}")

    dd = report.get("drawdown_action", "—")
    dd_icon = "✓" if dd == "hold" else "⚠" if dd == "reduce" else "✗"
    print(f"  {'DrawdownGuard action':<30}  {dd_icon} {dd:>9}")

    if report.get("liquidated"):
        liq = report.get("liquidation_orders", [])
        print()
        print(f"  *** EMERGENCY LIQUIDATION: {len(liq)} order(s) submitted ***")

    if report.get("error"):
        print()
        print(f"  ERROR: {report['error']}")

    print(bar)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live trading entry point for TradingOrchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["rebalance", "daily", "auto"],
        required=True,
        help="Execution mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Skip interactive confirmation prompt (use for scheduled / cron runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("run_live starting | mode=%s | force=%s", args.mode, args.force)

    try:
        orch = TradingOrchestrator()
    except OSError as exc:
        logger.error("Initialisation failed: %s", exc)
        print(f"\n  ERROR: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected init error: %s", exc, exc_info=True)
        sys.exit(1)

    mode = args.mode

    # Auto-detect mode based on today's date
    if mode == "auto":
        if orch.is_rebalance_day():
            logger.info("Auto mode: today is a rebalance day → running rebalance.")
            mode = "rebalance"
        else:
            logger.info("Auto mode: not a rebalance day → running daily check.")
            mode = "daily"

    # Execute
    if mode == "rebalance":
        report = orch.run_rebalance(force=args.force)
        _print_rebalance_summary(report)
        if not report.get("ok") and not report.get("cancelled"):
            sys.exit(2)

    elif mode == "daily":
        report = orch.run_daily_check()
        _print_daily_summary(report)
        if not report.get("ok"):
            sys.exit(2)

    logger.info("run_live complete | mode=%s | ok=%s", mode, report.get("ok"))


if __name__ == "__main__":
    main()
