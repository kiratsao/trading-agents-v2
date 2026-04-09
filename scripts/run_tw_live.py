"""台指期 live-trading CLI.

Usage
-----
    # Dry-run: generate signal, no orders
    python scripts/run_tw_live.py --dry-run

    # Daily signal + order (with confirmation prompt)
    python scripts/run_tw_live.py --mode daily

    # Auto: run daily if post-market on a trading day, otherwise health check
    python scripts/run_tw_live.py --mode auto

    # Health check only
    python scripts/run_tw_live.py --mode check

    # Skip confirmation prompt (e.g. cron job)
    python scripts/run_tw_live.py --mode daily --force

Modes
-----
daily   Run post-market signal generation + optional order execution.
check   Health check: position state, margin, kill-switch status.
auto    Decide automatically: daily if 13:45+ on a trading day, else check.

Flags
-----
--dry-run       Compute signal but do NOT submit orders.
--force         Skip order confirmation prompt.
--product TX    TX (大台) or MTX (小台).  Default: TX.
--equity N      Fallback equity TWD for position sizing (default: 2,000,000).
--simulation    Force simulation mode even if .env has SHIOAJI_SIMULATION=false.
--no-simulation Force live mode (overrides .env).  USE WITH CARE.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_tw_live")


# ── Formatting helpers ─────────────────────────────────────────────────────────

SEP = "=" * 68
SEP2 = "-" * 68


def _hdr(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _twd(v, prefix: str = "") -> str:
    if v is None:
        return "—"
    return f"TWD {prefix}{float(v):>14,.0f}"


def _print_signal(sig: dict) -> None:
    action = sig.get("action", "?").upper()
    if sys.stdout.isatty():
        colour = {
            "BUY": "\033[32m",
            "SELL": "\033[31m",
            "CLOSE": "\033[33m",
            "HOLD": "\033[90m",
        }.get(action, "")
        reset = "\033[0m"
    else:
        colour = reset = ""

    print(f"  Action    : {colour}{action}{reset}")
    print(f"  Contracts : {sig.get('contracts', 0)}")
    print(f"  Stop Loss : {sig.get('stop_loss', '—')}")
    print(f"  Reason    : {sig.get('reason', '—')}")


def _print_position(pos: dict, label: str = "Position") -> None:
    dir_map = {1: "LONG ▲", -1: "SHORT ▼", 0: "FLAT"}
    direction = int(pos.get("direction", 0))
    print(f"  {label:<18}: {dir_map.get(direction, '?')}")
    if direction != 0:
        print(f"  {'Contracts':<18}: {pos.get('contracts', 0)}")
        print(f"  {'Entry Price':<18}: {pos.get('entry_price', '—')}")
        print(f"  {'Entry Date':<18}: {str(pos.get('entry_date', '—'))[:10]}")
        print(f"  {'Trailing Stop':<18}: {pos.get('trailing_stop', '—')}")


def _print_account(acct: dict) -> None:
    equity = acct.get("equity", 0)
    if equity == 0:
        print("  Equity           : (simulation — broker returns 0)")
    else:
        print(f"  Equity           : {_twd(equity)}")
    print(f"  Margin Used      : {_twd(acct.get('margin_used', 0))}")
    print(f"  Available Margin : {_twd(acct.get('available_margin', 0))}")
    print(f"  Unrealised PnL   : {_twd(acct.get('unrealized_pnl', 0), '+')}")


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print(f"\n  ⚠  Warnings ({len(warnings)}):")
    for w in warnings:
        print(f"     • {w}")


# ── Confirmation prompt ────────────────────────────────────────────────────────


def _confirm_order(sig: dict, product: str) -> bool:
    """Display order preview and ask for confirmation. Returns True if confirmed."""
    action = sig.get("action", "?").upper()
    contracts = sig.get("contracts", 0)
    stop = sig.get("stop_loss")

    print(f"\n{SEP2}")
    print("  ORDER PREVIEW")
    print(f"  Product   : {product}")
    print(f"  Action    : {action}")
    print(f"  Contracts : {contracts}")
    if stop:
        print(f"  Stop Loss : {stop:.0f}")
    print("  Price     : MARKET")
    print(f"{SEP2}")

    try:
        ans = input("  Confirm order? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return ans in ("y", "yes")


# ── Report printer ─────────────────────────────────────────────────────────────


def _print_daily_report(report: dict) -> None:
    today = report.get("date", "?")
    product = report.get("product", "TX")
    dry_run = report.get("dry_run", False)
    last_close = report.get("last_close", 0)

    _hdr(f"台指期 {product}  {today}{'  [DRY-RUN]' if dry_run else ''}")
    print(f"  Last Close       : {last_close:,.0f}")

    ks = report.get("kill_switch", "ACTIVE")
    ks_str = "ACTIVE ✓" if ks == "ACTIVE" else "KILLED ✗"
    print(f"  Kill Switch      : {ks_str}")

    _hdr("Signal")
    _print_signal(report.get("signal", {}))

    _hdr("Position Before → After")
    _print_position(report.get("position_before", {}), "Before")
    print(f"  {'↓':>20}")
    _print_position(report.get("position_after", {}), "After")

    _hdr("Account (Broker)")
    _print_account(report.get("account", {}))

    order = report.get("order")
    if order:
        _hdr("Order Result")
        if order.get("dry_run"):
            print(
                f"  [DRY-RUN] would {order.get('shioaji_action')} "
                f"{order.get('contracts')} × {order.get('product')} @ MKT"
            )
        else:
            print(f"  Status    : {order.get('status', '?')}")
            print(f"  Order ID  : {order.get('order_id', '?')}")
            print(f"  Action    : {order.get('action', '?')}")
            print(f"  Contracts : {order.get('contracts', 0)}")

    _print_warnings(report.get("warnings", []))
    print()


def _print_check_report(report: dict) -> None:
    ts = report.get("timestamp", "")[:19]
    ks = report.get("kill_switch", "?")

    _hdr(f"Health Check  {ts}")
    print(f"  Kill Switch : {'ACTIVE ✓' if ks == 'ACTIVE' else 'KILLED ✗'}")
    if ks != "ACTIVE":
        print(f"  Reason      : {report.get('kill_switch_reason', '?')}")

    _hdr("Position")
    _print_position(report.get("position", {}))

    _hdr("Account (Broker)")
    _print_account(report.get("account", {}))

    mg = report.get("margin", {})
    if mg:
        _hdr("Margin")
        util = mg.get("utilisation_pct", 0)
        print(f"  Utilisation  : {util:.1f}%")
        print(f"  Open Contracts: {mg.get('open_contracts', 0)}")
        if mg.get("margin_call_risk"):
            print("  ⚠  MARGIN CALL RISK")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="台指期 live trading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["daily", "check", "auto"],
        help="Execution mode (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate signal but do NOT submit orders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip order confirmation prompt.",
    )
    parser.add_argument(
        "--product",
        default="TX",
        choices=["TX", "MTX"],
        help="Futures product (default: TX 大台)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=2_000_000.0,
        help="Fallback equity TWD for position sizing (default: 2,000,000)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        default=None,
        help="Force simulation mode.",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Force live mode. USE WITH CARE.",
    )
    args = parser.parse_args()

    # ── Determine simulation flag ──────────────────────────────────────
    if args.no_simulation:
        simulation = False
        logger.warning("Live mode requested (--no-simulation). Ensure credentials are set.")
    elif args.simulation:
        simulation = True
    else:
        simulation = settings.SHIOAJI_SIMULATION

    # ── Validate credentials ───────────────────────────────────────────
    if not settings.SHIOAJI_API_KEY or not settings.SHIOAJI_SECRET_KEY:
        print("[ERROR] SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY must be set in .env")
        sys.exit(1)

    # ── Initialise orchestrator ────────────────────────────────────────
    print(f"\n  Initialising orchestrator (sim={simulation}, dry_run={args.dry_run}) …")
    from tw_futures.orchestrator import TaifexOrchestrator, _is_post_market, _is_trading_day

    try:
        orch = TaifexOrchestrator(
            product=args.product,
            simulation=simulation,
            dry_run=args.dry_run,
            initial_equity=args.equity,
        )
    except Exception as exc:
        print(f"[ERROR] Orchestrator init failed: {exc}")
        logger.exception("Orchestrator init failed")
        sys.exit(1)

    # ── Resolve auto mode ──────────────────────────────────────────────
    mode = args.mode
    if mode == "auto":
        from datetime import date

        if _is_trading_day(date.today()) and _is_post_market():
            mode = "daily"
            logger.info("auto mode → daily (trading day, post-market)")
        else:
            mode = "check"
            logger.info("auto mode → check (non-trading day or pre-market)")

    # ── Execute ────────────────────────────────────────────────────────
    if mode == "check":
        report = orch.run_check()
        _print_check_report(report)

    elif mode == "daily":
        # For daily mode with orders: confirmation gate
        if not args.dry_run and not args.force:
            # Preview signal first without executing (use dry_run=True temporarily)
            logger.info("Running signal preview (no order yet) …")
            orch.dry_run = True
            preview = orch.run_daily()
            orch.dry_run = False

            # run_daily may redirect to run_check on non-trading days
            if preview.get("mode") == "check":
                _print_check_report(preview)
                sys.exit(0)

            if "error" in preview:
                print(f"\n[ERROR] {preview['error']}")
                sys.exit(1)

            _print_daily_report(preview)
            sig = preview.get("signal", {})
            act = sig.get("action", "hold")

            if act in ("buy", "sell", "close"):
                if not _confirm_order(sig, args.product):
                    print("  Order cancelled by user.")
                    sys.exit(0)
                # Now actually execute
                logger.info("Executing order …")
                import datetime as _dt

                report = orch._run_daily_inner(_dt.date.today())
            else:
                # No order to place; report is the preview
                report = preview
        else:
            report = orch.run_daily()

        # run_daily may have redirected to check mode (non-dry-run on non-trading day)
        if report.get("mode") == "check":
            _print_check_report(report)
        elif "error" in report:
            print(f"\n[ERROR] {report['error']}")
            sys.exit(1)
        else:
            _print_daily_report(report)

    # Print raw JSON for debug convenience
    if "--debug" in sys.argv:
        print("\n--- raw report JSON ---")
        print(json.dumps(report if "report" in dir() else {}, indent=2, default=str))


if __name__ == "__main__":
    main()
