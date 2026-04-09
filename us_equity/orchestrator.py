"""TradingOrchestrator — central scheduler that wires every agent together.

Pipeline (rebalance day)
------------------------
AlpacaFetcher → DataCleaner → MomentumLowTurnoverStrategy
    → RiskManager (KillSwitch → DrawdownGuard → ConcentrationGuard)
    → OrderManager.rebalance() → sleep(30) → Reconciler.check()

Pipeline (daily check)
-----------------------
AlpacaAdapter.get_account() → KillSwitch.check() → DrawdownGuard.check()
    → [optional] close_all_positions()

This file also keeps the original Orchestrator stub so existing imports
from other modules continue to work unchanged.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.config.settings import settings
from core.data.cleaner import fill_missing, filter_zero_volume
from core.monitor import Monitor
from core.risk import (
    DrawdownAction,
    DrawdownGuard,
    KillSwitch,
    TradingHaltedError,
)
from core.risk.slippage import DynamicSlippage
from us_equity.data.fetcher import AlpacaFetcher
from us_equity.executor.alpaca_adapter import AlpacaAdapter, ExecutionError
from us_equity.executor.order_manager import OrderManager
from us_equity.executor.reconciler import Reconciler
from us_equity.risk.risk_manager import RiskManager
from us_equity.strategies import MomentumLowTurnoverStrategy
from us_equity.strategies.dual_momentum import SP500_TOP50

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_SYMBOLS: list[str] = SP500_TOP50 + ["SPY", "SHY"]
_QUARTER_MONTHS: frozenset[int] = frozenset({3, 6, 9, 12})

# ~300 trading days expressed in calendar days
_DATA_LOOKBACK_CALENDAR_DAYS = 420

_ORDER_FILL_WAIT_SECS = 30

_EQUITY_HISTORY_PATH = Path("data/equity_history.json")
_EXECUTION_LOG_PATH = Path("data/execution_log.json")


# ---------------------------------------------------------------------------
# TradingOrchestrator
# ---------------------------------------------------------------------------


class TradingOrchestrator:
    """Central scheduler that wires every agent together.

    Parameters
    ----------
    config_path :
        Path to the ``.env`` file.  Passed through to ``Settings``; the
        default ``".env"`` is loaded automatically by pydantic-settings.

    Example
    -------
    >>> orch = TradingOrchestrator()
    >>> report = orch.run_rebalance(force=False)
    """

    def __init__(self, config_path: str = ".env") -> None:
        # pydantic-settings loads .env automatically; we just validate here.
        self._validate_credentials()

        # ---- Data layer -------------------------------------------------
        self.fetcher = AlpacaFetcher(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )

        # ---- Broker adapter (paper by default) --------------------------
        self.adapter = AlpacaAdapter(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=settings.ALPACA_PAPER,
        )

        # ---- Strategy ---------------------------------------------------
        self.strategy = MomentumLowTurnoverStrategy(universe=list(SP500_TOP50))

        # ---- Risk pipeline ----------------------------------------------
        self.kill_switch = KillSwitch()
        self.drawdown_guard = DrawdownGuard()
        self.risk_manager = RiskManager(
            db_path=settings.DB_PATH,
            kill_switch=self.kill_switch,
            drawdown_guard=self.drawdown_guard,
        )

        # ---- Execution layer --------------------------------------------
        self.slippage_model = DynamicSlippage()
        self.order_manager = OrderManager(db_path=settings.DB_PATH)
        self.reconciler = Reconciler(db_path=settings.DB_PATH)
        self.monitor = Monitor(db_path=settings.DB_PATH)

        # ---- Equity history (for daily-return computation) --------------
        self._equity_history: list[dict] = self._load_equity_history()

        mode_str = "PAPER" if settings.ALPACA_PAPER else "LIVE"
        logger.info(
            "TradingOrchestrator initialised | mode=%s | universe=%d symbols",
            mode_str,
            len(SP500_TOP50),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_rebalance(self, force: bool = False) -> dict:
        """Execute the full rebalance pipeline.

        Steps
        -----
        a. Fetch account snapshot (equity + live positions).
        b. Fetch ~300 trading days of market data for the full universe.
        c. Clean data (forward-fill, zero-volume filter).
        d. Generate target weights via MomentumLowTurnoverStrategy.
        e. Validate through RiskManager (KillSwitch → DrawdownGuard → ConcentrationGuard).
        f. Print preview and await confirmation (skipped when *force=True*).
        g. Submit orders via OrderManager (SELLs first, then BUYs).
        h. Wait 30 s for fills.
        i. Reconcile expected vs actual positions.
        j. Return execution report dict.

        Parameters
        ----------
        force :
            Skip the interactive confirmation prompt (use for scheduled runs).

        Returns
        -------
        dict
            Keys: ``timestamp``, ``mode``, ``equity``, ``target_weights``,
            ``final_weights``, ``orders``, ``drift_report``, ``risk_status``,
            ``ok`` (bool).
        """
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        report: dict[str, Any] = {
            "timestamp": timestamp,
            "mode": "rebalance",
            "ok": False,
        }

        try:
            # ---- a. Account snapshot ------------------------------------
            logger.info("[1/9] Fetching account snapshot ...")
            account = self.adapter.get_account()
            positions = self.adapter.get_positions()
            equity = account["equity"]
            report["equity"] = equity
            logger.info(
                "Account: equity=$%.2f  cash=$%.2f  positions=%d",
                equity,
                account["cash"],
                len(positions),
            )

            # ---- b. Market data fetch ------------------------------------
            logger.info("[2/9] Fetching market data (~300 trading days) ...")
            data = self._fetch_market_data()
            if not data:
                raise RuntimeError("No market data returned — cannot proceed.")
            logger.info("Market data: %d symbol(s) loaded.", len(data))

            # ---- c. Clean -----------------------------------------------
            logger.info("[3/9] Cleaning data ...")
            data = self._clean_data(data)

            # ---- d. Strategy signals ------------------------------------
            logger.info("[4/9] Generating signals ...")
            today_ts = pd.Timestamp.now(tz="UTC").normalize()
            raw_weights = self.strategy.generate_signals(data, today_ts)
            report["target_weights"] = raw_weights
            logger.info(
                "Strategy signals: %d position(s) — %s",
                len(raw_weights),
                ", ".join(f"{s}:{w:.1%}" for s, w in sorted(raw_weights.items())),
            )

            # ---- e. Risk validation -------------------------------------
            logger.info("[5/9] Running risk pipeline ...")
            daily_returns = self._get_daily_returns()
            final_weights = self.risk_manager.validate_trade(raw_weights, equity, daily_returns)
            report["final_weights"] = final_weights
            report["risk_status"] = "ok"
            logger.info(
                "Risk pipeline: approved %d position(s) — %s",
                len(final_weights),
                ", ".join(f"{s}:{w:.1%}" for s, w in sorted(final_weights.items())),
            )

            # ---- f. Confirmation ----------------------------------------
            logger.info("[6/9] Awaiting confirmation ...")
            self._print_rebalance_preview(final_weights, positions, equity)
            if not self._await_confirmation(force):
                logger.warning("Rebalance cancelled by user.")
                report["ok"] = False
                report["cancelled"] = True
                return report

            # ---- g. Submit orders ---------------------------------------
            logger.info("[7/9] Submitting orders ...")
            order_results = self.order_manager.rebalance(
                target_weights=final_weights,
                current_positions=positions,
                equity=equity,
                adapter=self.adapter,
            )
            report["orders"] = order_results
            ok_orders = [o for o in order_results if o.get("status") != "error"]
            err_orders = [o for o in order_results if o.get("status") == "error"]
            logger.info(
                "Orders: %d submitted (%d ok, %d error).",
                len(order_results),
                len(ok_orders),
                len(err_orders),
            )

            # ---- h. Wait for fills --------------------------------------
            logger.info("[8/9] Waiting %d s for orders to fill ...", _ORDER_FILL_WAIT_SECS)
            time.sleep(_ORDER_FILL_WAIT_SECS)

            # ---- i. Reconcile -------------------------------------------
            logger.info("[9/9] Reconciling positions ...")
            actual_positions = self.adapter.get_positions()
            post_equity = self.adapter.get_account()["equity"]
            drift_report = self.reconciler.check(final_weights, actual_positions, post_equity)
            report["drift_report"] = drift_report
            warnings = {s: v for s, v in drift_report.items() if v["status"] != "ok"}
            if warnings:
                logger.warning(
                    "Reconciler: %d drift warning(s) — %s",
                    len(warnings),
                    ", ".join(f"{s}:{v['drift_pct']:+.2f}pp" for s, v in warnings.items()),
                )

            # Update equity history after successful rebalance
            self._update_equity_history(post_equity)
            report["post_equity"] = post_equity
            report["ok"] = True

        except TradingHaltedError as exc:
            logger.error("KILL SWITCH TRIGGERED: %s — aborting rebalance.", exc)
            report["risk_status"] = "kill_switch_triggered"
            report["error"] = str(exc)

        except ExecutionError as exc:
            logger.error("Execution error during rebalance: %s", exc)
            report["error"] = str(exc)

        except Exception as exc:
            logger.error("Unexpected error during rebalance: %s", exc, exc_info=True)
            report["error"] = str(exc)

        finally:
            self._append_execution_log(report)

        return report

    def run_daily_check(self) -> dict:
        """Non-rebalance day check: monitor equity, trigger circuit breakers if needed.

        Steps
        -----
        a. Fetch current equity.
        b. Compute today's portfolio return vs yesterday's equity.
        c. Run KillSwitch.check() on recent daily returns.
        d. Run DrawdownGuard.check() on current equity.
        e. If either triggers → close_all_positions() immediately.
        f. Return status report.

        Returns
        -------
        dict
            Keys: ``timestamp``, ``mode``, ``equity``, ``daily_return``,
            ``kill_switch_status``, ``drawdown_action``, ``ok`` (bool).
        """
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        report: dict[str, Any] = {
            "timestamp": timestamp,
            "mode": "daily_check",
            "ok": False,
        }

        try:
            # ---- a. Account equity --------------------------------------
            account = self.adapter.get_account()
            equity = account["equity"]
            report["equity"] = equity
            logger.info("Daily check | equity=$%.2f", equity)

            # ---- b. Daily return ----------------------------------------
            daily_return = self._compute_daily_return(equity)
            report["daily_return"] = daily_return
            if daily_return is not None:
                logger.info("Daily return: %+.3f%%", daily_return * 100)

            # ---- c. KillSwitch ------------------------------------------
            daily_returns_series = self._get_daily_returns()
            ks_event = self.kill_switch.check(daily_returns_series)
            ks_status = (
                "triggered"
                if ks_event is not None
                else ("killed" if not self.kill_switch.is_active() else "active")
            )
            report["kill_switch_status"] = ks_status

            if ks_event is not None:
                logger.error("KillSwitch triggered: %s", ks_event.reason)

            # ---- d. DrawdownGuard ---------------------------------------
            dd_action = self.drawdown_guard.check(equity)
            report["drawdown_action"] = dd_action.value
            logger.info("DrawdownGuard: action=%s", dd_action.value)

            # ---- e. Emergency liquidation if any circuit breaker fires ---
            liquidated = False
            if ks_event is not None or not self.kill_switch.is_active():
                logger.error("EMERGENCY LIQUIDATION: KillSwitch active — closing all positions.")
                liq_results = self.adapter.close_all_positions()
                report["liquidation_orders"] = liq_results
                liquidated = True
            elif dd_action == DrawdownAction.EXIT:
                logger.error("EMERGENCY LIQUIDATION: DrawdownGuard EXIT — closing all positions.")
                liq_results = self.adapter.close_all_positions()
                report["liquidation_orders"] = liq_results
                liquidated = True

            report["liquidated"] = liquidated

            # Update equity history
            self._update_equity_history(equity)

            # ---- f. Monitor: PnL snapshot, anomaly detection, report ----
            try:
                positions = self.adapter.get_positions()
                monitor_result = self.monitor.run_daily(
                    equity=equity,
                    positions=positions,
                    risk_status={
                        "kill_switch": ks_status,
                        "drawdown_action": dd_action.value,
                        "liquidated": liquidated,
                    },
                    daily_returns=daily_returns_series,
                )
                report["monitor"] = monitor_result
            except Exception as exc:
                logger.error("Monitor.run_daily() failed: %s", exc)

            report["ok"] = True

        except Exception as exc:
            logger.error("Unexpected error during daily check: %s", exc, exc_info=True)
            report["error"] = str(exc)

        finally:
            self._append_execution_log(report)

        return report

    def is_rebalance_day(self, date: dt.date | pd.Timestamp | None = None) -> bool:
        """Return True if *date* is a quarter-end rebalance day.

        A rebalance day is the last *trading day* (Mon–Fri) of a quarter-end
        month (March, June, September, December).  Holidays are not excluded
        from the definition — this is a best-effort check without an exchange
        calendar dependency.

        Parameters
        ----------
        date :
            The date to test.  Defaults to today (UTC).

        Returns
        -------
        bool
        """
        if date is None:
            ts = pd.Timestamp.now(tz="UTC").normalize()
        else:
            ts = pd.Timestamp(date)

        if ts.month not in _QUARTER_MONTHS:
            return False

        # Is this the last business day of the month?
        # The next business day (BDay) falls in a different month → yes.
        next_bday = ts + pd.offsets.BDay(1)
        return next_bday.month != ts.month

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _fetch_market_data(self) -> dict[str, pd.DataFrame]:
        """Fetch recent OHLCV for the full universe; fall back to SQLite cache."""
        from core.data.store import MARKET_DB_PATH, load_ohlcv, save_ohlcv_bulk

        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=_DATA_LOOKBACK_CALENDAR_DAYS)

        try:
            logger.info(
                "Fetching from Alpaca: %d symbol(s) [%s → %s] ...",
                len(_ALL_SYMBOLS),
                start_date,
                end_date,
            )
            data = self.fetcher.fetch_ohlcv_bulk(
                _ALL_SYMBOLS,
                start=str(start_date),
                end=str(end_date),
                adjustment="all",
            )
            if data:
                # Persist to SQLite for offline use
                save_ohlcv_bulk(data, db_path=MARKET_DB_PATH)
                logger.info("Saved %d symbol(s) to cache.", len(data))
            return data

        except Exception as exc:
            logger.warning("Alpaca fetch failed (%s) — falling back to SQLite cache.", exc)
            data: dict[str, pd.DataFrame] = {}
            for sym in _ALL_SYMBOLS:
                df = load_ohlcv(
                    sym,
                    start=str(start_date),
                    end=str(end_date),
                    db_path=MARKET_DB_PATH,
                )
                if not df.empty:
                    data[sym] = df
            if data:
                logger.info("Loaded %d symbol(s) from SQLite cache.", len(data))
            else:
                logger.error("SQLite cache is also empty — no data available.")
            return data

    def _clean_data(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply forward-fill and zero-volume filter to each DataFrame."""
        cleaned: dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if df.empty:
                continue
            df = fill_missing(df, method="ffill")
            df = filter_zero_volume(df)
            if not df.empty:
                cleaned[sym] = df
        removed = set(data) - set(cleaned)
        if removed:
            logger.debug("Cleaner removed %d symbol(s): %s", len(removed), sorted(removed))
        return cleaned

    # ------------------------------------------------------------------
    # Equity history & return computation
    # ------------------------------------------------------------------

    def _load_equity_history(self) -> list[dict]:
        """Load equity history from disk; return empty list if absent."""
        if not _EQUITY_HISTORY_PATH.exists():
            return []
        try:
            return json.loads(_EQUITY_HISTORY_PATH.read_text())
        except Exception as exc:
            logger.warning("Could not load equity history: %s", exc)
            return []

    def _update_equity_history(self, equity: float) -> None:
        """Append today's equity value to the history file (atomic write)."""
        today_str = dt.date.today().isoformat()
        # Replace an existing entry for today if present
        self._equity_history = [e for e in self._equity_history if e.get("date") != today_str]
        self._equity_history.append({"date": today_str, "equity": equity})

        # Keep at most 60 entries (~3 months)
        self._equity_history = self._equity_history[-60:]

        _EQUITY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(_EQUITY_HISTORY_PATH, self._equity_history)

    def _compute_daily_return(self, current_equity: float) -> float | None:
        """Return today's portfolio return (fraction), or None if no history."""
        if not self._equity_history:
            return None
        prev_equity = self._equity_history[-1]["equity"]
        if prev_equity <= 0:
            return None
        return (current_equity - prev_equity) / prev_equity

    def _get_daily_returns(self) -> pd.Series:
        """Build a pd.Series of recent daily returns from equity history."""
        if len(self._equity_history) < 2:
            return pd.Series(dtype=float)
        equities = pd.Series(
            [e["equity"] for e in self._equity_history],
            index=pd.to_datetime([e["date"] for e in self._equity_history]),
        )
        return equities.pct_change().dropna()

    # ------------------------------------------------------------------
    # Execution log
    # ------------------------------------------------------------------

    def _append_execution_log(self, entry: dict) -> None:
        """Append *entry* to the execution log JSON file (atomic write)."""
        _EXECUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing: list[dict] = []
            if _EXECUTION_LOG_PATH.exists():
                existing = json.loads(_EXECUTION_LOG_PATH.read_text())
            existing.append(entry)
            _atomic_json_write(_EXECUTION_LOG_PATH, existing)
        except Exception as exc:
            logger.error("Could not append to execution log: %s", exc)

    # ------------------------------------------------------------------
    # Confirmation UI
    # ------------------------------------------------------------------

    def _print_rebalance_preview(
        self,
        final_weights: dict[str, float],
        current_positions: dict[str, dict],
        equity: float,
    ) -> None:
        """Print a human-readable preview of the proposed rebalance."""
        W = 72
        print()
        print("=" * W)
        print(f"  {'REBALANCE PREVIEW':^{W - 4}}")
        print(f"  {f'Portfolio equity: ${equity:,.2f}':^{W - 4}}")
        print("=" * W)

        # Current portfolio
        print(f"\n  {'CURRENT POSITIONS':}")
        print(f"  {'Symbol':<8}  {'Weight':>8}  {'Market Value':>14}")
        print("  " + "-" * 34)
        total_mv = sum(p.get("market_value", 0) for p in current_positions.values())
        for sym in sorted(current_positions):
            mv = current_positions[sym].get("market_value", 0)
            w = mv / equity if equity > 0 else 0
            print(f"  {sym:<8}  {w:>7.2%}  ${mv:>13,.2f}")
        if not current_positions:
            print("  (no current positions — all cash)")

        # Target portfolio
        print(f"\n  {'TARGET WEIGHTS (after risk pipeline)':}")
        print(f"  {'Symbol':<8}  {'Weight':>8}  {'Target Value':>14}  {'Delta':>12}")
        print("  " + "-" * 48)
        all_syms = sorted(set(final_weights) | set(current_positions))
        for sym in all_syms:
            tw = final_weights.get(sym, 0.0)
            cur_mv = current_positions.get(sym, {}).get("market_value", 0.0)
            target_mv = tw * equity
            delta = target_mv - cur_mv
            delta_str = f"${delta:>+10,.0f}"
            print(f"  {sym:<8}  {tw:>7.2%}  ${target_mv:>13,.2f}  {delta_str}")

        # Estimated order count
        sells = sum(
            1
            for sym in all_syms
            if final_weights.get(sym, 0.0) * equity
            < current_positions.get(sym, {}).get("market_value", 0.0) - 1.0
        )
        buys = sum(
            1
            for sym in all_syms
            if final_weights.get(sym, 0.0) * equity
            > current_positions.get(sym, {}).get("market_value", 0.0) + 1.0
        )
        print()
        print(f"  Estimated orders: {sells} SELL + {buys} BUY")
        print("=" * W)

    def _await_confirmation(self, force: bool) -> bool:
        """Prompt user to confirm the rebalance; return True to proceed."""
        if force:
            logger.info("--force flag set — skipping confirmation prompt.")
            return True
        try:
            answer = input("\n  Confirm rebalance? [y/N]  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # ------------------------------------------------------------------
    # Credential validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_credentials() -> None:
        missing = []
        if not settings.ALPACA_API_KEY:
            missing.append("ALPACA_API_KEY")
        if not settings.ALPACA_SECRET_KEY:
            missing.append("ALPACA_SECRET_KEY")
        if missing:
            raise OSError(
                f"Missing Alpaca credentials in .env: {', '.join(missing)}. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _atomic_json_write(path: Path, obj: Any) -> None:
    """Write *obj* as JSON to *path* atomically (tmp file → rename)."""
    payload = json.dumps(obj, indent=2, default=str)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with open(tmp_fd, "w") as fh:
            fh.write(payload)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Legacy Orchestrator stub (keeps existing imports working)
# ---------------------------------------------------------------------------


class Orchestrator:
    """Legacy stub — kept so existing ``from orchestrator import Orchestrator``
    imports continue to work.  New code should use :class:`TradingOrchestrator`.
    """

    def __init__(self) -> None:
        from core.data.database import init_db

        init_db(settings.DB_PATH)

    def run_once(self) -> None:
        raise NotImplementedError("Use TradingOrchestrator for live trading.")

    def run_daemon(self) -> None:
        raise NotImplementedError("Use TradingOrchestrator for live trading.")


# ---------------------------------------------------------------------------
# CLI (legacy entry point — use scripts/run_live.py for production)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="TradingOrchestrator (use run_live.py instead)")
    parser.add_argument("--rebalance", action="store_true")
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    orch = TradingOrchestrator()
    if args.rebalance:
        orch.run_rebalance(force=args.force)
    elif args.daily:
        orch.run_daily_check()
    else:
        parser.print_help()
