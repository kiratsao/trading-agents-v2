"""TW Futures Orchestrator — daily swing signal generation and execution for 台指期.

Architecture
------------
One instance is created per run.  It connects to Shioaji (simulation or live),
loads historical daily bars from SQLite, fetches the latest bar from TAIFEX,
runs the strategy, applies risk guards, and optionally submits orders.

Position state
--------------
Open position is persisted to ``data/tw_position_state.json`` so state survives
process restarts between the fetch and the next day's run.

Schema::

    {
        "direction": 1,          # 1=long, -1=short, 0=flat
        "contracts": 2,          # absolute contract count (current, signed by direction)
        "entry_price": 20500.0,
        "entry_date": "2026-04-01T00:00:00+08:00",
        "trailing_stop": 20100.0,
        "product": "TX",
        "current_stage": 1,      # pyramid stage: 0=flat, 1=L1, 2=L2, 3=L3
        "max_contracts": 4,      # max contracts for this trade (from equity×risk)
        "entry_atr": 85.5,       # ATR snapshot at initial entry (for L2/L3 thresholds)
        "stage1_size": 2         # L1 contract count (kept for L3 reduce-back target)
    }

Execution log
-------------
Every ``run_daily()`` call appends one record to ``data/tw_execution_log.json``.

Trading calendar
----------------
A simple weekend filter is applied.  A complete holiday list is a TODO.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_POSITION_FILE = _REPO_ROOT / "data" / "tw_position_state.json"
_EXEC_LOG_FILE = _REPO_ROOT / "data" / "tw_execution_log.json"
_MARKET_DB_PATH = _REPO_ROOT / "data" / "db" / "market_data.db"

# TAIFEX day session (Asia/Taipei)
_DAY_OPEN = time(8, 45)
_DAY_CLOSE = time(13, 45)

# Minimum daily returns history for KillSwitch (need ≥1)
_MIN_RETURNS_FOR_KS = 1

# TXF product code → backtester symbol mapping
_PRODUCT_MAP = {"TXF": "TX", "MXF": "MTX"}
_PRODUCT_MAP_REV = {v: k for k, v in _PRODUCT_MAP.items()}  # TX→TXF


# ---------------------------------------------------------------------------
# Trading calendar helpers
# ---------------------------------------------------------------------------


def _is_trading_day(d: date | None = None) -> bool:
    """True if *d* is a Monday–Friday (simplified — no holiday check).

    TODO: add Taiwan Exchange holiday calendar.
    """
    d = d or date.today()
    return d.weekday() < 5  # 0=Mon … 4=Fri


def _is_post_market(ts: datetime | None = None) -> bool:
    """True if current time (Asia/Taipei) is at or after 13:45."""
    import zoneinfo

    tz = zoneinfo.ZoneInfo("Asia/Taipei")
    now = ts or datetime.now(tz=tz)
    now_local = now.astimezone(tz)
    return now_local.time() >= _DAY_CLOSE


# ---------------------------------------------------------------------------
# Position state I/O
# ---------------------------------------------------------------------------

_EMPTY_STATE: dict[str, Any] = {
    "direction": 0,
    "contracts": 0,
    "entry_price": None,
    "entry_date": None,
    "trailing_stop": None,
    "product": "TX",
    # Pyramid state (BreakoutSwingScaledStrategy)
    "current_stage": 0,  # 0=flat, 1=L1, 2=L2, 3=L3
    "max_contracts": 0,  # max_contracts for current trade
    "entry_atr": 0.0,  # ATR snapshot at initial entry
    "stage1_size": 0,  # contracts at L1 (target for L3 reduce)
}


def _load_position(path: Path = _POSITION_FILE) -> dict[str, Any]:
    if not path.exists():
        return dict(_EMPTY_STATE)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return {**_EMPTY_STATE, **state}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load position state (%s) — treating as flat.", exc)
        return dict(_EMPTY_STATE)


def _save_position(state: dict[str, Any], path: Path = _POSITION_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.error("Failed to save position state: %s", exc)


# ---------------------------------------------------------------------------
# Execution log
# ---------------------------------------------------------------------------


def _append_exec_log(record: dict[str, Any], path: Path = _EXEC_LOG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(record)
    try:
        path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to append execution log: %s", exc)


def _load_daily_returns(path: Path = _EXEC_LOG_FILE) -> pd.Series:
    """Build a Series of daily equity returns from the execution log."""
    if not path.exists():
        return pd.Series([], dtype=float)
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return pd.Series([], dtype=float)
    returns = [r.get("equity_return") for r in records if r.get("equity_return") is not None]
    return pd.Series(returns, dtype=float)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TaifexOrchestrator:
    """Daily swing-signal orchestrator for TAIFEX 台指期.

    Strategy: TrendFollowV2Strategy — EMA(50)/EMA(150) Long-Only, 2×ATR hard stop.
    Sizing:   fixed_contracts (default 2 口 TX 大台).
    Capital:  initial_equity (default 200 萬 TWD).

    Parameters
    ----------
    product :
        ``"TX"`` (大台) or ``"MTX"`` (小台).
    simulation :
        If ``True`` (default), connect Shioaji in paper-trading mode.
    dry_run :
        If ``True``, generate signals but do NOT submit any orders.
    initial_equity :
        Fallback equity for position sizing when broker returns 0 (simulation).
    fixed_contracts :
        Number of contracts per trade (default 2).
    """

    def __init__(
        self,
        product: str = "TX",
        simulation: bool = True,
        dry_run: bool = False,
        initial_equity: float = 2_000_000.0,
        fixed_contracts: int = 2,
    ) -> None:
        from core.config.settings import settings
        from core.risk.kill_switch import KillSwitch
        from tw_futures.data.fetcher import TaifexFetcher
        from tw_futures.executor.shioaji_adapter import ShioajiAdapter
        from tw_futures.risk.margin_manager import MarginManager
        from tw_futures.strategies.swing.trend_follow_v2 import TrendFollowV2Strategy

        self.product = product.upper()
        self.simulation = simulation
        self.dry_run = dry_run
        self.initial_equity = initial_equity
        self.fixed_contracts = max(1, fixed_contracts)

        # ── Strategy: EMA(50)/EMA(150) Long-Only, 150pt fixed stop [B] ──
        self.strategy = TrendFollowV2Strategy(
            product=self.product,
            ema_fast=50,
            ema_slow=150,
            long_only=True,
            adx_entry=20.0,
            use_fixed_stop=True,
            fixed_stop_points=150.0,
        )

        # ── Data fetcher (public TAIFEX data — no auth needed) ────────
        self.fetcher = TaifexFetcher(request_delay=1.0)

        # ── Broker adapter ────────────────────────────────────────────
        logger.info("TaifexOrchestrator: initialising ShioajiAdapter (sim=%s)", simulation)
        self.adapter = ShioajiAdapter(
            api_key=settings.SHIOAJI_API_KEY,
            secret_key=settings.SHIOAJI_SECRET_KEY,
            cert_path=settings.SHIOAJI_CERT_PATH or None,
            cert_password=settings.SHIOAJI_CERT_PASSWORD or None,
            person_id=settings.SHIOAJI_PERSON_ID or None,
            simulation=simulation,
        )

        # ── Risk ──────────────────────────────────────────────────────
        self.kill_switch = KillSwitch()
        self.margin_manager = MarginManager(product=self.product)

        # ── Notifier (LINE push, SMTP fallback) ───────────────────────
        from core.monitor.notifier import Notifier

        self.notifier = Notifier()

        logger.info(
            "TaifexOrchestrator ready  product=%s  sim=%s  dry_run=%s",
            self.product,
            simulation,
            dry_run,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_daily(self) -> dict[str, Any]:
        """Post-market daily signal check and (optional) order execution.

        Automatically falls back to ``run_check()`` when:
        - today is not a trading day (weekend)
        - current time is before 13:45 Asia/Taipei

        Returns
        -------
        dict
            Execution report with keys: ``mode``, ``date``, ``signal``,
            ``order``, ``position_after``, ``account``, ``warnings``.
        """
        now = datetime.now(tz=_tz())
        today = now.date()

        if not _is_trading_day(today):
            if self.dry_run:
                logger.info(
                    "run_daily (dry-run): non-trading day (%s %s) — proceeding anyway.",
                    today.strftime("%A"),
                    today,
                )
            else:
                logger.warning(
                    "run_daily called on non-trading day (%s %s) — switching to run_check.",
                    today.strftime("%A"),
                    today,
                )
                return self.run_check()

        if not _is_post_market(now):
            if self.dry_run:
                logger.info(
                    "run_daily (dry-run): before market close (%s) — proceeding anyway.",
                    now.strftime("%H:%M"),
                )
            else:
                logger.warning(
                    "run_daily called before market close (now=%s, market closes 13:45) "
                    "— switching to run_check.",
                    now.strftime("%H:%M"),
                )
                return self.run_check()

        return self._run_daily_inner(today)

    def run_check(self) -> dict[str, Any]:
        """Health check: position status, margin level, kill-switch state.

        Safe to call at any time.
        """
        now = datetime.now(tz=_tz())
        position = _load_position()

        # Broker health (may return zeros in simulation)
        try:
            broker_acct = self.adapter.get_account()
            broker_positions = self.adapter.get_positions()
        except Exception as exc:
            logger.warning("run_check: broker query failed: %s", exc)
            broker_acct = {
                "equity": 0,
                "margin_used": 0,
                "available_margin": 0,
                "unrealized_pnl": 0,
            }
            broker_positions = []

        snap = self.margin_manager.snapshot(broker_acct, broker_positions)

        report: dict[str, Any] = {
            "mode": "check",
            "timestamp": now.isoformat(),
            "kill_switch": "KILLED" if not self.kill_switch.is_active() else "ACTIVE",
            "position": {
                "direction": position["direction"],
                "contracts": position["contracts"],
                "entry_price": position["entry_price"],
                "entry_date": position["entry_date"],
                "trailing_stop": position["trailing_stop"],
                "current_stage": position.get("current_stage", 0),
                "max_contracts": position.get("max_contracts", 0),
            },
            "account": {
                "equity": broker_acct["equity"],
                "margin_used": broker_acct["margin_used"],
                "available_margin": broker_acct["available_margin"],
                "unrealized_pnl": broker_acct["unrealized_pnl"],
            },
            "margin": {
                "utilisation_pct": round(snap.utilisation * 100, 1),
                "open_contracts": snap.open_contracts,
                "excess_margin": snap.excess_margin,
                "margin_call_risk": self.margin_manager.check_margin_call(snap),
            },
        }

        if not self.kill_switch.is_active():
            ev = self.kill_switch._last_event
            report["kill_switch_reason"] = ev.reason if ev else "unknown"

        return report

    # ------------------------------------------------------------------
    # Internal daily logic
    # ------------------------------------------------------------------

    def _run_daily_inner(self, today: date) -> dict[str, Any]:
        warnings: list[str] = []
        now_iso = datetime.now(tz=_tz()).isoformat()

        # ── 1. Fetch latest bar and update SQLite ──────────────────────
        logger.info("run_daily: fetching latest %s bar …", self.product)
        try:
            new_bar = self.fetcher.fetch_daily(
                self.product,
                str(today),
                str(today),
            )
            if not new_bar.empty:
                _upsert_bars(self.product, new_bar)
                logger.info("run_daily: fetched %d bar(s) for %s.", len(new_bar), today)
            else:
                logger.warning("run_daily: no new bar returned for %s %s.", self.product, today)
                warnings.append(f"No bar returned from TAIFEX for {today}")
        except Exception as exc:
            logger.warning("run_daily: bar fetch failed (%s) — using cached data only.", exc)
            warnings.append(f"Bar fetch failed: {exc}")

        # ── 2. Load full history from SQLite ──────────────────────────
        df = _load_history(self.product)
        if df.empty or len(df) < 50:
            msg = f"Insufficient history: {len(df)} rows (need ≥50)"
            logger.error("run_daily: %s", msg)
            return {"mode": "daily", "date": str(today), "error": msg, "warnings": warnings}

        # ── 3. Broker account + positions ─────────────────────────────
        try:
            broker_acct = self.adapter.get_account()
            broker_positions = self.adapter.get_positions()
        except Exception as exc:
            logger.warning("run_daily: broker account query failed: %s", exc)
            warnings.append(f"Broker account query failed: {exc}")
            broker_acct = {
                "equity": 0,
                "margin_used": 0,
                "available_margin": 0,
                "unrealized_pnl": 0,
            }
            broker_positions = []

        # Equity for sizing: use broker value if non-zero, else fallback
        equity = float(broker_acct.get("equity") or 0)
        if equity <= 0:
            equity = self.initial_equity
            warnings.append(
                f"Broker equity=0 (simulation mode) — using fallback equity {equity:,.0f} TWD"
            )

        # ── 4. Load persisted position state ──────────────────────────
        pos_state = _load_position()
        direction = int(pos_state["direction"])  # 1 / -1 / 0
        contracts = int(pos_state["contracts"])  # absolute count
        entry_price = pos_state["entry_price"]  # float | None
        entry_date = pos_state["entry_date"]  # ISO str | None

        current_position = direction * contracts  # signed int for strategy
        entry_price_f = float(entry_price) if entry_price is not None else None
        entry_date_ts = (
            pd.Timestamp(entry_date, tz="Asia/Taipei") if entry_date is not None else None
        )

        # ── 4b. TrendFollowV2Strategy is stateless — no pyramid restore needed ──

        # ── 5. Generate signal ─────────────────────────────────────────
        logger.info(
            "run_daily: generating signal  position=%+d  entry_price=%s  equity=%.0f",
            current_position,
            f"{entry_price_f:.0f}" if entry_price_f else "None",
            equity,
        )
        # Primary signal from V2b strategy (2 contracts)
        signal = self.strategy.generate_signal(
            data=df,
            current_position=current_position,
            entry_price=entry_price_f,
            entry_date=entry_date_ts,
            equity=equity,
        )
        # NOTE: strategy_fast (EMA20/60 × 1口) signal is logged separately;
        # full dual-slot execution requires per-slot position state.
        # Phase-1: orchestrator manages V2b slow only; fast slot added in Phase-2.
        logger.info("run_daily: signal = %s", signal)

        # ── 6. KillSwitch check ────────────────────────────────────────
        ks_event = None
        if not self.kill_switch.is_active():
            warnings.append("KillSwitch is KILLED — no orders will be placed.")
            signal_for_order = None
        else:
            daily_returns = _load_daily_returns()
            if len(daily_returns) >= _MIN_RETURNS_FOR_KS:
                ks_event = self.kill_switch.check(daily_returns)
                if ks_event:
                    warnings.append(f"KillSwitch TRIGGERED: {ks_event.reason}")
                    signal_for_order = None
                else:
                    signal_for_order = signal
            else:
                signal_for_order = signal

        # ── 7. Margin check ────────────────────────────────────────────
        margin_snap = self.margin_manager.snapshot(broker_acct, broker_positions)
        if self.margin_manager.check_margin_call(margin_snap):
            warnings.append("MARGIN CALL detected — no new orders.")
            if signal_for_order and signal_for_order.action in ("buy", "sell"):
                signal_for_order = None

        # ── 8. Execute order ───────────────────────────────────────────
        order_result: dict | None = None
        new_pos_state = dict(pos_state)

        if signal_for_order is not None and signal_for_order.action in (
            "buy",
            "sell",
            "close",
            "add",
            "reduce",
        ):
            # Override signal.contracts with fixed_contracts for buy/sell
            from tw_futures.strategies.swing.trend_follow_v2 import Signal as TFSignal

            if signal_for_order.action in ("buy", "sell"):
                signal_for_order = TFSignal(
                    action=signal_for_order.action,
                    contracts=self.fixed_contracts,
                    reason=signal_for_order.reason,
                    stop_loss=signal_for_order.stop_loss,
                    hard_stop=signal_for_order.hard_stop,
                )
            order_result = self._execute_signal(
                signal=signal_for_order,
                pos_state=pos_state,
                df=df,
                warnings=warnings,
            )
            if order_result:
                # Update position state after successful order
                new_pos_state = _update_position_state(
                    pos_state=pos_state,
                    signal=signal_for_order,
                    order_result=order_result,
                    df=df,
                    product=self.product,
                    strategy=self.strategy,
                )
                _save_position(new_pos_state)
        else:
            # Hold — update trailing stop if in position
            if current_position != 0 and signal.stop_loss is not None:
                new_pos_state["trailing_stop"] = signal.stop_loss
                _save_position(new_pos_state)

        # ── 9. Compute equity return for KillSwitch ────────────────────
        last_close = float(df.iloc[-1]["close"])
        equity_return = _estimate_daily_return(
            pos_state=pos_state,
            current_close=last_close,
            equity=equity,
        )

        # ── 10. Build and log report ───────────────────────────────────
        report: dict[str, Any] = {
            "mode": "daily",
            "date": str(today),
            "timestamp": now_iso,
            "product": self.product,
            "dry_run": self.dry_run,
            "last_close": last_close,
            "signal": {
                "action": signal.action,
                "contracts": signal.contracts,
                "reason": signal.reason,
                "stop_loss": signal.stop_loss,
            },
            "position_before": {
                "direction": direction,
                "contracts": contracts,
                "entry_price": entry_price,
                "entry_date": entry_date,
                "trailing_stop": pos_state.get("trailing_stop"),
                "current_stage": pos_state.get("current_stage", 0),
                "max_contracts": pos_state.get("max_contracts", 0),
            },
            "position_after": {
                "direction": new_pos_state["direction"],
                "contracts": new_pos_state["contracts"],
                "entry_price": new_pos_state["entry_price"],
                "trailing_stop": new_pos_state["trailing_stop"],
                "current_stage": new_pos_state.get("current_stage", 0),
                "max_contracts": new_pos_state.get("max_contracts", 0),
            },
            "order": order_result,
            "account": {
                "equity": broker_acct["equity"],
                "margin_used": broker_acct["margin_used"],
                "available_margin": broker_acct["available_margin"],
                "unrealized_pnl": broker_acct["unrealized_pnl"],
            },
            "equity_return": equity_return,
            "kill_switch": "KILLED" if ks_event else "ACTIVE",
            "warnings": warnings,
        }

        _append_exec_log(report)

        # ── 11. LINE notifications ─────────────────────────────────────
        # a) KillSwitch triggered → critical alert
        if ks_event:
            self.notifier.notify_kill_switch(reason=getattr(ks_event, "reason", str(ks_event)))

        # b) Trade executed (buy / sell / close) → trade notification
        act = signal.action
        if order_result and act in ("buy", "sell", "close"):
            self.notifier.notify_tw_trade(
                action=act,
                contracts=self.fixed_contracts if act in ("buy", "sell") else contracts,
                price=last_close,
                reason=signal.reason or "",
                hard_stop=signal.hard_stop,
            )

        # c) Daily summary → always push (hard stop close has dedicated msg above too)
        try:
            ind_df = self.strategy._compute_indicators(df)
            ind_row = ind_df.iloc[-1]
            indicators = {
                "ema_fast": float(ind_row["ema_fast"]),
                "ema_slow": float(ind_row["ema_slow"]),
                "adx": float(ind_row["adx"]),
                "atr": float(ind_row["atr"]),
                "close": last_close,
            }
        except Exception:
            indicators = {"close": last_close}

        self.notifier.notify_tw_daily(
            date_str=str(today),
            signal=report["signal"],
            position=report["position_after"],
            indicators=indicators,
            kill_switch=report["kill_switch"],
            dry_run=self.dry_run,
        )

        return report

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_signal(
        self,
        signal,
        pos_state: dict,
        df: pd.DataFrame,
        warnings: list[str],
    ) -> dict | None:
        """Submit or simulate an order from *signal*.

        Returns order result dict, or None on failure.
        """
        from tw_futures.executor.shioaji_adapter import ExecutionError

        action = signal.action
        shioaji_product = _PRODUCT_MAP_REV.get(self.product, "TXF")

        direction = int(pos_state.get("direction", 0))

        if action == "close":
            if direction == 0:
                warnings.append("Signal=close but no open position — skipping.")
                return None
            close_dir = "Buy" if direction == -1 else "Sell"
            n_contracts = pos_state["contracts"]
            action_str = f"close ({close_dir} {n_contracts} contracts)"
        elif action == "buy":
            close_dir = "Buy"
            n_contracts = max(1, signal.contracts)
            action_str = f"buy {n_contracts} contracts"
        elif action == "sell":
            close_dir = "Sell"
            n_contracts = max(1, signal.contracts)
            action_str = f"sell {n_contracts} contracts"
        elif action == "add":
            # Scale-in: same direction as existing position
            if direction == 0:
                warnings.append("Signal=add but no open position — skipping.")
                return None
            close_dir = "Buy" if direction == 1 else "Sell"
            n_contracts = max(1, signal.contracts)
            action_str = f"add {n_contracts} contracts (L2 scale-in)"
        elif action == "reduce":
            # Partial close: opposite direction of existing position
            if direction == 0:
                warnings.append("Signal=reduce but no open position — skipping.")
                return None
            close_dir = "Sell" if direction == 1 else "Buy"
            n_contracts = max(1, signal.contracts)
            action_str = f"reduce {n_contracts} contracts (L3 lock-profit)"
        else:
            return None

        if self.dry_run:
            logger.info("DRY-RUN: would %s %s @ market", action_str, shioaji_product)
            return {
                "dry_run": True,
                "action": action,
                "shioaji_action": close_dir,
                "product": shioaji_product,
                "contracts": n_contracts,
                "price_type": "MKT",
                "status": "DRY_RUN",
            }

        try:
            result = self.adapter.submit_order(
                product=shioaji_product,
                action=close_dir,
                contracts=n_contracts,
                price_type="MKT",
                octype="Cover" if action in ("close", "reduce") else "Auto",
            )
            logger.info("Order submitted: %s", result)
            return result
        except ExecutionError as exc:
            msg = f"Order submission failed: {exc}"
            logger.error(msg)
            warnings.append(msg)
            return None


# ---------------------------------------------------------------------------
# Position state update logic
# ---------------------------------------------------------------------------


def _update_position_state(
    pos_state: dict,
    signal,
    order_result: dict,
    df: pd.DataFrame,
    product: str,
    strategy=None,
) -> dict:
    """Return new position state after a successful order.

    For pyramid strategies, ``strategy`` should be the live
    BreakoutSwingScaledStrategy instance so its post-signal pyramid
    state can be persisted.
    """
    action = signal.action
    current_close = float(df.iloc[-1]["close"])
    now_iso = datetime.now(tz=_tz()).isoformat()

    def _pyramid_fields(strat) -> dict:
        """Extract pyramid state from strategy instance."""
        if strat is None:
            return {"current_stage": 0, "max_contracts": 0, "entry_atr": 0.0, "stage1_size": 0}
        return {
            "current_stage": int(getattr(strat, "_pyramid_level", 0)),
            "max_contracts": int(getattr(strat, "_max_contracts", 0)),
            "entry_atr": float(getattr(strat, "_entry_atr", 0.0)),
            "stage1_size": int(getattr(strat, "_stage1_size", 0)),
        }

    if action == "close":
        return {**_EMPTY_STATE, "product": product}

    if action == "buy":
        return {
            "direction": 1,
            "contracts": max(1, signal.contracts),
            "entry_price": current_close,
            "entry_date": now_iso,
            "trailing_stop": signal.stop_loss,
            "product": product,
            **_pyramid_fields(strategy),
        }

    if action == "sell":
        return {
            "direction": -1,
            "contracts": max(1, signal.contracts),
            "entry_price": current_close,
            "entry_date": now_iso,
            "trailing_stop": signal.stop_loss,
            "product": product,
            **_pyramid_fields(strategy),
        }

    if action == "add":
        # Scale in: increment contracts, keep entry_price/date, update pyramid state
        prev_contracts = int(pos_state.get("contracts", 0))
        new_contracts = prev_contracts + max(1, signal.contracts)
        return {
            **pos_state,
            "contracts": new_contracts,
            "trailing_stop": signal.stop_loss
            if signal.stop_loss is not None
            else pos_state.get("trailing_stop"),
            **_pyramid_fields(strategy),
        }

    if action == "reduce":
        # Partial close: decrement contracts, update pyramid state
        prev_contracts = int(pos_state.get("contracts", 0))
        removed = max(1, signal.contracts)
        new_contracts = max(0, prev_contracts - removed)
        if new_contracts == 0:
            return {**_EMPTY_STATE, "product": product}
        return {
            **pos_state,
            "contracts": new_contracts,
            "trailing_stop": signal.stop_loss
            if signal.stop_loss is not None
            else pos_state.get("trailing_stop"),
            **_pyramid_fields(strategy),
        }

    return dict(pos_state)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _upsert_bars(product: str, df: pd.DataFrame) -> None:
    """Save new bars to SQLite."""
    from core.data.store import save_ohlcv_bulk

    df_utc = df.copy()
    if df_utc.index.tz is not None:
        df_utc.index = df_utc.index.tz_convert("UTC")
    else:
        df_utc.index = df_utc.index.tz_localize("UTC")
    save_ohlcv_bulk({product: df_utc}, db_path=str(_MARKET_DB_PATH))


def _load_history(product: str) -> pd.DataFrame:
    """Load all available history from SQLite."""
    from core.data.store import load_ohlcv

    try:
        df = load_ohlcv(product, db_path=str(_MARKET_DB_PATH))
    except Exception as exc:
        logger.error("Failed to load history from SQLite: %s", exc)
        return pd.DataFrame()
    if df.empty:
        return df
    # Ensure tz is Asia/Taipei
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Taipei")
    elif str(df.index.tz) != "Asia/Taipei":
        df.index = df.index.tz_convert("Asia/Taipei")
    return df.sort_index()


def _estimate_daily_return(
    pos_state: dict,
    current_close: float,
    equity: float,
) -> float | None:
    """Rough estimate of today's return for KillSwitch (0 if no position/entry)."""
    from tw_futures.strategies.swing.breakout_swing_scaled import _TICK_VALUE

    direction = int(pos_state.get("direction", 0))
    entry_price = pos_state.get("entry_price")
    contracts = int(pos_state.get("contract", pos_state.get("contracts", 0)))
    product = pos_state.get("product", "TX")

    if direction == 0 or entry_price is None or equity <= 0:
        return None

    tick_val = _TICK_VALUE.get(product, 200.0)
    unrealised = direction * (current_close - float(entry_price)) * contracts * tick_val
    return unrealised / equity


# ---------------------------------------------------------------------------
# Timezone helper
# ---------------------------------------------------------------------------


def _tz():
    import zoneinfo

    return zoneinfo.ZoneInfo("Asia/Taipei")
