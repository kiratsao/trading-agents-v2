"""V2b 排程主程式 — 每日盤後執行策略邏輯。

Usage
-----
    python -m src.scheduler.main               # simulation (paper trading, daemon)
    python -m src.scheduler.main --run-once    # simulation, 執行一次後退出
    python -m src.scheduler.main --live        # live trading
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = "config/accounts.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2b trading scheduler")
    p.add_argument("--live", action="store_true", help="Enable live trading (default: simulation)")
    p.add_argument(
        "--run-once",
        action="store_true",
        help="Run one daily cycle then exit (useful for cron/dry-run)",
    )
    p.add_argument("--config", default=_CONFIG_PATH, help="Accounts config YAML")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise SystemExit("PyYAML is required: uv add pyyaml")
    path = Path(config_path)
    if not path.exists():
        logger.error("Config not found: %s", config_path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Build orchestrator from config
# ---------------------------------------------------------------------------


def _data_path_for(product: str) -> Path:
    """Canonical parquet path for a product (matches scripts/init_data.py)."""
    return Path(f"data/{product}_Daily_Clean_2020_to_now.parquet")


def _build_orchestrators(cfg: dict, live: bool = False) -> dict:
    """Build one V2bOrchestrator per account in cfg['accounts'].

    Returns dict[account_name → V2bOrchestrator]. All accounts share a single
    notifier instance so one LINE channel sees everything.
    """
    from src.scheduler.orchestrator import V2bOrchestrator
    from src.state.state_manager import StateManager
    from src.strategy.v2b_engine import V2bEngine

    notify_fn = _build_notifier()
    orchestrators: dict = {}

    for name, acc in cfg.get("accounts", {}).items():
        product = acc.get("product", "MXF")
        params = acc.get("strategy_params", {})
        ladder = [
            {"equity": entry["equity"], "contracts": entry["contracts"]}
            for entry in acc.get("scale_ladder", [])
        ]

        engine = V2bEngine(
            product=product,
            ema_fast=params.get("ema_fast", 30),
            ema_slow=params.get("ema_slow", 100),
            trail_atr_mult=params.get("atr_stop_mult", 2.0),
            confirm_days=params.get("confirm_days", 2),
            adx_threshold=params.get("adx_threshold", 25),
            ladder=ladder,
            max_contracts=acc.get("max_contracts"),
            margin_per_contract=acc.get("margin_per_contract"),
        )
        # Per-account override of the product-level default for settlement
        # behavior (e.g. you could disable rollover for an MXF account that
        # holds across expiry intentionally).
        if "settlement_force_close" in acc:
            engine.settlement_force_close = bool(acc["settlement_force_close"])

        state_mgr = StateManager(
            path=f"data/state_{name}.json",
            initial_equity=float(acc.get("equity", 350_000)),
        )

        sessions = acc.get("sessions", {})
        day_cfg = sessions.get("day", {})

        orchestrators[name] = V2bOrchestrator(
            strategy=engine,
            state_mgr=state_mgr,
            notify_fn=notify_fn,
            enable_tsmc_signal=day_cfg.get("enable_tsmc_signal", False),
            data_path=_data_path_for(product),
            decision_time=day_cfg.get("decision_time", "14:30"),
            execution_timing=day_cfg.get("execution_timing", "next_open"),
            live=live,
        )
        logger.info(
            "built account %s: product=%s state=%s data=%s timing=%s",
            name, product, state_mgr.path, _data_path_for(product),
            day_cfg.get("execution_timing", "next_open"),
        )

    if not orchestrators:
        raise SystemExit("No accounts found in config — check accounts.yaml")
    return orchestrators


# ---------------------------------------------------------------------------
# Broker factory
# ---------------------------------------------------------------------------


def _build_broker(live: bool):
    """Return a ShioajiAdapter when live=True, else None (paper mode)."""
    if not live:
        return None
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        logger.error("SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY not set — cannot build live broker")
        return None
    cert_path = os.environ.get("SHIOAJI_CERT_PATH") or None
    cert_password = os.environ.get("SHIOAJI_CERT_PASSWORD") or None
    person_id = os.environ.get("SHIOAJI_PERSON_ID") or None
    from tw_futures.executor.shioaji_adapter import ShioajiAdapter

    adapter = ShioajiAdapter(
        api_key=api_key,
        secret_key=secret_key,
        simulation=False,  # live mode
        cert_path=cert_path,
        cert_password=cert_password,
        person_id=person_id,
    )
    logger.info("Live broker (ShioajiAdapter) initialised")
    return adapter


# ---------------------------------------------------------------------------
# LINE notifier
# ---------------------------------------------------------------------------


def _build_notifier():
    """Build LINE push message notifier if env vars present, else no-op."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")
    if not token or not user_id:
        logger.info("LINE notifier disabled (LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID not set)")
        return lambda msg: None

    def _notify(msg: str) -> None:
        import json
        import urllib.request

        payload = {
            "to": user_id,
            "messages": [{"type": "text", "text": msg}],
        }
        req = urllib.request.Request(
            "https://api.line.me/v2/bot/message/push",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("LINE notified: %s", resp.status)
        except Exception as exc:
            logger.warning("LINE notify failed: %s", exc)

    return _notify


# ---------------------------------------------------------------------------
# Startup notification
# ---------------------------------------------------------------------------


def _query_live_equity(broker) -> tuple[float, str]:
    """Query real-time equity from broker API.

    Returns (equity_value, source_label) where source_label is "即時" or "估算".
    Uses api.margin() via get_account() → margin.equity (= balance + unrealized PnL).
    Falls back to get_positions() unrealized PnL sum if get_account() fails.
    """
    if broker is None:
        return 0.0, "估算"

    # Primary: get_account() uses api.margin() which returns equity directly
    try:
        acct = broker.get_account()
        equity = float(acct.get("equity", 0))
        if equity > 0:
            logger.info("Live equity from api.margin(): %.0f", equity)
            return equity, "即時"
    except Exception as exc:
        logger.warning("get_account() failed: %s — trying get_positions()", exc)

    # Fallback: sum unrealized PnL from positions (incomplete — no cash balance)
    try:
        positions = broker.get_positions()
        total_pnl = sum(float(p.get("unrealized_pnl", 0)) for p in positions)
        if positions:
            logger.info(
                "Live positions PnL: %.0f (no balance — partial data)", total_pnl
            )
            return total_pnl, "持倉PnL"
    except Exception as exc:
        logger.warning("get_positions() also failed: %s", exc)

    return 0.0, "估算"


def _send_startup_notification(
    orchestrators: dict, notify_fn, mode: str, broker=None
) -> None:
    """Send a single multi-account startup notification.

    One LINE message lists every configured account: product, position, and
    last close from its product-specific parquet. Per-account live snapshot
    via Shioaji is deferred to Step 5/6 — the snapshot helper is still
    MXF-hardcoded.
    """
    try:
        from pathlib import Path

        import pandas as pd

        timings = {o.execution_timing for o in orchestrators.values()}
        if timings == {"night_open"}:
            schedule_str = "14:30 信號 + 15:05 下單"
        elif timings == {"next_open"}:
            schedule_str = "14:30 信號+下單"
        else:
            schedule_str = "/".join(sorted(timings))

        lines = [
            "🚀 trading-agents-v2 啟動",
            "━━━━━━━━━━━━",
            f"模式: {mode}",
            f"帳戶數: {len(orchestrators)}",
            f"排程: {schedule_str}",
            "━━━━━━━━━━━━",
        ]

        for name, orch in orchestrators.items():
            state = orch.state_mgr.load()
            product = orch.strategy.product
            position_str = (
                f"{state.position}口 @ {state.entry_price:,.0f}"
                if state.position > 0
                else "空倉"
            )

            # Last close from this account's parquet
            last_close: float | None = None
            try:
                if Path(orch.data_path).exists():
                    df = pd.read_parquet(orch.data_path)
                    if len(df) > 0:
                        last_close = float(df["close"].iloc[-1])
            except Exception:
                pass

            lines.append(f"[{name} · {product}]")
            if last_close is not None:
                lines.append(f"  最新收盤: {last_close:,.2f}")
            lines.append(f"  持倉: {position_str}")
            if state.position > 0 and state.entry_price and last_close is not None:
                tick_val = orch.strategy.point_value
                unrealized = (last_close - state.entry_price) * state.position * tick_val
                pct = (unrealized / state.equity * 100) if state.equity > 0 else 0.0
                icon = "🟢" if unrealized >= 0 else "🔴"
                lines.append(f"  持倉損益: {icon} {unrealized:+,.0f} NTD ({pct:+.1f}%)")
            lines.append(f"  淨值: {state.equity:,.0f} NTD")

        lines.append("━━━━━━━━━━━━")
        notify_fn("\n".join(lines))
        logger.info("Startup notification sent (%d accounts).", len(orchestrators))
    except Exception as exc:
        logger.warning("Startup notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = _parse_args(argv)
    mode = "LIVE" if args.live else "SIMULATION"
    run_once = args.run_once

    logger.info(
        "V2b Scheduler starting | mode=%s | run_once=%s | config=%s",
        mode,
        run_once,
        args.config,
    )

    cfg = _load_config(args.config)
    is_live = args.live
    orchestrators = _build_orchestrators(cfg, live=is_live)
    # All orchestrators share one notifier (built once inside _build_orchestrators).
    notify_fn = next(iter(orchestrators.values())).notify_fn

    # ── Startup notification ──────────────────────────────────────────
    startup_broker = _build_broker(is_live)
    _send_startup_notification(orchestrators, notify_fn, mode, broker=startup_broker)
    if startup_broker:
        startup_broker.logout()

    # Partition accounts by execution timing — each timing has its own cron set.
    night_accounts = {
        n: o for n, o in orchestrators.items() if o.execution_timing == "night_open"
    }
    next_open_accounts = {
        n: o for n, o in orchestrators.items() if o.execution_timing == "next_open"
    }

    # ── Run-once mode ─────────────────────────────────────────────────
    if run_once:
        broker = _build_broker(is_live)
        try:
            if night_accounts:
                logger.info("── run-once night_open: Phase 1 (signal) ──")
                for name, orch in night_accounts.items():
                    sig_result = orch.run_signal(broker=broker)
                    logger.info("run_signal[%s] result: %s", name, sig_result)
                    print(f"\n  [14:30 signal · {name}]")
                    print(f"  action   : {sig_result.get('action')}")
                    print(f"  contracts: {sig_result.get('contracts', 0)}")
                    print(f"  reason   : {sig_result.get('reason')}")

                logger.info("── run-once night_open: Phase 2 (execution) ──")
                for name, orch in night_accounts.items():
                    exec_result = orch.run_execution(broker=broker)
                    logger.info("run_execution[%s] result: %s", name, exec_result)
                    print(f"\n  [15:05 execution · {name}]")
                    print(f"  action   : {exec_result.get('action')}")
                    if "pnl_twd" in exec_result:
                        print(f"  pnl_twd  : {exec_result['pnl_twd']:.0f} NTD")

            if next_open_accounts:
                logger.info("── run-once next_open: daily cycle ──")
                for name, orch in next_open_accounts.items():
                    result = orch.run_daily(broker=broker)
                    logger.info("run_daily[%s] result: %s", name, result)
                    print(f"\n  [{name}] action: {result.get('action')}")
        finally:
            if broker:
                broker.logout()
        return

    # ── Daemon mode (APScheduler) ─────────────────────────────────────
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore[import]
        from apscheduler.triggers.cron import CronTrigger  # type: ignore[import]
    except ImportError:
        logger.error("apscheduler not installed — run: uv add apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Asia/Taipei")

    # 14:25 data update — iterates every unique product in accounts.yaml.
    # Per-product failure is isolated (logged + LINE alert) and does not
    # block the others or the 14:30 signal job.
    def _safe_data_update():
        try:
            from src.data.daily_updater import update_all

            results = update_all(config_path=args.config, notify_fn=notify_fn)
            for r in results:
                if not r.get("success"):
                    logger.error(
                        "14:25 data update [%s] failed: %s",
                        r.get("product"), r.get("error"),
                    )
                else:
                    logger.info("14:25 data update [%s]: %s", r.get("product"), r)
        except Exception as exc:
            logger.error("🔴 資料更新失敗: %s", exc)
            try:
                notify_fn(f"🔴 資料更新失敗: {exc}")
            except Exception:
                pass

    scheduler.add_job(
        func=_safe_data_update,
        trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=25),
        id="v2b_data_update",
        name="V2b 14:25 資料更新",
    )

    # Each job creates a fresh broker to avoid token expiry, then iterates the
    # accounts whose execution timing matches.
    def _run_signal_all():
        b = _build_broker(is_live)
        try:
            for name, orch in night_accounts.items():
                try:
                    orch.run_signal(broker=b)
                except Exception as exc:
                    logger.exception("run_signal[%s] failed: %s", name, exc)
        finally:
            if b:
                b.logout()

    def _run_execution_all():
        b = _build_broker(is_live)
        try:
            for name, orch in night_accounts.items():
                try:
                    orch.run_execution(broker=b)
                except Exception as exc:
                    logger.exception("run_execution[%s] failed: %s", name, exc)
        finally:
            if b:
                b.logout()

    def _run_daily_all():
        b = _build_broker(is_live)
        try:
            for name, orch in next_open_accounts.items():
                try:
                    orch.run_daily(broker=b)
                except Exception as exc:
                    logger.exception("run_daily[%s] failed: %s", name, exc)
        finally:
            if b:
                b.logout()

    # 14:28 connection warm-up (pre-build broker to verify connectivity)
    def _warmup_connection():
        logger.info("14:28 連線預熱開始")
        for attempt in range(1, 3):
            try:
                b = _build_broker(is_live)
                if b is None:
                    logger.info("14:28 warmup: simulation mode, no broker needed")
                    return
                contract = b.get_contract("MXF")
                logger.info(
                    "14:28 warmup OK: contract=%s (attempt %d)", contract.code, attempt
                )
                b.logout()
                return
            except Exception as exc:
                logger.warning("14:28 warmup attempt %d failed: %s", attempt, exc)
                if attempt >= 2:
                    msg = f"🔴 14:28 連線預熱失敗（2次），14:30 信號可能失敗: {exc}"
                    logger.error(msg)
                    try:
                        notify_fn(msg)
                    except Exception:
                        pass

    scheduler.add_job(
        func=_warmup_connection,
        trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=28),
        id="v2b_warmup",
        name="V2b 14:28 連線預熱",
    )

    if night_accounts:
        scheduler.add_job(
            func=_run_signal_all,
            trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
            id="v2b_signal",
            name=f"V2b 14:30 信號 ×{len(night_accounts)}",
        )
        scheduler.add_job(
            func=_run_execution_all,
            trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=5),
            id="v2b_execution",
            name=f"V2b 15:05 夜盤下單 ×{len(night_accounts)}",
        )

        # 15:10 post-execution verification
        def _post_verify():
            try:
                from scripts.post_execution_verify import run_verify

                run_verify(skip_external=not is_live, notify_fn=notify_fn)
            except Exception as exc:
                logger.error("15:10 post-execution verify failed: %s", exc)

        scheduler.add_job(
            func=_post_verify,
            trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=10),
            id="v2b_post_verify",
            name="V2b 15:10 執行後驗證",
        )

        logger.info(
            "Scheduler registered night ×%d: 14:30 signal + 15:05 execution + "
            "15:10 verify",
            len(night_accounts),
        )

    if next_open_accounts:
        scheduler.add_job(
            func=_run_daily_all,
            trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
            id="v2b_daily",
            name=f"V2b 14:30 每日執行 ×{len(next_open_accounts)}",
        )
        logger.info(
            "Scheduler registered next_open ×%d: 14:30 daily",
            len(next_open_accounts),
        )

    # 20:00 pre-settlement check (runs every weekday, skips if not pre-settlement)
    def _pre_settlement_check():
        try:
            from scripts.pre_settlement_check import run_check

            run_check(skip_external=not is_live, notify_fn=notify_fn)
        except Exception as exc:
            logger.error("20:00 pre-settlement check failed: %s", exc)

    scheduler.add_job(
        func=_pre_settlement_check,
        trigger=CronTrigger(day_of_week="mon-fri", hour=20, minute=0),
        id="v2b_pre_settlement",
        name="V2b 20:00 結算日前檢查",
    )

    logger.info(
        "Scheduler registered: 14:25 (data) + 14:28 (warmup) + night=%d + "
        "next_open=%d + 20:00 (pre-settlement)",
        len(night_accounts), len(next_open_accounts),
    )

    logger.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
