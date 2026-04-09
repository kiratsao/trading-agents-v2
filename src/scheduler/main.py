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
        import json

        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    path = Path(config_path)
    if not path.exists():
        logger.error("Config not found: %s", config_path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Build orchestrator from config
# ---------------------------------------------------------------------------


def _build_orchestrator(cfg: dict, live: bool = False):
    from src.scheduler.orchestrator import V2bOrchestrator
    from src.state.state_manager import StateManager
    from src.strategy.v2b_engine import V2bEngine

    acc = cfg["accounts"]["aggressive"]
    params = acc.get("strategy_params", {})
    ladder = [
        {"equity": entry["equity"], "contracts": entry["contracts"]}
        for entry in acc.get("scale_ladder", [])
    ]

    engine = V2bEngine(
        product="MXF",
        ema_fast=params.get("ema_fast", 30),
        ema_slow=params.get("ema_slow", 100),
        trail_atr_mult=params.get("atr_stop_mult", 2.0),
        confirm_days=params.get("confirm_days", 3),
        ladder=ladder,
    )
    state_mgr = StateManager(path="data/paper_state.json")
    notify_fn = _build_notifier()

    sessions = acc.get("sessions", {})
    day_cfg = sessions.get("day", {})
    execution_timing = day_cfg.get("execution_timing", "next_open")
    decision_time = day_cfg.get("decision_time", "14:30")

    return V2bOrchestrator(
        strategy=engine,
        state_mgr=state_mgr,
        notify_fn=notify_fn,
        enable_tsmc_signal=day_cfg.get("enable_tsmc_signal", False),
        decision_time=decision_time,
        execution_timing=execution_timing,
        live=live,
    )


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


def _send_startup_notification(orchestrator, notify_fn, mode: str) -> None:
    """Send startup notification with current position and schedule info."""
    try:
        state = orchestrator.state_mgr.load()
        position_str = (
            f"{state.position}口 @ {state.entry_price:,.0f}" if state.position > 0 else "空倉"
        )

        # Try to get latest price from Shioaji or parquet
        latest_price = _get_latest_price(orchestrator, live=mode == "LIVE")

        timing = orchestrator.execution_timing
        if timing == "night_open":
            schedule_str = "14:30 信號 + 15:05 下單"
        else:
            schedule_str = "14:30 信號+下單"

        msg = "\n".join(
            [
                "🚀 trading-agents-v2 啟動",
                "━━━━━━━━━━━━",
                f"模式: {mode}",
                f"台指即時: {latest_price}",
                f"目前持倉: {position_str}",
                f"帳戶淨值: {state.equity:,.0f} NTD",
                f"排程: {schedule_str}",
                "━━━━━━━━━━━━",
            ]
        )
        notify_fn(msg)
        logger.info("Startup notification sent.")
    except Exception as exc:
        logger.warning("Startup notification failed: %s", exc)


def _get_latest_price(orchestrator, live: bool = False) -> str:
    """Try Shioaji live snapshot first, fallback to last parquet bar.

    Priority:
      1. ShioajiAdapter.get_snapshots("MXF") → snap["close"]  (即時 last price)
      2. Parquet last close (with multi-path fallback)
      3. "N/A"
    """
    # 1. Try Shioaji snapshot via ShioajiAdapter (live mode only — simulation mode skips
    #    Shioaji to avoid segfault from rapid connect/disconnect in the C extension)
    try:
        import os

        api_key = os.environ.get("SHIOAJI_API_KEY", "")
        secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
        if live and api_key and secret_key:
            from tw_futures.executor.shioaji_adapter import ShioajiAdapter

            adapter = ShioajiAdapter(
                api_key=api_key,
                secret_key=secret_key,
                simulation=False,
                cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
                cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
                person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
            )
            snap = adapter.get_snapshots("MXF")
            adapter.logout()
            close = snap.get("close") or snap.get("last_price")
            ts = snap.get("ts")
            if close:
                if ts:
                    import pandas as pd

                    ts_dt = pd.Timestamp(ts, unit="ns").tz_localize("Asia/Taipei")
                    ts_str = ts_dt.strftime("%m/%d %H:%M")
                else:
                    ts_str = "即時"
                return f"{close:,.0f} ({ts_str})"
    except Exception as exc:
        logger.debug("_get_latest_price Shioaji snapshot failed: %s", exc)
        pass

    # 2. Fallback to parquet last bar (try multiple paths)
    try:
        from pathlib import Path

        import pandas as pd

        candidates = [
            orchestrator.data_path,
            Path("data/TXF_Daily_Real.parquet"),
            Path("../trading-agents-v2/data/TXF_Daily_Real.parquet"),
            Path.home() / "trading-agents-v2" / "data" / "TXF_Daily_Real.parquet",
        ]
        for data_path in candidates:
            if data_path.exists():
                df = pd.read_parquet(data_path)
                if len(df) > 0:
                    last_close = float(df["close"].iloc[-1])
                    last_date = df.index[-1]
                    return f"{last_close:,.0f} ({last_date:%Y-%m-%d} 收盤)"
                break
    except Exception:
        pass

    return "N/A"


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
    orchestrator = _build_orchestrator(cfg, live=args.live)
    broker = _build_broker(args.live)
    notify_fn = orchestrator.notify_fn

    # ── Startup notification ──────────────────────────────────────────
    _send_startup_notification(orchestrator, notify_fn, mode)

    # ── Run-once mode ─────────────────────────────────────────────────
    if run_once:
        if orchestrator.execution_timing == "night_open":
            logger.info("── run-once night_open: Phase 1 (signal) ──")
            sig_result = orchestrator.run_signal()
            logger.info("run_signal result: %s", sig_result)
            print("\n  [14:30 signal]")
            print(f"  action   : {sig_result.get('action')}")
            print(f"  contracts: {sig_result.get('contracts', 0)}")
            print(f"  reason   : {sig_result.get('reason')}")

            logger.info("── run-once night_open: Phase 2 (execution) ──")
            exec_result = orchestrator.run_execution(broker=broker)
            logger.info("run_execution result: %s", exec_result)
            print("\n  [15:05 execution]")
            print(f"  action   : {exec_result.get('action')}")
            if "pnl_twd" in exec_result:
                print(f"  pnl_twd  : {exec_result['pnl_twd']:.0f} NTD")
        else:
            logger.info("── run-once mode: executing one daily cycle ──")
            result = orchestrator.run_daily(broker=broker)
            logger.info("run_daily result: %s", result)
            print(f"\n  action   : {result.get('action')}")
        return

    # ── Daemon mode (APScheduler) ─────────────────────────────────────
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore[import]
        from apscheduler.triggers.cron import CronTrigger  # type: ignore[import]
    except ImportError:
        logger.error("apscheduler not installed — run: uv add apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Asia/Taipei")

    if orchestrator.execution_timing == "night_open":
        # Phase 1: 14:30 signal
        scheduler.add_job(
            func=orchestrator.run_signal,
            trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
            id="v2b_signal",
            name="V2b 14:30 信號",
        )
        # Phase 2: 15:05 execution
        scheduler.add_job(
            func=lambda: orchestrator.run_execution(broker=broker),
            trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=5),
            id="v2b_execution",
            name="V2b 15:05 夜盤下單",
        )
        logger.info(
            "Scheduler registered: mon-fri 14:30 (signal) + 15:05 (night execution) Asia/Taipei"
        )
    else:
        scheduler.add_job(
            func=lambda: orchestrator.run_daily(broker=broker),
            trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
            id="v2b_daily",
            name="V2b 14:30 每日執行",
        )
        logger.info("Scheduler registered: mon-fri 14:30 Asia/Taipei")

    logger.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
