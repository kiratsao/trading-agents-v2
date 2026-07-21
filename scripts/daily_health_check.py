"""Daily Health Check — 每天 08:00 自動跑，檢查系統狀態。

Usage: python scripts/daily_health_check.py

異常時透過 LINE 告警。加到 cron：
0 8 * * 1-5 cd ~/trading-agents-v2 && python scripts/daily_health_check.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.state.state_manager import resolve_state_path
from src.utils.freshness import check_parquet_freshness

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA = Path("data/MXF_Daily_Clean_2020_to_now.parquet")
# Daemon's canonical per-account state file (single source of truth via
# accounts.yaml), NOT the old orphaned data/paper_state.json.
_STATE = resolve_state_path()

# Parquet freshness now lives in src.utils.freshness (single source of truth).


def _send_line(msg: str) -> None:
    """Shared deduped LINE notifier (see src/notify/line.py)."""
    from src.notify.line import build_line_notifier

    build_line_notifier()(msg)


def _refresh_state_equity_from_broker() -> str:
    """08:00 hook: read live equity from Shioaji and cache it into the daemon's
    canonical state file (``_STATE``) so downstream consumers (sizing, the
    health-check report below) don't drift on stale values.

    This writes ONLY ``state.equity`` (position/contracts are preserved from the
    loaded state) — the same operation the daemon's own ``_persist_live_equity``
    performs. At 08:00 the daemon is idle (signal runs 14:30), so there is no
    write race on the canonical file.

    No-ops cleanly when credentials / network are unavailable; the
    existing cached value is preserved on every failure mode.
    Returns a one-line summary for the health-check report."""
    if not _STATE.exists():
        return ""
    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        return "equity refresh skipped (no Shioaji credentials)"

    try:
        from src.scheduler.orchestrator import _persist_live_equity
        from src.state.state_manager import StateManager
        from tw_futures.executor.shioaji_adapter import ShioajiAdapter
    except Exception as exc:
        logger.warning("equity-refresh import failed: %s", exc)
        return ""

    state_mgr = StateManager(path=str(_STATE))
    state = state_mgr.load()

    broker = None
    try:
        broker = ShioajiAdapter(
            api_key=api_key,
            secret_key=secret_key,
            simulation=False,
            cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
            cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
            person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
        )
        equity, src = _persist_live_equity(broker, state, state_mgr)
    except Exception as exc:
        logger.warning("equity-refresh broker step failed: %s", exc)
        return "equity refresh failed (read error)"
    finally:
        if broker is not None:
            try:
                broker.logout()
            except Exception:
                pass

    if src == "即時":
        return f"equity refreshed: {equity:,.0f} (live)"
    return "equity refresh: read failed, cached value preserved"


def main():
    issues = []
    warns = []

    # 0. Refresh state.equity from live broker BEFORE reading state below,
    # so the report reflects the most recent margin balance.
    refresh_summary = _refresh_state_equity_from_broker()
    if refresh_summary:
        logger.info(refresh_summary)

    # 1. Parquet freshness — single source of truth (src.utils.freshness).
    #    2-tier: fresh (✅) or stale (🔴 → exit 1). Time-of-day aware so the
    #    08:00 run no longer false-alarms before the 14:30 update window.
    is_fresh, fresh_msg, _expected = check_parquet_freshness(_DATA)
    if not is_fresh:
        issues.append(fresh_msg)
    logger.info("Parquet freshness: %s", fresh_msg)

    # 2. State sanity
    if _STATE.exists():
        try:
            raw = json.loads(_STATE.read_text())
            s = raw.get("state", {})
            pos = s.get("position", 0)
            eq = s.get("equity", 0)
            entry = s.get("entry_price")
            logger.info("State: position=%d, equity=%.0f, entry=%s",
                        pos, eq, entry)
            if pos > 0 and not entry:
                issues.append(f"State 不一致: position={pos} 但 entry_price=None")
            if pos < 0:
                issues.append(f"State 異常: position={pos} (負數)")
            if eq <= 0:
                issues.append(f"State 異常: equity={eq} (<=0)")
        except Exception as exc:
            issues.append(f"State 讀取失敗: {exc}")
    else:
        logger.info("State file not found (new deployment?)")

    # 3. Build status lines
    if _DATA.exists():
        df = pd.read_parquet(_DATA)
        df.index = pd.to_datetime(df.index)
        bars = len(df)
        latest = df.index[-1].date()
    else:
        bars = 0
        latest = "N/A"

    state_line = "空倉"
    eq_line = "N/A"
    if _STATE.exists():
        try:
            raw = json.loads(_STATE.read_text())
            s = raw.get("state", {})
            pos = s.get("position", 0)
            eq = float(s.get("equity", 0))
            entry = s.get("entry_price")
            eq_line = f"{eq:,.0f}"
            if pos > 0 and entry:
                state_line = f"{pos}口 @ {entry:,.0f}"
            elif pos > 0:
                state_line = f"{pos}口"
        except Exception:
            pass

    # 4. PnL tracking (optional — only if investors.yaml exists)
    pnl_line = ""
    try:
        from scripts.pnl_tracker import format_pnl_line, track_pnl
        pnl_result = track_pnl()
        if pnl_result:
            pnl_line = "\n" + format_pnl_line(pnl_result)
    except Exception as exc:
        logger.debug("pnl_tracker skipped: %s", exc)

    # 4b. Light deep health check (Round 1 data integrity + Round 3 state).
    #     REPORT ONLY (do_fix=False) — auto-fix is opt-in via
    #     `python scripts/deep_health_check.py --fix`. Escalates any 🔴 into the
    #     08:00 report. Never breaks the core check (wrapped).
    try:
        from scripts.deep_health_check import run_deep_health_check
        deep = run_deep_health_check(light=True, do_fix=False)
        if deep.get("alert"):
            issues.append(f"Deep check (light): {deep['alert']} 項 🔴 — 見 log")
    except Exception as exc:
        logger.debug("deep_health_check (light) skipped: %s", exc)

    # 5. Report — always notify
    status_block = (
        f"資料: {bars} bars ({latest})\n"
        f"持倉: {state_line}\n"
        f"淨值: {eq_line}"
        f"{pnl_line}"
    )
    if issues:
        lines = [f"• {i}" for i in issues] + [f"• ⚠️ {w}" for w in warns]
        msg = "🔴 每日健康檢查異常\n" + "\n".join(lines)
        logger.error(msg)
        _send_line(msg)
        sys.exit(1)
    elif warns:
        msg = (
            "⚠️ 每日健康檢查警告\n"
            + "\n".join(f"• {w}" for w in warns)
            + "\n"
            + status_block
        )
        logger.warning(msg)
        _send_line(msg)
    else:
        msg = f"✅ 每日健康檢查通過\n{status_block}"
        logger.info(msg)
        _send_line(msg)


if __name__ == "__main__":
    main()
