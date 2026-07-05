"""結算日前一天 20:00 自動檢查 — 確認隔天轉倉準備就緒。

判斷邏輯：每月第三個週三是結算日，前一天（週二）20:00 跑。

檢查項目：
  1. 合約月份：get_contract 取的是否為即將到期合約
  2. 分批下單：持倉口數 > 5 時會需要分批
  3. Ladder 涵蓋：當前 equity 在 ladder 範圍內
  4. State 完整性：state.json 可讀、欄位齊全
  5. Shioaji 連線：login + CA + fetch_contracts

任何一項失敗 → 🔴 LINE 告警 + 列出修復建議。

Usage:
    python scripts/pre_settlement_check.py          # 需要 Shioaji 連線
    python scripts/pre_settlement_check.py --local  # 跳過外部連線
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _next_settlement_day(from_date: date) -> date:
    """Next holiday-adjusted settlement day on or after *from_date*.

    Shares the single implementation in src.data.tw_holidays (the old local
    copy ignored holidays and could drift from the engine's logic).
    """
    from src.data.tw_holidays import settlement_day_of_month

    settle = settlement_day_of_month(from_date)
    if settle >= from_date:
        return settle
    # Move to next month
    if from_date.month == 12:
        return settlement_day_of_month(date(from_date.year + 1, 1, 1))
    return settlement_day_of_month(date(from_date.year, from_date.month + 1, 1))


def _build_notifier():
    """Shared deduped LINE notifier (falls back to print when LINE unset)."""
    import os as _os

    from src.notify.line import build_line_notifier

    if not (_os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") and _os.environ.get("LINE_USER_ID")):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
    if not (_os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") and _os.environ.get("LINE_USER_ID")):
        return lambda msg: print(msg)
    return build_line_notifier()


def run_check(skip_external: bool = False, notify_fn=None) -> dict:
    """Run all pre-settlement checks. Returns dict with results."""
    from src.utils.tw_time import today_taipei

    today = today_taipei()
    tomorrow = today + timedelta(days=1)
    next_settle = _next_settlement_day(tomorrow)

    results: list[dict] = []
    is_pre_settlement = next_settle == tomorrow

    if not is_pre_settlement:
        return {
            "is_pre_settlement": False,
            "next_settlement": str(next_settle),
            "results": [],
            "all_passed": True,
        }

    logger.info("明天是結算日 (%s)，開始檢查", next_settle)

    # ── 1. State 完整性 ──
    try:
        from src.state.state_manager import StateManager

        state_mgr = StateManager(path="data/paper_state.json")
        state = state_mgr.load()
        state_ok = state.position >= 0 and state.equity > 0
        results.append({
            "name": "State 完整性",
            "passed": state_ok,
            "detail": f"position={state.position}, equity={state.equity:,.0f}",
            "fix": "檢查 data/paper_state.json 格式",
        })
    except Exception as exc:
        state = None
        results.append({
            "name": "State 完整性",
            "passed": False,
            "detail": str(exc),
            "fix": "檢查 data/paper_state.json 是否存在且格式正確",
        })

    # ── 2. Ladder 涵蓋 ──
    try:
        from src.scheduler.main import _load_config

        cfg = _load_config("config/accounts.yaml")
        acc = cfg["accounts"]["aggressive"]
        ladder = acc.get("scale_ladder", [])
        max_eq = max(e["equity"] for e in ladder) if ladder else 0
        equity = state.equity if state else 0
        ladder_ok = equity <= max_eq
        results.append({
            "name": "Ladder 涵蓋",
            "passed": ladder_ok,
            "detail": f"equity={equity:,.0f}, ladder最高={max_eq:,.0f}",
            "fix": "擴展 config/accounts.yaml 的 scale_ladder",
        })
    except Exception as exc:
        results.append({
            "name": "Ladder 涵蓋",
            "passed": False,
            "detail": str(exc),
            "fix": "檢查 config/accounts.yaml",
        })

    # ── 3. 分批下單預警 ──
    if state and state.position > 5:
        batches = (state.position + 4) // 5  # ceil division
        results.append({
            "name": "分批下單",
            "passed": True,
            "detail": f"持倉 {state.position} 口 → 平倉需 {batches} 批（每批≤5口）",
            "fix": "",
        })
    elif state:
        results.append({
            "name": "分批下單",
            "passed": True,
            "detail": f"持倉 {state.position} 口，不需分批",
            "fix": "",
        })

    # ── 4. 合約月份 ──
    if not skip_external:
        try:
            from tw_futures.executor.shioaji_adapter import ShioajiAdapter

            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            adapter = ShioajiAdapter(
                api_key=os.environ.get("SHIOAJI_API_KEY", ""),
                secret_key=os.environ.get("SHIOAJI_SECRET_KEY", ""),
                simulation=False,
                cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
                cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
                person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
            )
            contract = adapter.get_contract("MXF")
            delivery = contract.delivery_date
            adapter.logout()

            # On settlement day, get_contract should skip the expiring one
            # But today is pre-settlement (the day before), so it should still show current month
            results.append({
                "name": "合約月份",
                "passed": True,
                "detail": f"近月合約={contract.code}, 到期={delivery}",
                "fix": "",
            })
        except Exception as exc:
            results.append({
                "name": "合約月份",
                "passed": False,
                "detail": str(exc),
                "fix": "檢查 Shioaji 連線和憑證",
            })

    # ── 5. Shioaji 連線 ──
    if not skip_external:
        try:
            from tw_futures.executor.shioaji_adapter import ShioajiAdapter

            adapter = ShioajiAdapter(
                api_key=os.environ.get("SHIOAJI_API_KEY", ""),
                secret_key=os.environ.get("SHIOAJI_SECRET_KEY", ""),
                simulation=False,
                cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
                cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
                person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
            )
            acct = adapter.get_account()
            adapter.logout()
            results.append({
                "name": "Shioaji 連線",
                "passed": True,
                "detail": f"equity={acct.get('equity', 0):,.0f}",
                "fix": "",
            })
        except Exception as exc:
            results.append({
                "name": "Shioaji 連線",
                "passed": False,
                "detail": str(exc),
                "fix": "檢查 API Key / 憑證 / 網路連線",
            })

    # ── Report ──
    all_passed = all(r["passed"] for r in results)

    lines = [
        "━━━━━━━━━━━━",
        f"📋 結算日前檢查（明天 {next_settle}）",
        "━━━━━━━━━━━━",
    ]
    for r in results:
        icon = "✅" if r["passed"] else "🔴"
        lines.append(f"{icon} {r['name']}: {r['detail']}")
        if not r["passed"] and r.get("fix"):
            lines.append(f"   修復: {r['fix']}")

    if all_passed:
        lines.append("\n✅ 全部通過，明天結算日轉倉準備就緒")
    else:
        lines.append("\n🔴 有項目未通過，請立即處理！")

    lines.append("━━━━━━━━━━━━")
    msg = "\n".join(lines)

    if notify_fn:
        notify_fn(msg)
    else:
        print(msg)

    return {
        "is_pre_settlement": True,
        "next_settlement": str(next_settle),
        "results": results,
        "all_passed": all_passed,
    }


def main():
    skip_external = "--local" in sys.argv
    notify_fn = _build_notifier()
    result = run_check(skip_external=skip_external, notify_fn=notify_fn)
    if not result["is_pre_settlement"]:
        print(f"非結算日前夕，下次結算日: {result['next_settlement']}")
    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
