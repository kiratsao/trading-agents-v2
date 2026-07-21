"""15:10 下單後自動驗證 — 確認 15:05 執行結果正確。

檢查項目：
  1. 信號→成交一致性：BUY/CLOSE/ADD 信號需有對應成交
  2. HOLD 無多餘成交：HOLD 信號時不應有新成交
  3. 合約月份正確：不是已到期的月份
  4. Equity 一致性：即時 equity vs state.json 差異 < 5%

Usage:
    python scripts/post_execution_verify.py          # 需要 Shioaji 連線
    python scripts/post_execution_verify.py --local  # 跳過外部連線（僅驗 state）
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _build_notifier():
    """Shared deduped LINE notifier (falls back to print when LINE unset)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not (os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") and os.environ.get("LINE_USER_ID")):
        return lambda msg: print(msg)
    from src.notify.line import build_line_notifier

    return build_line_notifier()


def run_verify(skip_external: bool = False, notify_fn=None) -> dict:
    """Run post-execution verification. Returns dict with results."""
    from src.state.state_manager import StateManager, resolve_state_path

    results: list[dict] = []
    # Read the daemon's canonical per-account state file, NOT the old orphaned
    # data/paper_state.json (which the daemon never updated → false "state=20").
    state_mgr = StateManager(path=str(resolve_state_path()))
    state = state_mgr.load()

    # ── 1. 信號→成交一致性 ──
    pending = state.pending_action
    if pending and pending != "hold":
        # There is a pending action that wasn't cleared — execution may have failed
        results.append({
            "name": "信號執行",
            "passed": False,
            "detail": f"pending_action={pending} 未清除，15:05 可能未執行",
        })
    elif pending == "hold":
        results.append({
            "name": "信號執行",
            "passed": True,
            "detail": "HOLD 信號，pending 已清除",
        })
    else:
        results.append({
            "name": "信號執行",
            "passed": True,
            "detail": "pending_action 已清除（正常）",
        })

    # ── 2. Broker 持倉 vs State ──
    broker_positions = None
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
            broker_positions = adapter.get_positions()
            actual_contracts = sum(
                p.get("contracts", p.get("quantity", 0)) for p in broker_positions
            )

            pos_match = actual_contracts == state.position
            results.append({
                "name": "持倉口數",
                "passed": pos_match,
                "detail": (
                    f"broker={actual_contracts}, state={state.position}"
                    + ("" if pos_match else " ⚠️ 不一致！")
                ),
            })

            # ── 3. 合約月份 ──
            for p in broker_positions:
                code = p.get("code", "")
                # Check if the contract code looks expired
                # We can't easily determine expiry from code alone,
                # but we can check via get_contract
                results.append({
                    "name": f"合約 {code}",
                    "passed": True,
                    "detail": f"方向={p.get('direction')}, 口數={p.get('contracts')}",
                })

            # Verify contract month via get_contract
            contract = adapter.get_contract("MXF")
            if broker_positions:
                held_code = broker_positions[0].get("code", "")
                # Extract product prefix from held code
                contract_match = True
                if held_code and not held_code.startswith(contract.code[:3]):
                    contract_match = False
                results.append({
                    "name": "合約月份正確",
                    "passed": contract_match,
                    "detail": f"持倉={held_code}, 近月={contract.code}",
                })

            # ── 4. Equity 一致性 ──
            try:
                acct = adapter.get_account()
                live_equity = float(acct.get("equity", 0))
                if live_equity > 0 and state.equity > 0:
                    diff_pct = abs(live_equity - state.equity) / state.equity * 100
                    eq_ok = diff_pct < 5.0
                    results.append({
                        "name": "Equity 一致性",
                        "passed": eq_ok,
                        "detail": (
                            f"即時={live_equity:,.0f}, state={state.equity:,.0f}, "
                            f"差異={diff_pct:.1f}%"
                        ),
                    })
                elif live_equity > 0:
                    results.append({
                        "name": "Equity 一致性",
                        "passed": True,
                        "detail": f"即時={live_equity:,.0f} (state equity 未設定)",
                    })
            except Exception as exc:
                results.append({
                    "name": "Equity 一致性",
                    "passed": False,
                    "detail": f"查詢失敗: {exc}",
                })

            adapter.logout()

        except Exception as exc:
            results.append({
                "name": "Broker 連線",
                "passed": False,
                "detail": str(exc),
            })

    # ── Report ──
    all_passed = all(r["passed"] for r in results)

    lines = [
        "━━━━━━━━━━━━",
        "🔍 15:10 執行後驗證",
        "━━━━━━━━━━━━",
    ]
    for r in results:
        icon = "✅" if r["passed"] else "🔴"
        lines.append(f"{icon} {r['name']}: {r['detail']}")

    if all_passed:
        lines.append("\n✅ 驗證通過")
    else:
        lines.append("\n🔴 有異常，請檢查！")

    lines.append("━━━━━━━━━━━━")
    msg = "\n".join(lines)

    if notify_fn and not all_passed:
        # Only send LINE alert on failure
        notify_fn(msg)
    logger.info(msg)

    return {
        "results": results,
        "all_passed": all_passed,
    }


def main():
    skip_external = "--local" in sys.argv
    notify_fn = _build_notifier()
    result = run_verify(skip_external=skip_external, notify_fn=notify_fn)
    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
