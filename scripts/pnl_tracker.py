"""損益分帳追蹤器。

讀取 config/investors.yaml + state equity，計算各人損益，
追加到 data/pnl_history.csv。

config/investors.yaml 範例：
    investors:
      - name: "Kira"
        capital: 530000
      - name: "Dad"
        capital: 350000

此檔案不在 git 中。如果不存在，此腳本靜默跳過。
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_INVESTORS_YAML = Path("config/investors.yaml")
_PNL_CSV = Path("data/pnl_history.csv")
_STATE_JSON = Path("data/paper_state.json")


def load_investors() -> list[dict] | None:
    """Load investors from YAML. Returns None if file missing."""
    if not _INVESTORS_YAML.exists():
        return None
    try:
        import yaml
        with open(_INVESTORS_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        investors = cfg.get("investors", [])
        if not investors:
            return None
        return investors
    except Exception as exc:
        logger.warning("pnl_tracker: failed to load investors.yaml: %s", exc)
        return None


def get_equity() -> tuple[float, str]:
    """Get current equity: Shioaji live first, state file fallback.

    Returns (equity, source) where source is "即時" or "估算".
    """
    # 1. Try Shioaji live equity
    try:
        import os

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("SHIOAJI_API_KEY", "")
        secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
        if api_key and secret_key:
            from tw_futures.executor.shioaji_adapter import ShioajiAdapter
            adapter = ShioajiAdapter(
                api_key=api_key,
                secret_key=secret_key,
                simulation=False,
                cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
                cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
                person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
            )
            try:
                acct = adapter.get_account()
                eq = float(acct.get("equity", 0))
                if eq > 0:
                    return eq, "即時"
            finally:
                adapter.logout()
    except Exception as exc:
        logger.debug("pnl_tracker: Shioaji equity failed: %s", exc)

    # 2. Fallback to state file
    import json
    if _STATE_JSON.exists():
        try:
            raw = json.loads(_STATE_JSON.read_text(encoding="utf-8"))
            eq = float(raw.get("state", {}).get("equity", 0))
            if eq > 0:
                return eq, "估算"
        except Exception:
            pass

    return 0.0, "估算"


def track_pnl() -> dict | None:
    """Calculate PnL splits and append to CSV.

    Returns dict with total and per-investor PnL, or None if no investors.yaml.
    """
    investors = load_investors()
    if not investors:
        return None

    equity, equity_src = get_equity()
    total_capital = sum(inv["capital"] for inv in investors)

    if total_capital <= 0:
        return None

    total_pnl = equity - total_capital
    today_str = str(date.today())

    result = {
        "date": today_str,
        "total_equity": equity,
        "equity_source": equity_src,
        "total_capital": total_capital,
        "total_pnl": total_pnl,
        "investors": [],
    }

    # Per-investor split by capital share
    for inv in investors:
        share = inv["capital"] / total_capital
        inv_pnl = total_pnl * share
        inv_equity = inv["capital"] + inv_pnl
        result["investors"].append({
            "name": inv["name"],
            "capital": inv["capital"],
            "share": share,
            "pnl": inv_pnl,
            "equity": inv_equity,
        })

    # Append to CSV
    _PNL_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _PNL_CSV.exists()

    # Build row: date, total_equity, total_pnl, then per-investor
    row = {"date": today_str, "total_equity": f"{equity:.0f}",
           "total_pnl": f"{total_pnl:.0f}"}
    for inv_data in result["investors"]:
        name = inv_data["name"].lower()
        row[f"{name}_pnl"] = f"{inv_data['pnl']:.0f}"
        row[f"{name}_equity"] = f"{inv_data['equity']:.0f}"

    fieldnames = list(row.keys())

    with open(_PNL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info("pnl_tracker: %s total_pnl=%+.0f", today_str, total_pnl)
    return result


def format_pnl_line(result: dict) -> str:
    """Format PnL split for LINE notification."""
    src = result.get("equity_source", "估算")
    eq = result["total_equity"]
    lines = [f"📊 損益分帳 (淨值: {eq:,.0f} {src})"]
    for inv in result["investors"]:
        pct = inv["share"] * 100
        icon = "📈" if inv["pnl"] >= 0 else "📉"
        lines.append(
            f"{inv['name']} ({pct:.1f}%): {icon} {inv['pnl']:+,.0f} NTD"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = track_pnl()
    if result:
        print(format_pnl_line(result))
    else:
        print("No config/investors.yaml found — skipped.")
