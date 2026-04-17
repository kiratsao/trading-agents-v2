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


def get_equity() -> float:
    """Get current equity from state file."""
    import json
    if not _STATE_JSON.exists():
        return 0.0
    try:
        raw = json.loads(_STATE_JSON.read_text(encoding="utf-8"))
        return float(raw.get("state", {}).get("equity", 0))
    except Exception:
        return 0.0


def track_pnl() -> dict | None:
    """Calculate PnL splits and append to CSV.

    Returns dict with total and per-investor PnL, or None if no investors.yaml.
    """
    investors = load_investors()
    if not investors:
        return None

    equity = get_equity()
    total_capital = sum(inv["capital"] for inv in investors)

    if total_capital <= 0:
        return None

    total_pnl = equity - total_capital
    today_str = str(date.today())

    result = {
        "date": today_str,
        "total_equity": equity,
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
    lines = ["📊 損益分帳"]
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
