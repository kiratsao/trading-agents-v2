"""損益分帳追蹤器。

讀取 config/investors.yaml + state equity，計算各人損益，
追加到 data/pnl_history.csv。

config/investors.yaml 範例（withdrawals 為選填）：
    investors:
      - name: "Kira"
        capital: 380000
      - name: "Wife"
        capital: 150000
      - name: "Dad"
        capital: 350000

    # 出金紀錄（選填）。本金 capital 維持不變，提領金額單獨累計。
    withdrawals:
      - date: "2026-06-02"
        amounts:
          Kira: 1079545
          Wife: 426136
          Dad: 994318

分帳公式（每人 i）：
    比例   ratio_i  = capital_i / Σcapital            # 持分比例固定為本金占比
    持分   holding_i = equity × ratio_i               # 在場資金中的當前持分
    累計提領 wd_i     = Σ withdrawals[*].amounts[i]    # 已提領總額
    淨投入 net_i     = capital_i − wd_i
    總獲利 pnl_i     = (holding_i + wd_i) − capital_i  # 提領加回，不漏算已落袋獲利

沒有 withdrawals 時 wd_i = 0，退化為舊版 (pnl_i = holding_i − capital_i)。

此檔案不在 git 中（含本金隱私，gitignore）。如果不存在，此腳本靜默跳過。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_INVESTORS_YAML = Path("config/investors.yaml")
_PNL_CSV = Path("data/pnl_history.csv")
_STATE_JSON = Path("data/paper_state.json")


def load_investors() -> list[dict] | None:
    """Load investors list from YAML. Returns None if file missing/empty."""
    cfg = _load_config()
    if cfg is None:
        return None
    investors = cfg.get("investors", [])
    return investors or None


def _load_config() -> dict | None:
    """Load the full investors.yaml (investors + optional withdrawals).

    Returns None when the file is missing or unreadable so callers degrade
    silently (the daemon never depends on this).
    """
    if not _INVESTORS_YAML.exists():
        return None
    try:
        import yaml
        with open(_INVESTORS_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("pnl_tracker: failed to load investors.yaml: %s", exc)
        return None


def cumulative_withdrawals(cfg: dict, names: set[str]) -> dict[str, float]:
    """Sum each investor's withdrawals across all withdrawal events.

    ``withdrawals`` is an optional top-level list of ``{date, amounts: {name:
    amount}}``. Unknown names (typo / departed investor) are logged and ignored
    so a stray entry never silently mis-allocates a real investor's share.
    """
    totals: dict[str, float] = {n: 0.0 for n in names}
    for event in cfg.get("withdrawals", []) or []:
        amounts = (event or {}).get("amounts", {}) or {}
        for name, amt in amounts.items():
            if name not in totals:
                logger.warning(
                    "pnl_tracker: withdrawal for unknown investor %r (date=%s) — ignored",
                    name, (event or {}).get("date"),
                )
                continue
            totals[name] += float(amt)
    return totals


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
    """Calculate PnL splits (withdrawal-aware) and append to CSV.

    Returns dict with totals and per-investor PnL, or None if no investors.yaml.
    """
    cfg = _load_config()
    if not cfg:
        return None
    investors = cfg.get("investors", [])
    if not investors:
        return None

    equity, equity_src = get_equity()
    total_capital = sum(inv["capital"] for inv in investors)

    if total_capital <= 0:
        return None

    names = {inv["name"] for inv in investors}
    withdrawn = cumulative_withdrawals(cfg, names)
    total_withdrawn = sum(withdrawn.values())

    # Total profit adds back what has already been withdrawn so realised payouts
    # are not mistaken for losses: (equity + Σwithdrawn) − Σcapital.
    total_pnl = (equity + total_withdrawn) - total_capital
    from src.utils.tw_time import today_taipei

    today_str = str(today_taipei())

    result = {
        "date": today_str,
        "total_equity": equity,
        "equity_source": equity_src,
        "total_capital": total_capital,
        "total_withdrawn": total_withdrawn,
        "total_pnl": total_pnl,
        "investors": [],
    }

    # Per-investor split by capital share (ratio fixed to capital weight).
    for inv in investors:
        name = inv["name"]
        capital = inv["capital"]
        share = capital / total_capital
        wd = withdrawn.get(name, 0.0)
        holding = equity * share                 # current stake in the fund
        inv_pnl = (holding + wd) - capital       # total profit incl. withdrawn
        inv_equity = capital + inv_pnl           # = holding + wd
        result["investors"].append({
            "name": name,
            "capital": capital,
            "share": share,
            "withdrawn": wd,
            "net_invested": capital - wd,
            "holding": holding,
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
    """Format PnL split for LINE notification (shows withdrawals when present)."""
    src = result.get("equity_source", "估算")
    eq = result["total_equity"]
    total_wd = result.get("total_withdrawn", 0.0)
    header = f"📊 損益分帳 (淨值: {eq:,.0f} {src}"
    if total_wd > 0:
        header += f"，已提領 {total_wd:,.0f}"
    header += ")"
    lines = [header]
    for inv in result["investors"]:
        pct = inv["share"] * 100
        icon = "📈" if inv["pnl"] >= 0 else "📉"
        line = f"{inv['name']} ({pct:.1f}%): {icon} {inv['pnl']:+,.0f} NTD"
        wd = inv.get("withdrawn", 0.0)
        if wd > 0:
            # 總獲利已含提領加回；附註已落袋金額讓分帳透明。
            line += f"（已提領 {wd:,.0f}）"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = track_pnl()
    if result:
        print(format_pnl_line(result))
    else:
        print("No config/investors.yaml found — skipped.")
