"""Spot-validation band margin report — standing monitor, re-run after ANY band change.

The band ``basis_band_for(spot) = max(500, 1.6% × spot)`` (``src/data/spot_ref.py``)
gates two write/act paths:
  * ``orchestrator._validate_today_bar``  — live gate (today's bar accepted or not)
  * ``daily_updater._spot_flags_bar``     — write gate (parquet row accepted or not)
A band that is too tight false-trips legitimate day bars (the flat-500 rule did, on
high-index 2026 days); too loose and a mislabeled night bar slips through.

本腳本對整條日K parquet 逐日重算 basis = |MXF 日收 − ^TWII 現貨收|，列出離門檻最近
的幾天（margin = band − basis，越小越危險），並與舊的 flat-500 規則對照。
**任何 band 參數變動後必須重跑**：現行 band 只要有 1 天誤觸就 exit 1，可直接當閘門。

Usage:
    python scripts/band_margin_report.py
    python scripts/band_margin_report.py --parquet /path/to/other.parquet --top 15

Exit code: 0 = 現行 band 全數通過；1 = 有誤觸（band 太緊，不可上線）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.spot_ref import _BASIS_BAND, basis_band_for

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_PARQUET = _REPO / "data" / "MXF_Daily_Clean_2020_to_now.parquet"


def _build(parquet: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """讀 parquet + 抓現貨 → 每日 basis / band / margin（只留有現貨的交易日）。"""
    from src.data.spot_ref import fetch_spot_range

    df = pd.read_parquet(parquet).sort_index()
    df.index = pd.to_datetime(df.index)
    start, end = df.index[0].date(), df.index[-1].date()

    spot = fetch_spot_range(start, end)
    spot_s = pd.Series({pd.Timestamp(d): float(v) for d, v in spot.items()}).sort_index()

    j = df[["close"]].copy()
    j["move_pct"] = df["close"].pct_change() * 100.0
    j = j.join(spot_s.rename("spot"), how="left")
    j = j.dropna(subset=["spot"])

    j["basis"] = (j["close"] - j["spot"]).abs()
    j["band"] = j["spot"].map(basis_band_for)
    j["margin"] = j["band"] - j["basis"]
    j["margin500"] = _BASIS_BAND - j["basis"]
    return j, df, spot_s


def _table(rows: pd.DataFrame, band_col: str, margin_col: str) -> str:
    """固定寬度表：date / close / spot / basis / band / margin / move%。"""
    head = (f"  {'date':<12}{'close':>9}{'spot':>10}{'basis':>9}"
            f"{'band':>9}{'margin':>9}{'move%':>9}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for d, r in rows.iterrows():
        mv = f"{r['move_pct']:+.2f}" if pd.notna(r["move_pct"]) else "n/a"
        flag = "  TRIP" if r[margin_col] < 0 else ""
        lines.append(
            f"  {d.date().isoformat():<12}{r['close']:>9.0f}{r['spot']:>10.1f}"
            f"{r['basis']:>9.1f}{r[band_col]:>9.1f}{r[margin_col]:>9.1f}{mv:>9}{flag}"
        )
    return "\n".join(lines)


def _report_rule(j: pd.DataFrame, label: str, band_col: str, margin_col: str, top: int) -> int:
    """印出單一規則的誤觸數與最緊的 top 天；回傳誤觸天數。"""
    trips = j[j[margin_col] < 0].sort_index()
    print(f"\n-- {label} --")
    print(f"TRIPS: {len(trips)}" + ("  OK" if len(trips) == 0 else "  <-- false trips"))
    if len(trips):
        print("  trip dates: " + ", ".join(d.date().isoformat() for d in trips.index))
    tight = j.nsmallest(top, margin_col)
    print(f"tightest {min(top, len(j))} (smallest margin = {band_col} - basis first):")
    print(_table(tight, band_col, margin_col))
    return len(trips)


def main() -> int:
    ap = argparse.ArgumentParser(description="Spot-validation band margin report")
    ap.add_argument("--parquet", default=str(_DEFAULT_PARQUET), help="day-session parquet path")
    ap.add_argument("--top", type=int, default=10, help="how many tightest days to list")
    args = ap.parse_args()

    pq = Path(args.parquet)
    if not pq.exists():
        print(f"parquet not found: {pq}", file=sys.stderr)
        return 2

    j, df, spot_s = _build(pq)
    print("== spot-validation band margin report ==")
    print(f"parquet  : {pq}")
    print(f"bars     : {len(df)}  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"spot     : {len(spot_s)} ^TWII closes (cache-first)")
    print(f"compared : {len(j)} spot-days  (skipped {len(df) - len(j)} bars with no spot)")
    print(f"current band: max({_BASIS_BAND:.0f}, 1.6% x spot)  "
          f"[src.data.spot_ref.basis_band_for] "
          f"range {j['band'].min():.0f}..{j['band'].max():.0f}")

    n_cur = _report_rule(j, "CURRENT band  max(500, 1.6% x spot)", "band", "margin", args.top)
    legacy = j.copy()
    legacy["band500"] = _BASIS_BAND
    n_500 = _report_rule(legacy, "LEGACY flat-500", "band500", "margin500", args.top)

    print(f"\nimprovement: flat-500 false-trips {n_500} day(s) -> current band {n_cur} day(s)")
    worst = j["margin"].min()
    worst_day = j["margin"].idxmin()
    print(f"tightest margin under current band: {worst:.1f}pt on {worst_day.date()}")
    if n_cur:
        print("RESULT: FAIL — current band false-trips legitimate day bars. Do NOT ship.")
        return 1
    print("RESULT: PASS — 0 false trips under the current band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
