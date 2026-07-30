"""+DI>−DI 方向濾網回測矩陣 — report-only,不動生產碼、不動任何預設值。

背景 (2026-07-29/30): ADX 只量趨勢強度、不辨方向 — 崩盤把 −DI 推高即令 ADX
跨過 25,在 close 低於 EMA30 數千點處放行做多接刀 (+DI=10 vs −DI=45)。
本濾網在既有進場條件之上加「+DI > −DI (多方主導) 才進場」,以 DMI 方向
判定補上 ADX 的方向盲區。引擎子類只存在於本腳本 (研究用)。

Usage:
    python scripts/backtest_di_matrix.py [--parquet PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.backtest_gate_matrix import _july_2026, _whipsaws
from src.backtest.engine import BacktestEngine
from src.data.daily_updater import _PRIMARY_PARQUET
from src.strategy.v2b_engine import V2bEngine


def _di(data: pd.DataFrame, period: int = 14) -> tuple[float, float]:
    """(+DI, -DI) at the last bar — same EWM-span smoothing as indicators.adx."""
    h, low, c = data["high"], data["low"], data["close"]
    pdm, mdm = h.diff(), -low.diff()
    plus_dm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    minus_dm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    pdi = float((100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s).iloc[-1])
    mdi = float((100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s).iloc[-1])
    return pdi, mdi


class DIFilterEngine(V2bEngine):
    """進場加 +DI>−DI;其餘 (含濾網A/B、risk-cap 開關) 繼承 V2bEngine。"""

    def _entry_filter_block(self, data, close, ema_f):
        block = super()._entry_filter_block(data, close, ema_f)
        if block:
            return block
        pdi, mdi = _di(data)
        if pdi <= mdi:
            return f"DI 方向濾網: +DI={pdi:.1f} ≤ −DI={mdi:.1f} — 空方主導,不做多"
        return None


def _mk(cls=V2bEngine, **kw):
    return cls(product="MXF", ema_fast=30, ema_slow=100, confirm_days=2,
               adx_threshold=25, **kw)


VARIANTS = [
    ("baseline",        lambda: _mk()),
    ("DI",              lambda: _mk(DIFilterEngine)),
    ("DI+cap20+buf1.0", lambda: _mk(DIFilterEngine, risk_cap_pct=0.20,
                                    margin_buffer_atr=1.0)),
    ("cap20+buf1.0",    lambda: _mk(risk_cap_pct=0.20, margin_buffer_atr=1.0)),
    ("B(close>EMA30)",  lambda: _mk(reentry_require_above_ema_fast=True)),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(_PRIMARY_PARQUET))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"data: {len(df)} bars {df.index[0].date()} → {df.index[-1].date()}\n")

    rows, trades_by = [], {}
    for name, make in VARIANTS:
        res = BacktestEngine(strategy=make(), initial_capital=350_000,
                             exec_timing="same_day_close").run(df)
        m = res.metrics
        trades_by[name] = res.trades
        rows.append({
            "variant": name, "CAGR%": m["CAGR_%"], "MDD%": m["MDD_%"],
            "Sharpe": m["Sharpe"], "Win%": m["Win_Rate_%"], "PF": m["Profit_Factor"],
            "Trades": m["Total_Trades"], "Whipsaw": _whipsaws(df, res),
            "Final": f"{m['Final_Equity']:,.0f}",
            "Jul2026": len(_july_2026(res)),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    base_entries = {t.entry_date for t in trades_by["baseline"]}
    di_entries = {t.entry_date for t in trades_by["DI"]}
    blocked = sorted(base_entries - di_entries)
    added = sorted(di_entries - base_entries)
    pnl_by_entry: dict[str, float] = {}
    for t in trades_by["baseline"]:
        pnl_by_entry[t.entry_date] = pnl_by_entry.get(t.entry_date, 0.0) + t.pnl_twd
    print(f"\nDI 濾網擋掉的 baseline 進場 ({len(blocked)}):")
    for d in blocked:
        print(f"  {d}  baseline 該進場後續實現損益 {pnl_by_entry.get(d, 0):+,.0f} NTD")
    if added:
        print(f"DI 濾網新增(延後)的進場 ({len(added)}): {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
