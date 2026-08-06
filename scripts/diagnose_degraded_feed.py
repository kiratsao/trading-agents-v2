"""Shioaji degraded-feed 診斷 — 只讀不下單不寫檔。

背景 (2026-07-25 起): Shioaji day-session 回夜值,7/29/7/30 pre-rescue bar 逐位
等於 TAIFEX 盤後 close → 疑似夜盤價格帶著日盤時間戳 (feed-level mis-stamp),
或 kbars 全落在日盤窗外。本腳本 dump 原始 kbars 的時間戳分佈與日盤窗聚合值,
對照 TAIFEX 一般/盤後 與 ^TWII spot,給出確診分類。

Usage (GCP, system python3):
    python3 scripts/diagnose_degraded_feed.py            # 預設: 最近一個交易日
    python3 scripts/diagnose_degraded_feed.py --day 2026-08-05
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

_TOL = 1.0  # 與 daily_updater._PROV_TOL 同義: TAIFEX 價格為整數


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default=None, help="YYYY-MM-DD (預設: 最近交易日)")
    ap.add_argument("--product", default="MXF")
    args = ap.parse_args(argv)

    from src.data.daily_updater import _taifex_day_bar, _taifex_night_close
    from src.data.spot_ref import basis_band_for, fetch_spot_close
    from src.data.tw_holidays import last_trading_day_before
    from src.utils.tw_time import today_taipei

    day = date.fromisoformat(args.day) if args.day else \
        last_trading_day_before(today_taipei())
    print(f"=== diagnose_degraded_feed: {args.product} {day} ===")

    # ── Shioaji 原始 kbars(不經任何濾網)────────────────────────────────
    import os

    from src.data.shioaji_fetcher import fetch_via_env  # noqa: F401 (同 env 慣例)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("🔴 SHIOAJI_API_KEY/SECRET 未設 — 無法診斷 (需在 GCP .env 環境執行)")
        return 1
    from tw_futures.executor.shioaji_adapter import ShioajiAdapter

    adapter = ShioajiAdapter(
        api_key=api_key, secret_key=secret_key, simulation=False,
        cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
        cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
        person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
    )
    try:
        contract = adapter.get_contract(args.product)
        print(f"contract 解析: code={getattr(contract, 'code', '?')} "
              f"delivery={getattr(contract, 'delivery_month', '?')} "
              f"symbol={getattr(contract, 'symbol', '?')}")
        kbars = adapter._api.kbars(contract, start=str(day), end=str(day),
                                   timeout=30_000)
        ts = getattr(kbars, "ts", None)
        if not kbars or ts is None or len(ts) == 0:
            print("verdict: 🔴 Shioaji kbars 空 — 無資料 (fallback 路徑應接手)")
            return 0
        raw = pd.DataFrame({
            "ts": kbars.ts, "open": kbars.Open, "high": kbars.High,
            "low": kbars.Low, "close": kbars.Close, "volume": kbars.Volume,
        })
        raw["ts"] = pd.to_datetime(raw["ts"], unit="ns", utc=True) \
            .dt.tz_convert("Asia/Taipei")
        raw = raw.sort_values("ts")
    finally:
        adapter.logout()

    print(f"\n原始 kbars: {len(raw)} bars  ts {raw['ts'].iloc[0]} → {raw['ts'].iloc[-1]}")
    hours = raw["ts"].dt.strftime("%H").value_counts().sort_index()
    print("每小時 bar 數:", ", ".join(f"{h}:{c}" for h, c in hours.items()))

    from datetime import time as _t
    t = raw["ts"].dt.time
    win = raw[(raw["ts"].dt.date == day) & (t >= _t(8, 45)) & (t < _t(13, 45))]
    out_win = len(raw) - len(win)
    print(f"日盤窗內 (date=={day}, 08:45–13:45): {len(win)} bars; 窗外 {out_win} bars")

    # ── 參考真值 ────────────────────────────────────────────────────────
    tx_day = _taifex_day_bar(day)
    day_close = float(tx_day["close"].iloc[-1]) if tx_day is not None else None
    night_close = _taifex_night_close(day)
    spot = fetch_spot_close(day)
    print(f"\nTAIFEX 一般 close: {day_close}   盤後 close: {night_close}   "
          f"^TWII spot: {spot}")

    if win.empty:
        print("verdict: ⚠️ kbars 有資料但全在日盤窗外 (時間戳落在夜盤時段) — "
              "濾網會正確回 None,TAIFEX fallback 應接手;確認 updater log 是否如此")
        return 0

    w_close = float(win.iloc[-1]["close"])
    w_last = win.iloc[-1]["ts"].strftime("%H:%M")
    print(f"日盤窗聚合: O={float(win.iloc[0]['open']):,.0f} "
          f"H={float(win['high'].max()):,.0f} L={float(win['low'].min()):,.0f} "
          f"C={w_close:,.0f} V={int(win['volume'].sum())} 末根={w_last}")

    if night_close is not None and abs(w_close - night_close) <= _TOL and (
            day_close is None or abs(w_close - day_close) > _TOL):
        print("verdict: 🔴 確診 feed-level mis-stamp — 日盤窗內的價格 == TAIFEX 盤後值 "
              "(夜盤資料帶日盤時間戳)。live 防線: provenance gate + spot gate 會攔; "
              "請回報永豐並附本輸出")
    elif day_close is not None and abs(w_close - day_close) <= _TOL:
        print("verdict: ✅ 正常 — 日盤窗 close == TAIFEX 一般")
    else:
        dev = (w_close - spot) if spot is not None else None
        band = basis_band_for(spot) if spot is not None else None
        print(f"verdict: ❓ 與一般/盤後皆不符 (spot偏差 {dev}, band {band}) — "
              f"附本輸出人工判讀")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
