"""2026-07-25 Shioaji kbars 時戳語義切換 (UTC ns → 台北 naive ns) — per-response
結構偵測 regression。

事故: 硬編 utc=True 對 naive 時戳多加 8h,「日盤窗」實際選到前夜盤 00:45–05:45
尾段 → 每天日收 == 夜收 (provenance 四次實戰攔截的上游根因)。偵測器只用結構事實:
(a) 未來時戳殺解讀 (b) 全部 bar 須落在合法 TAIFEX session 窗;兩解讀同時合法但
選出不同日盤集合 → 拒用 (絕不猜)。日盤聚合含 13:45 收盤集合競價 bar。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from src.data.shioaji_fetcher import (
    _kbars_ts_interpretations,
    fetch_day_session_bar,
)
from tests.fakes import _FakeKbars


def _mk(bars, *, naive: bool) -> _FakeKbars:
    """naive=True → ts 直接是台北 wall-clock ns (新語義);
    naive=False → ts 是真 UTC ns (舊語義)。"""
    ts = [(pd.Timestamp(t) if naive else pd.Timestamp(t, tz="Asia/Taipei")).value
          for t, *_ in bars]
    return _FakeKbars(ts, [b[1] for b in bars], [b[2] for b in bars],
                      [b[3] for b in bars], [b[4] for b in bars],
                      [b[5] for b in bars])


class _Api:
    def __init__(self, kb):
        self.kb = kb

    def kbars(self, contract, start=None, end=None, timeout=None):
        return self.kb


# 2026-08-05 真實形狀: 前夜盤尾 + 日盤(13:44=44,535、13:45 集合競價=44,526 官方
# close、真夜收 44,546=當日 TAIFEX 盤後值) + 當日夜盤。
_D = "2026-08-05"
_FULL_DAY = [
    (f"{_D} 00:45", 44_600, 44_620, 44_580, 44_610, 5_000),
    (f"{_D} 04:59", 44_560, 44_570, 44_540, 44_546, 4_000),   # 真夜盤收 (盤後值)
    (f"{_D} 08:45", 44_400, 44_420, 44_380, 44_400, 9_000),
    (f"{_D} 13:44", 44_530, 44_540, 44_520, 44_535, 8_000),
    (f"{_D} 13:45", 44_526, 44_526, 44_526, 44_526, 3_000),   # 收盤集合競價
    (f"{_D} 15:00", 44_500, 44_510, 44_490, 44_505, 2_000),
    (f"{_D} 22:00", 44_450, 44_460, 44_440, 44_450, 2_000),
]
_NOW_EOD = pd.Timestamp(f"{_D} 23:00")


def test_dual_encoding_identical_output():
    """裁示案: 同一天資料兩種包裝,輸出必須相同 + 語義判定正確。"""
    day = date(2026, 8, 5)
    a = fetch_day_session_bar(_Api(_mk(_FULL_DAY, naive=True)), None, day,
                              _now=_NOW_EOD)
    b = fetch_day_session_bar(_Api(_mk(_FULL_DAY, naive=False)), None, day,
                              _now=_NOW_EOD)
    assert a is not None and b is not None
    assert a["ts_semantics"] == "taipei-naive"
    assert b["ts_semantics"] == "utc-legacy"
    for k in ("open", "high", "low", "close", "volume", "n_bars"):
        assert a[k] == b[k], k
    assert a["close"] == 44_526                 # 13:45 集合競價 = 官方 close
    assert a["open"] == 44_400
    assert a["n_bars"] == 3                     # 08:45, 13:44, 13:45
    assert pd.Timestamp(a["last_ts"]).strftime("%H:%M") == "13:45"


def test_night_close_never_selected_as_day_close():
    """核心事故斷言: naive 資料下,舊 +8 解讀會把 00:45–04:59 夜尾當日盤
    (close 44,546 == 盤後值) — 新偵測必須回真日盤 44,526。"""
    bar = fetch_day_session_bar(_Api(_mk(_FULL_DAY, naive=True)), None,
                                date(2026, 8, 5), _now=_NOW_EOD)
    assert bar["close"] == 44_526
    assert bar["close"] != 44_546


def test_monday_shape_no_predawn_tail():
    """7/27 形狀: 週日無夜盤 → payload 無 00:00–05:00 段,判定不受影響。"""
    # 深夜 bar (22:30) 是判定根: +8 讀成次日 06:30 非法 → 殺掉 utc-legacy。
    # (真實 payload 840 根必含此類 bar;稀疏到無判定根時 fetcher 正確拒用。)
    bars = [("2026-07-27 08:45", 46_000, 46_020, 45_980, 46_000, 9_000),
            ("2026-07-27 13:45", 45_900, 45_910, 45_890, 45_900, 3_000),
            ("2026-07-27 15:00", 45_850, 45_860, 45_840, 45_850, 2_000),
            ("2026-07-27 22:30", 45_800, 45_810, 45_790, 45_800, 1_500)]
    bar = fetch_day_session_bar(_Api(_mk(bars, naive=True)), None,
                                date(2026, 7, 27),
                                _now=pd.Timestamp("2026-07-27 23:00"))
    assert bar is not None
    assert bar["ts_semantics"] == "taipei-naive"
    assert bar["close"] == 45_900


def test_live_midsession_shape():
    """8/06 22:08 形狀: 夜盤進行中查詢 — +8 解讀產生未來 bar 被殺。"""
    bars = [("2026-08-06 00:45", 44_300, 44_320, 44_280, 44_300, 4_000),
            ("2026-08-06 08:45", 44_200, 44_240, 44_180, 44_220, 9_000),
            ("2026-08-06 13:45", 44_274, 44_274, 44_274, 44_274, 3_000),
            ("2026-08-06 15:00", 44_250, 44_260, 44_240, 44_255, 2_000),
            ("2026-08-06 22:08", 44_100, 44_110, 44_090, 44_100, 1_500)]
    bar = fetch_day_session_bar(_Api(_mk(bars, naive=True)), None,
                                date(2026, 8, 6),
                                _now=pd.Timestamp("2026-08-06 22:08"))
    assert bar is not None
    assert bar["ts_semantics"] == "taipei-naive"
    assert bar["close"] == 44_274


def test_monday_1430_day_only_payload_resolved_by_future_test():
    """週一 14:30 決策時的真實形狀: 純日盤 payload (週日無夜盤、當日夜盤未開)。
    +8 解讀落在 16:45–21:45 — 合法夜盤時段, 結構測不出 — 由未來時戳測試殺掉。
    此案保證週一的進場能力不因偵測器而 fail-closed。"""
    bars = [("2026-07-27 08:45", 46_000, 46_020, 45_980, 46_000, 9_000),
            ("2026-07-27 13:44", 45_910, 45_915, 45_905, 45_910, 8_000),
            ("2026-07-27 13:45", 45_900, 45_900, 45_900, 45_900, 3_000)]
    bar = fetch_day_session_bar(_Api(_mk(bars, naive=True)), None,
                                date(2026, 7, 27),
                                _now=pd.Timestamp("2026-07-27 14:30"))
    assert bar is not None
    assert bar["ts_semantics"] == "taipei-naive"
    assert bar["close"] == 45_900


def test_future_timestamp_kills_interpretation():
    """結構皆合法時,未來時戳測試單獨定勝負: 夜盤 15:00–21:00 naive,
    +8 解讀 = 23:00–05:00(次日) 全在未來。"""
    ts = [pd.Timestamp(f"2026-08-06 {h:02d}:00").value for h in (15, 17, 19, 21)]
    surv = _kbars_ts_interpretations(ts, now=pd.Timestamp("2026-08-06 21:05"))
    assert [n for n, _ in surv] == ["taipei-naive"]


def test_both_interpretations_illegal_rejected():
    """兩解讀皆有非法時段 bar → 🔴 拒用回 None。"""
    bars = [("2026-08-05 06:30", 1_000, 1_000, 1_000, 1_000, 1_000),
            ("2026-08-05 07:10", 1_000, 1_000, 1_000, 1_000, 1_000)]
    bar = fetch_day_session_bar(_Api(_mk(bars, naive=True)), None,
                                date(2026, 8, 5), _now=_NOW_EOD)
    assert bar is None


def test_pure_predawn_tail_materially_ambiguous_rejected():
    """危險退化案: 純 00:45–04:50 夜尾 — 兩解讀皆結構合法,但 +8 解讀會把它
    扮成 08:45–12:50「日盤」。選集不同 → 絕不猜,拒用。"""
    bars = [("2026-08-05 00:45", 44_600, 44_620, 44_580, 44_610, 5_000),
            ("2026-08-05 04:50", 44_560, 44_570, 44_540, 44_546, 4_000)]
    bar = fetch_day_session_bar(_Api(_mk(bars, naive=True)), None,
                                date(2026, 8, 5), _now=_NOW_EOD)
    assert bar is None


def test_settlement_auction_1330_included(monkeypatch):
    monkeypatch.setattr("src.strategy.v2b_engine._is_settlement_day",
                        lambda d: True)
    bars = [("2026-08-19 13:25", 44_000, 44_010, 43_990, 44_000, 4_000),
            ("2026-08-19 13:30", 43_990, 43_990, 43_990, 43_990, 2_000),
            ("2026-08-19 13:35", 44_100, 44_100, 44_100, 44_100, 1_000),
            # 深夜判定根 (+8 → 次日 06:30 非法, 殺 utc-legacy 解讀)
            ("2026-08-19 22:30", 44_050, 44_060, 44_040, 44_050, 1_500)]
    bar = fetch_day_session_bar(_Api(_mk(bars, naive=True)), None,
                                date(2026, 8, 19),
                                _now=pd.Timestamp("2026-08-19 23:00"))
    assert bar is not None
    assert bar["close"] == 43_990               # 13:30 結算集合競價含入, 13:35 排除


def test_completeness_accepts_1345_auction_bar():
    """日盤聚合含 13:45 後, P0 completeness 窗必須同步接受末根 13:45。"""
    from src.scheduler.orchestrator import _validate_today_bar

    with patch("src.data.spot_ref.fetch_spot_close", return_value=44_400.0):
        meta = _validate_today_bar(
            {"close": 44_526.0, "last_ts": "2026-08-05T13:45:00"},
            date(2026, 8, 5), source="kbars",
        )
    assert meta["complete"] is True
    assert meta["validated"] is True
