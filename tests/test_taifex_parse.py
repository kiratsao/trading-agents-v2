"""Tests for TAIFEX CSV parsing — esp. the dual-row 交易時段 day-session fix.

Phase 1 (verified) findings encoded here:
  * 舊格式 (單列): one row/day labelled 一般 → that row IS the day session.
  * 新格式 (雙列): [一般=夜盤, 盤後=日盤] → 盤後 close aligns with Shioaji day.
    e.g. 2026-05-28 盤後=45,249 (日盤) vs 一般=43,846 (夜盤),
         2026-05-29 盤後=44,790 (日盤) vs 一般=45,292 (夜盤).
"""

from __future__ import annotations

import pandas as pd

from scripts.init_data import _parse_taifex_csv, fetch_taifex_month

# Header mirrors the columns the parser references (plus a couple of extras).
_HEADER = (
    "交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,"
    "漲跌價,漲跌%,成交量,交易時段"
)


def _csv(*rows: str) -> str:
    # Real TAIFEX lines carry a trailing comma — keep it to exercise the strip.
    return "\n".join([_HEADER] + [r + "," for r in rows])


def test_single_row_old_format_uses_the_only_row():
    """舊格式單列 (一般) → take that row as the day session."""
    raw = _csv("2024/12/02,MTX,202412,22000,22100,21900,22050,+50,0.23,5000,一般")
    daily = _parse_taifex_csv(raw, year=2024, month=12)

    assert len(daily) == 1
    ts = pd.Timestamp("2024-12-02")
    assert daily.loc[ts, "close"] == 22050
    assert daily.loc[ts, "open"] == 22000


def test_dual_row_new_format_picks_panhou_day_session():
    """新格式雙列 → 盤後 (日盤) close, NOT 一般 (夜盤)."""
    raw = _csv(
        "2026/05/28,MTX,202506,43800,43900,43700,43846,-100,-0.2,8000,一般",
        "2026/05/28,MTX,202506,45200,45300,45100,45249,+200,0.4,9000,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert len(daily) == 1
    ts = pd.Timestamp("2026-05-28")
    # day session = 盤後 row (aligns Shioaji), night 一般=43846 must be dropped.
    assert daily.loc[ts, "close"] == 45249
    assert daily.loc[ts, "open"] == 45200
    assert daily.loc[ts, "high"] == 45300
    assert daily.loc[ts, "low"] == 45100


def test_dual_row_multiple_days_real_values():
    """Two verified trading days, dual-row each → day-session closes."""
    raw = _csv(
        "2026/05/28,MTX,202506,43800,43900,43700,43846,-100,-0.2,8000,一般",
        "2026/05/28,MTX,202506,45200,45300,45100,45249,+200,0.4,9000,盤後",
        "2026/05/29,MTX,202506,45300,45400,45200,45292,+50,0.1,7000,一般",
        "2026/05/29,MTX,202506,44800,44900,44700,44790,-460,-1.0,9500,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 45249  # 日盤
    assert daily.loc[pd.Timestamp("2026-05-29"), "close"] == 44790  # 日盤


def test_nearest_month_selected_under_dual_row():
    """With two expiries (both dual-row), keep nearest month's day session."""
    raw = _csv(
        # near month 202506
        "2026/05/28,MTX,202506,43800,43900,43700,43846,-100,-0.2,8000,一般",
        "2026/05/28,MTX,202506,45200,45300,45100,45249,+200,0.4,9000,盤後",
        # far month 202507 (must be ignored)
        "2026/05/28,MTX,202507,43900,44000,43800,43950,-90,-0.2,500,一般",
        "2026/05/28,MTX,202507,45300,45400,45200,45350,+210,0.5,400,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert len(daily) == 1
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 45249


def test_weekly_contracts_excluded():
    """Weekly contract (YYYYMMW1) rows must be dropped; only monthly kept."""
    raw = _csv(
        "2026/05/28,MX1,202505W5,45000,45100,44900,45050,0,0,300,盤後",
        "2026/05/28,MTX,202506,45200,45300,45100,45249,+200,0.4,9000,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert len(daily) == 1
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 45249


def test_unknown_session_label_warns_and_is_ranked_last(capsys):
    """Unknown label alongside 一般 → known wins, warning printed."""
    raw = _csv(
        "2026/05/28,MTX,202506,43800,43900,43700,43846,-100,-0.2,8000,一般",
        "2026/05/28,MTX,202506,99000,99100,98900,99050,0,0,1,試撮",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)
    out = capsys.readouterr().out

    assert "unknown TAIFEX 交易時段" in out
    assert "試撮" in out
    # known label (一般) chosen over the unknown row, never silently wrong.
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846


def test_empty_and_no_data_inputs():
    assert _parse_taifex_csv("").empty
    assert _parse_taifex_csv("查無資料").empty


def test_fetch_taifex_month_wires_download_to_parser(monkeypatch):
    """End-to-end: fetch_taifex_month should parse whatever the download returns."""
    raw = _csv("2026/05/28,MTX,202506,45200,45300,45100,45249,+200,0.4,9000,盤後")
    monkeypatch.setattr(
        "scripts.init_data._download_taifex_csv", lambda y, m, product="MTX": raw
    )
    daily = fetch_taifex_month(2026, 5, product="MTX")

    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 45249
