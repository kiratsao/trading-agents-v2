"""Tests for TAIFEX CSV parsing — esp. the dual-row 交易時段 day-session rule.

Verified semantics (exact-OHLCV match vs Shioaji day-session parquet on
2020-06-15, 2023-06-15, 2025-06-16, 2026-05-28/29):
  * 一般 = 該交易日的日盤 (08:45–13:45) → keep.
  * 盤後 = 前一交易日 15:00 起的夜盤, booked to the NEXT trading date → drop,
    even when it is the only row for a date (e.g. Monday's 盤後 row appears
    right after Friday's night session, before Monday's day session exists).
Real values used below: 2026-05-28 一般=43,846 日盤 / 盤後=45,249 夜盤;
2026-07-06 fetched on Sat 07-05 had only the 盤後 (Fri night) row.
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


def test_single_yiban_row_is_kept():
    """Date with only a 一般 row → that row is the day session."""
    raw = _csv("2024/12/02,MTX,202412,22000,22100,21900,22050,+50,0.23,5000,一般")
    daily = _parse_taifex_csv(raw, year=2024, month=12)

    assert len(daily) == 1
    ts = pd.Timestamp("2024-12-02")
    assert daily.loc[ts, "close"] == 22050
    assert daily.loc[ts, "open"] == 22000


def test_dual_row_picks_yiban_day_session():
    """雙列 → 一般 (日盤) kept, 盤後 (夜盤) dropped. Real 2026-05-28 values."""
    raw = _csv(
        "2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,一般",
        "2026/05/28,MTX,202606,44669,45300,44600,45249,+455,1.0,215770,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert len(daily) == 1
    ts = pd.Timestamp("2026-05-28")
    # day session = 一般 row (matches Shioaji day OHLCV exactly).
    assert daily.loc[ts, "close"] == 43846
    assert daily.loc[ts, "open"] == 45188
    assert daily.loc[ts, "high"] == 45350
    assert daily.loc[ts, "low"] == 43464


def test_dual_row_multiple_days_real_values():
    """Two verified trading days, dual-row each → 一般 (day) closes."""
    raw = _csv(
        "2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,一般",
        "2026/05/28,MTX,202606,44669,45300,44600,45249,+455,1.0,215770,盤後",
        "2026/05/29,MTX,202606,44660,45469,44654,45292,+1446,3.3,144792,一般",
        "2026/05/29,MTX,202606,43958,44900,43900,44790,-459,-1.0,161797,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846  # 日盤
    assert daily.loc[pd.Timestamp("2026-05-29"), "close"] == 45292  # 日盤


def test_panhou_only_date_yields_no_bar():
    """盤後-only date (day session not traded yet) must emit NO bar.

    Real case: fetching July 2026 on Sat 07-05 returns a 2026/07/06 row
    labelled 盤後 (Friday's night session booked to Monday) and no 一般 row —
    writing it would create a future-dated night bar that blocks the updater.
    """
    raw = _csv(
        "2026/07/03,MTX,202607,45711,47086,45606,46985,+1314,2.9,139592,一般",
        "2026/07/03,MTX,202607,46300,47221,45267,45671,-640,-1.4,284238,盤後",
        "2026/07/06,MTX,202607,47042,47470,46536,47102,+117,0.2,78539,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=7)

    assert len(daily) == 1
    assert pd.Timestamp("2026-07-06") not in daily.index
    assert daily.loc[pd.Timestamp("2026-07-03"), "close"] == 46985


def test_nearest_month_selected_under_dual_row():
    """With two expiries (both dual-row), keep nearest month's day session."""
    raw = _csv(
        # near month 202606
        "2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,一般",
        "2026/05/28,MTX,202606,44669,45300,44600,45249,+455,1.0,215770,盤後",
        # far month 202607 (must be ignored)
        "2026/05/28,MTX,202607,45250,45400,43500,43900,-940,-2.1,5000,一般",
        "2026/05/28,MTX,202607,44700,45350,44650,45300,+450,1.0,4000,盤後",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert len(daily) == 1
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846


def test_weekly_contracts_excluded():
    """Weekly contract (YYYYMMW1) rows must be dropped; only monthly kept."""
    raw = _csv(
        "2026/05/28,MX1,202505W5,45000,45100,44900,45050,0,0,300,一般",
        "2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,一般",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)

    assert len(daily) == 1
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846


def test_unknown_session_label_warns_and_ranks_after_yiban(capsys):
    """Unknown label alongside 一般 → 一般 wins, warning printed."""
    raw = _csv(
        "2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,一般",
        "2026/05/28,MTX,202606,99000,99100,98900,99050,0,0,1,試撮",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)
    out = capsys.readouterr().out

    assert "unknown TAIFEX 交易時段" in out
    assert "試撮" in out
    # known label (一般) chosen over the unknown row, never silently wrong.
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846


def test_unknown_only_label_still_flows_with_warning(capsys):
    """If TAIFEX renames the day-session label, data keeps flowing + warning."""
    raw = _csv(
        "2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,日盤新名",
    )
    daily = _parse_taifex_csv(raw, year=2026, month=5)
    out = capsys.readouterr().out

    assert "unknown TAIFEX 交易時段" in out
    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846


def test_empty_and_no_data_inputs():
    assert _parse_taifex_csv("").empty
    assert _parse_taifex_csv("查無資料").empty


def test_fetch_taifex_month_wires_download_to_parser(monkeypatch):
    """End-to-end: fetch_taifex_month should parse whatever the download returns."""
    raw = _csv("2026/05/28,MTX,202606,45188,45350,43464,43846,-948,-2.1,230563,一般")
    monkeypatch.setattr(
        "scripts.init_data._download_taifex_csv", lambda y, m, product="MTX": raw
    )
    daily = fetch_taifex_month(2026, 5, product="MTX")

    assert daily.loc[pd.Timestamp("2026-05-28"), "close"] == 43846
