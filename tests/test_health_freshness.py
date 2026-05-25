"""Freshness check for daily_health_check — trading-day aware, not calendar.

Regression for the bug that fired a false "🔴 Parquet 過期 gap=4天" every
Monday/post-holiday 08:00, because gap was measured in calendar days.

Anchor week (no holidays): 2026-05-21 Thu … 2026-05-27 Wed, all trading.
Holiday anchor: Mon 2026-05-04, prev Fri 2026-05-01 (勞動節) closed, Thu 2026-04-30 trading.
"""

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_health_check import check_freshness


def test_monday_morning_latest_thursday_ok():
    # 週一 08:00, latest=上週四 → ✅ (pre-14:25 → expect 2nd-to-last trading day)
    assert check_freshness(datetime(2026, 5, 25, 8, 0), date(2026, 5, 21))[0] == "ok"


def test_monday_morning_friday_holiday_latest_thursday_ok():
    # 週一 08:00, 上週五(5/1 勞動節)放假, latest=上週四(4/30) → ✅
    assert check_freshness(datetime(2026, 5, 4, 8, 0), date(2026, 4, 30))[0] == "ok"


def test_wednesday_morning_latest_last_friday_warn():
    # 週三 08:00, latest=上週五 → ⚠️ (落後 1 交易日, 缺週一 bar)
    assert check_freshness(datetime(2026, 5, 27, 8, 0), date(2026, 5, 22))[0] == "warn"


def test_monday_afternoon_latest_friday_ok():
    # 週一 15:00 (>=14:25), latest=上週五 → ✅
    assert check_freshness(datetime(2026, 5, 25, 15, 0), date(2026, 5, 22))[0] == "ok"


def test_genuinely_stale_alerts():
    # Wed 15:00 expects Tue 5/26; latest 5/15 is ~7 trading days behind → 🔴
    assert check_freshness(datetime(2026, 5, 27, 15, 0), date(2026, 5, 15))[0] == "alert"


def test_fresh_data_never_warns():
    # latest ahead of expected (today's bar already pulled) → ✅
    assert check_freshness(datetime(2026, 5, 25, 8, 0), date(2026, 5, 22))[0] == "ok"
