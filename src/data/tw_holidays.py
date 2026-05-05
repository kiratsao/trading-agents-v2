"""台灣期交所 (TAIFEX) 休市日清單。

期交所每年 12 月公布隔年休市日。本模組維護 2020–2027 年休市清單，
供 init_data / daily_updater / verify_data 判斷是否為交易日。

來源: https://www.taifex.com.tw/cht/5/holidaySchedule

維護方式: 每年 12 月底追加下一年度。
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

# 格式: (month, day) 固定假日 + 年度不固定假日以完整 date 列表補充
# 僅含非週末的休市日（週末本來就不交易）

_FIXED_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),    # 元旦 (有時會調移)
    (2, 28),   # 和平紀念日
    (5, 1),    # 勞動節
    (10, 10),  # 國慶日
}

# 年度變動假日（農曆春節、清明、端午、中秋、調整放假等）
# 來源: TAIFEX 歷年休市日公告
_ANNUAL_HOLIDAYS: dict[int, list[date]] = {
    2020: [
        date(2020, 1, 1),
        date(2020, 1, 23), date(2020, 1, 24), date(2020, 1, 27),
        date(2020, 1, 28), date(2020, 1, 29),  # 春節
        date(2020, 2, 28),  # 和平紀念日
        date(2020, 4, 2), date(2020, 4, 3),  # 兒童節+清明
        date(2020, 5, 1),  # 勞動節
        date(2020, 6, 25), date(2020, 6, 26),  # 端午
        date(2020, 10, 1), date(2020, 10, 2),  # 中秋
        date(2020, 10, 9),  # 國慶連假
    ],
    2021: [
        date(2021, 1, 1),
        date(2021, 2, 8), date(2021, 2, 9), date(2021, 2, 10),
        date(2021, 2, 11), date(2021, 2, 12), date(2021, 2, 15), date(2021, 2, 16),  # 春節
        date(2021, 3, 1),  # 228 補假
        date(2021, 4, 2), date(2021, 4, 5),  # 兒童+清明
        # 2021-05-01 falls on Saturday; TAIFEX traded on 5/3 (no compensatory day off)
        date(2021, 6, 14),  # 端午
        date(2021, 9, 20), date(2021, 9, 21),  # 中秋
        date(2021, 10, 11),  # 國慶補假
    ],
    2022: [
        # 2022-01-03: TAIFEX traded (1/1 Sat, no Mon compensatory for futures market)
        date(2022, 1, 31), date(2022, 2, 1), date(2022, 2, 2),
        date(2022, 2, 3), date(2022, 2, 4),  # 春節
        date(2022, 2, 28),  # 和平紀念日
        date(2022, 4, 4), date(2022, 4, 5),  # 兒童+清明
        date(2022, 5, 2),  # 勞動節補假
        date(2022, 6, 3),  # 端午
        date(2022, 9, 9),  # 中秋
        date(2022, 10, 10),  # 國慶
    ],
    2023: [
        date(2023, 1, 2),  # 元旦補假
        date(2023, 1, 20), date(2023, 1, 23), date(2023, 1, 24),
        date(2023, 1, 25), date(2023, 1, 26), date(2023, 1, 27),  # 春節
        date(2023, 2, 27), date(2023, 2, 28),  # 和平紀念日連假
        date(2023, 4, 3), date(2023, 4, 4), date(2023, 4, 5),  # 兒童+清明
        date(2023, 5, 1),  # 勞動節
        date(2023, 6, 22), date(2023, 6, 23),  # 端午
        date(2023, 9, 29),  # 中秋
        date(2023, 10, 9), date(2023, 10, 10),  # 國慶
    ],
    2024: [
        date(2024, 1, 1),  # 元旦
        date(2024, 2, 8), date(2024, 2, 9),
        date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14),  # 春節
        date(2024, 2, 28),  # 和平紀念日
        date(2024, 4, 4), date(2024, 4, 5),  # 兒童+清明
        date(2024, 5, 1),  # 勞動節
        date(2024, 6, 10),  # 端午
        date(2024, 9, 17),  # 中秋
        date(2024, 10, 10),  # 國慶
    ],
    2025: [
        date(2025, 1, 1),  # 元旦
        date(2025, 1, 27), date(2025, 1, 28), date(2025, 1, 29),
        date(2025, 1, 30), date(2025, 1, 31),  # 春節
        date(2025, 2, 28),  # 和平紀念日
        date(2025, 4, 3), date(2025, 4, 4),  # 兒童+清明
        date(2025, 5, 1),  # 勞動節
        date(2025, 5, 30),  # 端午 (5/31 六; 6/2 Mon TAIFEX traded)
        date(2025, 10, 6), date(2025, 10, 10),  # 中秋+國慶
    ],
    2026: [
        date(2026, 1, 1),  # 元旦 (1/2 Fri TAIFEX traded)
        date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
        date(2026, 2, 19), date(2026, 2, 20),  # 春節
        # 2026-03-02: 228 補假 — TAIFEX traded (no compensatory for futures market)
        date(2026, 4, 3), date(2026, 4, 6),  # 兒童+清明
        date(2026, 5, 1),  # 勞動節
        date(2026, 6, 19),  # 端午
        date(2026, 10, 5),  # 中秋
        date(2026, 10, 12),  # 國慶補假 (10/10 六)
    ],
    2027: [
        date(2027, 1, 1),  # 元旦
        date(2027, 2, 5), date(2027, 2, 8), date(2027, 2, 9),
        date(2027, 2, 10), date(2027, 2, 11), date(2027, 2, 12),  # 春節
        date(2027, 3, 1),  # 228 補假 (2/28 日)
        date(2027, 4, 5), date(2027, 4, 6),  # 兒童+清明
        date(2027, 5, 3),  # 勞動節補假 (5/1 六)
        date(2027, 6, 9),  # 端午
        date(2027, 9, 24),  # 中秋 (9/25 六 → 週五放假)
        date(2027, 10, 11),  # 國慶補假 (10/10 日)
    ],
}


@lru_cache(maxsize=1)
def _all_holidays() -> frozenset[date]:
    """Build the complete set of known TAIFEX holidays."""
    holidays: set[date] = set()
    for year_dates in _ANNUAL_HOLIDAYS.values():
        holidays.update(year_dates)
    return frozenset(holidays)


def is_taifex_holiday(d: date) -> bool:
    """Return True if *d* is a known TAIFEX holiday (non-trading day).

    Also returns True for weekends.
    """
    if d.weekday() >= 5:
        return True
    return d in _all_holidays()


def is_trading_day(d: date) -> bool:
    """Return True if *d* is a TAIFEX trading day."""
    return not is_taifex_holiday(d)


def last_trading_day_before(d: date) -> date:
    """Return the most recent trading day strictly before *d*."""
    from datetime import timedelta

    d = d - timedelta(days=1)
    while is_taifex_holiday(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(start: date, end: date) -> list[date]:
    """Return list of trading days in [start, end] inclusive."""
    from datetime import timedelta

    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days
