"""台灣期交所 (TAIFEX) 休市日清單。

期交所每年 12 月公布隔年休市日。本模組維護 2020–2027 年休市清單，
供 init_data / daily_updater / verify_data 判斷是否為交易日。

來源: https://www.taifex.com.tw/cht/5/holidaySchedule

維護方式: 每年 12 月底追加下一年度。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)

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
    # 2026-07-04 全表用真實 parquet 實測稽核過(表=交易日但市場無 bar 的日子
    # 全數補入):含颱風臨時停市、春節前封關日、2025 起新增之國定假日。
    2020: [
        date(2020, 1, 1),
        date(2020, 1, 21), date(2020, 1, 22),  # 春節前封關 (實測: 無交易)
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
        date(2021, 4, 30),  # 勞動節補假 (5/1 六; 實測: 無交易)
        date(2021, 6, 14),  # 端午
        date(2021, 9, 20), date(2021, 9, 21),  # 中秋
        date(2021, 10, 11),  # 國慶補假
        date(2021, 12, 31),  # 2022 元旦補假 (1/1 六; 實測: 無交易)
    ],
    2022: [
        # 2022-01-03: TAIFEX traded (1/1 Sat, no Mon compensatory for futures market)
        date(2022, 1, 27), date(2022, 1, 28),  # 春節前封關 (實測: 無交易)
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
        date(2023, 1, 18), date(2023, 1, 19),  # 春節前封關 (實測: 無交易)
        date(2023, 1, 20), date(2023, 1, 23), date(2023, 1, 24),
        date(2023, 1, 25), date(2023, 1, 26), date(2023, 1, 27),  # 春節
        date(2023, 2, 27), date(2023, 2, 28),  # 和平紀念日連假
        date(2023, 4, 3), date(2023, 4, 4), date(2023, 4, 5),  # 兒童+清明
        date(2023, 5, 1),  # 勞動節
        date(2023, 6, 22), date(2023, 6, 23),  # 端午
        date(2023, 8, 3),  # 颱風卡努停市 (實測: 無交易)
        date(2023, 9, 29),  # 中秋
        date(2023, 10, 9), date(2023, 10, 10),  # 國慶
    ],
    2024: [
        date(2024, 1, 1),  # 元旦
        date(2024, 2, 6), date(2024, 2, 7),  # 春節前封關 (實測: 無交易)
        date(2024, 2, 8), date(2024, 2, 9),
        date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14),  # 春節
        date(2024, 2, 28),  # 和平紀念日
        date(2024, 4, 4), date(2024, 4, 5),  # 兒童+清明
        date(2024, 5, 1),  # 勞動節
        date(2024, 6, 10),  # 端午
        date(2024, 7, 24), date(2024, 7, 25),  # 颱風凱米停市 (實測: 無交易)
        date(2024, 9, 17),  # 中秋
        date(2024, 10, 2), date(2024, 10, 3),  # 颱風山陀兒停市 (實測: 無交易)
        date(2024, 10, 10),  # 國慶
        date(2024, 10, 31),  # 颱風康芮停市 (實測: 無交易)
    ],
    2025: [
        date(2025, 1, 1),  # 元旦
        date(2025, 1, 23), date(2025, 1, 24),  # 春節前封關 (實測: 無交易)
        date(2025, 1, 27), date(2025, 1, 28), date(2025, 1, 29),
        date(2025, 1, 30), date(2025, 1, 31),  # 春節
        date(2025, 2, 28),  # 和平紀念日
        date(2025, 4, 3), date(2025, 4, 4),  # 兒童+清明
        date(2025, 5, 1),  # 勞動節
        date(2025, 5, 30),  # 端午 (5/31 六; 6/2 Mon TAIFEX traded)
        date(2025, 9, 29),  # 教師節補假 (9/28 日; 2025 起新增國定假日; 實測: 無交易)
        date(2025, 10, 6), date(2025, 10, 10),  # 中秋+國慶
        date(2025, 10, 24),  # 光復節補假 (10/25 六; 2025 起新增; 實測: 無交易)
        date(2025, 12, 25),  # 行憲紀念日 (2025 起恢復放假; 實測: 無交易)
    ],
    2026: [
        date(2026, 1, 1),  # 元旦 (1/2 Fri TAIFEX traded)
        date(2026, 2, 12), date(2026, 2, 13),  # 春節前封關 (實測: 無交易)
        date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
        date(2026, 2, 19), date(2026, 2, 20),  # 春節
        date(2026, 2, 27),  # 228 彈性放假 (2/28 六; 實測: 無交易)
        # 2026-03-02: TAIFEX traded (no compensatory for futures market)
        date(2026, 4, 3), date(2026, 4, 6),  # 兒童+清明
        date(2026, 5, 1),  # 勞動節
        date(2026, 6, 19),  # 端午
        date(2026, 9, 25),  # 中秋 (原表誤植 10/5 — 依期貨商 2026 休市表修正)
        date(2026, 9, 28),  # 教師節 (2025 起新增國定假日)
        date(2026, 10, 9),  # 國慶補假 (10/10 六 → 前一日週五補; 原表誤植 10/12)
        date(2026, 10, 26),  # 光復節補假 (10/25 日; 2025 起新增)
        date(2026, 12, 25),  # 行憲紀念日
    ],
    # ⚠️ 2027 為預估值(政府/期交所行事曆通常於 2026 年中後公告)。
    # 公告後務必核對:封關日、補假規則、中秋農曆換算。
    2027: [
        date(2027, 1, 1),  # 元旦
        date(2027, 2, 5), date(2027, 2, 8), date(2027, 2, 9),
        date(2027, 2, 10), date(2027, 2, 11), date(2027, 2, 12),  # 春節
        date(2027, 3, 1),  # 228 補假 (2/28 日)
        date(2027, 4, 5), date(2027, 4, 6),  # 兒童+清明
        date(2027, 5, 3),  # 勞動節補假 (5/1 六)
        date(2027, 6, 9),  # 端午
        date(2027, 9, 15),  # 中秋 (2027 農曆八月十五; 原表 9/24 有誤, 待官方公告確認)
        date(2027, 9, 28),  # 教師節 (2025 起新增國定假日)
        date(2027, 10, 11),  # 國慶補假 (10/10 日)
        date(2027, 10, 25),  # 光復節 (2025 起新增)
        date(2027, 12, 24),  # 行憲紀念日補假 (12/25 六 → 前一日週五補, 待公告確認)
    ],
}


@lru_cache(maxsize=1)
def _all_holidays() -> frozenset[date]:
    """Build the complete set of known TAIFEX holidays."""
    holidays: set[date] = set()
    for year_dates in _ANNUAL_HOLIDAYS.values():
        holidays.update(year_dates)
    return frozenset(holidays)


_MAX_KNOWN_YEAR = max(_ANNUAL_HOLIDAYS)
_warned_uncovered_years: set[int] = set()


def holidays_cover(d: date) -> bool:
    """True when the holiday table has been maintained for *d*'s year."""
    return d.year in _ANNUAL_HOLIDAYS


def is_taifex_holiday(d: date) -> bool:
    """Return True if *d* is a known TAIFEX holiday (non-trading day).

    Also returns True for weekends. Queries beyond the maintained table
    (_MAX_KNOWN_YEAR) silently treat weekdays as trading days — that would
    make the bot "trade" through Chinese New Year, so it logs loudly once
    per year value until the table is extended.
    """
    if d.weekday() >= 5:
        return True
    if d.year > _MAX_KNOWN_YEAR and d.year not in _warned_uncovered_years:
        _warned_uncovered_years.add(d.year)
        logger.error(
            "tw_holidays: %d 年假日表未維護（僅到 %d）— 週間日一律視為交易日，"
            "請更新 _ANNUAL_HOLIDAYS", d.year, _MAX_KNOWN_YEAR,
        )
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
    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def settlement_day_of_month(d: date) -> date:
    """TAIFEX monthly settlement day for *d*'s month.

    3rd Wednesday, deferred to the NEXT trading day when that Wednesday is a
    TAIFEX holiday (e.g. 2026-02-18 fell in the Chinese New Year break — the
    actual settlement was the next trading day). Callers that only compare
    ``d == third_wednesday`` silently skip settlement in such months.
    """
    import calendar

    c = calendar.monthcalendar(d.year, d.month)
    wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
    s = date(d.year, d.month, wednesdays[2])
    while is_taifex_holiday(s):
        s += timedelta(days=1)
    return s


def is_settlement_date(d: date) -> bool:
    """True when *d* is the (holiday-adjusted) monthly settlement day."""
    return d == settlement_day_of_month(d)
