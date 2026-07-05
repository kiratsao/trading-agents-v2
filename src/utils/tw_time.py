"""台北時區時間的單一來源。

程式跑在 GCP VM 上,系統時區不保證是 Asia/Taipei(UTC 的話,台北 00:00–08:00
之間裸 ``date.today()`` 會少一天)。所有「今天/現在」一律經過這裡,
禁止在業務邏輯直接呼叫 ``datetime.now()`` / ``date.today()``。
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")


def now_taipei() -> datetime:
    """Current tz-aware datetime in Asia/Taipei."""
    return datetime.now(tz=TAIPEI)


def today_taipei() -> date:
    """Current calendar date in Asia/Taipei (not the VM's local date)."""
    return now_taipei().date()
