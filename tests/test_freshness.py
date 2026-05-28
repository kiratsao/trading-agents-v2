"""src.utils.freshness — single source of truth for expected-latest / freshness."""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.freshness import check_parquet_freshness, expected_parquet_latest


def _write(path, latest):
    pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1.0], "volume": [9]},
        index=pd.DatetimeIndex([pd.Timestamp(latest)], name="date"),
    ).to_parquet(path, index=True)


# ── expected_parquet_latest (time-of-day + holiday aware) ───────────────────
def test_expected_monday_before_window():
    # Mon 2026-05-25 08:00 < 14:30 → T-2 = prev(prev(Mon)) = Thu 5/21
    assert expected_parquet_latest(datetime(2026, 5, 25, 8, 0)) == date(2026, 5, 21)


def test_expected_monday_after_window():
    # Mon 2026-05-25 14:30 ≥ window → T-1 = prev(Mon) = Fri 5/22
    assert expected_parquet_latest(datetime(2026, 5, 25, 14, 30)) == date(2026, 5, 22)


def test_expected_saturday():
    # Sat 2026-05-30 (non-trading) → prev(prev(Sat)) = prev(Fri 5/29) = Thu 5/28
    assert expected_parquet_latest(datetime(2026, 5, 30, 10, 0)) == date(2026, 5, 28)


def test_expected_post_holiday():
    # Mon 2026-05-04 15:00; Fri 2026-05-01 = 勞動節 holiday → T-1 = Thu 4/30
    assert expected_parquet_latest(datetime(2026, 5, 4, 15, 0)) == date(2026, 4, 30)


# ── check_parquet_freshness ─────────────────────────────────────────────────
def test_check_fresh(tmp_path):
    p = tmp_path / "p.parquet"
    now = datetime(2026, 5, 28, 15, 0)  # expected = 5/27
    _write(p, "2026-05-27")
    is_fresh, msg, expected = check_parquet_freshness(p, now)
    assert is_fresh is True and expected == date(2026, 5, 27)


def test_check_stale(tmp_path):
    p = tmp_path / "p.parquet"
    now = datetime(2026, 5, 28, 15, 0)
    _write(p, "2026-05-20")
    is_fresh, msg, _ = check_parquet_freshness(p, now)
    assert is_fresh is False and "過期" in msg


def test_check_missing(tmp_path):
    is_fresh, msg, _ = check_parquet_freshness(tmp_path / "nope.parquet")
    assert is_fresh is False and "不存在" in msg
