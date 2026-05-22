"""Tests for src.data.daily_updater."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.daily_updater import update


def _make_existing_parquet(path: Path, last_date: str = "2026-04-07") -> None:
    dates = pd.bdate_range(end=last_date, periods=3)
    df = pd.DataFrame(
        {
            "open": [33000.0, 33100.0, 33200.0],
            "high": [33200.0, 33300.0, 33400.0],
            "low": [32800.0, 32900.0, 33000.0],
            "close": [33100.0, 33200.0, 33300.0],
            "volume": [100000, 110000, 120000],
        },
        index=dates,
    )
    df.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


class TestDailyUpdater:
    @pytest.fixture(autouse=True)
    def _silence_gap_scan(self):
        """The append-path tests below use a sparse 3-bar fixture parquet
        which the new gap-fill code would treat as 7+ missing days. Skip
        the gap scan for these tests; gap-detection coverage lives in
        TestGapDetection below."""
        with patch(
            "src.data.daily_updater._detect_and_fill_gaps",
            return_value=(0, []),
        ):
            yield

    def test_append_new_bars(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-08")], name="date",
                ),
            )
            mock_fetch.return_value = new_bar
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["bars_added"] == 1
        assert result["error"] is None
        df = pd.read_parquet(pq)
        assert len(df) == 4
        # Fetch range should be start→yesterday (2026-04-08)
        mock_fetch.assert_called_once_with(date(2026, 4, 8), date(2026, 4, 8))

    def test_only_fetches_up_to_yesterday(self, tmp_path):
        """Core fix: today=2026-04-08 (Tuesday) → fetch only up to 2026-04-07."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-04")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-07")], name="date",
                ),
            )
            mock_fetch.return_value = new_bar
            result = update(parquet_path=pq)

        # Should fetch up to yesterday=2026-04-07, NOT today=2026-04-08
        # fetch_start = last_date+1 (bdate after 2026-04-04 is 2026-04-05)
        args = mock_fetch.call_args[0]
        assert args[1] == date(2026, 4, 7), "end must be yesterday, not today"
        assert result["bars_added"] == 1

    def test_no_duplicate(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-08")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            dup_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-08")], name="date",
                ),
            )
            mock_fetch.return_value = dup_bar
            result = update(parquet_path=pq)

        assert result["bars_added"] == 0

    def test_weekend_filtered(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-10")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 14)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            bars = pd.DataFrame(
                {"open": [33300.0, 33300.0, 33400.0],
                 "high": [33500.0, 33500.0, 33600.0],
                 "low": [33250.0, 33250.0, 33350.0],
                 "close": [33450.0, 33450.0, 33550.0],
                 "volume": [100, 100, 120000]},
                index=pd.DatetimeIndex([
                    pd.Timestamp("2026-04-11"),
                    pd.Timestamp("2026-04-12"),
                    pd.Timestamp("2026-04-13"),
                ], name="date"),
            )
            mock_fetch.return_value = bars
            result = update(parquet_path=pq)

        assert result["bars_added"] == 1

    def test_empty_kbars(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  return_value=None),
        ):
            notify = MagicMock()
            result = update(parquet_path=pq, notify_fn=notify)

        # bars_added=0 AND latest < yesterday → should FAIL with 🔴 alert
        assert result["success"] is False
        assert result["bars_added"] == 0
        notify.assert_called_once()
        assert "🔴" in notify.call_args[0][0]

    def test_empty_kbars_already_current(self, tmp_path):
        """bars_added=0 but latest_date = yesterday → OK."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-08")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
        ):
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["bars_added"] == 0

    def test_up_to_date(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-08")

        with patch("src.data.daily_updater._today_taipei",
                   return_value=date(2026, 4, 9)):
            result = update(parquet_path=pq)

        assert result["success"] is True

    def test_concat_preserves_datetime_index(self, tmp_path):
        """After append, parquet index is DatetimeIndex (not RangeIndex)."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-08")], name="date",
                ),
            )
            mock_fetch.return_value = new_bar
            update(parquet_path=pq)

        # Reload and verify index type
        df = pd.read_parquet(pq)
        assert isinstance(df.index, pd.DatetimeIndex), (
            f"Expected DatetimeIndex, got {type(df.index)}"
        )

        # Second update should not crash on .normalize()
        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 10)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock2,
        ):
            bar2 = pd.DataFrame(
                {"open": [33500.0], "high": [33600.0], "low": [33400.0],
                 "close": [33550.0], "volume": [100000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-09")], name="date",
                ),
            )
            mock2.return_value = bar2
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["bars_added"] == 1

    def test_fetch_failure_notifies(self, tmp_path):
        """Failure → success=False, error set, notify_fn called with 🔴."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")
        notify = MagicMock()

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  side_effect=ConnectionError("timeout")),
        ):
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["success"] is False
        assert "timeout" in result["error"]
        notify.assert_called_once()
        assert "🔴" in notify.call_args[0][0]

    def test_success_notifies(self, tmp_path):
        """Success → notify_fn called with ✅ and close price."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")
        notify = MagicMock()

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-08")], name="date",
                ),
            )
            mock_fetch.return_value = new_bar
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["success"] is True
        notify.assert_called_once()
        assert "✅" in notify.call_args[0][0]
        assert "33,450" in notify.call_args[0][0]

    def test_data_gap_alerts(self, tmp_path):
        """If parquet is 3 days behind and no bars fetched → 🔴 alert."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-03")
        notify = MagicMock()

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  return_value=None),
        ):
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["success"] is False
        assert "🔴" in notify.call_args[0][0]
        assert "資料缺口" in notify.call_args[0][0]


def _seed_parquet_with_hole(path: Path, hole_date: str, *, anchor: str) -> None:
    """Build a parquet contiguous around *anchor* except *hole_date* is
    missing. Used to reproduce a 05/18-style intra-window gap."""
    days = pd.bdate_range(end=anchor, periods=10)
    keep = [d for d in days if d.date().isoformat() != hole_date]
    df = pd.DataFrame(
        {
            "open":  [33000.0] * len(keep),
            "high":  [33200.0] * len(keep),
            "low":   [32800.0] * len(keep),
            "close": [33100.0] * len(keep),
            "volume": [100_000] * len(keep),
        },
        index=pd.DatetimeIndex(keep, name="date"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


class TestGapDetection:
    """05/18-style intra-window gap detection and back-fill."""

    def test_gap_detection_back_fills_missing_day(self, tmp_path):
        """Parquet missing one day inside the look-back window: gap scan
        must (1) detect, (2) fetch with the exact missing date, (3)
        append, (4) report gaps_filled >= 1."""
        pq = tmp_path / "test.parquet"
        # Anchor 2026-04-07; drop 2026-04-01 (this is the test's "05/18").
        _seed_parquet_with_hole(pq, hole_date="2026-04-01", anchor="2026-04-07")

        per_date_calls: list[date] = []

        def fake_fetch(start: date, end: date):
            per_date_calls.append(start)
            return pd.DataFrame(
                {"open": [33500.0], "high": [33700.0], "low": [33400.0],
                 "close": [33600.0], "volume": [80_000]},
                index=pd.DatetimeIndex([pd.Timestamp(start)], name="date"),
            )

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  side_effect=fake_fetch),
        ):
            result = update(parquet_path=pq)

        assert result["success"] is True, result
        assert result["gaps_filled"] >= 1
        assert date(2026, 4, 1) in per_date_calls, (
            "back-fill must explicitly request the missing date"
        )
        df = pd.read_parquet(pq)
        assert pd.Timestamp("2026-04-01") in df.index

    def test_gap_alert_on_failure(self, tmp_path):
        """Back-fill fetcher returns None for the gap day → LINE alert
        with ⚠️ {date} 資料缺失 marker; success=False."""
        pq = tmp_path / "test.parquet"
        _seed_parquet_with_hole(pq, hole_date="2026-04-01", anchor="2026-04-07")
        notify = MagicMock()

        def fake_fetch(start: date, end: date):
            # Main append (2026-04-08) succeeds; gap back-fill returns None.
            if start == date(2026, 4, 8):
                return pd.DataFrame(
                    {"open": [33500.0], "high": [33700.0], "low": [33400.0],
                     "close": [33600.0], "volume": [80_000]},
                    index=pd.DatetimeIndex(
                        [pd.Timestamp("2026-04-08")], name="date",
                    ),
                )
            return None

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  side_effect=fake_fetch),
        ):
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["success"] is False
        assert "unfilled" in (result["error"] or "")
        alerts = [c.args[0] for c in notify.call_args_list]
        assert any(
            "2026-04-01" in m and "資料缺失" in m and "⚠️" in m
            for m in alerts
        ), f"expected unfilled-gap alert, got: {alerts}"

    def test_no_gap_when_window_contiguous(self, tmp_path):
        """Contiguous parquet (no holes) → gap scan no-ops.

        Use periods=20 so the look-back window's 10 TAIFEX trading days
        are fully covered even when holidays expand calendar reach."""
        pq = tmp_path / "test.parquet"
        days = pd.bdate_range(end="2026-04-07", periods=20)
        df = pd.DataFrame(
            {"open": [33000.0] * len(days), "high": [33200.0] * len(days),
             "low": [32800.0] * len(days), "close": [33100.0] * len(days),
             "volume": [100_000] * len(days)},
            index=pd.DatetimeIndex(days, name="date"),
        )
        df.to_parquet(pq, index=True)

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate") as m,
        ):
            m.return_value = pd.DataFrame(
                {"open": [33500.0], "high": [33700.0], "low": [33400.0],
                 "close": [33600.0], "volume": [80_000]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-08")], name="date",
                ),
            )
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["gaps_filled"] == 0
        # Parquet already at yesterday → already-up-to-date branch, no
        # main fetch; window is contiguous → no back-fill probes.
        assert m.call_count == 0
