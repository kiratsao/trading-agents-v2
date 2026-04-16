"""Tests for src.data.daily_updater."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

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
    def test_append_new_bars(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")

        with (
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-08")], name="date"),
            )
            mock_fetch.return_value = new_bar
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["bars_added"] == 1
        assert result["error"] is None
        df = pd.read_parquet(pq)
        assert len(df) == 4

    def test_no_duplicate(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-08")

        with (
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            dup_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-08")], name="date"),
            )
            mock_fetch.return_value = dup_bar
            result = update(parquet_path=pq)

        assert result["bars_added"] == 0

    def test_weekend_filtered(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-10")

        with (
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 13)),
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
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate", return_value=None),
        ):
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["bars_added"] == 0

    def test_up_to_date(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-08")

        with patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)):
            result = update(parquet_path=pq)

        assert result["success"] is True

    def test_concat_preserves_datetime_index(self, tmp_path):
        """After append, parquet index is DatetimeIndex (not RangeIndex)."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")

        with (
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-08")], name="date"),
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
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 9)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch2,
        ):
            bar2 = pd.DataFrame(
                {"open": [33500.0], "high": [33600.0], "low": [33400.0],
                 "close": [33550.0], "volume": [100000]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-09")], name="date"),
            )
            mock_fetch2.return_value = bar2
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["bars_added"] == 1

    def test_fetch_failure_notifies(self, tmp_path):
        """Failure → success=False, error set, notify_fn called with 🔴."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-04-07")
        notify = MagicMock()

        with (
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)),
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
            patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 8)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            new_bar = pd.DataFrame(
                {"open": [33300.0], "high": [33500.0], "low": [33250.0],
                 "close": [33450.0], "volume": [120000]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-08")], name="date"),
            )
            mock_fetch.return_value = new_bar
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["success"] is True
        notify.assert_called_once()
        assert "✅" in notify.call_args[0][0]
        assert "33,450" in notify.call_args[0][0]
