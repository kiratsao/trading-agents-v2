"""Tests for src.data.daily_updater."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from src.data.daily_updater import _fetch_and_aggregate, update


def _one_day_frame(day: str, close: float) -> pd.DataFrame:
    ts = pd.Timestamp(day)
    df = pd.DataFrame(
        {"open": [close], "high": [close], "low": [close],
         "close": [close], "volume": [45055]},
        index=pd.DatetimeIndex([ts], name="date"),
    )
    return df


class TestTaifexFallback:
    """Task 1: when Shioaji has no day-session data (2026-06-01 class), the
    fetch must fall back to TAIFEX and loudly flag the missing cross-validation.
    """

    def test_shioaji_empty_falls_back_to_taifex(self):
        notes: list[str] = []
        d = date(2026, 6, 1)
        with (
            patch("src.data.shioaji_fetcher.fetch_via_env",
                  return_value=pd.DataFrame()),
            patch("src.data.validation.fetch_taifex_day_session_range",
                  return_value=_one_day_frame("2026-06-01", 45055.0)) as mock_tx,
        ):
            out = _fetch_and_aggregate(d, d, notify_fn=notes.append)

        assert out is not None and not out.empty
        assert float(out["close"].iloc[-1]) == 45055.0
        mock_tx.assert_called_once_with(d, d)
        # Must surface that the bar is unvalidated (user requirement).
        assert any("TAIFEX" in m and "無獨立交叉驗證" in m for m in notes), notes

    def test_shioaji_exception_falls_back_to_taifex(self):
        d = date(2026, 6, 1)
        with (
            patch("src.data.shioaji_fetcher.fetch_via_env",
                  side_effect=RuntimeError("creds missing")),
            patch("src.data.validation.fetch_taifex_day_session_range",
                  return_value=_one_day_frame("2026-06-01", 45055.0)),
        ):
            out = _fetch_and_aggregate(d, d)
        assert out is not None and float(out["close"].iloc[-1]) == 45055.0

    def test_shioaji_present_skips_taifex(self):
        d = date(2026, 6, 2)
        with (
            patch("src.data.shioaji_fetcher.fetch_via_env",
                  return_value=_one_day_frame("2026-06-02", 44900.0)),
            patch("src.data.validation.fetch_taifex_day_session_range") as mock_tx,
        ):
            out = _fetch_and_aggregate(d, d)
        assert out is not None and float(out["close"].iloc[-1]) == 44900.0
        mock_tx.assert_not_called()

    def test_both_empty_returns_none(self):
        d = date(2026, 6, 1)
        with (
            patch("src.data.shioaji_fetcher.fetch_via_env",
                  return_value=pd.DataFrame()),
            patch("src.data.validation.fetch_taifex_day_session_range",
                  return_value=None),
        ):
            out = _fetch_and_aggregate(d, d)
        assert out is None


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
        # Fetch range should be start→yesterday (2026-04-08). notify_fn is
        # threaded so a TAIFEX fallback can surface its no-cross-validation alert.
        mock_fetch.assert_called_once_with(
            date(2026, 4, 8), date(2026, 4, 8), notify_fn=ANY,
        )

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


class TestFutureBarGuard:
    """A future-dated last bar = a night-session (盤後) row written as a day K
    (real case: 2026-07-06 bar written while today was 07-05). Left alone the
    updater reports "already up-to-date" forever — it must alert instead."""

    def test_future_last_bar_alerts_and_aborts(self, tmp_path):
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-07-06")  # Monday, after "today"
        notes = []

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 7, 5)),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock_fetch,
        ):
            result = update(parquet_path=pq, notify_fn=notes.append)

        assert result["success"] is False
        assert "在未來" in result["error"]
        assert any("🔴" in n for n in notes)
        mock_fetch.assert_not_called()

    def test_last_bar_equal_today_is_not_flagged(self, tmp_path):
        """last == today is legitimate (14:25 run after today's close)."""
        pq = tmp_path / "test.parquet"
        _make_existing_parquet(pq, "2026-07-06")

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=date(2026, 7, 6)),
            patch("src.data.daily_updater._detect_and_fill_gaps",
                  return_value=(0, [])),
        ):
            result = update(parquet_path=pq)

        assert result["success"] is True
        assert result["error"] is None


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

        def fake_fetch(start: date, end: date, notify_fn=None):
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
        with ⚠️ {date} 日K缺失 marker; success=False."""
        pq = tmp_path / "test.parquet"
        _seed_parquet_with_hole(pq, hole_date="2026-04-01", anchor="2026-04-07")
        notify = MagicMock()

        def fake_fetch(start: date, end: date, notify_fn=None):
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
            "2026-04-01" in m and "日K缺失" in m and "⚠️" in m
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


class TestShioajiFetchEndDate:
    """Regression for the 14:25 data-contamination bug: the previous
    `end_plus = end + 1 day` pulled today's intra-session bar through
    Shioaji's INCLUSIVE end, then aggregated it as the day-session close.
    These tests pin both the `end` arg passed to Shioaji AND the
    post-aggregation `> yesterday` cutoff filter."""

    def _fake_adapter_module(self, kbars_obj):
        """Build a stand-in shioaji_adapter module that records the
        kbars(end=...) argument and returns kbars_obj."""
        captured = {}

        class _FakeKbarsApi:
            def kbars(self, contract, start, end, timeout):
                captured["start"] = start
                captured["end"] = end
                return kbars_obj

        class _FakeAdapter:
            def __init__(self, *args, **kwargs): self._api = _FakeKbarsApi()
            def get_contract(self, product): return object()
            def logout(self): pass

        return _FakeAdapter, captured

    def _make_kbars(self, ts_list, prices):
        """Build a duck-typed kbars-like object out of two parallel lists."""
        class _K:
            ts = [int(pd.Timestamp(t, tz="Asia/Taipei").tz_convert("UTC").value)
                  for t in ts_list]
            Open = list(prices)
            High = list(prices)
            Low = list(prices)
            Close = list(prices)
            Volume = [50_000] * len(prices)  # above the day-session volume floor
        return _K()

    def test_end_date_is_yesterday_only(self):
        """The kbars call's `end` argument must equal yesterday, NOT
        yesterday+1 (today). yesterday=2026-04-08."""
        from src.data import daily_updater as du

        FakeAdapter, captured = self._fake_adapter_module(
            self._make_kbars(
                ["2026-04-08 13:00:00"], [21000.0],
            )
        )
        with (
            patch.dict("os.environ", {
                "SHIOAJI_API_KEY": "k", "SHIOAJI_SECRET_KEY": "s",
            }, clear=False),
            patch("tw_futures.executor.shioaji_adapter.ShioajiAdapter",
                  FakeAdapter),
        ):
            du._fetch_and_aggregate(date(2026, 4, 8), date(2026, 4, 8))

        assert captured["end"] == "2026-04-08", (
            f"end must be yesterday (2026-04-08), got {captured['end']!r}; "
            "leaking today's intra-session bar was the data-contamination bug"
        )

    def test_fetch_does_not_include_today(self):
        """Even if Shioaji leaks bars dated past `end` (today / future),
        the post-aggregation filter must drop them. yesterday=2026-04-08;
        Shioaji is forced to return both 2026-04-08 (good) and
        2026-04-09 (today, must be filtered out)."""
        from src.data import daily_updater as du

        FakeAdapter, _ = self._fake_adapter_module(
            self._make_kbars(
                [
                    "2026-04-08 09:00:00",  # yesterday day-session open
                    "2026-04-08 13:00:00",  # yesterday day-session close
                    "2026-04-09 09:00:00",  # TODAY -- must be dropped
                    "2026-04-09 13:00:00",
                ],
                [21000.0, 21100.0, 19800.0, 19850.0],
            )
        )
        with (
            patch.dict("os.environ", {
                "SHIOAJI_API_KEY": "k", "SHIOAJI_SECRET_KEY": "s",
            }, clear=False),
            patch("tw_futures.executor.shioaji_adapter.ShioajiAdapter",
                  FakeAdapter),
        ):
            df = du._fetch_and_aggregate(date(2026, 4, 8), date(2026, 4, 8))

        assert df is not None and not df.empty
        idx_dates = [t.date() for t in df.index]
        assert date(2026, 4, 9) not in idx_dates, (
            f"today's bar (2026-04-09) leaked through aggregation: {idx_dates}"
        )
        assert date(2026, 4, 8) in idx_dates, (
            "yesterday's bar must still be returned"
        )
