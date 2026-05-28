"""daily_updater.ensure_parquet_fresh + daily_updater_cli exit codes (offline)."""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import daily_updater
from src.data.daily_updater import ensure_parquet_fresh
from src.utils.freshness import DataIntegrityError

_NOW = datetime(2026, 5, 28, 15, 0)  # ≥14:30 trading day → target = Wed 5/27
_NOSLEEP = lambda *_a, **_k: None  # noqa: E731


def _write(path, latest):
    pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [100.0], "volume": [99999]},
        index=pd.DatetimeIndex([pd.Timestamp(latest)], name="date"),
    ).to_parquet(path, index=True)


def _bar(d0, _d1):
    return pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [200.0], "volume": [99999]},
        index=pd.DatetimeIndex([pd.Timestamp(d0)], name="date"),
    )


def test_noop_when_at_target(tmp_path):
    p = tmp_path / "p.parquet"
    _write(p, "2026-05-27")
    r = ensure_parquet_fresh(p, now=_NOW, fetch_override=_bar, sleep=_NOSLEEP)
    assert r["status"] == "noop" and r["filled"] == 0


def test_gap1_fills(tmp_path):
    p = tmp_path / "p.parquet"
    _write(p, "2026-05-26")
    r = ensure_parquet_fresh(p, now=_NOW, fetch_override=_bar, sleep=_NOSLEEP)
    assert r["status"] == "ok" and r["filled"] == 1
    assert pd.read_parquet(p).index.max().date() == date(2026, 5, 27)


def test_gap3_with_holiday(tmp_path, monkeypatch):
    import src.data.tw_holidays as h

    # 5/26 forced holiday (still treat weekends as non-trading).
    monkeypatch.setattr(
        h, "is_taifex_holiday", lambda d: d == date(2026, 5, 26) or d.weekday() >= 5
    )
    p = tmp_path / "p.parquet"
    _write(p, "2026-05-21")  # → missing 5/22,5/25,5/27 (5/26 holiday)
    fetched: list[date] = []

    def fetch(d0, _d1):
        fetched.append(d0)
        return _bar(d0, _d1)

    r = ensure_parquet_fresh(p, now=_NOW, fetch_override=fetch, sleep=_NOSLEEP)
    assert r["status"] == "ok" and r["filled"] == 3
    assert date(2026, 5, 26) not in fetched  # holiday skipped


def test_gap_over_max_raises(tmp_path):
    p = tmp_path / "p.parquet"
    _write(p, "2026-04-01")  # > 10 trading days behind 5/27
    with pytest.raises(DataIntegrityError):
        ensure_parquet_fresh(p, now=_NOW, fetch_override=_bar, sleep=_NOSLEEP)


def test_partial_when_fetch_fails(tmp_path):
    p = tmp_path / "p.parquet"
    _write(p, "2026-05-26")
    r = ensure_parquet_fresh(p, now=_NOW, fetch_override=lambda *_: None, sleep=_NOSLEEP)
    assert r["status"] == "partial"
    assert pd.read_parquet(p).index.max().date() == date(2026, 5, 26)  # not advanced


def test_post_write_verify_fail_raises(tmp_path):
    p = tmp_path / "p.parquet"
    _write(p, "2026-05-26")
    # fetch "succeeds" but returns an already-present bar → nothing advances,
    # no still_missing → post-write verify (latest < target) must raise.
    with pytest.raises(DataIntegrityError):
        ensure_parquet_fresh(
            p, now=_NOW, sleep=_NOSLEEP, fetch_override=lambda d0, d1: _bar(date(2026, 5, 26), d1)
        )


def test_missing_parquet_raises(tmp_path):
    with pytest.raises(DataIntegrityError):
        ensure_parquet_fresh(tmp_path / "nope.parquet", now=_NOW, sleep=_NOSLEEP)


# ── daily_updater_cli exit codes ────────────────────────────────────────────
def test_cli_exit_codes(monkeypatch):
    from src.data import daily_updater_cli as cli

    monkeypatch.setattr(cli, "_line_notifier", lambda: None)

    monkeypatch.setattr(
        daily_updater,
        "ensure_parquet_fresh",
        lambda **k: {"status": "ok", "missing": [], "filled": 0, "latest": "x", "target": "y"},
    )
    assert cli.main(["--no-validate"]) == 0

    monkeypatch.setattr(
        daily_updater,
        "ensure_parquet_fresh",
        lambda **k: {
            "status": "partial",
            "missing": ["2026-05-27"],
            "filled": 0,
            "latest": "x",
            "target": "y",
        },
    )
    assert cli.main(["--no-validate"]) == 1

    def _raise(**k):
        raise DataIntegrityError("boom")

    monkeypatch.setattr(daily_updater, "ensure_parquet_fresh", _raise)
    assert cli.main(["--no-validate"]) == 2
