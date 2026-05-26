"""Part B: cross-source data validation (init_data B1 + daily_updater B2)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import daily_updater
from src.data.validation import (
    CloseDiff,
    validate_and_override_with_shioaji,
    validate_latest_bar,
)


def _ref(day: date, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": close, "high": close + 5, "low": close - 5, "close": close, "volume": 1}],
        index=pd.DatetimeIndex([pd.Timestamp(day)], name="date"),
    )


# ── validate_latest_bar (B2 decision) ───────────────────────────────────────
def test_validate_latest_bar_ok_when_sources_agree():
    d = date(2026, 5, 21)
    level, diffs = validate_latest_bar(
        d, 20_000.0,
        shioaji_fetch=lambda a, b: _ref(d, 20_000.0),
        taifex_fetch=lambda a, b: _ref(d, 20_010.0),  # 10pt ≤ 50
    )
    assert level == "ok" and diffs == []


def test_validate_latest_bar_warns_between_50_and_200():
    d = date(2026, 5, 21)
    level, diffs = validate_latest_bar(
        d, 20_000.0,
        shioaji_fetch=lambda a, b: _ref(d, 20_000.0),
        taifex_fetch=lambda a, b: _ref(d, 20_080.0),  # 80pt
    )
    assert level == "warn"
    assert [cd.source for cd in diffs] == ["taifex"]


def test_validate_latest_bar_alerts_above_200():
    d = date(2026, 5, 21)
    level, diffs = validate_latest_bar(
        d, 20_000.0,
        shioaji_fetch=lambda a, b: _ref(d, 20_350.0),  # 350pt → alert
        taifex_fetch=lambda a, b: _ref(d, 20_000.0),
    )
    assert level == "alert"
    assert any(cd.diff > 200 for cd in diffs)


def test_validate_latest_bar_skips_failed_reference():
    d = date(2026, 5, 21)

    def boom(a, b):
        raise RuntimeError("offline")

    # shioaji dead, taifex agrees → still ok (dead ref never blocks)
    level, diffs = validate_latest_bar(
        d, 20_000.0, shioaji_fetch=boom, taifex_fetch=lambda a, b: _ref(d, 20_000.0),
    )
    assert level == "ok"


def test_validate_latest_bar_both_dead_is_ok():
    d = date(2026, 5, 21)

    def boom(a, b):
        raise RuntimeError("offline")

    assert validate_latest_bar(d, 20_000.0, shioaji_fetch=boom, taifex_fetch=boom)[0] == "ok"


# ── B1: validate_and_override_with_shioaji ──────────────────────────────────
def test_b1_overrides_divergent_bar_with_shioaji():
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in
                            (date(2026, 5, 19), date(2026, 5, 20), date(2026, 5, 21))], name="date")
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                       "close": [20_000.0, 20_500.0, 21_000.0], "volume": [1, 2, 3]}, index=idx)

    def shioaji(a, b):
        # 5/20 diverges by 400pt; the other two agree
        rows = {date(2026, 5, 19): 20_000.0, date(2026, 5, 20): 20_900.0,
                date(2026, 5, 21): 21_000.0}
        return pd.DataFrame(
            [{"open": c, "high": c, "low": c, "close": c, "volume": 9} for c in rows.values()],
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in rows], name="date"),
        )

    logs: list[str] = []
    out, overridden = validate_and_override_with_shioaji(
        df, recent_days=20, shioaji_fetch=shioaji, threshold=50.0, log=logs.append,
    )
    assert [cd.day for cd in overridden] == [date(2026, 5, 20)]
    assert out.loc[pd.Timestamp("2026-05-20"), "close"] == 20_900.0   # overridden
    assert out.loc[pd.Timestamp("2026-05-21"), "close"] == 21_000.0   # untouched
    assert any("用 Shioaji 覆蓋" in m for m in logs)


def test_b1_no_override_when_agree():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-21")], name="date")
    df = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [21_000.0], "volume": [1]}, index=idx,
    )
    out, overridden = validate_and_override_with_shioaji(
        df, shioaji_fetch=lambda a, b: _ref(date(2026, 5, 21), 21_020.0), threshold=50.0,
    )
    assert overridden == []
    assert out.loc[pd.Timestamp("2026-05-21"), "close"] == 21_000.0


# ── B2 wired into daily_updater.update() ────────────────────────────────────
def _seed_parquet(path: Path, last_date: str) -> None:
    dates = pd.bdate_range(end=last_date, periods=3)
    df = pd.DataFrame({"open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3],
                       "close": [33_100.0, 33_200.0, 33_300.0], "volume": [1, 2, 3]}, index=dates)
    df.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def test_update_refuses_to_save_on_alert(tmp_path):
    pq = tmp_path / "t.parquet"
    _seed_parquet(pq, "2026-04-07")
    new_bar = pd.DataFrame({"open": [33_300.0], "high": [33_500.0], "low": [33_250.0],
                            "close": [33_450.0], "volume": [1]},
                           index=pd.DatetimeIndex([pd.Timestamp("2026-04-08")], name="date"))
    notes: list[str] = []
    with (
        patch("src.data.daily_updater._detect_and_fill_gaps", return_value=(0, [])),
        patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 9)),
        patch("src.data.daily_updater._fetch_and_aggregate", return_value=new_bar),
    ):
        res = daily_updater.update(
            parquet_path=pq, notify_fn=notes.append,
            validate_fn=lambda day, close: (
                "alert", [CloseDiff(day, close, close - 400, 400.0, "taifex")]),
        )
    assert res["success"] is False and res["bars_added"] == 0
    # parquet must be UNCHANGED (bad bar not written)
    assert len(pd.read_parquet(pq)) == 3
    assert any(m.startswith("🔴") and "不存入" in m for m in notes)


def test_update_saves_on_warn(tmp_path):
    pq = tmp_path / "t.parquet"
    _seed_parquet(pq, "2026-04-07")
    new_bar = pd.DataFrame({"open": [33_300.0], "high": [33_500.0], "low": [33_250.0],
                            "close": [33_450.0], "volume": [1]},
                           index=pd.DatetimeIndex([pd.Timestamp("2026-04-08")], name="date"))
    notes: list[str] = []
    with (
        patch("src.data.daily_updater._detect_and_fill_gaps", return_value=(0, [])),
        patch("src.data.daily_updater._today_taipei", return_value=date(2026, 4, 9)),
        patch("src.data.daily_updater._fetch_and_aggregate", return_value=new_bar),
    ):
        res = daily_updater.update(
            parquet_path=pq, notify_fn=notes.append,
            validate_fn=lambda day, close: (
                "warn", [CloseDiff(day, close, close - 80, 80.0, "taifex")]),
        )
    assert res["success"] is True and res["bars_added"] == 1
    assert len(pd.read_parquet(pq)) == 4
    assert any(m.startswith("⚠️") for m in notes)


# ── update_all (fixes the missing import in the 14:25 path) ─────────────────
def test_update_all_iterates_products_and_skips_non_mxf(tmp_path):
    cfg = {"accounts": {
        "mxf_aggressive": {"product": "MXF"},
        "tsmc_2330": {"product": "2330"},
    }}
    cfg_path = tmp_path / "accounts.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    calls: list = []

    def fake_update(parquet_path=None, notify_fn=None, validate_fn=None):
        calls.append(parquet_path)
        return {"success": True, "bars_added": 1, "gaps_filled": 0,
                "latest_date": "2026-05-21", "error": None}

    with patch("src.data.daily_updater.update", side_effect=fake_update):
        results = daily_updater.update_all(config_path=str(cfg_path), enable_validation=False)

    by_product = {r["product"]: r for r in results}
    assert set(by_product) == {"MXF", "2330"}
    assert by_product["MXF"]["success"] is True and by_product["MXF"]["bars_added"] == 1
    assert by_product["2330"].get("skipped") is True
    assert len(calls) == 1  # only MXF went through update()
