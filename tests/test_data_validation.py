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
    compare_to_shioaji,
    validate_latest_bar,
)


def _ref(day: date, close: float, volume: float = 90_000) -> pd.DataFrame:
    # volume defaults well above the reliability floor so the oracle counts;
    # pass a low volume to exercise the unreliable-ref skip path.
    return pd.DataFrame(
        [{"open": close, "high": close + 5, "low": close - 5, "close": close, "volume": volume}],
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


def test_validate_latest_bar_alerts_above_500():
    d = date(2026, 5, 21)
    level, diffs = validate_latest_bar(
        d, 20_000.0,
        shioaji_fetch=lambda a, b: _ref(d, 20_600.0),  # 600pt > max(500, 1.5%=300) → alert
        taifex_fetch=lambda a, b: _ref(d, 20_000.0),
    )
    assert level == "alert"
    assert any(cd.diff > 500 for cd in diffs)


def test_validate_latest_bar_warns_not_alerts_on_settlement_gap():
    # 5/26 regression: high-vol ref but 491pt day-close vs settlement gap →
    # warn (still written), NOT alert. close=44,386 vs 43,895 (vol=131,486).
    d = date(2026, 5, 26)
    level, _ = validate_latest_bar(
        d, 44_386.0,
        shioaji_fetch=lambda a, b: _ref(d, 44_386.0, volume=131_486),  # agrees
        taifex_fetch=lambda a, b: _ref(d, 43_895.0, volume=131_486),   # 491pt, reliable
    )
    assert level == "warn"          # 491 < max(500, 1.5%*44386≈666) → not blocked


def test_validate_latest_bar_skips_low_volume_ref():
    # The 5/27 regression: a ref with vol≈0 (rolling-contract / empty oracle)
    # must NOT block a legit update, even with a large close diff.
    d = date(2026, 5, 27)
    level, diffs = validate_latest_bar(
        d, 44_448.0,
        shioaji_fetch=lambda a, b: _ref(d, 44_448.0, volume=99_000),  # agrees, reliable
        taifex_fetch=lambda a, b: _ref(d, 44_794.0, volume=0),        # vol=0 → skipped
    )
    assert level == "ok" and diffs == []


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


# ── compare_to_shioaji (report-only — never mutates) ────────────────────────
def test_compare_flags_divergent_bar_without_mutating():
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in
                            (date(2026, 5, 19), date(2026, 5, 20), date(2026, 5, 21))], name="date")
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                       "close": [20_000.0, 20_500.0, 21_000.0], "volume": [1, 2, 3]}, index=idx)

    def shioaji(a, b):
        rows = {date(2026, 5, 19): (20_000.0, 90_000), date(2026, 5, 20): (20_900.0, 95_000),
                date(2026, 5, 21): (21_000.0, 90_000)}
        return pd.DataFrame(
            [{"open": c, "high": c, "low": c, "close": c, "volume": v} for c, v in rows.values()],
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in rows], name="date"),
        )

    diffs, ref = compare_to_shioaji(df, recent_days=20, shioaji_fetch=shioaji, threshold=50.0)
    assert [cd.day for cd in diffs] == [date(2026, 5, 20)]
    assert diffs[0].ref_close == 20_900.0
    assert diffs[0].ref_volume == 95_000          # volume carried for safety valve
    assert df.loc[pd.Timestamp("2026-05-20"), "close"] == 20_500.0   # df NOT mutated


def test_compare_no_diff_when_agree():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-21")], name="date")
    df = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [21_000.0], "volume": [1]}, index=idx,
    )
    diffs, _ = compare_to_shioaji(
        df, shioaji_fetch=lambda a, b: _ref(date(2026, 5, 21), 21_020.0), threshold=50.0,
    )
    assert diffs == []


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


# ── unification: validation routes through the single fetcher ───────────────
def test_validation_default_fetch_uses_shioaji_fetcher(monkeypatch):
    import src.data.shioaji_fetcher as sf
    from src.data.validation import _default_shioaji_fetch

    sentinel = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-05-15")]))
    monkeypatch.setattr(sf, "fetch_via_env", lambda s, e, product="MXF": sentinel)
    out = _default_shioaji_fetch(date(2026, 5, 15), date(2026, 5, 15))
    assert out is sentinel


# ── init_data holiday filtering no longer crashes ──────────────────────────
def test_init_data_remove_holidays_no_crash():
    from scripts.init_data import _remove_taifex_holidays

    idx = pd.DatetimeIndex([pd.Timestamp("2026-04-30"), pd.Timestamp("2026-05-01"),
                            pd.Timestamp("2026-05-04")], name="date")  # 5/1 = 勞動節
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                       "close": [1.0, 2, 3], "volume": [1, 2, 3]}, index=idx)
    out = _remove_taifex_holidays(df)
    assert pd.Timestamp("2026-05-01") not in out.index   # holiday dropped
    assert len(out) == 2
