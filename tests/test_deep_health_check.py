"""Part C: deep_health_check rounds + auto-fix + iteration (offline)."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_health_check import (
    round1_data_integrity,
    round2_shioaji_cross,
    round3_state,
    round4_config,
    round5_scheduler,
    run_deep_health_check,
)
from tests.fakes import FakeBroker, write_synthetic_parquet


def _status(checks, name):
    return next(c.status for c in checks if c.name == name)


# ── Round 1 ─────────────────────────────────────────────────────────────────
def test_round1_clean_data_all_ok(tmp_path):
    df = write_synthetic_parquet(tmp_path / "p.parquet", n_bars=120)
    checks = round1_data_integrity(df)
    assert _status(checks, "連續性") == "ok"
    assert _status(checks, "重複日期") == "ok"
    assert _status(checks, "NaN") == "ok"
    assert _status(checks, "OHLC 邏輯") == "ok"
    assert _status(checks, "Volume") == "ok"


def test_round1_detects_nan_and_ohlc_violation(tmp_path):
    df = write_synthetic_parquet(tmp_path / "p.parquet", n_bars=120).copy()
    df.iloc[-1, df.columns.get_loc("close")] = float("nan")
    df.iloc[-2, df.columns.get_loc("high")] = 0.0  # high < low/open/close
    checks = round1_data_integrity(df)
    assert _status(checks, "NaN") == "alert"
    assert _status(checks, "OHLC 邏輯") == "alert"


def test_round1_flags_gap_and_low_volume(tmp_path):
    df = write_synthetic_parquet(tmp_path / "p.parquet", n_bars=60).copy()
    df = df.drop(df.index[30])                       # create a gap
    df.iloc[-1, df.columns.get_loc("volume")] = 100  # below floor
    checks = round1_data_integrity(df)
    assert _status(checks, "連續性") == "warn"
    assert _status(checks, "Volume") == "warn"


# ── Round 3 ─────────────────────────────────────────────────────────────────
def test_round3_clean_state_ok():
    state = {"position": 0, "contracts": 0, "equity": 880_000}
    checks, fixes, _ = round3_state(state, broker=None, today=date(2026, 5, 26))
    assert _status(checks, "contracts==position") == "ok"
    assert _status(checks, "pending_action") == "ok"
    assert fixes == []


def test_round3_clears_stale_pending():
    state = {"position": 0, "contracts": 0, "equity": 880_000,
             "pending_action": "buy", "pending_contracts": 3,
             "pending_signal_date": "2026-05-01", "pending_reason": "x"}
    checks, fixes, fixed = round3_state(state, broker=None, today=date(2026, 5, 26))
    assert any("跨天殘留" in f for f in fixes)
    assert fixed["pending_action"] is None


def test_round3_contracts_mismatch_alerts():
    state = {"position": 5, "contracts": 3, "entry_price": 20_000, "highest_high": 20_100}
    checks, _, _ = round3_state(state, broker=None, today=date(2026, 5, 26))
    assert _status(checks, "contracts==position") == "alert"


def test_round3_autofixes_equity_and_entry_against_broker():
    state = {"position": 8, "contracts": 8, "entry_price": 20_000.0,
             "highest_high": 20_100.0, "equity": 1_000_000.0}
    fb = FakeBroker(equity=2_000_000.0)
    fb.seed_position(8, 20_500.0)                    # broker avg 20500 vs state 20000
    checks, fixes, fixed = round3_state(state, broker=fb, today=date(2026, 5, 26))
    assert fixed["equity"] == 2_000_000.0            # equity drift fixed
    assert fixed["entry_price"] == 20_500.0          # entry drift fixed
    assert any("equity" in f for f in fixes)
    assert any("entry_price" in f for f in fixes)
    assert _status(checks, "口數") == "ok"           # 8 == 8


# ── Round 4 ─────────────────────────────────────────────────────────────────
def test_round4_config_ok_and_skips_investors():
    cfg = {"accounts": {"mxf": {
        "equity": 880_000, "max_contracts": 15,
        "scale_ladder": [{"equity": 350_000, "contracts": 2},
                         {"equity": 1_920_000, "contracts": 15}],
    }}}
    checks = round4_config(cfg, investors=None)
    assert _status(checks, "mxf ladder 覆蓋") == "ok"
    assert _status(checks, "mxf max_contracts") == "ok"
    assert _status(checks, "investors.yaml") == "skip"


def test_round4_flags_ladder_exceeding_max_and_bad_investors():
    cfg = {"accounts": {"mxf": {
        "equity": 880_000, "max_contracts": 10,
        "scale_ladder": [{"equity": 350_000, "contracts": 2},
                         {"equity": 1_920_000, "contracts": 15}],  # 15 > max 10
    }}}
    checks = round4_config(cfg, investors={"shares": {"a": 60, "b": 30}})  # 90 != 100
    assert _status(checks, "mxf max_contracts") == "warn"
    assert _status(checks, "investors 比例") == "alert"


# ── Round 5 ─────────────────────────────────────────────────────────────────
def test_round5_skips_without_logs(tmp_path):
    checks = round5_scheduler(log_dir=None)
    assert checks[0].status == "skip"


def test_round5_warns_on_stale_log(tmp_path):
    log = tmp_path / "old.log"
    log.write_text("x")
    import os
    old = (datetime.now() - timedelta(hours=48)).timestamp()
    os.utime(log, (old, old))
    checks = round5_scheduler(log_dir=tmp_path, now=datetime.now())
    assert checks[0].status == "warn"


# ── Round 2 ─────────────────────────────────────────────────────────────────
def test_round2_ok_when_shioaji_agrees(tmp_path):
    p = tmp_path / "p.parquet"
    df = write_synthetic_parquet(p, n_bars=30, end=date(2026, 5, 21))

    def agree(a, b):
        return df.copy()  # identical → no diff

    checks, _, fixes = round2_shioaji_cross(df, parquet_path=p, shioaji_fetch=agree)
    assert checks[0].status == "ok" and fixes == []


def _diverge_factory(df, *, delta=400, volume=None):
    def diverge(a, b):
        ref = df.copy()
        ref.iloc[-1, ref.columns.get_loc("close")] += delta
        if volume is not None:
            ref.iloc[-1, ref.columns.get_loc("volume")] = volume
        return ref
    return diverge


def test_round2_overrides_and_backs_up_with_fix(tmp_path):
    p = tmp_path / "p.parquet"
    df = write_synthetic_parquet(p, n_bars=30, end=date(2026, 5, 21))  # volume ~50k
    notes: list[str] = []
    checks, new_df, fixes = round2_shioaji_cross(
        df, parquet_path=p, shioaji_fetch=_diverge_factory(df, delta=400),
        do_fix=True, notify_fn=notes.append)
    assert checks[0].status == "warn"
    assert len(fixes) == 1
    assert list(tmp_path.glob("parquet_backup_*.parquet"))   # original backed up
    assert any(m.startswith("🔧") for m in notes)            # LINE per override


def test_deep_health_auto_fix_off_by_default(tmp_path):
    p = tmp_path / "p.parquet"
    df = write_synthetic_parquet(p, n_bars=30, end=date(2026, 5, 21))
    # do_fix not passed → defaults False → divergence reported but NOT written
    checks, _, fixes = round2_shioaji_cross(
        df, parquet_path=p, shioaji_fetch=_diverge_factory(df, delta=400))
    assert fixes == []
    assert not list(tmp_path.glob("parquet_backup_*.parquet"))
    assert checks[0].status == "warn"


def test_deep_health_low_volume_ref_skips_not_overrides(tmp_path):
    p = tmp_path / "p.parquet"
    df = write_synthetic_parquet(p, n_bars=30, end=date(2026, 5, 21))
    # big diff but ref volume below reliability cutoff = rolling-contract data →
    # cannot trust → ⏭️ skip, never override, even with --fix.
    checks, _, fixes = round2_shioaji_cross(
        df, parquet_path=p, shioaji_fetch=_diverge_factory(df, delta=400, volume=1_222),
        do_fix=True)
    assert fixes == []
    assert not list(tmp_path.glob("parquet_backup_*.parquet"))
    assert checks[0].status == "skip"            # unreliable ref → skip, not error/warn


def test_deep_health_fix_skips_small_diff(tmp_path):
    p = tmp_path / "p.parquet"
    df = write_synthetic_parquet(p, n_bars=30, end=date(2026, 5, 21))
    # 80pt diff > warn(50) but < override(200) → reported, not overwritten
    checks, _, fixes = round2_shioaji_cross(
        df, parquet_path=p, shioaji_fetch=_diverge_factory(df, delta=80), do_fix=True)
    assert fixes == []
    assert checks[0].status == "warn"


def test_round2_skips_when_fetch_dead(tmp_path):
    p = tmp_path / "p.parquet"
    df = write_synthetic_parquet(p, n_bars=30)

    def boom(a, b):
        raise RuntimeError("offline")

    checks, _, fixes = round2_shioaji_cross(df, parquet_path=p, shioaji_fetch=boom)
    assert checks[0].status == "skip" and fixes == []


# ── run_deep_health_check: light mode + iteration convergence ────────────────
def test_run_light_converges_after_autofix(tmp_path):
    from src.utils.freshness import expected_parquet_latest

    parquet = tmp_path / "MXF.parquet"
    # End the parquet at the expected latest so the new freshness gate is ✅
    # (this test targets Round-3 auto-fix convergence, not freshness).
    write_synthetic_parquet(parquet, n_bars=120, end=expected_parquet_latest())
    cfg = {"accounts": {"mxf_aggressive": {"product": "MXF"}}}
    cfg_path = tmp_path / "accounts.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # state with a stale pending → round3 auto-fix, then converge
    state_path = tmp_path / "state_mxf_aggressive.json"
    state_path.write_text(json.dumps({"state": {
        "position": 0, "contracts": 0, "equity": 880_000,
        "pending_action": "buy", "pending_signal_date": "2026-05-01",
    }}), encoding="utf-8")

    result = run_deep_health_check(
        parquet_path=parquet, config_path=cfg_path, state_dir=tmp_path,
        light=True, do_fix=True, max_iters=3,
    )
    assert any("跨天殘留" in f for f in result["fixes"])
    assert result["iterations"] == 2          # fixed on 1, stable on 2
    assert result["alert"] == 0
    # the fix persisted to disk
    saved = json.loads(state_path.read_text())["state"]
    assert saved["pending_action"] is None


def test_run_alert_triggers_notify(tmp_path):
    parquet = tmp_path / "MXF.parquet"
    df = write_synthetic_parquet(parquet, n_bars=120).copy()
    df.iloc[-1, df.columns.get_loc("close")] = float("nan")  # NaN → Round 1 alert
    df.to_parquet(parquet, index=True)
    cfg_path = tmp_path / "accounts.yaml"
    cfg_path.write_text(yaml.safe_dump({"accounts": {}}), encoding="utf-8")

    msgs: list[str] = []
    result = run_deep_health_check(
        parquet_path=parquet, config_path=cfg_path, state_dir=tmp_path,
        light=True, do_fix=False, notify_fn=msgs.append,
    )
    assert result["alert"] >= 1
    assert any(m.startswith("🔴") for m in msgs)
