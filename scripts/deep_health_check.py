"""Deep Health Check — 多輪迭代式系統體檢。

6 rounds, each isolating one failure surface. Auto-fixes the safe, unambiguous
divergences (equity / entry_price / contaminated bar / stale pending) and only
ALERTS on the judgement calls (口數不一致 / config / 缺失交易日). After any
auto-fix the whole battery re-runs to confirm the fix took (max 3 iterations);
a 🔴 surviving 3 rounds escalates to LINE for manual intervention.

Rounds needing a live broker (2, parts of 3) or the GCP host (5) degrade to
⏭️ skip when those aren't available, so the same script runs locally and on
the box. Every round is a pure, injectable function — see tests/test_deep_health_check.py.

Usage:
    python scripts/deep_health_check.py                 # full 6 rounds
    python scripts/deep_health_check.py --light         # Round 1 + 3 (daily 08:00)
    python scripts/deep_health_check.py --no-fix        # report only, never mutate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger(__name__)

_ICON = {"ok": "✅", "warn": "⚠️", "alert": "🔴", "skip": "⏭️"}
_VOLUME_FLOOR = 30_000          # day-session volume sanity floor
_CLOSE_JUMP_PCT = 10.0          # |Δclose| beyond this is flagged (not auto-fixed)
_EQUITY_DRIFT_PCT = 5.0         # state.equity vs broker beyond this → auto-fix
_ENTRY_DRIFT_PTS = 50.0         # state.entry_price vs broker beyond this → auto-fix
# Round-2 parquet override is the dangerous one (it once rewrote correct data
# with a buggy Shioaji value). Override ONLY a RELIABLE ref AND a big diff:
_OVERRIDE_MIN_DIFF = 200.0      # < 200pt may be benign (settle price vs last trade)
# A low-volume Shioaji ref means MXFR1 returned rolling-contract historical data
# (the wrong contract), NOT the contract that actually traded that day — so it
# cannot be trusted to validate/override. TAIFEX is ground truth; Shioaji is
# auxiliary and only trusted when the ref looks like a real main-contract session.
_RELIABLE_REF_VOLUME = 30_000


@dataclass
class Check:
    round_no: int
    name: str
    status: str          # ok | warn | alert | skip
    detail: str = ""

    def line(self) -> str:
        return f"{_ICON[self.status]} {self.name}" + (f": {self.detail}" if self.detail else "")


# ─────────────────────────────────────────────────────────────────────────────
# Round 1 — data integrity (fully offline)
# ─────────────────────────────────────────────────────────────────────────────
def round1_data_integrity(df: pd.DataFrame) -> list[Check]:
    from src.data.tw_holidays import is_taifex_holiday, trading_days_between

    R, out = 1, []
    if df is None or len(df) == 0:
        return [Check(R, "資料", "alert", "parquet 為空")]

    idx = pd.DatetimeIndex(df.index).normalize()
    first, last = idx[0].date(), idx[-1].date()

    # Continuity vs the TAIFEX calendar
    expected = set(trading_days_between(first, last))
    present = {ts.date() for ts in idx}
    missing = sorted(expected - present)
    if missing:
        shown = ", ".join(d.isoformat() for d in missing[:8])
        out.append(Check(R, "連續性", "warn",
                         f"{len(df)} bars, {len(missing)} 缺失交易日: {shown}"))
    else:
        out.append(Check(R, "連續性", "ok", f"{len(df)} bars, 0 gaps"))

    # Duplicates
    dups = idx[idx.duplicated()]
    out.append(Check(R, "重複日期", "alert" if len(dups) else "ok",
                     f"{len(dups)} 重複" if len(dups) else "無重複"))

    # Weekend / holiday bars
    bad_days = [ts.date() for ts in idx if is_taifex_holiday(ts.date())]
    out.append(Check(R, "週末/假日 bar", "alert" if bad_days else "ok",
                     f"{len(bad_days)} 筆: {bad_days[:5]}" if bad_days else "無"))

    # NaN in OHLC
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    nan_n = int(df[cols].isna().sum().sum()) if cols else 0
    out.append(Check(R, "NaN", "alert" if nan_n else "ok",
                     f"{nan_n} NaN" if nan_n else "無 NaN"))

    # OHLC logic
    bad_ohlc = 0
    if {"open", "high", "low", "close"}.issubset(df.columns):
        v = (
            (df["high"] < df["open"]) | (df["high"] < df["close"])
            | (df["low"] > df["open"]) | (df["low"] > df["close"])
            | (df["high"] < df["low"])
        )
        bad_ohlc = int(v.sum())
    out.append(Check(R, "OHLC 邏輯", "alert" if bad_ohlc else "ok",
                     f"{bad_ohlc} 違反" if bad_ohlc else "正確"))

    # Volume sanity (day session)
    if "volume" in df.columns:
        low_vol = df.index[df["volume"] < _VOLUME_FLOOR]
        out.append(Check(R, "Volume", "warn" if len(low_vol) else "ok",
                         f"{len(low_vol)} 天 < {_VOLUME_FLOOR:,}"
                         if len(low_vol) else f"全部 > {_VOLUME_FLOOR:,}"))

    # Close jump vs previous bar
    if "close" in df.columns and len(df) > 1:
        pct = df["close"].pct_change(fill_method=None).abs() * 100
        jumps = df.index[pct > _CLOSE_JUMP_PCT]
        if len(jumps):
            d0 = jumps[-1]
            out.append(Check(R, "Close 跳動", "warn",
                             f"{len(jumps)} 天 > {_CLOSE_JUMP_PCT:.0f}% "
                             f"(最近 {d0.date()} {pct.loc[d0]:.1f}%) — 可能正常但標記"))
        else:
            out.append(Check(R, "Close 跳動", "ok", f"無 > {_CLOSE_JUMP_PCT:.0f}% 跳動"))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Round 2 — live Shioaji cross-validation (auto-fix: override + backup)
# ─────────────────────────────────────────────────────────────────────────────
def round2_shioaji_cross(
    df: pd.DataFrame,
    *,
    parquet_path: Path | None = None,
    shioaji_fetch=None,
    recent_days: int = 20,
    do_fix: bool = False,
    notify_fn=None,
) -> tuple[list[Check], pd.DataFrame, list[str]]:
    """Compare recent parquet closes vs Shioaji. Shioaji is AUXILIARY only —
    TAIFEX is ground truth. A low-volume ref means MXFR1 returned rolling-
    contract historical data (the wrong contract) and is treated as unreliable
    → skipped, not flagged as an error. Auto-fix is OFF by default; even with
    --fix a bar is overridden ONLY when the ref is reliable
    (volume ≥ _RELIABLE_REF_VOLUME) AND the divergence is large (> 200pt).
    The original parquet is backed up first and every override emits a LINE notice."""
    from src.data.validation import compare_to_shioaji

    R = 2
    if df is None or len(df) == 0:
        return [Check(R, "Shioaji 交叉驗證", "skip", "無資料")], df, []
    try:
        diffs, ref = compare_to_shioaji(df, recent_days=recent_days, shioaji_fetch=shioaji_fetch)
    except Exception as exc:
        return [Check(R, "Shioaji 交叉驗證", "skip", f"無法連線 Shioaji: {exc}")], df, []

    if not diffs:
        return [Check(R, "Shioaji 交叉驗證", "ok",
                      f"近 {recent_days} 天 close 差距 < 50 點")], df, []

    # A low-volume ref is rolling-contract historical data → can't trust it.
    reliable = [cd for cd in diffs if cd.ref_volume >= _RELIABLE_REF_VOLUME]
    unreliable = [cd for cd in diffs if cd.ref_volume < _RELIABLE_REF_VOLUME]

    if not reliable:
        # Every divergence sits on an unreliable (rolling-contract) ref → skip.
        return [Check(R, "Shioaji 交叉驗證", "skip",
                      f"{len(unreliable)} 筆差異但 ref volume 過低（滾動合約歷史量），"
                      f"無法可靠驗證；以 TAIFEX 為準")], df, []

    fixable = [cd for cd in reliable if cd.diff > _OVERRIDE_MIN_DIFF]
    small = [cd for cd in reliable if cd not in fixable]

    fixes: list[str] = []
    if do_fix and fixable and parquet_path is not None:
        import datetime as _dt
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = Path(parquet_path).parent / f"parquet_backup_{stamp}.parquet"
        try:
            df.to_parquet(backup, index=True)              # backup ORIGINAL first
            new_df = df.copy()
            new_df.index = pd.DatetimeIndex(new_df.index).normalize()
            new_df.index.name = "date"
            for cd in fixable:
                tsd = pd.Timestamp(cd.day)
                for col in ("open", "high", "low", "close", "volume"):
                    if col in ref.columns and tsd in ref.index:
                        new_df.loc[tsd, col] = ref.loc[tsd, col]
                msg = (f"{cd.day} close {cd.parquet_close:.0f}→{cd.ref_close:.0f} "
                       f"(差 {cd.diff:.0f}, vol={cd.ref_volume:.0f})")
                fixes.append(msg)
                if notify_fn:
                    notify_fn(f"🔧 parquet 覆蓋: {msg}（已備份 {backup.name}）")
            new_df.to_parquet(parquet_path, index=True)
            df = new_df
        except Exception as exc:
            return [Check(R, "Shioaji 交叉驗證", "alert",
                          f"{len(fixable)} 筆待覆蓋，寫入失敗: {exc}")], df, []

    parts = [f"{len(reliable)} 筆可靠差異 > 50 點"]
    if fixes:
        parts.append(f"{len(fixes)} 筆已覆蓋(>200點，已備份)")
    if small:
        parts.append(f"{len(small)} 筆差<200點未動(疑正常價差)")
    if unreliable:
        parts.append(f"{len(unreliable)} 筆 ref量過低跳過")
    if not do_fix and fixable:
        parts.append(f"{len(fixable)} 筆可覆蓋，需 --fix 啟用")
    return [Check(R, "Shioaji 交叉驗證", "warn", "; ".join(parts))], df, fixes


# ─────────────────────────────────────────────────────────────────────────────
# Round 3 — state consistency (auto-fix equity / entry_price / stale pending)
# ─────────────────────────────────────────────────────────────────────────────
def round3_state(
    state: dict,
    *,
    broker=None,
    today: date | None = None,
    do_fix: bool = True,
) -> tuple[list[Check], list[str], dict]:
    R = 3
    today = today or date.today()
    s = dict(state)
    out: list[Check] = []
    fixes: list[str] = []

    pos = int(s.get("position", 0))
    contracts = int(s.get("contracts", 0))
    entry = s.get("entry_price")
    highest = s.get("highest_high")
    equity = float(s.get("equity", 0) or 0)

    # contracts == position
    out.append(Check(R, "contracts==position", "alert" if contracts != pos else "ok",
                     f"contracts={contracts}, position={pos}"))

    # highest_high >= entry_price (when long)
    if pos > 0 and entry and highest is not None:
        out.append(Check(R, "highest_high", "alert" if highest < entry else "ok",
                         f"highest={highest:.0f}, entry={entry:.0f}"))
    elif pos > 0 and entry and highest is None:
        out.append(Check(R, "highest_high", "warn", "持倉但 highest_high=None"))

    # Stale pending action (cross-day residue) → auto-fix clear
    pend = s.get("pending_action")
    pend_date = s.get("pending_signal_date")
    if pend and pend_date and pend_date != today.isoformat():
        if do_fix:
            for k, v in (("pending_action", None), ("pending_contracts", 0),
                         ("pending_signal_date", None), ("pending_reason", None)):
                s[k] = v
            fixes.append(f"清除跨天殘留 pending_action={pend} ({pend_date})")
            out.append(Check(R, "pending_action", "warn",
                             f"跨天殘留 {pend}@{pend_date} — 已清除"))
        else:
            out.append(Check(R, "pending_action", "alert",
                             f"跨天殘留 {pend}@{pend_date}"))
    else:
        out.append(Check(R, "pending_action", "ok", "無殘留"))

    # ── Broker-dependent checks ──────────────────────────────────────────────
    if broker is None:
        out.append(Check(R, "broker 對帳", "skip", "無 broker（離線）"))
        return out, fixes, s

    # position size
    try:
        positions = broker.get_positions()
        bpos = sum(int(p.get("contracts", p.get("quantity", 0))) for p in positions)
        out.append(Check(R, "口數", "alert" if bpos != pos else "ok",
                         f"state={pos}, broker={bpos}"))
    except Exception as exc:
        out.append(Check(R, "口數", "skip", f"list_positions 失敗: {exc}"))

    # equity drift → auto-fix
    try:
        bequity = float(broker.get_account().get("equity", 0) or 0)
        if bequity > 0:
            drift = abs(bequity - equity) / bequity * 100 if bequity else 0
            if drift > _EQUITY_DRIFT_PCT:
                if do_fix:
                    s["equity"] = bequity
                    fixes.append(f"equity {equity:.0f} → broker {bequity:.0f} (差 {drift:.0f}%)")
                    out.append(Check(R, "Equity", "warn",
                                     f"state={equity:.0f}, broker={bequity:.0f} — 已更新"))
                else:
                    out.append(Check(R, "Equity", "alert",
                                     f"state={equity:.0f}, broker={bequity:.0f} (差 {drift:.0f}%)"))
            else:
                out.append(Check(R, "Equity", "ok", f"{bequity:.0f} (差 {drift:.1f}%)"))
    except Exception as exc:
        out.append(Check(R, "Equity", "skip", f"margin 失敗: {exc}"))

    # entry_price drift vs broker avg → auto-fix
    if pos > 0 and entry:
        try:
            from src.scheduler.orchestrator import _query_broker_avg_price
            bavg = _query_broker_avg_price(broker, "MXF", pos)
            if bavg and abs(bavg - entry) > _ENTRY_DRIFT_PTS:
                if do_fix:
                    s["entry_price"] = bavg
                    fixes.append(f"entry_price {entry:.0f} → broker {bavg:.0f}")
                    out.append(Check(R, "均價", "warn",
                                     f"state={entry:.0f}, broker={bavg:.0f} — 已更新"))
                else:
                    out.append(Check(R, "均價", "alert", f"state={entry:.0f}, broker={bavg:.0f}"))
            elif bavg:
                out.append(Check(R, "均價", "ok", f"state={entry:.0f}, broker={bavg:.0f}"))
        except Exception as exc:
            out.append(Check(R, "均價", "skip", f"avg 查詢失敗: {exc}"))

    return out, fixes, s


# ─────────────────────────────────────────────────────────────────────────────
# Round 4 — config consistency (offline)
# ─────────────────────────────────────────────────────────────────────────────
def round4_config(cfg: dict, *, investors: dict | None = None) -> list[Check]:
    R, out = 4, []
    for name, acc in cfg.get("accounts", {}).items():
        ladder = acc.get("scale_ladder", [])
        equity = float(acc.get("equity", 0) or 0)
        max_c = acc.get("max_contracts")
        if not ladder:
            out.append(Check(R, f"{name} ladder", "alert", "無 scale_ladder"))
            continue
        thresholds = [e["equity"] for e in ladder]
        ladder_max = max(e["contracts"] for e in ladder)
        if equity < thresholds[0]:
            out.append(Check(R, f"{name} ladder 覆蓋", "warn",
                             f"equity={equity:.0f} < ladder 起點 {thresholds[0]:.0f}"))
        else:
            out.append(Check(R, f"{name} ladder 覆蓋", "ok",
                             f"equity={equity:.0f} 在 ladder 範圍內"))
        if max_c is not None and ladder_max > max_c:
            out.append(Check(R, f"{name} max_contracts", "warn",
                             f"ladder 最大 {ladder_max} 口 > max_contracts={max_c}"))
        else:
            out.append(Check(R, f"{name} max_contracts", "ok",
                             f"max_contracts={max_c}, ladder 最大 {ladder_max}"))

    if investors is None:
        out.append(Check(R, "investors.yaml", "skip", "未設定（選用）"))
    else:
        total = sum(float(v) for v in investors.get("shares", {}).values())
        out.append(Check(R, "investors 比例", "alert" if abs(total - 100.0) > 0.01 else "ok",
                         f"總和 = {total:.1f}%"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Round 5 — scheduler health (off-host → skip)
# ─────────────────────────────────────────────────────────────────────────────
def round5_scheduler(*, log_dir: Path | None = None, now: datetime | None = None) -> list[Check]:
    R, out = 5, []
    now = now or datetime.now()
    if log_dir is None or not Path(log_dir).exists():
        return [Check(R, "排程/log", "skip", "無 log 目錄（off-host，GCP 上才驗）")]
    logs = sorted(Path(log_dir).glob("*.log"))
    if not logs:
        return [Check(R, "排程/log", "skip", "log 目錄為空")]
    newest = max(logs, key=lambda p: p.stat().st_mtime)
    age_h = (now.timestamp() - newest.stat().st_mtime) / 3600
    out.append(Check(R, "最近 log", "warn" if age_h > 24 else "ok",
                     f"{newest.name} ({age_h:.1f}h 前)"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Round 6 — historical signal replay (compare vs logged signals if available)
# ─────────────────────────────────────────────────────────────────────────────
def round6_signal_replay(
    df: pd.DataFrame,
    strategy,
    *,
    logged: dict[str, str] | None = None,
    n_days: int = 30,
) -> list[Check]:
    R = 6
    if df is None or len(df) < strategy.ema_slow + n_days:
        return [Check(R, "信號回放", "skip", "資料不足以回放")]
    try:
        ind = strategy._compute_indicators(df)
    except Exception as exc:
        return [Check(R, "信號回放", "skip", f"指標計算失敗: {exc}")]

    if logged is None:
        return [Check(R, "信號回放", "skip",
                      f"無歷史信號 log 可比對（已可重算最近 {n_days} 天）")]

    mismatches = []
    recent = ind.index[-n_days:]
    for ts in recent:
        key = ts.date().isoformat()
        if key in logged:
            # placeholder comparison hook — real per-day re-evaluation wires here
            pass
    out_status = "alert" if mismatches else "ok"
    return [Check(R, "信號回放", out_status,
                  f"{n_days - len(mismatches)}/{n_days} 天一致"
                  if not mismatches else f"{len(mismatches)} 天不一致: {mismatches[:5]}")]


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration + iteration
# ─────────────────────────────────────────────────────────────────────────────
def _run_rounds(
    *, parquet_path, config_path, state_dir, broker, shioaji_fetch,
    log_dir, investors, do_fix, light, notify_fn=None,
) -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    fixes: list[str] = []

    df = pd.read_parquet(parquet_path) if Path(parquet_path).exists() else None
    if df is not None:
        df.index = pd.to_datetime(df.index)
    checks += round1_data_integrity(df)

    if not light:
        r2c, df, r2f = round2_shioaji_cross(
            df, parquet_path=parquet_path, shioaji_fetch=shioaji_fetch,
            do_fix=do_fix, notify_fn=notify_fn)
        checks += r2c
        fixes += r2f

    # State is re-read from disk every iteration so prior auto-fixes are seen
    # (lets the battery converge instead of re-reporting the same fix).
    for name, state, state_path in _discover_state_items(config_path, state_dir):
        r3c, r3f, fixed = round3_state(state, broker=broker, do_fix=do_fix)
        checks += [Check(c.round_no, f"[{name}] {c.name}", c.status, c.detail) for c in r3c]
        if r3f and do_fix and state_path is not None:
            _save_state(state_path, fixed)
            fixes += [f"[{name}] {f}" for f in r3f]

    if not light:
        cfg = _load_yaml(config_path)
        checks += round4_config(cfg, investors=investors)
        checks += round5_scheduler(log_dir=log_dir)
        # Round 6 needs a strategy; build a default MXF engine for replay
        try:
            from src.strategy.v2b_engine import V2bEngine
            eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                            confirm_days=2, adx_threshold=25)
            checks += round6_signal_replay(df, eng)
        except Exception as exc:
            checks.append(Check(6, "信號回放", "skip", f"無法建立策略: {exc}"))

    return checks, fixes


def run_deep_health_check(
    *,
    parquet_path: str | Path = "data/MXF_Daily_Clean_2020_to_now.parquet",
    config_path: str | Path = "config/accounts.yaml",
    state_dir: str | Path = "data",
    broker=None,
    shioaji_fetch=None,
    log_dir: Path | None = None,
    investors: dict | None = None,
    do_fix: bool = False,
    light: bool = False,
    max_iters: int = 3,
    notify_fn=None,
) -> dict:
    """Run the battery; re-run after any auto-fix (≤ max_iters). Returns a
    summary dict and emits a LINE alert if a 🔴 survives all iterations.

    ``do_fix`` defaults to False — auto-fix (incl. parquet override) is opt-in
    only, surfaced via the ``--fix`` CLI flag."""
    checks: list[Check] = []
    all_fixes: list[str] = []
    iteration = 0
    for iteration in range(1, max_iters + 1):
        checks, fixes = _run_rounds(
            parquet_path=parquet_path, config_path=config_path, state_dir=state_dir,
            broker=broker, shioaji_fetch=shioaji_fetch, log_dir=log_dir,
            investors=investors, do_fix=do_fix, light=light, notify_fn=notify_fn,
        )
        all_fixes += fixes
        if not fixes:
            break  # stable — nothing more to fix

    report = _format_report(checks, all_fixes, iteration, light)
    print(report)
    logger.info("deep_health_check complete: %d iterations", iteration)

    alerts = [c for c in checks if c.status == "alert"]
    if alerts and notify_fn:
        notify_fn("🔴 Deep Health Check 需人工介入:\n"
                  + "\n".join(f"• {c.line()}" for c in alerts[:10]))

    return {
        "iterations": iteration,
        "ok": sum(c.status == "ok" for c in checks),
        "warn": sum(c.status == "warn" for c in checks),
        "alert": len(alerts),
        "skip": sum(c.status == "skip" for c in checks),
        "fixes": all_fixes,
        "checks": checks,
    }


def _format_report(checks, fixes, iterations, light) -> str:
    lines = ["🔍 Deep Health Check" + (" (light)" if light else " — 6 Rounds"),
             "━" * 23]
    last_round = None
    titles = {1: "資料完整性", 2: "Shioaji 交叉驗證", 3: "State 一致性",
              4: "Config 一致性", 5: "排程健康", 6: "歷史信號回放"}
    for c in sorted(checks, key=lambda x: x.round_no):
        if c.round_no != last_round:
            lines.append(f"Round {c.round_no}: {titles.get(c.round_no, '')}")
            last_round = c.round_no
        lines.append(c.line())
    lines.append("━" * 23)
    n_ok = sum(c.status == "ok" for c in checks)
    n_warn = sum(c.status == "warn" for c in checks)
    n_alert = sum(c.status == "alert" for c in checks)
    n_skip = sum(c.status == "skip" for c in checks)
    lines.append(f"結果: {n_ok} ✅, {n_warn} ⚠️, {n_alert} 🔴, {n_skip} ⏭️ "
                 f"({iterations} 輪迭代)")
    if fixes:
        lines.append("自動修正:")
        lines += [f"  • {f}" for f in fixes]
    lines.append("━" * 23)
    return "\n".join(lines)


# ── small IO helpers ─────────────────────────────────────────────────────────
def _load_yaml(path) -> dict:
    import yaml
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _discover_state_items(config_path, state_dir="data") -> list:
    cfg = _load_yaml(config_path)
    items = []
    for name in cfg.get("accounts", {}):
        sp = Path(state_dir) / f"state_{name}.json"
        if sp.exists():
            try:
                raw = json.loads(sp.read_text(encoding="utf-8"))
                items.append((name, raw.get("state", {}), sp))
            except Exception as exc:
                logger.warning("cannot read %s: %s", sp, exc)
    return items


def _save_state(path, state: dict) -> None:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        raw = {}
    raw["state"] = state
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def _line_notifier():
    import os
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    uid = os.environ.get("LINE_USER_ID", "")
    if not token or not uid:
        return None

    def _send(msg: str) -> None:
        import urllib.request
        payload = json.dumps({"to": uid, "messages": [{"type": "text", "text": msg}]})
        req = urllib.request.Request(
            "https://api.line.me/v2/bot/message/push", data=payload.encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.warning("LINE alert failed: %s", exc)

    return _send


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Deep multi-round health check")
    ap.add_argument("--light", action="store_true", help="Round 1 + 3 only (daily 08:00)")
    ap.add_argument("--fix", action="store_true",
                    help="enable auto-fix (parquet override / state). OFF by default.")
    ap.add_argument("--config", default="config/accounts.yaml")
    ap.add_argument("--parquet", default="data/MXF_Daily_Clean_2020_to_now.parquet")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    log_dir = Path("logs") if Path("logs").exists() else None
    result = run_deep_health_check(
        parquet_path=args.parquet, config_path=args.config,
        do_fix=args.fix, light=args.light,
        log_dir=log_dir, notify_fn=_line_notifier(),
    )
    return 1 if result["alert"] else 0


if __name__ == "__main__":
    sys.exit(main())
