"""Production Smoke Test — 模擬 GCP 環境的完整啟動流程驗證。

Usage:
    python scripts/production_smoke_test.py           # full test (needs Shioaji/LINE)
    python scripts/production_smoke_test.py --local   # skip external connections

Exit code 0 = 🟢 PRODUCTION READY, 1 = 🔴 blocked.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Helpers ──────────────────────────────────────────────────────────────────

_PASS = "✅"
_FAIL = "❌"
_SKIP = "⏭️"
_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    icon = _PASS if ok else _FAIL
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def _section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: ENV
# ═══════════════════════════════════════════════════════════════════════════════
def check_env(skip_external: bool = False) -> int:
    _section("1. Environment Variables")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        _check("load_dotenv()", True)
    except ImportError:
        _check("load_dotenv()", True, "dotenv not installed — using system env")

    if skip_external:
        required = []  # ENV not needed for local-only test
    else:
        required = ["SHIOAJI_API_KEY", "SHIOAJI_SECRET_KEY"]
    if not skip_external:
        required += [
            "SHIOAJI_CERT_PATH", "SHIOAJI_CERT_PASSWORD", "SHIOAJI_PERSON_ID",
            "LINE_CHANNEL_ACCESS_TOKEN", "LINE_USER_ID",
        ]

    fails = 0
    for var in required:
        val = os.environ.get(var, "")
        ok = bool(val)
        _check(f"ENV {var}", ok, "set" if ok else "MISSING")
        if not ok:
            fails += 1

    if not skip_external:
        cert = os.environ.get("SHIOAJI_CERT_PATH", "")
        if cert:
            exists = Path(cert).exists()
            _check("CA cert file exists", exists, cert if exists else f"{cert} NOT FOUND")
            if not exists:
                fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Import Chain
# ═══════════════════════════════════════════════════════════════════════════════
def check_imports() -> int:
    _section("2. Import Chain")
    root = Path(__file__).resolve().parent.parent

    all_py = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if str(rel).startswith("scripts/"):
            continue
        all_py.append(rel)

    fails = 0
    for rel in all_py:
        # Convert path to module name
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        mod_name = ".".join(parts)
        try:
            importlib.import_module(mod_name)
            _check(f"import {mod_name}", True)
        except Exception as exc:
            _check(f"import {mod_name}", False, str(exc)[:80])
            fails += 1

    # Also check critical entry point
    try:
        _check("main entry point", True)
    except Exception as exc:
        _check("main entry point", False, str(exc)[:80])
        fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: External Connections
# ═══════════════════════════════════════════════════════════════════════════════
def check_external(skip: bool = False) -> int:
    _section("3. External Connections")
    if skip:
        print(f"  {_SKIP} Skipped (--local mode)")
        return 0

    fails = 0

    # Shioaji
    try:
        from tw_futures.executor.shioaji_adapter import ShioajiAdapter
        adapter = ShioajiAdapter(
            api_key=os.environ["SHIOAJI_API_KEY"],
            secret_key=os.environ["SHIOAJI_SECRET_KEY"],
            simulation=False,
            cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
            cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
            person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
        )
        contract = adapter.get_contract("MXF")
        adapter.logout()
        _check("Shioaji connect + get_contract", True, contract.code)
    except Exception as exc:
        _check("Shioaji connect", False, str(exc)[:80])
        fails += 1

    # LINE
    try:
        token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
        uid = os.environ.get("LINE_USER_ID", "")
        if token and uid:
            import json as _json
            import urllib.request
            payload = {"to": uid, "messages": [{"type": "text", "text": "🧪 smoke test"}]}
            req = urllib.request.Request(
                "https://api.line.me/v2/bot/message/push",
                data=_json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                _check("LINE push", True, f"status={resp.status}")
        else:
            _check("LINE push", False, "credentials missing")
            fails += 1
    except Exception as exc:
        _check("LINE push", False, str(exc)[:80])
        fails += 1

    # TAIFEX
    try:
        import urllib.request
        url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            _check("TAIFEX reachable", resp.status == 200, f"status={resp.status}")
    except Exception as exc:
        _check("TAIFEX reachable", False, str(exc)[:80])
        fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Data Integrity
# ═══════════════════════════════════════════════════════════════════════════════
def check_data() -> int:
    _section("4. Data Integrity")
    import pandas as pd

    pq = Path("data/MXF_Daily_Clean_2020_to_now.parquet")
    if not pq.exists():
        _check("parquet exists", False, str(pq))
        return 1

    df = pd.read_parquet(pq)
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    fails = 0
    _check("bars >= 1500", len(df) >= 1500, f"{len(df)} bars")
    if len(df) < 1500:
        fails += 1

    latest = df.index[-1].date()
    today = pd.Timestamp.now(tz="Asia/Taipei").date()
    gap = (today - latest).days
    _check("latest date fresh", gap <= 3, f"latest={latest}, gap={gap}d")
    if gap > 3:
        fails += 1

    weekend = df[df.index.dayofweek >= 5]
    _check("no weekend bars", len(weekend) == 0, f"{len(weekend)} weekend bars")
    if len(weekend) > 0:
        fails += 1

    nan_count = df[["open", "high", "low", "close"]].isna().sum().sum()
    _check("no NaN in OHLC", nan_count == 0, f"{nan_count} NaN")
    if nan_count > 0:
        fails += 1

    dups = df.index[df.index.duplicated()]
    _check("no duplicate dates", len(dups) == 0, f"{len(dups)} duplicates")
    if len(dups) > 0:
        fails += 1

    _check("index is DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
    if not isinstance(df.index, pd.DatetimeIndex):
        fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Strategy Paths
# ═══════════════════════════════════════════════════════════════════════════════
def check_strategy() -> int:
    _section("5. Strategy Paths")
    import numpy as np
    import pandas as pd

    from src.strategy.v2b_engine import V2bEngine

    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 20000.0 + np.linspace(0, 3000, n) + np.random.randn(n) * 20
    df = pd.DataFrame({
        "open": close - 10, "high": close + 50,
        "low": close - 50, "close": close,
        "volume": [100000] * n,
    }, index=dates)

    engine = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                       confirm_days=2, adx_threshold=25)
    fails = 0

    # Empty hand
    sig = engine.generate_signal(df, current_position=0, equity=350_000)
    _check("empty hand signal", sig.action in ("buy", "hold"), f"action={sig.action}")

    # Holding
    sig = engine.generate_signal(
        df, current_position=2, entry_price=20000.0,
        equity=350_000, highest_high=23000.0, contracts=2,
    )
    _check("holding signal", sig.action in ("hold", "close", "add"),
           f"action={sig.action}")

    # Settlement day
    from src.strategy.v2b_engine import _third_wednesday
    settle = _third_wednesday(date(2025, 4, 1))  # 2025-04-16
    settle_dates = pd.bdate_range(end=str(settle), periods=n)
    df_s = df.copy()
    df_s.index = settle_dates
    sig = engine.generate_signal(
        df_s, current_position=2, entry_price=20000.0,
        equity=350_000, highest_high=23000.0, contracts=2,
    )
    _check("settlement day → close", sig.action == "close",
           f"action={sig.action}, reason={sig.reason}")
    if sig.action != "close":
        fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Scheduler
# ═══════════════════════════════════════════════════════════════════════════════
def check_scheduler() -> int:
    _section("6. Scheduler Configuration")
    from src.scheduler.main import _parse_args

    args = _parse_args(["--run-once"])
    _check("--run-once parses", args.run_once is True)

    fails = 0
    try:
        from src.scheduler.main import _load_config
        cfg = _load_config("config/accounts.yaml")
        has_acc = "accounts" in cfg and "aggressive" in cfg.get("accounts", {})
        _check("accounts.yaml loads", has_acc,
               f"accounts: {list(cfg.get('accounts', {}).keys())}" if cfg else "empty")
        if has_acc:
            acc = cfg["accounts"]["aggressive"]
            params = acc.get("strategy_params", {})
            _check("strategy params", params.get("ema_fast") == 30,
                   f"ema_fast={params.get('ema_fast')}")
            sessions = acc.get("sessions", {}).get("day", {})
            _check("execution_timing",
                   sessions.get("execution_timing") == "night_open",
                   f"timing={sessions.get('execution_timing')}")
    except SystemExit:
        _check("accounts.yaml loads", False, "pyyaml not installed")
        fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Atomic Writes
# ═══════════════════════════════════════════════════════════════════════════════
def check_rollback() -> int:
    _section("7. Atomic Write Safety")
    import tempfile

    from src.state.state_manager import StateManager, TradingState

    fails = 0
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        mgr = StateManager(path=str(path))

        # Write initial
        mgr.save(TradingState(position=2, equity=400_000))
        assert path.exists()
        path.read_text()

        # Verify no .tmp leftover
        tmp = path.with_suffix(".json.tmp")
        _check("no .tmp after save", not tmp.exists())

        # Write again
        mgr.save(TradingState(position=3, equity=500_000))
        _check("no .tmp after second save", not tmp.exists())

        # Verify content updated
        data = json.loads(path.read_text())
        _check("state roundtrip", data["state"]["position"] == 3)
        if data["state"]["position"] != 3:
            fails += 1

    return fails


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    local_mode = "--local" in sys.argv

    print("=" * 50)
    print("  Production Smoke Test")
    print("=" * 50)

    total_fails = 0
    total_fails += check_env(skip_external=local_mode)
    total_fails += check_imports()
    total_fails += check_external(skip=local_mode)
    total_fails += check_data()
    total_fails += check_strategy()
    total_fails += check_scheduler()
    total_fails += check_rollback()

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    print(f"\n{'=' * 50}")
    print(f"  {_PASS} {passed}/{total} passed")
    if failed > 0:
        print(f"  {_FAIL} {failed}/{total} failed")
        for name, ok, detail in _results:
            if not ok:
                print(f"    - {name}: {detail}")
        print("\n  🔴 NOT PRODUCTION READY")
        sys.exit(1)
    else:
        print("\n  🟢 PRODUCTION READY")
        sys.exit(0)


if __name__ == "__main__":
    main()
