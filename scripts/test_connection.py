"""Connection smoke test — verify Shioaji + LINE before going live.

Usage
-----
    python scripts/test_connection.py
    python scripts/test_connection.py --skip-line   # skip LINE notification
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _check_env() -> dict:
    """Check required env vars are present."""
    from dotenv import load_dotenv

    load_dotenv()

    required = {
        "SHIOAJI_API_KEY": os.getenv("SHIOAJI_API_KEY", ""),
        "SHIOAJI_SECRET_KEY": os.getenv("SHIOAJI_SECRET_KEY", ""),
    }
    optional = {
        "LINE_CHANNEL_ACCESS_TOKEN": os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""),
        "LINE_USER_ID": os.getenv("LINE_USER_ID", ""),
        "SHIOAJI_CERT_PATH": os.getenv("SHIOAJI_CERT_PATH", ""),
    }

    print("\n── Environment variables ──────────────────────────────")
    ok = True
    for k, v in required.items():
        status = "✓" if v else "✗ MISSING"
        print(f"  {k:<36} {status}")
        if not v:
            ok = False

    for k, v in optional.items():
        status = "✓ set" if v else "— not set (optional)"
        print(f"  {k:<36} {status}")

    if not ok:
        print("\n  → Fill in missing values in .env and retry.")
    return {**required, **optional}


def _test_shioaji(env: dict) -> bool:
    """Test Shioaji login, contract load, and MXF snapshot."""
    print("\n── Shioaji ────────────────────────────────────────────")
    try:
        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        simulation = os.getenv("SHIOAJI_SIMULATION", "true").lower() != "false"
        print(f"  mode: {'SIMULATION' if simulation else 'LIVE'}")

        adapter = ShioajiAdapter(
            api_key=env["SHIOAJI_API_KEY"],
            secret_key=env["SHIOAJI_SECRET_KEY"],
            simulation=simulation,
        )

        print("  login …", end="", flush=True)
        # Login happens lazily in _connect; trigger via get_snapshots
        snap = adapter.get_snapshots("MXF")
        print(" OK")

        close = snap.get("close") or snap.get("last_price")
        ts = snap.get("ts")
        if close:
            import pandas as pd

            if ts:
                ts_dt = pd.Timestamp(ts, unit="ns").tz_localize("Asia/Taipei")
                ts_str = ts_dt.strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = "unknown time"
            print(f"  MXF snapshot: {close:,.0f}  ({ts_str})")
        else:
            print("  MXF snapshot: no price (market may be closed)")

        contract = adapter.get_contract("MXF")
        print(f"  near-month contract: {contract.code if contract else 'N/A'}")

        adapter.logout()
        print("  ✓ Shioaji OK")
        return True

    except Exception as exc:
        print(f"\n  ✗ Shioaji FAILED: {exc}")
        logger.debug("Shioaji error", exc_info=True)
        return False


def _test_line(env: dict) -> bool:
    """Send a test push message via LINE Messaging API."""
    print("\n── LINE Messaging API ─────────────────────────────────")
    token = env.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    user_id = env.get("LINE_USER_ID", "")

    if not token or not user_id:
        print("  — skipped (LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID not set)")
        return True

    try:
        import json
        import urllib.request

        payload = {
            "to": user_id,
            "messages": [{"type": "text", "text": "✅ trading-agents-v2 連線測試成功"}],
        }
        req = urllib.request.Request(
            "https://api.line.me/v2/bot/message/push",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  ✓ LINE push sent (HTTP {resp.status})")
        return True

    except Exception as exc:
        print(f"  ✗ LINE FAILED: {exc}")
        return False


def _test_data() -> bool:
    """Verify canonical data file is accessible."""
    print("\n── Market data ────────────────────────────────────────")
    candidates = [
        ROOT / "data" / "MXF_Daily_Clean_2020_to_now.parquet",
        ROOT / "data" / "TXF_Daily_Real.parquet",
    ]
    for p in candidates:
        if p.exists():
            try:
                import pandas as pd

                df = pd.read_parquet(p)
                print(f"  ✓ {p.name}  {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
                return True
            except Exception as exc:
                print(f"  ✗ {p.name}: {exc}")
    print("  ✗ No canonical parquet found — run `python scripts/fetch_data.py` first")
    return False


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Connection smoke test")
    p.add_argument("--skip-line", action="store_true", help="Skip LINE notification test")
    args = p.parse_args(argv)

    print("=" * 55)
    print("  trading-agents-v2  connection smoke test")
    print("=" * 55)

    env = _check_env()

    results = {}
    results["env"] = all([env["SHIOAJI_API_KEY"], env["SHIOAJI_SECRET_KEY"]])
    results["shioaji"] = _test_shioaji(env) if results["env"] else False
    results["line"] = _test_line(env) if not args.skip_line else None
    results["data"] = _test_data()

    print("\n── Summary ────────────────────────────────────────────")
    all_ok = True
    for name, ok in results.items():
        if ok is None:
            icon = "—"
        elif ok:
            icon = "✓"
        else:
            icon = "✗"
            all_ok = False
        print(f"  {icon}  {name}")

    if all_ok:
        print("\n  All checks passed. Ready to start the scheduler.\n")
        sys.exit(0)
    else:
        print("\n  Some checks failed. Fix above errors before going live.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
