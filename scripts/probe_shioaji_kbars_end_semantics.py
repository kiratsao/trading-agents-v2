"""Probe Shioaji's kbars `end` parameter semantics.

Run this once before/after changing the daily_updater fetch range to
confirm Shioaji treats `end` as inclusive or exclusive. The 14:25 cron
data-contamination bug (parquet close off by up to ~1,300pt) was caused
by passing end=today, which pulled today's intra-session bar.

Usage:
    python scripts/probe_shioaji_kbars_end_semantics.py [YYYY-MM-DD]

If no date is given, defaults to the most recent completed trading day.

Output:
    Reports which dates are present in the returned bars when querying
    start == end == <probe_date>:
      * `<probe_date>` present  -> end is INCLUSIVE  (current code is correct
                                   with end=yesterday, no +1 day)
      * `<probe_date>` missing  -> end is EXCLUSIVE  (need end=yesterday+1
                                   AND filter > yesterday post-aggregate)
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd  # noqa: E402

from src.data.tw_holidays import last_trading_day_before  # noqa: E402


def _resolve_probe_date(argv: list[str]) -> date:
    if len(argv) > 1:
        return date.fromisoformat(argv[1])
    today = pd.Timestamp.now(tz="Asia/Taipei").date()
    return last_trading_day_before(today)


def main() -> int:
    probe = _resolve_probe_date(sys.argv)
    print(f"probe date (start == end): {probe}")

    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("ERROR: SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY not set", file=sys.stderr)
        return 2

    from tw_futures.executor.shioaji_adapter import ShioajiAdapter

    adapter = ShioajiAdapter(
        api_key=api_key,
        secret_key=secret_key,
        simulation=False,
        cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
        cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
        person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
    )
    try:
        contract = adapter.get_contract("MXF")
        kbars = adapter._api.kbars(
            contract,
            start=str(probe),
            end=str(probe),
            timeout=30_000,
        )
    finally:
        adapter.logout()

    if not kbars or len(kbars.ts) == 0:
        print("Shioaji returned NO bars.")
        print("  -> if you're confident the probe date IS a trading day,")
        print("     `end` is most likely EXCLUSIVE: use end=yesterday+1 and")
        print("     keep the post-aggregation `> yesterday` filter.")
        return 0

    ts = pd.to_datetime(kbars.ts, unit="ns", utc=True).tz_convert("Asia/Taipei")
    dates = sorted({t.date() for t in ts})
    print(f"distinct bar dates returned: {dates}")
    if probe in dates:
        print(f"  -> {probe} IS present.")
        print("     `end` is INCLUSIVE. Current daily_updater (end=yesterday) is correct.")
    else:
        print(f"  -> {probe} is NOT present.")
        print("     `end` is EXCLUSIVE. Revert to end=yesterday+1 AND keep the")
        print("     post-aggregation `> yesterday` filter to drop today's bar.")
    nxt = probe + timedelta(days=1)
    if nxt in dates:
        print(f"  -> WARNING: bars for {nxt} (one day past `end`) leaked in;")
        print("     the post-aggregation cutoff filter is essential here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
