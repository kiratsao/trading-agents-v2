"""Daemon-independent daily parquet updater (CLI entry for a systemd timer).

Run by a Mon–Fri 14:30 systemd timer so the parquet stays fresh **even when the
trading daemon is down** (the daemon's in-process 14:25 job froze the parquet
for 2 days when it crash-looped). Exit codes are for systemd to act on:

  0 = ok / noop (parquet at expected latest)
  1 = partial  (some trading days could not be filled — investigate)
  2 = DataIntegrityError (gap too large / post-write verify failed — manual)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger("daily_updater_cli")


def _line_notifier():
    """Shared deduped notifier — same journal as the daemon, so an alert the
    daemon's 14:25 pass already sent is not re-sent by this 14:30 timer."""
    from src.notify.line import build_line_notifier

    return build_line_notifier()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Daemon-independent daily parquet updater")
    ap.add_argument("--parquet", default=None, help="Parquet path (default: MXF primary)")
    ap.add_argument("--no-validate", action="store_true", help="Skip B2 cross-validation")
    ap.add_argument(
        "--dry-run", action="store_true", help="Report freshness only; do not fetch or write"
    )
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from src.data.daily_updater import _PRIMARY_PARQUET, ensure_parquet_fresh
    from src.utils.freshness import DataIntegrityError, check_parquet_freshness

    notify = _line_notifier()
    parquet = args.parquet or _PRIMARY_PARQUET

    if args.dry_run:
        is_fresh, msg, expected = check_parquet_freshness(parquet)
        logger.info("dry-run: %s (expected=%s)", msg, expected)
        return 0 if is_fresh else 1

    validate_fn = None
    if not args.no_validate:
        from src.data.validation import validate_latest_bar

        validate_fn = validate_latest_bar

    try:
        result = ensure_parquet_fresh(
            parquet_path=parquet,
            notify_fn=notify,
            validate_fn=validate_fn,
        )
    except DataIntegrityError as exc:
        logger.error("🔴 DataIntegrityError: %s", exc)
        if notify:
            notify(f"🔴 daily_updater_cli: {exc}")
        return 2

    logger.info("daily_updater_cli: %s", result)
    if result["status"] == "partial":
        if notify:
            notify(f"⚠️ daily_updater_cli partial: 未補 {result['missing']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
