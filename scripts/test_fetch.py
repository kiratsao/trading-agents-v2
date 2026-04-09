"""Quick smoke test for the data pipeline.

Fetches 30 days of daily OHLCV for AAPL and MSFT via Alpaca,
saves to SQLite, then reads back and prints the first 5 rows.

Usage:
    python scripts/test_fetch.py

Prerequisites:
    - .env file with ALPACA_API_KEY and ALPACA_SECRET_KEY
    - pip install -e ".[dev]"
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.cleaner import check_completeness, filter_zero_volume, validate_ohlcv
from core.data.store import MARKET_DB_PATH, coverage, load_ohlcv, save_ohlcv_bulk
from us_equity.data.fetcher import AlpacaFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_fetch")

SYMBOLS = ["AAPL", "MSFT"]
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)


def main() -> None:
    logger.info("=== Data pipeline smoke test ===")
    logger.info("Symbols : %s", SYMBOLS)
    logger.info("Range   : %s → %s", START_DATE, END_DATE)
    logger.info("DB path : %s", MARKET_DB_PATH)
    print()

    # ------------------------------------------------------------------
    # 1. Fetch from Alpaca
    # ------------------------------------------------------------------
    logger.info("Step 1 — Fetching from Alpaca ...")
    fetcher = AlpacaFetcher()
    raw_data = fetcher.fetch_ohlcv_bulk(SYMBOLS, start=START_DATE, end=END_DATE)

    for sym, df in raw_data.items():
        logger.info("  %-6s  %d bars fetched", sym, len(df))

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    logger.info("Step 2 — Cleaning ...")
    cleaned: dict = {}
    for sym, df in raw_data.items():
        df = filter_zero_volume(df)

        errors = validate_ohlcv(df)
        if errors:
            logger.warning("  %-6s  validation issues: %s", sym, errors)
        else:
            logger.info("  %-6s  validation OK", sym)

        report = check_completeness(df, symbol=sym)
        logger.info("  %-6s  %s", sym, report)

        cleaned[sym] = df

    # ------------------------------------------------------------------
    # 3. Save to SQLite
    # ------------------------------------------------------------------
    logger.info("Step 3 — Saving to SQLite (%s) ...", MARKET_DB_PATH)
    written = save_ohlcv_bulk(cleaned, db_path=MARKET_DB_PATH)
    for sym, n in written.items():
        logger.info("  %-6s  %d rows upserted", sym, n)

    # ------------------------------------------------------------------
    # 4. Read back and print
    # ------------------------------------------------------------------
    logger.info("Step 4 — Reading back from SQLite ...")
    print()

    for sym in SYMBOLS:
        cov = coverage(sym, db_path=MARKET_DB_PATH)
        logger.info(
            "  %-6s  DB coverage: %s → %s",
            sym,
            cov[0].date() if cov else "—",
            cov[1].date() if cov else "—",
        )

        df_loaded = load_ohlcv(sym, start=START_DATE, end=END_DATE, db_path=MARKET_DB_PATH)

        print(f"\n{'─' * 60}")
        print(f"  {sym}  —  first 5 rows (loaded from SQLite)")
        print(f"{'─' * 60}")
        print(df_loaded[["open", "high", "low", "close", "volume"]].head(5).to_string())

    print()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
