"""SQLite OHLCV cache with upsert support.

Schema (table: ohlcv):
  symbol  TEXT    NOT NULL
  date    TEXT    NOT NULL  -- YYYY-MM-DD (UTC date of the bar)
  open    REAL    NOT NULL
  high    REAL    NOT NULL
  low     REAL    NOT NULL
  close   REAL    NOT NULL
  volume  INTEGER NOT NULL
  PRIMARY KEY (symbol, date)

Separate from shared/database.py, which manages agent-state tables.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MARKET_DB_PATH = "data/db/market_data.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol  TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);
"""

_CREATE_IDX_SYMBOL = "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv (symbol);"
_CREATE_IDX_DATE = "CREATE INDEX IF NOT EXISTS idx_ohlcv_date   ON ohlcv (date);"

_UPSERT = """
INSERT OR REPLACE INTO ohlcv (symbol, date, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

_SELECT = """
SELECT date, open, high, low, close, volume
FROM   ohlcv
WHERE  symbol = ?
{date_filter}
ORDER BY date ASC;
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection, ensure the schema exists, and auto-commit/close."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_IDX_SYMBOL)
        conn.execute(_CREATE_IDX_DATE)
        conn.commit()
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _df_to_rows(symbol: str, df: pd.DataFrame) -> list[tuple]:
    """Convert a fetcher DataFrame to a list of (symbol, date, o, h, l, c, v) tuples."""
    rows = []
    for ts, row in df.iterrows():
        # ts is a pd.Timestamp (UTC); we store date as YYYY-MM-DD string
        date_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        rows.append(
            (
                symbol,
                date_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            )
        )
    return rows


def _rows_to_df(rows: list[sqlite3.Row], symbol: str) -> pd.DataFrame:
    """Convert SQLite rows back to a normalised OHLCV DataFrame."""
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "symbol"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("timestamp").drop(columns=["date"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(int)
    df["symbol"] = symbol
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_ohlcv(
    symbol: str,
    df: pd.DataFrame,
    db_path: str = MARKET_DB_PATH,
) -> int:
    """Upsert OHLCV rows for a symbol into the market data cache.

    Uses INSERT OR REPLACE so existing rows for the same (symbol, date)
    are overwritten with fresh data.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        df:     DataFrame produced by data/fetcher.py (UTC DatetimeIndex,
                columns: open, high, low, close, volume).
        db_path: Path to the SQLite database file.

    Returns:
        Number of rows written.

    Raises:
        ValueError: df is empty or missing required columns.
    """
    if df.empty:
        logger.warning("save_ohlcv: empty DataFrame for %s — nothing written", symbol)
        return 0

    missing_cols = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"save_ohlcv: DataFrame missing columns: {missing_cols}")

    rows = _df_to_rows(symbol, df)

    with _connect(db_path) as conn:
        conn.executemany(_UPSERT, rows)
        conn.commit()

    logger.info("save_ohlcv: upserted %d rows for %s into %s", len(rows), symbol, db_path)
    return len(rows)


def save_ohlcv_bulk(
    data: dict[str, pd.DataFrame],
    db_path: str = MARKET_DB_PATH,
) -> dict[str, int]:
    """Upsert OHLCV data for multiple symbols in a single DB connection.

    Args:
        data:    Dict mapping symbol → DataFrame (same schema as save_ohlcv).
        db_path: Path to the SQLite database file.

    Returns:
        Dict mapping symbol → number of rows written.
    """
    if not data:
        return {}

    result: dict[str, int] = {}

    with _connect(db_path) as conn:
        for symbol, df in data.items():
            if df.empty:
                logger.warning("save_ohlcv_bulk: empty DataFrame for %s — skipped", symbol)
                result[symbol] = 0
                continue

            missing_cols = [
                c for c in ("open", "high", "low", "close", "volume") if c not in df.columns
            ]
            if missing_cols:
                logger.error(
                    "save_ohlcv_bulk: %s missing columns %s — skipped", symbol, missing_cols
                )
                result[symbol] = 0
                continue

            rows = _df_to_rows(symbol, df)
            conn.executemany(_UPSERT, rows)
            result[symbol] = len(rows)

        conn.commit()

    total = sum(result.values())
    logger.info(
        "save_ohlcv_bulk: upserted %d total rows across %d symbols",
        total,
        len([v for v in result.values() if v > 0]),
    )
    return result


def load_ohlcv(
    symbol: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    db_path: str = MARKET_DB_PATH,
) -> pd.DataFrame:
    """Load cached OHLCV for a symbol, optionally filtered by date range.

    Args:
        symbol: Ticker symbol.
        start:  Optional start date, inclusive (str YYYY-MM-DD or Timestamp).
        end:    Optional end date, inclusive.
        db_path: Path to the SQLite database file.

    Returns:
        DataFrame with UTC DatetimeIndex and columns:
        open, high, low, close, volume, symbol.
        Returns an empty DataFrame (same schema) if symbol is not cached.
    """
    conditions: list[str] = []
    params: list[str | None] = [symbol]

    if start is not None:
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        conditions.append("date >= ?")
        params.append(start_str)

    if end is not None:
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        conditions.append("date <= ?")
        params.append(end_str)

    date_filter = ("AND " + " AND ".join(conditions)) if conditions else ""
    query = _SELECT.format(date_filter=date_filter)

    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    df = _rows_to_df(rows, symbol)
    logger.debug("load_ohlcv: %s → %d rows", symbol, len(df))
    return df


def list_symbols(db_path: str = MARKET_DB_PATH) -> list[str]:
    """Return all distinct symbol names present in the market data cache."""
    with _connect(db_path) as conn:
        cursor = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol;")
        return [row[0] for row in cursor.fetchall()]


def coverage(
    symbol: str, db_path: str = MARKET_DB_PATH
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (earliest_date, latest_date) for a symbol, or None if not cached."""
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT MIN(date), MAX(date) FROM ohlcv WHERE symbol = ?;",
            (symbol,),
        )
        row = cursor.fetchone()

    if row is None or row[0] is None:
        return None

    return (
        pd.Timestamp(row[0], tz="UTC"),
        pd.Timestamp(row[1], tz="UTC"),
    )


def delete_symbol(symbol: str, db_path: str = MARKET_DB_PATH) -> int:
    """Delete all cached data for a symbol. Returns number of rows deleted."""
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM ohlcv WHERE symbol = ?;", (symbol,))
        conn.commit()
        deleted = cursor.rowcount

    logger.info("delete_symbol: removed %d rows for %s", deleted, symbol)
    return deleted
