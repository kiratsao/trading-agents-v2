"""Data cleaning — missing values, zero-volume filtering, and completeness checks.

All functions operate on DataFrames produced by data/fetcher.py:
  - UTC DatetimeIndex named "timestamp"
  - columns: open, high, low, close, volume, symbol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLS = ["open", "high", "low", "close"]
_REQUIRED_COLS = [*_PRICE_COLS, "volume"]


# ---------------------------------------------------------------------------
# Fill missing values
# ---------------------------------------------------------------------------


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing OHLCV values.

    Args:
        df:     OHLCV DataFrame (may contain NaN rows introduced by reindexing).
        method: "ffill" — carry the last known close forward for OHLC;
                          volume is filled with 0 (market was closed / no trades).
                "drop"  — remove any row with NaN in required columns.

    Returns:
        Cleaned DataFrame with the same index.

    Raises:
        ValueError: Unknown method.
    """
    if method not in ("ffill", "drop"):
        raise ValueError(f"Unknown fill method '{method}'. Use 'ffill' or 'drop'.")

    df = df.copy()

    if method == "drop":
        before = len(df)
        df = df.dropna(subset=_REQUIRED_COLS)
        dropped = before - len(df)
        if dropped:
            logger.info("fill_missing(drop): removed %d rows with NaN", dropped)
        return df

    # ffill: price columns carry forward, volume defaults to 0
    df[_PRICE_COLS] = df[_PRICE_COLS].ffill()
    df["volume"] = df["volume"].fillna(0).astype(int)

    remaining_na = df[_REQUIRED_COLS].isna().sum().sum()
    if remaining_na:
        logger.warning(
            "fill_missing(ffill): %d NaN values remain after forward-fill "
            "(likely leading NaNs at the start of the series)",
            remaining_na,
        )

    return df


# ---------------------------------------------------------------------------
# Zero-volume filter
# ---------------------------------------------------------------------------


def filter_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trading days with zero volume.

    Zero-volume days typically represent non-trading days that slipped through
    (e.g. holidays, early market closures, or data errors).

    Returns:
        DataFrame with zero-volume rows removed.
    """
    before = len(df)
    df = df[df["volume"] > 0].copy()
    removed = before - len(df)
    if removed:
        logger.info("filter_zero_volume: removed %d zero-volume rows", removed)
    return df


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------


@dataclass
class CompletenessReport:
    symbol: str
    expected_days: int
    actual_days: int
    missing_dates: list[pd.Timestamp] = field(default_factory=list)

    @property
    def missing_count(self) -> int:
        return len(self.missing_dates)

    @property
    def coverage_pct(self) -> float:
        if self.expected_days == 0:
            return 100.0
        return round(100 * self.actual_days / self.expected_days, 2)

    def __str__(self) -> str:
        return (
            f"[{self.symbol}] {self.actual_days}/{self.expected_days} days "
            f"({self.coverage_pct}%), missing {self.missing_count} dates"
        )


def check_completeness(
    df: pd.DataFrame,
    symbol: str | None = None,
) -> CompletenessReport:
    """Report trading days present in the date range but missing from the DataFrame.

    Uses pandas business-day calendar as a proxy for US trading days.
    This does NOT account for US market holidays (e.g. Thanksgiving, Christmas).
    For precise holiday-aware checks, replace pd.bdate_range with
    pandas_market_calendars (NYSE calendar).

    Args:
        df:     OHLCV DataFrame with UTC DatetimeIndex.
        symbol: Symbol name for the report. If None, reads df["symbol"].iloc[0].

    Returns:
        CompletenessReport with missing dates listed.
    """
    if df.empty:
        sym = symbol or "UNKNOWN"
        logger.warning("check_completeness: empty DataFrame for %s", sym)
        return CompletenessReport(symbol=sym, expected_days=0, actual_days=0)

    sym = symbol or str(df["symbol"].iloc[0])

    # Normalise index to date-only for comparison
    actual_dates = pd.DatetimeIndex(
        df.index.normalize().tz_localize(None) if df.index.tz else df.index.normalize()
    )
    start_date = actual_dates.min()
    end_date = actual_dates.max()

    expected_dates = pd.bdate_range(start=start_date, end=end_date)
    missing = expected_dates.difference(actual_dates)

    report = CompletenessReport(
        symbol=sym,
        expected_days=len(expected_dates),
        actual_days=len(actual_dates),
        missing_dates=missing.tolist(),
    )

    if report.missing_count:
        logger.warning(
            "check_completeness: %s — %d missing business days (%.1f%% coverage)",
            sym,
            report.missing_count,
            report.coverage_pct,
        )
    else:
        logger.debug("check_completeness: %s — complete (%d days)", sym, report.actual_days)

    return report


def check_completeness_bulk(
    data: dict[str, pd.DataFrame],
) -> dict[str, CompletenessReport]:
    """Run check_completeness for each symbol in a bulk result dict."""
    return {sym: check_completeness(df, symbol=sym) for sym, df in data.items()}


# ---------------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------------


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages. Empty list = clean data.

    Checks:
      - No NaN in required columns
      - high >= low, high >= open, high >= close
      - low <= open, low <= close
      - volume >= 0
      - Index is monotonically increasing
      - No duplicate timestamps
    """
    errors: list[str] = []

    if df.empty:
        return errors

    # NaN check
    na_counts = df[_REQUIRED_COLS].isna().sum()
    for col, count in na_counts.items():
        if count > 0:
            errors.append(f"Column '{col}' has {count} NaN value(s)")

    # OHLC consistency
    bad_high_low = (df["high"] < df["low"]).sum()
    if bad_high_low:
        errors.append(f"high < low in {bad_high_low} row(s)")

    bad_high_open = (df["high"] < df["open"]).sum()
    if bad_high_open:
        errors.append(f"high < open in {bad_high_open} row(s)")

    bad_high_close = (df["high"] < df["close"]).sum()
    if bad_high_close:
        errors.append(f"high < close in {bad_high_close} row(s)")

    bad_low_open = (df["low"] > df["open"]).sum()
    if bad_low_open:
        errors.append(f"low > open in {bad_low_open} row(s)")

    bad_low_close = (df["low"] > df["close"]).sum()
    if bad_low_close:
        errors.append(f"low > close in {bad_low_close} row(s)")

    # Volume
    bad_volume = (df["volume"] < 0).sum()
    if bad_volume:
        errors.append(f"Negative volume in {bad_volume} row(s)")

    # Index ordering
    if not df.index.is_monotonic_increasing:
        errors.append("Index is not monotonically increasing")

    # Duplicate timestamps
    dupes = df.index.duplicated().sum()
    if dupes:
        errors.append(f"{dupes} duplicate timestamp(s) in index")

    if errors:
        sym = df["symbol"].iloc[0] if "symbol" in df.columns else "?"
        logger.warning("validate_ohlcv [%s]: %d issue(s) found", sym, len(errors))

    return errors
