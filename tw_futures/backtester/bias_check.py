"""Look-ahead bias detection for TW futures strategies.

Two functions are provided:

1. ``check_signal_date`` — called by :class:`FuturesBacktester` before every
   ``generate_signal`` invocation to assert that the data slice passed to the
   strategy contains **only** rows ≤ *signal_date*.  Raises immediately on
   violation so bugs are caught during development, not masked by silent
   mis-pricing.

2. ``assert_donchian_shifted`` — vectorised sanity check that confirms the
   Donchian bands stored in an indicator DataFrame are shifted by at least
   one bar relative to the raw rolling window.  Useful in unit tests.
"""

from __future__ import annotations

import pandas as pd


def check_signal_date(
    data: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> None:
    """Assert that *data* contains no rows strictly after *signal_date*.

    The :class:`~tw_futures.backtester.backtester.FuturesBacktester` calls
    this immediately before passing a data slice to ``generate_signal``.
    Because the backtester pre-slices to ``data.iloc[:i+1]`` (i = today's
    bar index), this check should never fail in normal operation; it is an
    invariant that surfaces implementation bugs early.

    Parameters
    ----------
    data :
        OHLCV DataFrame that will be passed to ``generate_signal``.
    signal_date :
        The bar date for which the signal is being generated (inclusive
        upper bound).

    Raises
    ------
    ValueError
        If ``data.index.max()`` is strictly greater than *signal_date*
        (same-day timestamps are allowed regardless of time component).
    """
    if data.empty:
        return

    max_ts = data.index.max()

    # Normalise both to tz-aware UTC so comparison is valid regardless of the
    # timezone stored in each index.
    def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    max_utc = _to_utc(max_ts)
    cutoff_utc = _to_utc(pd.Timestamp(signal_date))

    # Compare at day granularity — intraday timestamps on the same date are fine.
    if max_utc.normalize() > cutoff_utc.normalize():
        raise ValueError(
            f"Look-ahead bias detected: data extends to {max_utc.date()} "
            f"but signal_date is {cutoff_utc.date()}.  "
            f"Only rows with date ≤ signal_date may be passed to generate_signal."
        )


def assert_donchian_shifted(
    indicators: pd.DataFrame,
    raw_high: pd.Series,
    raw_low: pd.Series,
    period: int = 20,
) -> None:
    """Assert that the Donchian bands in *indicators* are correctly shifted.

    Confirms ``indicators["upper"] == raw_high.rolling(period).max().shift(1)``
    and likewise for ``lower``.  Raises ``AssertionError`` with a descriptive
    message on first violation.

    Parameters
    ----------
    indicators :
        DataFrame containing ``"upper"`` and ``"lower"`` columns.
    raw_high, raw_low :
        The original ``high`` and ``low`` price Series (unshifted).
    period :
        Donchian look-back in bars (default 20).
    """
    expected_upper = raw_high.rolling(period).max().shift(1)
    expected_lower = raw_low.rolling(period).min().shift(1)

    try:
        pd.testing.assert_series_equal(
            indicators["upper"].dropna().reset_index(drop=True),
            expected_upper.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-6,
        )
    except AssertionError as exc:
        raise AssertionError(f"Donchian upper band is not shifted by 1 bar: {exc}") from exc

    try:
        pd.testing.assert_series_equal(
            indicators["lower"].dropna().reset_index(drop=True),
            expected_lower.dropna().reset_index(drop=True),
            check_names=False,
            rtol=1e-6,
        )
    except AssertionError as exc:
        raise AssertionError(f"Donchian lower band is not shifted by 1 bar: {exc}") from exc
