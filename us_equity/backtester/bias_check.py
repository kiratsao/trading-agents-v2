"""Assertions to detect look-ahead bias in signal generation.

Two distinct checks are provided:

1. ``check_no_lookahead`` — verifies that every DataFrame in a data snapshot
   passed to a strategy contains **only** rows whose timestamps are ≤
   ``rebalance_date``.  Called by the backtester before every
   ``generate_signals`` invocation; raises ``ValueError`` on violation.

2. ``assert_no_lookahead`` / ``assert_signal_shifted`` — vectorbt-style
   assertions for signal series used in the walk-forward engine.
"""

from __future__ import annotations

import pandas as pd


def check_no_lookahead(
    data: dict[str, pd.DataFrame],
    rebalance_date: pd.Timestamp,
) -> None:
    """Verify that no DataFrame in *data* contains rows after *rebalance_date*.

    Called by the backtester immediately before passing the data snapshot to
    ``strategy.generate_signals``.  Because the backtester pre-slices all
    DataFrames to ``index <= rebalance_date``, this should never fail in
    normal operation; it is an invariant check that surfaces implementation
    bugs early.

    Parameters
    ----------
    data:
        Dict of ``symbol → OHLCV DataFrame`` that will be passed to the
        strategy.
    rebalance_date:
        The rebalance cutoff date (inclusive).  Any row whose timestamp is
        *strictly greater* than this date constitutes look-ahead bias.

    Raises
    ------
    ValueError
        If any DataFrame in *data* contains a row timestamped after
        *rebalance_date*.
    """
    # Normalise the cutoff to UTC so comparison works regardless of the
    # timezone stored in each DataFrame's index.
    if rebalance_date.tz is None:
        cutoff: pd.Timestamp = rebalance_date.tz_localize("UTC")
    else:
        cutoff = rebalance_date.tz_convert("UTC")

    for symbol, df in data.items():
        if df.empty:
            continue

        max_ts = df.index.max()

        # Normalise to UTC for comparison
        if max_ts.tz is None:
            max_ts = max_ts.tz_localize("UTC")
        else:
            max_ts = max_ts.tz_convert("UTC")

        if max_ts > cutoff:
            raise ValueError(
                f"Look-ahead bias detected for '{symbol}': "
                f"data extends to {max_ts.date()} but rebalance_date is "
                f"{cutoff.date()}.  Only rows with timestamp ≤ rebalance_date "
                f"may be passed to generate_signals."
            )


def assert_no_lookahead(signals: pd.Series, ohlcv: pd.DataFrame) -> None:
    """Assert that all entry signals use only data available at the signal bar.

    Checks that signals are shifted by at least 1 bar relative to the
    generating factor — raises AssertionError if look-ahead bias is detected.
    """
    raise NotImplementedError


def assert_signal_shifted(raw_signal: pd.Series, entry_signal: pd.Series) -> None:
    """Assert that entry_signal == raw_signal.shift(1)."""
    pd.testing.assert_series_equal(
        entry_signal,
        raw_signal.shift(1),
        check_names=False,
    )
