"""Mean-Reversion Intraday Strategy for TAIFEX futures (台指期當沖策略).

.. TODO:: This is a skeleton.  Implement once Shioaji minute-bar data is
          available via ``tw_futures.data.fetcher.TaifexFetcher.fetch_minute()``.

Strategy concept (to be implemented)
--------------------------------------
VWAP Mean-Reversion with opening-range filter:

1. **Opening Range** (first 30 minutes: 08:45–09:15 CST):
   - Define opening_high and opening_low.
   - No new positions during the opening range.

2. **VWAP Calculation** (cumulative intraday):
   - VWAP = cumulative(price × volume) / cumulative(volume)
   - Computed from market open (08:45 CST) on each bar.

3. **Entry Signals**:
   - Long : close < VWAP − 1×ATR(14, 1-min)  AND  close > opening_low
             (overextended below VWAP but above opening-range support)
   - Short: close > VWAP + 1×ATR(14, 1-min)  AND  close < opening_high
             (overextended above VWAP but below opening-range resistance)

4. **Exit Rules**:
   a. Price returns to VWAP → take profit (target = VWAP)
   b. ATR trailing stop (1.5×ATR)
   c. Hard stop: opening_low (for longs) / opening_high (for shorts)
   d. End-of-day forced exit at 13:20 CST (10 minutes before close)

5. **Position Sizing**:
   - Same fixed-risk formula as swing strategy (1% equity per trade for intraday).
   - Max 2 contracts per signal.

6. **Filters**:
   - Do not trade within 15 minutes of major news (placeholder).
   - Skip if intraday volume < 20-day average at the same time of day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntradaySignal:
    """Trade instruction from MeanReversionIntradayStrategy.

    Attributes
    ----------
    action :
        ``"buy"`` | ``"sell"`` | ``"close"`` | ``"hold"``
    contracts :
        Number of contracts (0 for hold).
    reason :
        Human-readable explanation.
    stop_loss :
        Price level for the initial stop (``None`` for hold).
    target :
        Price target for the take-profit (VWAP level).
    """

    action: str
    contracts: int = 0
    reason: str = ""
    stop_loss: float | None = None
    target: float | None = None


class MeanReversionIntradayStrategy:
    """VWAP Mean-Reversion intraday strategy for TAIFEX TX / MTX futures.

    .. TODO:: Implement ``generate_signal()`` once Shioaji minute-bar data
              is available.  Currently all methods raise ``NotImplementedError``.

    Parameters
    ----------
    product :
        ``"TX"`` (大台指, 200 TWD/pt) or ``"MTX"`` (小台指, 50 TWD/pt).
    atr_period :
        ATR look-back for intraday bars (default 14).
    vwap_entry_mult :
        ATR multiple away from VWAP needed to trigger entry (default 1.0).
    atr_stop_mult :
        ATR multiple for trailing stop (default 1.5).
    risk_per_trade :
        Fraction of equity to risk per intraday trade (default 0.01 = 1 %).
    max_contracts :
        Hard cap on contracts per signal (default 2).
    eod_exit_time :
        Time to force-exit all positions before close (default 13:20 CST).
    """

    def __init__(
        self,
        product: str = "TX",
        atr_period: int = 14,
        vwap_entry_mult: float = 1.0,
        atr_stop_mult: float = 1.5,
        risk_per_trade: float = 0.01,
        max_contracts: int = 2,
        eod_exit_time: str = "13:20",
    ) -> None:
        self.product = product.upper()
        self.atr_period = atr_period
        self.vwap_entry_mult = vwap_entry_mult
        self.atr_stop_mult = atr_stop_mult
        self.risk_per_trade = risk_per_trade
        self.max_contracts = max_contracts
        self.eod_exit_time = eod_exit_time

    # ------------------------------------------------------------------
    # Public interface (TODO: implement once minute data is available)
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int,
        entry_price: float | None,
        entry_time: pd.Timestamp | None,
        equity: float,
    ) -> IntradaySignal:
        """Generate an intraday trade signal from *data*.

        .. TODO:: Implement this method.

        Parameters
        ----------
        data :
            Intraday OHLCV DataFrame with 1-minute (or N-minute) bars.
            Index: ``pd.DatetimeIndex`` (Asia/Taipei), intraday bars only.
            Columns: ``open, high, low, close, volume``.
        current_position :
            Net contracts: positive = long, negative = short, 0 = flat.
        entry_price :
            Average entry price of current position.  ``None`` if flat.
        entry_time :
            Bar timestamp when current position was opened.  ``None`` if flat.
        equity :
            Current account equity in TWD.

        Returns
        -------
        IntradaySignal

        Raises
        ------
        NotImplementedError
            Until Shioaji minute-bar data is integrated.
        """
        raise NotImplementedError(
            "MeanReversionIntradayStrategy.generate_signal() is not yet implemented.  "
            "Waiting for Shioaji minute-bar data from TaifexFetcher.fetch_minute()."
        )

    def compute_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Compute cumulative intraday VWAP from *data*.

        .. TODO:: Implement.

        Parameters
        ----------
        data :
            Intraday bars with ``close`` and ``volume`` columns.

        Returns
        -------
        pd.Series
            VWAP at each bar.
        """
        raise NotImplementedError

    def compute_opening_range(
        self, data: pd.DataFrame, open_time: str = "08:45", range_end_time: str = "09:15"
    ) -> tuple[float, float]:
        """Return (opening_high, opening_low) for the first 30-minute range.

        .. TODO:: Implement.

        Parameters
        ----------
        data :
            Intraday bars with ``high`` and ``low`` columns.
            Index must be tz-aware (Asia/Taipei).
        open_time, range_end_time :
            Opening range window times (CST).

        Returns
        -------
        tuple[float, float]
            ``(opening_high, opening_low)``
        """
        raise NotImplementedError

    def is_end_of_day_exit(self, current_time: pd.Timestamp) -> bool:
        """Return True if *current_time* is at or past the EOD exit cutoff.

        .. TODO:: Implement.

        Parameters
        ----------
        current_time :
            Current bar timestamp (Asia/Taipei).
        """
        raise NotImplementedError
