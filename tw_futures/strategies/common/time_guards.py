"""Time-based guards for TAIFEX futures trading.

Centralises all calendar and session-time rules so every strategy and the
backtester share one consistent implementation.

Rules encoded
-------------
* **Friday close** (13:15): force-flatten before the weekend gap.
* **Settlement day** (3rd Wednesday of each month): force-flatten before TAIFEX
  final settlement to avoid holding through the mark-to-settlement move.
* **No-trade zones**:
  - Opening 30 min (08:45–09:15): high volatility, many false breakouts.
  - Closing 15 min (13:30–13:45): thin liquidity, wide spreads.
* **EOD force-close** (13:30 for intraday strategies): prevents overnight
  positions from an intraday-only strategy.

Usage
-----
>>> from tw_futures.strategies.common.time_guards import TimeGuards
>>> tg = TimeGuards()
>>> force, reason = tg.should_force_close(current_time, current_date)
>>> if force:
...     # emit close signal

TODO
----
A complete Taiwan Exchange holiday calendar.  Currently only the weekend rule
(Mon–Fri) is applied.  Add a ``TAIFEX_HOLIDAYS: set[date]`` constant covering
each year's official closures once confirmed by the exchange.
"""

from __future__ import annotations

from datetime import date, time

import pandas as pd

# ── Session boundary constants ─────────────────────────────────────────────────

_DAY_OPEN = time(8, 45)
_OPENING_RANGE_END = time(9, 15)  # no-trade zone ends here
_EOD_FORCE_CLOSE = time(13, 30)  # intraday EOD exit
_SWING_FORCE_CLOSE = time(13, 15)  # Friday / settlement exit (swing)
_DAY_CLOSE = time(13, 45)


class TimeGuards:
    """Stateless helper that answers "should I act now?" for TAIFEX session rules.

    Parameters
    ----------
    eod_force_close :
        Time at which intraday strategies must flatten regardless of signals.
        Default 13:30 (gives 15 min before TAIFEX close for order execution).
    swing_force_close :
        Time at which Friday / settlement-day swing positions must be closed.
        Default 13:15.
    """

    def __init__(
        self,
        eod_force_close: time = _EOD_FORCE_CLOSE,
        swing_force_close: time = _SWING_FORCE_CLOSE,
    ) -> None:
        self._eod_close = eod_force_close
        self._swing_close = swing_force_close

    # ------------------------------------------------------------------
    # Individual predicates
    # ------------------------------------------------------------------

    def is_friday_close(self, current_time: pd.Timestamp) -> bool:
        """True when it is Friday **and** time ≥ 13:15 (Asia/Taipei).

        Positions should be closed before the weekend gap risk.

        Parameters
        ----------
        current_time :
            Current bar timestamp.  Must be tz-aware or will be treated as
            Asia/Taipei local time.
        """
        local = _to_taipei(current_time)
        return local.weekday() == 4 and local.time() >= self._swing_close

    def is_settlement_day(self, current_date: date | pd.Timestamp) -> bool:
        """True if *current_date* falls on the **3rd Wednesday** of its month.

        TAIFEX TX / MTX front-month contracts settle on the 3rd Wednesday.
        Positions held through settlement are marked to the TAIFEX settlement
        price, which can gap significantly from the last traded price.

        Parameters
        ----------
        current_date :
            Any date-like; only the date component is used.
        """
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
        if current_date.weekday() != 2:  # 2 = Wednesday
            return False
        # Is it the 3rd Wednesday? Count Wednesdays in this month up to this date.
        wed_count = sum(
            1
            for d in range(1, current_date.day + 1)
            if date(current_date.year, current_date.month, d).weekday() == 2
        )
        return wed_count == 3

    def is_settlement_close(self, current_time: pd.Timestamp) -> bool:
        """True when it is a settlement day **and** time ≥ 13:15 (Asia/Taipei)."""
        local = _to_taipei(current_time)
        return self.is_settlement_day(local.date()) and local.time() >= self._swing_close

    def is_no_trade_zone(self, current_time: pd.Timestamp) -> bool:
        """True during high-risk session boundaries where new entries are blocked.

        No-trade zones
        --------------
        * 08:45–09:15 — opening 30 minutes: wide spreads, spikes, pre-session
          order-book imbalances.
        * 13:30–13:45 — closing 15 minutes: thin liquidity, marking-the-close
          effects.

        Parameters
        ----------
        current_time :
            Current bar timestamp (tz-aware preferred).
        """
        local = _to_taipei(current_time)
        t = local.time()
        in_opening = _DAY_OPEN <= t < _OPENING_RANGE_END
        in_closing = _EOD_FORCE_CLOSE <= t <= _DAY_CLOSE
        return in_opening or in_closing

    def is_eod_force_close(self, current_time: pd.Timestamp) -> bool:
        """True when it is at or past the intraday EOD exit cutoff (13:30).

        Intraday strategies must not hold positions past this bar to avoid
        carrying a futures position overnight.
        """
        local = _to_taipei(current_time)
        return local.time() >= self._eod_close

    def is_trading_day(self, current_date: date | pd.Timestamp) -> bool:
        """True if *current_date* is Monday–Friday.

        TODO: incorporate the full Taiwan Exchange holiday calendar.
        """
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
        return current_date.weekday() < 5

    # ------------------------------------------------------------------
    # Composite decision
    # ------------------------------------------------------------------

    def should_force_close(
        self,
        current_time: pd.Timestamp,
        current_date: date | pd.Timestamp | None = None,
    ) -> tuple[bool, str]:
        """Return ``(True, reason)`` when an open position must be closed NOW.

        Checks are applied in priority order:
        1. Friday close (13:15+)
        2. Settlement-day close (13:15+)
        3. Intraday EOD close (13:30+)

        Parameters
        ----------
        current_time :
            Current bar timestamp.
        current_date :
            Optional explicit date override.  If ``None``, derived from
            *current_time*.

        Returns
        -------
        tuple[bool, str]
            ``(should_close, reason_string)``.  If ``should_close`` is
            ``False``, *reason_string* is an empty string.
        """
        local = _to_taipei(current_time)
        d = (
            local.date()
            if current_date is None
            else (current_date.date() if isinstance(current_date, pd.Timestamp) else current_date)
        )

        if self.is_friday_close(local):
            return True, "Friday close — flattening before weekend gap"

        if self.is_settlement_close(local):
            return True, f"Settlement day {d} — closing before TAIFEX final settlement"

        if self.is_eod_force_close(local):
            return True, "Intraday EOD force-close (13:30)"

        return False, ""

    def should_block_entry(
        self,
        current_time: pd.Timestamp,
        current_date: date | pd.Timestamp | None = None,
    ) -> tuple[bool, str]:
        """Return ``(True, reason)`` when new entries must be blocked.

        Blocks during:
        1. No-trade zones (opening / closing windows)
        2. Friday close window (no point opening new trades)
        3. Settlement close window
        4. EOD force-close window
        """
        if self.is_no_trade_zone(current_time):
            return True, "No-trade zone (opening or closing window)"

        force, reason = self.should_force_close(current_time, current_date)
        if force:
            return True, f"Entry blocked — {reason}"

        return False, ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_taipei(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalise *ts* to Asia/Taipei — localise naive timestamps."""
    import zoneinfo

    tz = zoneinfo.ZoneInfo("Asia/Taipei")
    if ts.tz is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)
