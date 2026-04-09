"""Contract rollover helpers for TAIFEX monthly futures (TX / MTX).

TAIFEX TX/MTX monthly futures expire on the **third Wednesday** of the
contract month.  Standard convention: roll the position to the next
month 3 trading days before expiry.

Usage example
-------------
>>> from tw_futures.data.rollover import get_expiry_date, get_front_month, should_rollover
>>> get_expiry_date(2025, 4)          # → date(2025, 4, 16)
>>> get_front_month(date(2025, 4, 10))  # → "202504"
>>> get_front_month(date(2025, 4, 14))  # → "202505"  (within 3-day rollover window)
>>> should_rollover(date(2025, 4, 14), "202504")  # → True
"""

from __future__ import annotations

import calendar
import logging
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_ROLLOVER_TRADING_DAYS_BEFORE: int = 3  # trading days before expiry to roll


# ---------------------------------------------------------------------------
# Core date helpers
# ---------------------------------------------------------------------------


def get_expiry_date(year: int, month: int) -> date:
    """Return the expiry date for a TAIFEX monthly futures contract.

    Expiry = **third Wednesday** of the contract month.

    Parameters
    ----------
    year, month :
        Contract year (e.g. 2025) and month (1–12).

    Returns
    -------
    date

    Examples
    --------
    >>> get_expiry_date(2025, 3)
    datetime.date(2025, 3, 19)
    >>> get_expiry_date(2025, 4)
    datetime.date(2025, 4, 16)
    """
    if not (1 <= month <= 12):
        raise ValueError(f"month must be 1–12, got {month}")
    cal = calendar.monthcalendar(year, month)
    # calendar.WEDNESDAY = 2; 0 means the day falls outside the month
    wednesdays = [week[calendar.WEDNESDAY] for week in cal if week[calendar.WEDNESDAY] != 0]
    if len(wednesdays) < 3:
        raise ValueError(f"Cannot find third Wednesday for {year}/{month:02d}")
    return date(year, month, wednesdays[2])


def get_front_month(ref_date: date | None = None) -> str:
    """Return the front-month contract code as of *ref_date*.

    Rolls to next month's contract when *ref_date* reaches the rollover
    window (``_ROLLOVER_TRADING_DAYS_BEFORE`` trading days before expiry).

    Parameters
    ----------
    ref_date :
        Reference date (defaults to today in Asia/Taipei).

    Returns
    -------
    str
        Six-digit contract month string, e.g. ``"202604"``.

    Examples
    --------
    >>> get_front_month(date(2025, 4, 10))
    '202504'
    >>> get_front_month(date(2025, 4, 14))  # within rollover window
    '202505'
    """
    today = ref_date or _today_taipei()
    expiry = get_expiry_date(today.year, today.month)
    rollover_date = _subtract_trading_days(expiry, _ROLLOVER_TRADING_DAYS_BEFORE)

    if today >= rollover_date:
        # Switch to next month
        if today.month == 12:
            return f"{today.year + 1}01"
        return f"{today.year}{today.month + 1:02d}"
    return f"{today.year}{today.month:02d}"


def get_next_contract(current_contract: str) -> str:
    """Return the next month's contract code from *current_contract*.

    Parameters
    ----------
    current_contract :
        Six-digit string ``"YYYYMM"`` (e.g. ``"202603"``).
        Weekly suffixes (e.g. ``"202603W1"``) are not supported.

    Returns
    -------
    str
        Next month, e.g. ``"202604"``.

    Raises
    ------
    ValueError
        If *current_contract* is not a valid ``"YYYYMM"`` string.
    """
    c = current_contract.strip()
    if len(c) != 6 or not c.isdigit():
        raise ValueError(
            f"Expected 'YYYYMM' (e.g. '202603'), got {current_contract!r}.  "
            "Weekly contract codes (e.g. '202503W1') are not supported."
        )
    year, month = int(c[:4]), int(c[4:])
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month {month} in contract {current_contract!r}")
    if month == 12:
        return f"{year + 1}01"
    return f"{year}{month + 1:02d}"


def should_rollover(current_date: date, current_contract: str) -> bool:
    """Return True if *current_contract* should be rolled over on *current_date*.

    Rollover is triggered when *current_date* is within
    ``_ROLLOVER_TRADING_DAYS_BEFORE`` trading days of the contract expiry.

    Parameters
    ----------
    current_date :
        Reference date (e.g. today).
    current_contract :
        Six-digit contract month string ``"YYYYMM"``.

    Returns
    -------
    bool

    Examples
    --------
    >>> should_rollover(date(2025, 4, 10), "202504")  # expiry = Apr 16, rollover = Apr 11
    False
    >>> should_rollover(date(2025, 4, 14), "202504")
    True
    """
    c = current_contract.strip()
    # Silently ignore weekly contracts (not applicable for monthly rollover)
    if len(c) < 6 or not c[:6].isdigit():
        return False
    year = int(c[:4])
    month = int(c[4:6])
    try:
        expiry = get_expiry_date(year, month)
    except ValueError:
        return False
    rollover_date = _subtract_trading_days(expiry, _ROLLOVER_TRADING_DAYS_BEFORE)
    return current_date >= rollover_date


# ---------------------------------------------------------------------------
# RolloverManager — execution skeleton (needs ShioajiAdapter)
# ---------------------------------------------------------------------------


class RolloverManager:
    """Detect and execute contract rollovers automatically.

    Call :meth:`check_and_roll` once per trading day (pre-market) to detect
    positions that need rolling and submit the corresponding orders.

    Parameters
    ----------
    days_before_expiry :
        Trading days before expiry to initiate rollover (default 3).
    """

    def __init__(self, days_before_expiry: int = _ROLLOVER_TRADING_DAYS_BEFORE) -> None:
        self.days_before_expiry = days_before_expiry

    def check_and_roll(
        self,
        positions: dict[str, int],
        adapter: Any,
        as_of: date | None = None,
    ) -> list[dict[str, Any]]:
        """Check positions for expiring contracts and roll them.

        Parameters
        ----------
        positions :
            ``{contract_code: net_contracts}`` where contract_code is ``"YYYYMM"``.
            E.g. ``{"202603": 5, "202604": -2}``.
        adapter :
            ShioajiAdapter instance for order submission.
        as_of :
            Reference date (defaults to today).

        Returns
        -------
        list[dict]
            One dict per rollover action.  Empty list if nothing to roll.
            Each dict: ``{action, from, to, net_contracts, status, note}``.
        """
        today = as_of or _today_taipei()
        results: list[dict[str, Any]] = []

        for contract, net in list(positions.items()):
            if net == 0:
                continue
            if not should_rollover(today, contract):
                continue

            try:
                next_c = get_next_contract(contract)
            except ValueError as exc:
                logger.warning("Cannot determine next contract for %r: %s", contract, exc)
                continue

            logger.info(
                "RolloverManager: rollover triggered %s → %s  (net=%+d, as_of=%s)",
                contract,
                next_c,
                net,
                today,
            )
            # TODO: submit close of `contract` + open of `next_c` via adapter
            # once ShioajiAdapter.submit_order() is implemented.
            results.append(
                {
                    "action": "rollover",
                    "from": contract,
                    "to": next_c,
                    "net_contracts": net,
                    "status": "pending",
                    "note": "ShioajiAdapter not yet connected — order not submitted",
                }
            )

        return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _subtract_trading_days(ref: date, n: int) -> date:
    """Return the date *n* Taiwan trading days before *ref*.

    Uses pandas BDay (Mon–Fri) as a calendar approximation.
    For production, replace with a proper Taiwan public holidays calendar.
    """
    ts = pd.Timestamp(ref) - pd.offsets.BDay(n)
    return ts.date()


def _today_taipei() -> date:
    """Return today's date in Asia/Taipei timezone."""
    return pd.Timestamp.now(tz="Asia/Taipei").date()
