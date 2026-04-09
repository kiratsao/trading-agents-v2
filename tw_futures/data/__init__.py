"""TW futures data layer — TAIFEX public data fetcher and rollover manager."""

from .fetcher import TaifexFetcher, TaifexFetchError
from .rollover import (
    RolloverManager,
    get_expiry_date,
    get_front_month,
    get_next_contract,
    should_rollover,
)

__all__ = [
    "TaifexFetcher",
    "TaifexFetchError",
    "RolloverManager",
    "get_expiry_date",
    "get_front_month",
    "get_next_contract",
    "should_rollover",
]
