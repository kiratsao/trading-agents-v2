"""TSM ADR + SOX 價格抓取器。

* 使用 yfinance 抓取 TSM 和 ^SOX 的最新收盤/即時價格
* 同一天只抓取一次（in-memory cache）
* 任何異常均 graceful fallback → 回傳 None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class FetchedPrices:
    tsm_change_pct: float
    sox_change_pct: float
    tsm_price: float
    sox_price: float


_CACHE: dict[str, object] = {}


def _fetch_ticker_change(ticker_sym: str) -> tuple[float, float]:
    """Fetch latest % change for a ticker. Raises on failure."""
    import yfinance as yf  # type: ignore[import]

    tk = yf.Ticker(ticker_sym)
    info = tk.fast_info
    prev_close = getattr(info, "previous_close", None) or info.get("previousClose", 0)
    last_price = getattr(info, "last_price", None) or info.get("lastPrice", prev_close)
    if not prev_close:
        raise ValueError(f"previousClose=0 for {ticker_sym}")
    change_pct = (last_price - prev_close) / prev_close * 100.0
    return change_pct, last_price


def fetch_prices(
    today: date | None = None,
    force_refresh: bool = False,
) -> FetchedPrices | None:
    """Fetch TSM ADR and SOX changes.

    Parameters
    ----------
    today :
        Date key for cache. Defaults to today.
    force_refresh :
        Bypass cache when True.

    Returns
    -------
    FetchedPrices or None if any fetch fails.
    """
    today = today or date.today()
    cache_key = str(today)
    if not force_refresh and cache_key in _CACHE:
        return _CACHE[cache_key]  # type: ignore[return-value]
    try:
        tsm_change, tsm_price = _fetch_ticker_change("TSM")
        sox_change, sox_price = _fetch_ticker_change("^SOX")
        result = FetchedPrices(
            tsm_change_pct=tsm_change,
            sox_change_pct=sox_change,
            tsm_price=tsm_price,
            sox_price=sox_price,
        )
        _CACHE[cache_key] = result
        logger.info(
            "Fetched TSM %.2f%% / SOX %.2f%% (date=%s)",
            tsm_change,
            sox_change,
            today,
        )
        return result
    except Exception as exc:
        logger.warning("fetch_prices failed (%s): %s — returning None", type(exc).__name__, exc)
        return None
