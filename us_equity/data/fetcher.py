"""Market data fetching for US stocks via alpaca-py (Alpaca Data API v2).

All public methods return DataFrames with:
  - UTC DatetimeIndex named "timestamp"
  - columns: open, high, low, close, volume (int), symbol (str)

Note on paper vs live:
  StockHistoricalDataClient uses the same data endpoint for both paper and live
  accounts — the paper/live distinction only affects the trading client
  (used in agents/executor). API keys from either account type work here.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime, pd.Timestamp]

# Columns guaranteed in every returned DataFrame
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume", "symbol"]

# Max symbols per request (Alpaca imposes no hard per-call limit, but batching
# large universes keeps memory and response size manageable)
_CHUNK_SIZE = 50


# ---------------------------------------------------------------------------
# Enum helpers — resolved lazily to avoid import-time crash if alpaca-py is
# not installed in the current environment
# ---------------------------------------------------------------------------


def _get_timeframe(key: str) -> object:
    from alpaca.data.timeframe import TimeFrame  # type: ignore[import]

    _map = {
        "1d": TimeFrame.Day,
        "1h": TimeFrame.Hour,
        "1m": TimeFrame.Minute,
    }
    if key not in _map:
        raise ValueError(f"Unsupported timeframe '{key}'. Valid options: {list(_map)}")
    return _map[key]


def _get_adjustment(key: str) -> object:
    from alpaca.data.enums import Adjustment  # type: ignore[import]

    _map = {
        "raw": Adjustment.RAW,
        "split": Adjustment.SPLIT,
        "dividend": Adjustment.DIVIDEND,
        "all": Adjustment.ALL,
    }
    if key not in _map:
        raise ValueError(f"Unsupported adjustment '{key}'. Valid options: {list(_map)}")
    return _map[key]


def _get_feed(key: str) -> object:
    from alpaca.data.enums import DataFeed  # type: ignore[import]

    _map = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "otc": DataFeed.OTC,
    }
    if key not in _map:
        raise ValueError(f"Unsupported data feed '{key}'. Valid options: {list(_map)}")
    return _map[key]


# ---------------------------------------------------------------------------
# Internal DataFrame helpers
# ---------------------------------------------------------------------------


def _to_datetime(d: DateLike) -> datetime:
    """Convert any DateLike to a timezone-naive datetime for Alpaca requests."""
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime().replace(tzinfo=None)
    if isinstance(d, datetime):
        return d.replace(tzinfo=None)
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day)
    # str: parse ISO YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
    return datetime.fromisoformat(str(d).split("T")[0])


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OHLCV_COLUMNS,
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )


def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Standardise an alpaca-py BarSet slice to the project OHLCV schema.

    alpaca-py may include trade_count and vwap columns — we drop those here.
    The input df already has a flat DatetimeIndex (symbol level already dropped).
    """
    df = df[["open", "high", "low", "close", "volume"]].copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(int)
    df["symbol"] = symbol

    return df


def _barset_to_dict(raw: pd.DataFrame, requested: list[str]) -> dict[str, pd.DataFrame]:
    """Split an alpaca-py BarSet DataFrame into per-symbol DataFrames.

    alpaca-py always returns a MultiIndex DataFrame with levels
    (symbol, timestamp), even for single-symbol requests.
    """
    result: dict[str, pd.DataFrame] = {}

    if not isinstance(raw.index, pd.MultiIndex):
        # Should not happen with alpaca-py, but guard defensively
        logger.warning(
            "_barset_to_dict: expected MultiIndex but got flat index "
            "(alpaca-py version mismatch?). Attributing to first symbol."
        )
        if requested:
            result[requested[0]] = _normalise(raw, requested[0])
        return result

    sym_level = raw.index.names.index("symbol") if "symbol" in raw.index.names else 0

    for sym in raw.index.get_level_values(sym_level).unique():
        try:
            sym_df = raw.xs(sym, level=sym_level)
        except KeyError:
            logger.warning("Symbol %s missing from BarSet response", sym)
            continue
        result[sym] = _normalise(sym_df, sym)

    return result


# ---------------------------------------------------------------------------
# Public fetcher class
# ---------------------------------------------------------------------------


class AlpacaFetcher:
    """Fetches historical OHLCV data from Alpaca using the alpaca-py SDK.

    Uses StockHistoricalDataClient, which connects to the same data endpoint
    regardless of whether the account is paper or live.

    Example:
        fetcher = AlpacaFetcher()
        df   = fetcher.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
        bulk = fetcher.fetch_ohlcv_bulk(["AAPL", "TSLA"], "2024-01-01", "2024-12-31")
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        data_feed: str | None = None,
        # paper is intentionally accepted but unused: data API has no paper/live split.
        # It remains so callers can pass settings.ALPACA_PAPER without KeyError.
        paper: bool | None = None,  # noqa: ARG002
    ) -> None:
        """
        Args:
            api_key:    Override ALPACA_API_KEY from .env.
            secret_key: Override ALPACA_SECRET_KEY from .env.
            data_feed:  "iex" (free, default) | "sip" (paid consolidated tape).
            paper:      Accepted for interface compatibility; has no effect on the
                        data client (paper/live only matters for the trading client).
        """
        from alpaca.data import StockHistoricalDataClient  # type: ignore[import]

        from config.settings import settings

        resolved_key = api_key or settings.ALPACA_API_KEY
        resolved_secret = secret_key or settings.ALPACA_SECRET_KEY

        if not resolved_key or not resolved_secret:
            raise ValueError(
                "Alpaca credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
            )

        self._data_feed = data_feed or settings.ALPACA_DATA_FEED

        # StockHistoricalDataClient does not accept a base_url —
        # it always connects to data.alpaca.markets.
        self._client = StockHistoricalDataClient(
            api_key=resolved_key,
            secret_key=resolved_secret,
        )

        logger.info("AlpacaFetcher initialised (feed=%s)", self._data_feed)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        timeframe: str = "1d",
        adjustment: str = "all",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single US stock symbol.

        Args:
            symbol:     Ticker, e.g. "AAPL".
            start:      Start date, inclusive (str YYYY-MM-DD, date, or Timestamp).
            end:        End date, inclusive.
            timeframe:  "1d" | "1h" | "1m".
            adjustment: "raw" | "split" | "dividend" | "all" (recommended).

        Returns:
            DataFrame with UTC DatetimeIndex named "timestamp" and columns:
            open, high, low, close, volume, symbol.
            Returns an empty DataFrame (same schema) if no data is available.

        Raises:
            ValueError:   Unsupported timeframe or adjustment.
            RuntimeError: Alpaca API returned an error.
        """
        from alpaca.data.requests import StockBarsRequest  # type: ignore[import]

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_get_timeframe(timeframe),
            start=_to_datetime(start),
            end=_to_datetime(end),
            adjustment=_get_adjustment(adjustment),
            feed=_get_feed(self._data_feed),
        )

        logger.debug(
            "fetch_ohlcv: symbol=%s start=%s end=%s tf=%s adj=%s feed=%s",
            symbol,
            start,
            end,
            timeframe,
            adjustment,
            self._data_feed,
        )

        try:
            barset = self._client.get_stock_bars(request)
        except Exception as exc:
            raise RuntimeError(
                f"Alpaca API error fetching {symbol} [{start} → {end}]: {exc}"
            ) from exc

        raw = barset.df

        if raw.empty:
            logger.warning("No data returned for %s [%s → %s]", symbol, start, end)
            return _empty_df()

        # alpaca-py returns MultiIndex (symbol, timestamp) even for single symbols
        result = _barset_to_dict(raw, [symbol])
        df = result.get(symbol, _empty_df())

        logger.info("fetch_ohlcv: %s → %d bars", symbol, len(df))
        return df

    def fetch_ohlcv_bulk(
        self,
        symbols: list[str],
        start: DateLike,
        end: DateLike,
        timeframe: str = "1d",
        adjustment: str = "all",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols in batched requests.

        Chunks the symbol list into batches of up to 50 to keep individual
        responses at a manageable size.

        Args:
            symbols:    List of US ticker symbols.
            start / end / timeframe / adjustment: Same as fetch_ohlcv.

        Returns:
            Dict mapping symbol → DataFrame (same schema as fetch_ohlcv).
            Symbols with no data are omitted.

        Raises:
            ValueError:   Unsupported timeframe or adjustment.
            RuntimeError: Alpaca API returned an error on any chunk.
        """
        from alpaca.data.requests import StockBarsRequest  # type: ignore[import]

        if not symbols:
            return {}

        tf = _get_timeframe(timeframe)
        adj = _get_adjustment(adjustment)
        feed = _get_feed(self._data_feed)
        start_dt = _to_datetime(start)
        end_dt = _to_datetime(end)

        deduped = list(dict.fromkeys(symbols))  # preserve order, remove duplicates
        chunks = [deduped[i : i + _CHUNK_SIZE] for i in range(0, len(deduped), _CHUNK_SIZE)]

        result: dict[str, pd.DataFrame] = {}

        for idx, chunk in enumerate(chunks, start=1):
            logger.debug(
                "fetch_ohlcv_bulk: chunk %d/%d (%d symbols) [%s → %s]",
                idx,
                len(chunks),
                len(chunk),
                start,
                end,
            )

            request = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                adjustment=adj,
                feed=feed,
            )

            try:
                barset = self._client.get_stock_bars(request)
            except Exception as exc:
                raise RuntimeError(
                    f"Alpaca API error on bulk fetch chunk {idx}/{len(chunks)}: {exc}"
                ) from exc

            raw = barset.df

            if raw.empty:
                logger.warning("No data returned for chunk %d (%s)", idx, chunk)
                continue

            result.update(_barset_to_dict(raw, chunk))

        logger.info(
            "fetch_ohlcv_bulk: %d/%d symbols returned data",
            len(result),
            len(deduped),
        )
        return result
