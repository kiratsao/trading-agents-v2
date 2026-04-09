"""TAIFEX public data fetcher — daily OHLCV from Taiwan Futures Exchange open data.

Data source: POST https://www.taifex.com.tw/cht/3/futDataDown
Response: MS950-encoded CSV, 19 columns.

No broker account required.  Supports TX (大台指期) and MTX (小台指期).

Observed response columns (index):
  0  交易日期       1  契約           2  到期月份(週別)
  3  開盤價         4  最高價         5  最低價
  6  收盤價         7  漲跌價         8  漲跌%
  9  成交量         10 結算價         11 未沖銷契約數
  12 最後最佳買價   13 最後最佳賣價   14 歷史最高價
  15 歷史最低價     16 是否因訊息面暫停交易
  17 交易時段       18 價差對單式委託成交量

Rate limiting: 1-second delay between monthly requests.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterator
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TAIFEX_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
_ENCODING = "ms950"
_REQUEST_DELAY = 1.0  # seconds between requests
_TIMEOUT = 30  # seconds per request
_MAX_RETRIES = 3  # per-chunk retry attempts on transient errors

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.taifex.com.tw/cht/3/futDataDown",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded",
}

_SESSION_REGULAR = "一般"  # regular trading session (排除盤後)

# Expected number of columns in the CSV response
_EXPECTED_COLS = 19
_COL_NAMES = [
    "date",
    "product_code",
    "contract",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_pct",
    "volume",
    "settlement",
    "oi",
    "last_bid",
    "last_ask",
    "hist_high",
    "hist_low",
    "halted",
    "session",
    "spread_vol",
]


class TaifexFetchError(RuntimeError):
    """Raised when a TAIFEX data request fails unexpectedly."""


class TaifexFetcher:
    """Fetch TAIFEX futures daily OHLCV from the exchange's public data portal.

    No broker account required.  Data is fetched month-by-month to stay
    within the exchange's query limit.

    Parameters
    ----------
    request_delay :
        Seconds to wait between consecutive HTTP requests (default 1.0).

    Example
    -------
    >>> fetcher = TaifexFetcher()
    >>> df = fetcher.fetch_daily("TX", "2025-01-01", "2025-03-31")
    >>> print(df.tail())
    """

    def __init__(self, request_delay: float = _REQUEST_DELAY) -> None:
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        product: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for *product* over [start_date, end_date].

        Automatically splits into monthly chunks, filters to regular-session
        (一般交易時段) data, and returns only the front-month contract
        (highest volume) for each trading date.

        Parameters
        ----------
        product :
            TAIFEX product code: ``"TX"`` (大台) or ``"MTX"`` (小台).
        start_date, end_date :
            Inclusive date range, ISO format ``"YYYY-MM-DD"``.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume, oi, product, contract.
            Index: ``pd.DatetimeIndex`` named ``"date"``, tz=Asia/Taipei.
            Sorted ascending.  Empty DataFrame if no data found.

        Raises
        ------
        TaifexFetchError
            On non-recoverable HTTP or parsing failures.
        ValueError
            If start_date > end_date or product is not recognised.
        """
        product = product.upper()
        if product not in ("TX", "MTX"):
            raise ValueError(f"Unsupported product {product!r}.  Use 'TX' or 'MTX'.")

        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        if start > end:
            raise ValueError(f"start_date {start_date!r} must be <= end_date {end_date!r}")

        chunks: list[pd.DataFrame] = []
        month_ranges = list(_monthly_chunks(start, end))

        for idx, (chunk_start, chunk_end) in enumerate(month_ranges):
            logger.info(
                "TaifexFetcher: fetching %s  %s → %s  (%d/%d)",
                product,
                chunk_start,
                chunk_end,
                idx + 1,
                len(month_ranges),
            )
            raw = self._fetch_chunk_with_retry(product, chunk_start, chunk_end)
            if raw is not None and not raw.empty:
                chunks.append(raw)
            # Polite delay between requests (skip after the last chunk)
            if idx < len(month_ranges) - 1:
                time.sleep(self.request_delay)

        if not chunks:
            logger.warning("TaifexFetcher: no data returned for %s %s → %s", product, start, end)
            return _empty_df()

        df = pd.concat(chunks)
        df = _select_front_month(df)
        df = df.sort_index()
        # Drop duplicate index entries (same date appears in overlapping chunks)
        df = df[~df.index.duplicated(keep="first")]
        return df

    def fetch_minute(
        self,
        product: str,
        date_str: str,
        interval: int = 1,
    ) -> pd.DataFrame:
        """Fetch intraday minute bars for *product* on *date_str*.

        .. TODO:: Replace with Shioaji real-time / tick data once the broker
                  account is live.  TAIFEX does not publish free minute-bar
                  data; this method returns the daily OHLCV as a single
                  placeholder bar timestamped at 08:45 CST (TAIFEX open).

        Parameters
        ----------
        product :
            TAIFEX product code (``"TX"`` or ``"MTX"``).
        date_str :
            Trading date, ISO format ``"YYYY-MM-DD"``.
        interval :
            Desired bar interval in minutes (stored in column metadata only;
            not used for aggregation in the placeholder implementation).

        Returns
        -------
        pd.DataFrame
            Same columns as :meth:`fetch_daily`.
            Single-row index at 08:45 CST (Asia/Taipei).
        """
        logger.warning(
            "fetch_minute: using daily-bar placeholder for %s %s  "
            "(TODO: replace with Shioaji tick data)",
            product,
            date_str,
        )
        df = self.fetch_daily(product, date_str, date_str)
        if df.empty:
            return df

        # Re-stamp to 08:45 CST — TAIFEX day-session open
        open_ts = pd.Timestamp(date_str, tz="Asia/Taipei").replace(hour=8, minute=45)
        df.index = pd.DatetimeIndex([open_ts], name="date")
        df.attrs["interval_minutes"] = interval
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_chunk_with_retry(
        self,
        product: str,
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        """Attempt _fetch_chunk up to _MAX_RETRIES times on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._fetch_chunk(product, start, end)
            except TaifexFetchError as exc:
                last_exc = exc
                logger.warning(
                    "TaifexFetcher: attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(attempt * 2.0)  # back-off: 2s, 4s
        logger.error("TaifexFetcher: all retries exhausted for %s %s→%s", product, start, end)
        if last_exc is not None:
            raise last_exc
        return None

    def _fetch_chunk(
        self,
        product: str,
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        """Fetch one calendar-month chunk and return a parsed DataFrame."""
        payload = {
            "queryStartDate": start.strftime("%Y/%m/%d"),
            "queryEndDate": end.strftime("%Y/%m/%d"),
            "commodity_id": product,
            "down_type": "1",
            "commodity_id2": "",
        }
        try:
            resp = self._session.post(_TAIFEX_URL, data=payload, timeout=_TIMEOUT)
            resp.raise_for_status()
        except requests.Timeout as exc:
            raise TaifexFetchError(f"Request timed out ({_TIMEOUT}s): {exc}") from exc
        except requests.HTTPError as exc:
            raise TaifexFetchError(f"HTTP {resp.status_code}: {exc}") from exc
        except requests.RequestException as exc:
            raise TaifexFetchError(f"Network error: {exc}") from exc

        raw_bytes = resp.content
        # Decode MS950; fall back to UTF-8 for error pages
        try:
            text = raw_bytes.decode(_ENCODING)
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("utf-8", errors="replace")

        # TAIFEX returns HTML when the query is invalid or blocked
        stripped = text.lstrip()
        if stripped.startswith("<") or "<html" in stripped[:200].lower():
            raise TaifexFetchError(
                f"TAIFEX returned HTML (possible invalid range or rate-limit) "
                f"for {product} {start}→{end}"
            )

        return _parse_csv(text, product)


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _monthly_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) pairs, each within one calendar month."""
    cursor = start
    while cursor <= end:
        # Last day of cursor's month
        if cursor.month == 12:
            month_end = date(cursor.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(cursor.year, cursor.month + 1, 1) - timedelta(days=1)
        yield cursor, min(month_end, end)
        cursor = min(month_end, end) + timedelta(days=1)


def _parse_csv(text: str, product: str) -> pd.DataFrame | None:
    """Parse the MS950 CSV text into a clean DataFrame.

    TAIFEX quirk: data rows end with two trailing commas (20 fields) while
    the header has only 19 column names.  When pandas sees n_data_fields =
    n_header_fields + 1 it silently promotes field[0] to a row index,
    shifting every column value one position to the right.  We avoid this
    by reading without a header row and providing explicit column names.
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None  # header-only or empty

    # Count actual fields in the first data row to decide how many names needed
    first_data_fields = len(lines[1].split(","))
    # Provide enough names to cover all data fields (extras become "extra_N")
    n_names = max(first_data_fields, _EXPECTED_COLS)
    col_names = _COL_NAMES + [f"extra_{i}" for i in range(n_names - len(_COL_NAMES))]

    try:
        df = pd.read_csv(
            io.StringIO(text),
            header=None,  # don't use first row as header
            skiprows=1,  # skip the actual header line
            names=col_names[:n_names],
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        logger.error("TaifexFetcher: CSV parse error: %s", exc)
        return None

    if df.empty:
        return None

    # ── Filter: regular trading session only ──────────────────────────
    df = df[df["session"].str.strip() == _SESSION_REGULAR].copy()
    if df.empty:
        return None

    # ── Parse date ────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize("Asia/Taipei")

    # ── Clean contract column (strip trailing spaces) ─────────────────
    df["contract"] = df["contract"].str.strip()

    # ── Numeric conversion ("-" → NaN, strip commas from thousands) ───
    for col in ("open", "high", "low", "close", "volume", "oi"):
        df[col] = pd.to_numeric(
            df[col].str.replace(",", "", regex=False).replace("-", pd.NA),
            errors="coerce",
        )

    df["product"] = product

    result = (
        df[["date", "open", "high", "low", "close", "volume", "oi", "product", "contract"]]
        .copy()
        .set_index("date")
    )
    return result


def _select_front_month(df: pd.DataFrame) -> pd.DataFrame:
    """For each trading date retain only the front-month (max-volume) contract."""
    reset = df.reset_index()
    # Sum volume per (date, contract) across any duplicate rows
    vol = reset.groupby(["date", "contract"])["volume"].sum().reset_index()
    # Index of max-volume contract per date
    front_idx = vol.groupby("date")["volume"].idxmax()
    front = vol.loc[front_idx, ["date", "contract"]].set_index("date")

    merged = reset.merge(front.reset_index(), on=["date", "contract"], how="inner").set_index(
        "date"
    )
    return merged


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "oi", "product", "contract"],
        index=pd.DatetimeIndex([], name="date", tz="Asia/Taipei"),
    )


# ---------------------------------------------------------------------------
# Shioaji minute-bar fetcher
# ---------------------------------------------------------------------------


class ShioajiFetcher:
    """Fetch intraday minute K-bars from Shioaji (永豐).

    Requires an active :class:`~tw_futures.executor.shioaji_adapter.ShioajiAdapter`
    instance (already logged in).

    Parameters
    ----------
    adapter :
        A connected ``ShioajiAdapter`` instance.

    Example
    -------
    >>> from tw_futures.executor.shioaji_adapter import ShioajiAdapter
    >>> adapter = ShioajiAdapter(api_key="...", secret_key="...", simulation=True)
    >>> fetcher = ShioajiFetcher(adapter)
    >>> df = fetcher.fetch_minute_kbars("TXF", "2026-04-03")
    >>> print(df.head())
    """

    def __init__(self, adapter) -> None:
        self._adapter = adapter

    def fetch_minute_kbars(
        self,
        product: str,
        date: str,
        interval: int = 1,
    ) -> pd.DataFrame:
        """Fetch minute K-bars for *product* on *date* via Shioaji ``kbars()``.

        Parameters
        ----------
        product :
            ``"TXF"`` (大台) or ``"MXF"`` (小台).
        date :
            Trading date, ISO format ``"YYYY-MM-DD"``.
        interval :
            Bar interval in minutes.  ``1`` = 1-minute bars (default).
            Note: Shioaji's ``kbars()`` always returns 1-minute resolution;
            this parameter is used to resample the result.

        Returns
        -------
        pd.DataFrame
            Columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
            Index: ``pd.DatetimeIndex`` named ``"ts"``, tz=Asia/Taipei.
            Sorted ascending.  Empty DataFrame if no data.

        Raises
        ------
        ExecutionError
            If the Shioaji API call fails.
        """
        from tw_futures.executor.shioaji_adapter import ExecutionError

        contract = self._adapter.get_contract(product)
        api = self._adapter._api

        try:
            kbars = api.kbars(contract, start=date, end=date)
        except Exception as exc:
            raise ExecutionError(f"kbars() failed for {product} {date}: {exc}") from exc

        if not kbars or not kbars.ts:
            logger.warning("ShioajiFetcher: no kbar data for %s %s", product, date)
            return _empty_minute_df()

        ts_index = pd.to_datetime(kbars.ts, unit="ns", utc=True).tz_convert("Asia/Taipei")
        df = pd.DataFrame(
            {
                "open": kbars.Open,
                "high": kbars.High,
                "low": kbars.Low,
                "close": kbars.Close,
                "volume": kbars.Volume,
            },
            index=ts_index,
        )
        df.index.name = "ts"
        df = df.sort_index()

        if interval > 1:
            df = _resample_bars(df, interval)

        df.attrs["interval_minutes"] = interval
        df.attrs["product"] = product
        logger.info(
            "ShioajiFetcher: %s %s  %d bars (interval=%dm)",
            product,
            date,
            len(df),
            interval,
        )
        return df


def _empty_minute_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="ts", tz="Asia/Taipei"),
    )


def _resample_bars(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Resample 1-minute bars to *interval*-minute bars."""
    rule = f"{interval}min"
    resampled = df.resample(rule, closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return resampled.dropna(subset=["open"])
