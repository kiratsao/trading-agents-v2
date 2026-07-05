"""首次部署或資料損壞時，重建完整日K parquet。

Usage:
    python scripts/init_data.py                  # 預設 MXF (TAIFEX)
    python scripts/init_data.py --product MXF    # 期貨日K (TAIFEX MTX)
    python scripts/init_data.py --product 2330   # 台積電現貨日K (yfinance 2330.TW)

資料來源：
    MXF  → TAIFEX 期貨每日交易行情 (https://www.taifex.com.tw/cht/3/futDataDown)
    2330 → yfinance 2330.TW 現貨日K（僅供信號生成；執行使用 CDF 股票期貨）
"""

from __future__ import annotations

import argparse
import io
import sys
import time as _time
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

START_YEAR = 2020
START_MONTH = 1

_TAIFEX_URL = "https://www.taifex.com.tw/cht/3/futDataDown"


def _download_taifex_csv(year: int, month: int, product: str = "MTX") -> str:
    """POST to TAIFEX futDataDown and return the decoded (big5) CSV body."""
    import calendar

    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}/{month:02d}/01"
    end_date = f"{year}/{month:02d}/{last_day:02d}"

    params = (
        f"down_type=1&queryStartDate={start_date}"
        f"&queryEndDate={end_date}&commodity_id={product}"
    )
    req = urllib.request.Request(
        _TAIFEX_URL,
        data=params.encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("big5", errors="replace")


def _parse_taifex_csv(
    raw: str, *, year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    """Parse a TAIFEX futDataDown CSV body into a day-session daily OHLCV frame.

    ``交易時段`` semantics — stable across 2020→2026, verified by exact-OHLCV
    match against the Shioaji day-session parquet on 2020-06-15, 2023-06-15,
    2025-06-16, 2026-05-28/29:

    * ``一般`` — 該交易日的日盤 (08:45–13:45). The row we keep.
    * ``盤後`` — 前一交易日 15:00 起的夜盤, booked by TAIFEX to the **next**
      trading date. Never a day session — dropped outright. Consequence: a
      date whose day session hasn't traded yet (e.g. Monday's 盤後 row is
      published right after Friday's night session ends) yields NO bar,
      instead of a night bar masquerading as a future day K.

    Commit 6bdd195 had this inverted — its "Shioaji day close" oracle was
    itself a night-session fetch. Disproof via 2026-06-08: 盤後 low 40,761 is
    exactly -10% of the 06-05 settlement (weekend night session limit-down),
    while 一般 42,318→43,060 is Monday's actual day session.

    Unknown session labels are ranked after ``一般`` and a warning is printed
    — a future format change surfaces loudly instead of silently selecting
    the wrong row.
    """
    if not raw.strip() or "查無資料" in raw:
        return pd.DataFrame()

    # TAIFEX CSV has trailing commas on data lines → strip before parsing
    lines = raw.strip().split("\n")
    fixed = "\n".join(line.rstrip(", ") for line in lines)
    df = pd.read_csv(io.StringIO(fixed), index_col=False)

    # TAIFEX columns: 交易日期, 契約, 到期月份(週別), 開盤價, 最高價, 最低價, 收盤價,
    #                 漲跌價, 漲跌%, 成交量, 結算價, ..., 交易時段, ...
    required = {"交易日期", "開盤價", "最高價", "最低價", "收盤價", "成交量"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Parse date early — per-date session selection needs it.
    df["date"] = pd.to_datetime(
        df["交易日期"].astype(str).str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()

    # Filter: monthly contracts only (YYYYMM, 6 digits), exclude weekly (YYYYMMW1)
    if "到期月份(週別)" in df.columns:
        df["_expire"] = df["到期月份(週別)"].astype(str).str.strip()
        df = df[df["_expire"].str.match(r"^\d{6}$")]
    if df.empty:
        return pd.DataFrame()

    # Session selection: 一般 = 日盤 (keep); 盤後 = 夜盤 booked to the next
    # trading date (drop outright — never a valid day bar, even when it is
    # the only row for a date). Unknown labels rank after 一般, with a warning.
    if "交易時段" in df.columns:
        df["_session"] = df["交易時段"].astype(str).str.strip()
        df = df[df["_session"] != "盤後"]
        if df.empty:
            return pd.DataFrame()
        rank = df["_session"].map({"一般": 0})
        unknown = sorted(df.loc[rank.isna(), "_session"].unique())
        if unknown:
            where = f"{year}-{month:02d} " if year and month else ""
            print(
                f"  WARNING: {where}unknown TAIFEX 交易時段 labels {unknown} — "
                f"ranked after 一般; verify day-session alignment"
            )
        df["_sess_rank"] = rank.fillna(1)
        subset = ["date"] + (["_expire"] if "_expire" in df.columns else [])
        df = df.sort_values(subset + ["_sess_rank"])
        df = df.drop_duplicates(subset=subset, keep="first")

    # Keep only the nearest delivery month per date.
    if "_expire" in df.columns:
        df = df.sort_values(["date", "_expire"])
        df = df.drop_duplicates(subset=["date"], keep="first")

    for col in ["開盤價", "最高價", "最低價", "收盤價", "成交量"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ""),
            errors="coerce",
        )

    df = df.dropna(subset=["開盤價", "最高價", "最低價", "收盤價"])
    if df.empty:
        return pd.DataFrame()

    # One row per date after selection — groupby is a defensive no-op.
    daily = df.groupby("date").agg(
        open=("開盤價", "first"),
        high=("最高價", "max"),
        low=("最低價", "min"),
        close=("收盤價", "last"),
        volume=("成交量", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily


def fetch_taifex_month(year: int, month: int, product: str = "MTX") -> pd.DataFrame:
    """Fetch one month of day-session daily OHLCV from the TAIFEX CSV download."""
    raw = _download_taifex_csv(year, month, product)
    return _parse_taifex_csv(raw, year=year, month=month)


def fetch_mxf() -> pd.DataFrame:
    """Build MXF daily K from TAIFEX MTX month-by-month."""
    today = date.today()
    all_bars = []

    print(
        f"Fetching MTX daily from TAIFEX: "
        f"{START_YEAR}-{START_MONTH:02d} → {today.year}-{today.month:02d}"
    )

    y, m = START_YEAR, START_MONTH
    while (y, m) <= (today.year, today.month):
        sys.stdout.write(f"  {y}-{m:02d} ... ")
        sys.stdout.flush()
        try:
            df = fetch_taifex_month(y, m, product="MTX")
            if not df.empty:
                all_bars.append(df)
                print(f"{len(df)} bars")
            else:
                print("no data")
        except Exception as exc:
            print(f"ERROR: {exc}")

        _time.sleep(0.3)

        m += 1
        if m > 12:
            m = 1
            y += 1

    if not all_bars:
        print("ERROR: No data fetched from TAIFEX")
        sys.exit(1)

    daily = pd.concat(all_bars).sort_index()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily = daily[daily.index.dayofweek < 5]
    daily = daily[~daily.index.duplicated(keep="last")]
    return daily


def fetch_2330() -> pd.DataFrame:
    """Fetch 2330.TW daily OHLCV from yfinance.

    Cash stock (現貨), used for signal generation only — execution uses the
    CDF stock-futures contract with its own tick value & margin. auto_adjust
    smooths splits/dividends so EMA/ADX don't see fake gaps on ex-div days.
    """
    import yfinance as yf

    start = f"{START_YEAR}-{START_MONTH:02d}-01"
    today = date.today().isoformat()

    print(f"Fetching 2330.TW daily from yfinance: {start} → {today}")
    df = yf.download(
        "2330.TW",
        start=start,
        end=today,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        print("ERROR: No data fetched from yfinance")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df[df.index.dayofweek < 5]
    df = df[~df.index.duplicated(keep="last")]
    print(f"  fetched {len(df)} bars")
    return df


PRODUCT_FETCHERS = {
    "MXF": (fetch_mxf, "MXF_Daily_Clean_2020_to_now.parquet"),
    "2330": (fetch_2330, "2330_Daily_Clean_2020_to_now.parquet"),
}


def _remove_taifex_holidays(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop bars dated on a TAIFEX holiday (e.g. a 夜盤 bar mis-attributed to a
    closed day). Uses an explicit numpy boolean mask — the old
    ``Index.map(...).sum()`` crashed on the object-dtype result."""
    import numpy as np

    from src.data.tw_holidays import is_taifex_holiday

    if daily is None or len(daily) == 0:
        return daily
    idx = pd.DatetimeIndex(daily.index)
    mask = np.array([is_taifex_holiday(ts.date()) for ts in idx], dtype=bool)
    n = int(mask.sum())
    if n > 0:
        removed = idx[mask].strftime("%Y-%m-%d").tolist()
        print(f"  Removed {n} holiday bars: {removed[:10]}")
        daily = daily[~mask]
    return daily


def main():
    parser = argparse.ArgumentParser(description="Rebuild daily K parquet for a product.")
    parser.add_argument(
        "--product",
        default="MXF",
        choices=list(PRODUCT_FETCHERS.keys()),
        help="Which product to fetch (default: MXF)",
    )
    args = parser.parse_args()

    fetcher, filename = PRODUCT_FETCHERS[args.product]
    daily = fetcher()
    out_path = DATA_DIR / filename

    # TAIFEX is ground truth — the build does NOT pass through Shioaji.
    # (Cross-validation against Shioaji is a separate, opt-in step via
    # deep_health_check; init_data must not let a buggy Shioaji fetch
    # overwrite authoritative exchange data.)
    daily = _remove_taifex_holidays(daily)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=True)
    print(f"\nSaved: {out_path}")
    print(f"  Bars: {len(daily)}")
    print(f"  Range: {daily.index[0].date()} → {daily.index[-1].date()}")
    print(f"  Last close: {daily['close'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
