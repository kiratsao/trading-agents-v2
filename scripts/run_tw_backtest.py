"""Backtest BreakoutSwing V1/V2/V3, ORB intraday, and combo strategies on TAIFEX futures.

Usage
-----
    # Swing strategies (daily bars)
    python scripts/run_tw_backtest.py                       # V1 vs V3
    python scripts/run_tw_backtest.py --strategy v1
    python scripts/run_tw_backtest.py --strategy v3

    # ORB intraday (synthetic 15-min bars, for smoke-testing)
    python scripts/run_tw_backtest.py --strategy orb

    # Combo: Swing V1 (50%) + ORB (50%) — synthetic 15-min for ORB
    python scripts/run_tw_backtest.py --strategy combo

    # Use cached data
    python scripts/run_tw_backtest.py --no-fetch --strategy v3

    # Custom range and capital
    python scripts/run_tw_backtest.py --start 2020-01-01 --end 2026-04-01 \\
        --capital 5000000 --strategy v1v3

Output
------
* Comparison metrics table or single-strategy metrics.
* Yearly performance table.
* Equity curve + drawdown chart saved to results/tw_futures_equity.png.

Notes
-----
--strategy orb and --strategy combo always use SYNTHETIC 15-min bars built from
daily TAIFEX data.  Suitable for smoke-testing logic only, NOT performance evaluation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from core.data.store import load_ohlcv, save_ohlcv_bulk
from tw_futures.backtester.backtester import FuturesBacktester, FuturesBacktestResult
from tw_futures.data.fetcher import TaifexFetcher, TaifexFetchError
from tw_futures.strategies.intraday.orb_intraday import ORBIntradayStrategy
from tw_futures.strategies.swing.breakout_swing import BreakoutSwingStrategy
from tw_futures.strategies.swing.breakout_swing_scaled import BreakoutSwingScaledStrategy
from tw_futures.strategies.swing.breakout_swing_v2 import BreakoutSwingV2Strategy
from tw_futures.strategies.swing.breakout_swing_v3 import BreakoutSwingV3Strategy

_REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _REPO_ROOT / "data" / "db" / "market_data.db"
RESULTS_DIR = _REPO_ROOT / "results"

_METRIC_KEYS = [
    "Total Return (%)",
    "CAGR (%)",
    "Sharpe Ratio",
    "Max Drawdown (%)",
    "Calmar Ratio",
    "Total Trades",
    "Win Rate (%)",
    "Profit Factor",
    "Avg Win (TWD)",
    "Avg Loss (TWD)",
    "Max Single Loss (TWD)",
    "Total Cost (TWD)",
    "Avg Holding Days",
    "Max Consecutive Losses",
    "Peak Leverage (×)",
    "Margin Call Count",
    "Benchmark (%)",
]


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest TX Breakout Swing V1/V2 strategies")
    parser.add_argument(
        "--no-fetch", action="store_true", help="Use SQLite cache; skip TAIFEX HTTP fetch."
    )
    parser.add_argument(
        "--strategy",
        default="v1v3",
        choices=["v1", "v2", "v3", "both", "v1v3", "scaled", "v1scaled", "orb", "combo"],
        help="Which strategy to run (default: v1v3).",
    )
    parser.add_argument("--product", default="TX", choices=["TX", "MTX"])
    parser.add_argument(
        "--start", default="2024-01-01", help="Backtest start date (default: 2024-01-01)."
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="Backtest end date (default: 2025-12-31)."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=2_000_000.0,
        help="Initial capital TWD (default: 2,000,000).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_tw_backtest")

    # ── Load data ─────────────────────────────────────────────────────────
    df = _load_data(args.product, args.start, args.end, args.no_fetch, logger)
    if df is None:
        sys.exit(1)
    logger.info(
        "Data ready: %d rows  [%s → %s]", len(df), df.index.min().date(), df.index.max().date()
    )

    # ── Intraday / Combo strategies — need synthetic 15-min data ─────────
    if args.strategy in ("orb", "combo"):
        _run_intraday_strategies(df, args, logger)
        return

    # ── Run swing strategy/strategies ────────────────────────────────────
    bt = FuturesBacktester()

    results: dict[str, FuturesBacktestResult] = {}

    if args.strategy in ("v1", "both", "v1v3", "v1scaled"):
        logger.info("Running V1 …")
        results["V1"] = bt.run(
            BreakoutSwingStrategy(product=args.product), df, args.capital, args.product
        )

    if args.strategy in ("v2", "both"):
        logger.info("Running V2 …")
        results["V2"] = bt.run(
            BreakoutSwingV2Strategy(product=args.product), df, args.capital, args.product
        )

    if args.strategy in ("v3", "v1v3"):
        logger.info("Running V3 …")
        results["V3"] = bt.run(
            BreakoutSwingV3Strategy(product=args.product), df, args.capital, args.product
        )

    if args.strategy in ("scaled", "v1scaled"):
        logger.info("Running Scaled (3-stage pyramid) …")
        results["Scaled"] = bt.run(
            BreakoutSwingScaledStrategy(product=args.product), df, args.capital, args.product
        )

    # ── Print output ──────────────────────────────────────────────────────
    if "V1" in results and "V3" in results and "V2" not in results and "Scaled" not in results:
        _print_comparison_v1v3(
            results["V1"], results["V3"], args.start, args.end, args.product, args.capital
        )
        _print_yearly_both_labeled(results["V1"], results["V3"], "V1", "V3", args.start, args.end)
    elif "V1" in results and "Scaled" in results and len(results) == 2:
        _print_comparison_v1v3(
            results["V1"],
            results["Scaled"],
            args.start,
            args.end,
            args.product,
            args.capital,
            label1="V1",
            label2="Scaled",
        )
        _print_yearly_both_labeled(
            results["V1"], results["Scaled"], "V1", "Scaled", args.start, args.end
        )
    elif len(results) == 2 and "V1" in results and "V2" in results:
        _print_comparison(
            results["V1"], results["V2"], args.start, args.end, args.product, args.capital
        )
        _print_yearly_both(results["V1"], results["V2"], args.start, args.end)
    else:
        label, result = next(iter(results.items()))
        _print_single(result, label, args.start, args.end, args.product, args.capital)
        _print_yearly_single(result, label, args.capital)

    # Final equity + profit
    for label, result in results.items():
        eq = result.equity_curve
        final = float(eq.iloc[-1])
        profit = final - args.capital
        print(
            f"  {label}  Final Equity: TWD {final:>12,.0f}   "
            f"Total Profit: TWD {profit:>+12,.0f}  "
            f"({(final / args.capital - 1) * 100:+.2f}%)"
        )
    print()

    # ── Chart ─────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = RESULTS_DIR / "tw_futures_equity.png"
    _plot_results(results, args.start, args.end, chart_path)
    print(f"  Chart → {chart_path}")


# ── Intraday / Combo runner ───────────────────────────────────────────────────


def _run_intraday_strategies(
    daily_df: pd.DataFrame,
    args,
    logger: logging.Logger,
) -> None:
    """Run ORB (and optionally Swing V1 combo) using synthetic 15-min bars."""
    import warnings as _warnings

    from scripts.build_synthetic_15min import build_synthetic_15min

    _warnings.filterwarnings("ignore")  # suppress synthetic data warning in output

    # Build synthetic 15-min data (printed warning suppressed above)
    logger.info("Building synthetic 15-min bars from daily data …")
    data_15min = build_synthetic_15min(daily_df, seed=42)
    logger.info(
        "Synthetic 15-min bars: %d rows  [%s → %s]",
        len(data_15min),
        data_15min.index.min().strftime("%Y-%m-%d") if not data_15min.empty else "—",
        data_15min.index.max().strftime("%Y-%m-%d") if not data_15min.empty else "—",
    )

    bt = FuturesBacktester()

    # ── ORB only ──────────────────────────────────────────────────────────
    if args.strategy == "orb":
        print("\n  ⚠  SYNTHETIC DATA — results are NOT suitable for strategy evaluation.\n")
        logger.info("Running ORB intraday (synthetic 15-min) …")
        orb_result = bt.run_intraday(
            ORBIntradayStrategy(product=args.product),
            data_15min,
            args.capital,
            args.product,
        )
        _print_single(
            orb_result, "ORB (synthetic)", args.start, args.end, args.product, args.capital
        )
        eq = orb_result.equity_curve
        final = float(eq.iloc[-1])
        print(
            f"  ORB  Final Equity: TWD {final:>12,.0f}   "
            f"Total Profit: TWD {(final - args.capital):>+12,.0f}  "
            f"({(final / args.capital - 1) * 100:+.2f}%)\n"
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = RESULTS_DIR / "tw_orb_equity.png"
        _plot_results({"ORB": orb_result}, args.start, args.end, chart_path)
        print(f"  Chart → {chart_path}")
        return

    # ── Combo: Swing V1 (50%) + ORB (50%) ────────────────────────────────
    assert args.strategy == "combo"
    print(
        "\n  ⚠  COMBO SMOKE TEST — ORB uses SYNTHETIC 15-min data.\n"
        "  Swing V1 uses real daily data.  Combo metrics are illustrative only.\n"
    )

    swing_cap = args.capital * 0.5
    orb_cap = args.capital * 0.5

    logger.info("Running Swing V1 (50%% of capital = TWD %.0f) …", swing_cap)
    swing_result = bt.run(
        BreakoutSwingStrategy(product=args.product),
        daily_df,
        swing_cap,
        args.product,
    )

    logger.info("Running ORB intraday (50%% of capital = TWD %.0f, synthetic) …", orb_cap)
    orb_result = bt.run_intraday(
        ORBIntradayStrategy(product=args.product),
        data_15min,
        orb_cap,
        args.product,
    )

    # ── Print individual metrics ───────────────────────────────────────────
    _print_single(swing_result, "Swing V1 (50%)", args.start, args.end, args.product, swing_cap)
    _print_single(orb_result, "ORB (50%, synthetic)", args.start, args.end, args.product, orb_cap)

    # ── Combine equity curves ─────────────────────────────────────────────
    eq_swing = swing_result.equity_curve.rename("swing")
    eq_orb = orb_result.equity_curve.rename("orb")

    # Resample both to daily, forward-fill gaps, then align
    eq_swing_d = eq_swing.resample("D").last().ffill()
    eq_orb_d = eq_orb.resample("D").last().ffill()

    # Align on common date range
    common_start = max(eq_swing_d.index.min(), eq_orb_d.index.min())
    common_end = min(eq_swing_d.index.max(), eq_orb_d.index.max())
    eq_swing_d = eq_swing_d.loc[common_start:common_end]
    eq_orb_d = eq_orb_d.loc[common_start:common_end].reindex(eq_swing_d.index, method="ffill")

    eq_combo = eq_swing_d + eq_orb_d  # combined equity

    # Returns for correlation
    ret_swing = eq_swing_d.pct_change().dropna()
    ret_orb = eq_orb_d.pct_change().reindex(ret_swing.index).dropna()
    ret_swing = ret_swing.reindex(ret_orb.index)
    correlation = ret_swing.corr(ret_orb)

    # Combined metrics
    total_ret = (float(eq_combo.iloc[-1]) / args.capital - 1.0) * 100.0
    n_days = max((eq_combo.index[-1] - eq_combo.index[0]).days, 1)
    cagr = ((float(eq_combo.iloc[-1]) / args.capital) ** (365.0 / n_days) - 1.0) * 100.0
    max_dd = abs(float((eq_combo / eq_combo.cummax() - 1.0).min())) * 100.0

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  Combo Portfolio  [Swing V1 50% + ORB 50%]")
    print(SEP)
    print(f"  {'Total Capital':<32}: TWD {args.capital:>12,.0f}")
    print(f"  {'Total Return (%)':<32}: {total_ret:>12.2f}")
    print(f"  {'CAGR (%)':<32}: {cagr:>12.2f}")
    print(f"  {'Max Drawdown (%)':<32}: {max_dd:>12.2f}")
    print(f"  {'Swing–ORB Return Correlation':<32}: {correlation:>12.3f}")
    swing_fin = float(eq_swing_d.iloc[-1])
    orb_fin = float(eq_orb_d.iloc[-1])
    combo_fin = float(eq_combo.iloc[-1])
    print(
        f"  {'Swing V1 Final':<32}: TWD {swing_fin:>12,.0f}  ({(swing_fin / swing_cap - 1) * 100:+.2f}%)"
    )
    print(
        f"  {'ORB Final (synthetic)':<32}: TWD {orb_fin:>12,.0f}  ({(orb_fin / orb_cap - 1) * 100:+.2f}%)"
    )
    print(
        f"  {'Combo Final':<32}: TWD {combo_fin:>12,.0f}  ({(combo_fin / args.capital - 1) * 100:+.2f}%)"
    )
    print(f"\n{SEP}\n")

    # ── Chart ─────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = RESULTS_DIR / "tw_combo_equity.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    combo_m = eq_combo / 1_000_000
    swing_m = eq_swing_d / 1_000_000
    orb_m = eq_orb_d / 1_000_000

    ax1.plot(
        eq_combo.index,
        combo_m,
        color="#1a73e8",
        linewidth=2.0,
        label=f"Combo  {total_ret:+.1f}%  DD={max_dd:.1f}%",
    )
    ax1.plot(
        eq_swing_d.index,
        swing_m,
        color="#e8711a",
        linewidth=1.2,
        linestyle="--",
        label="Swing V1 (50%)",
    )
    ax1.plot(
        eq_orb_d.index,
        orb_m,
        color="#16a34a",
        linewidth=1.2,
        linestyle="--",
        label="ORB (50%, synthetic)",
    )
    ax1.axhline(
        y=args.capital / 1_000_000,
        color="#888",
        linestyle=":",
        linewidth=0.8,
        label="Initial capital",
    )
    ax1.set_ylabel("Equity (TWD million)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.22)

    dd_combo = (eq_combo / eq_combo.cummax() - 1.0) * 100.0
    ax2.fill_between(
        dd_combo.index,
        dd_combo,
        0,
        where=dd_combo < 0,
        color="#1a73e8",
        alpha=0.4,
        label="Combo DD",
    )
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_ylim(top=0)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.22)

    import matplotlib.dates as mdates

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        f"TX Combo Portfolio [Swing V1 + ORB]  [{args.start} → {args.end}]\n"
        f"Corr={correlation:.3f}  CAGR={cagr:.2f}%  MaxDD={max_dd:.1f}%",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart → {chart_path}")


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_data(
    product: str,
    start: str,
    end: str,
    no_fetch: bool,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    if no_fetch:
        logger.info("--no-fetch: loading from SQLite cache")
        try:
            df = load_ohlcv(product, start, end, db_path=str(DB_PATH))
        except Exception as exc:
            logger.error("SQLite load failed: %s", exc)
            return None
        if df.empty:
            logger.error("No cached data. Run without --no-fetch first.")
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Taipei")
        elif str(df.index.tz) == "UTC":
            df.index = df.index.tz_convert("Asia/Taipei")
        return df

    logger.info("Fetching %s  [%s → %s] from TAIFEX …", product, start, end)
    fetcher = TaifexFetcher(request_delay=1.0)
    try:
        df = fetcher.fetch_daily(product, start, end)
    except TaifexFetchError as exc:
        logger.error("Fetch failed: %s", exc)
        return None
    if df.empty:
        logger.error("No data returned.")
        return None
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_utc = df.copy()
        df_utc.index = df_utc.index.tz_convert("UTC")
        save_ohlcv_bulk({product: df_utc}, db_path=str(DB_PATH))
        logger.info("Saved %d rows to SQLite.", len(df))
    except Exception as exc:
        logger.warning("SQLite save failed (continuing): %s", exc)
    return df


# ── Print helpers ─────────────────────────────────────────────────────────────


def _print_comparison(
    r1: FuturesBacktestResult,
    r2: FuturesBacktestResult,
    start: str,
    end: str,
    product: str,
    capital: float,
) -> None:
    SEP = "=" * 68
    m1, m2 = r1.metrics, r2.metrics
    print(f"\n{SEP}")
    print(f"  V1 vs V2 Comparison  [{start} → {end}]")
    print(f"  Product: {product}   Capital: TWD {capital:,.0f}")
    print(SEP)
    print(f"  {'Metric':<32} {'V1':>12} {'V2':>12}  {'Δ':>10}")
    print("  " + "-" * 64)

    for k in _METRIC_KEYS:
        v1 = m1.get(k, float("nan"))
        v2 = m2.get(k, float("nan"))
        if isinstance(v1, float) and isinstance(v2, float):
            delta = v2 - v1
            # Metrics where lower is better (show negative delta as good)
            lower_better = k in (
                "Max Drawdown (%)",
                "Avg Loss (TWD)",
                "Max Consecutive Losses",
                "Margin Call Count",
            )
            delta_str = f"{delta:>+10.2f}"
            print(f"  {k:<32} {v1:>12.2f} {v2:>12.2f} {delta_str}")
        else:
            v1i = int(v1) if isinstance(v1, (int, float)) and not isinstance(v1, bool) else v1
            v2i = int(v2) if isinstance(v2, (int, float)) and not isinstance(v2, bool) else v2
            delta_i = (v2i - v1i) if isinstance(v1i, int) and isinstance(v2i, int) else "—"
            delta_s = f"{delta_i:>+10}" if isinstance(delta_i, int) else f"{'—':>10}"
            print(f"  {k:<32} {str(v1i):>12} {str(v2i):>12} {delta_s}")

    print(f"\n{SEP}\n")


def _print_single(
    result: FuturesBacktestResult,
    label: str,
    start: str,
    end: str,
    product: str,
    capital: float,
) -> None:
    SEP = "=" * 60
    m = result.metrics
    print(f"\n{SEP}")
    print(f"  Breakout Swing {label}  [{start} → {end}]")
    print(f"  Product: {product}   Capital: TWD {capital:,.0f}")
    print(SEP)
    for k in _METRIC_KEYS:
        v = m.get(k, "n/a")
        if isinstance(v, float):
            print(f"  {k:<32}: {v:>10.2f}")
        else:
            print(f"  {k:<32}: {v:>10}")
    print(f"\n{SEP}\n")


def _yearly_stats(result: FuturesBacktestResult, year: int, capital: float) -> dict:
    eq = result.equity_curve
    trades_list = result.trades

    mask = eq.index.year == year
    yr_eq = eq[mask]
    if yr_eq.empty:
        return {"ret": float("nan"), "dd": float("nan"), "n": 0, "wr": float("nan")}

    prior = eq[eq.index.year < year]
    base = float(prior.iloc[-1]) if not prior.empty else capital
    ret = (float(yr_eq.iloc[-1]) / base - 1.0) * 100.0
    dd = abs(float((yr_eq / yr_eq.cummax() - 1.0).min())) * 100.0

    if trades_list:
        import pandas as _pd

        tdf = _pd.DataFrame(trades_list)
        ymask = tdf["entry_date"].apply(lambda d: d.year if d is not None else 0) == year
        yt = tdf[ymask]
        n = len(yt)
        wr = (yt["net_pnl"] > 0).sum() / n * 100.0 if n > 0 else 0.0
    else:
        n, wr = 0, 0.0
    return {"ret": ret, "dd": dd, "n": n, "wr": wr}


def _print_comparison_v1v3(
    r1: FuturesBacktestResult,
    r3: FuturesBacktestResult,
    start: str,
    end: str,
    product: str,
    capital: float,
    label1: str = "V1",
    label2: str = "V3",
) -> None:
    SEP = "=" * 68
    m1, m3 = r1.metrics, r3.metrics
    print(f"\n{SEP}")
    print(f"  {label1} vs {label2} Comparison  [{start} → {end}]")
    print(f"  Product: {product}   Capital: TWD {capital:,.0f}")
    print(SEP)
    print(f"  {'Metric':<32} {label1:>12} {label2:>12}  {'Δ':>10}")
    print("  " + "-" * 64)

    for k in _METRIC_KEYS:
        v1 = m1.get(k, float("nan"))
        v3 = m3.get(k, float("nan"))
        if isinstance(v1, float) and isinstance(v3, float):
            delta = v3 - v1
            delta_str = f"{delta:>+10.2f}"
            print(f"  {k:<32} {v1:>12.2f} {v3:>12.2f} {delta_str}")
        else:
            v1i = int(v1) if isinstance(v1, (int, float)) and not isinstance(v1, bool) else v1
            v3i = int(v3) if isinstance(v3, (int, float)) and not isinstance(v3, bool) else v3
            delta_i = (v3i - v1i) if isinstance(v1i, int) and isinstance(v3i, int) else "—"
            delta_s = f"{delta_i:>+10}" if isinstance(delta_i, int) else f"{'—':>10}"
            print(f"  {k:<32} {str(v1i):>12} {str(v3i):>12} {delta_s}")

    print(f"\n{SEP}\n")


def _print_yearly_both_labeled(
    r1: FuturesBacktestResult,
    r2: FuturesBacktestResult,
    label1: str,
    label2: str,
    start: str,
    end: str,
) -> None:
    capital = 2_000_000.0
    SEP = "=" * 80
    print(f"\n{SEP}")
    print("  Yearly Performance")
    print(SEP)
    print(
        f"  {'Year':<10} "
        f"{label1 + ' Ret%':>9} {label1 + ' DD%':>8} {label1 + ' Tr':>7}  "
        f"{label2 + ' Ret%':>9} {label2 + ' DD%':>8} {label2 + ' Tr':>7}  "
        f"{'ΔRet':>7} {'ΔDD':>6}"
    )
    print("  " + "-" * 76)

    start_year = int(start[:4])
    end_year = int(end[:4])

    for year in range(start_year, end_year + 1):
        s1 = _yearly_stats(r1, year, capital)
        s2 = _yearly_stats(r2, year, capital)

        def _fmt(v: float, fmt: str = "+.2f") -> str:
            return f"{v:{fmt}}" if not (v != v) else "  —"

        lbl = f"{year} YTD" if year == end_year else str(year)
        dr = (
            s2["ret"] - s1["ret"]
            if not (s1["ret"] != s1["ret"] or s2["ret"] != s2["ret"])
            else float("nan")
        )
        ddd = (
            s2["dd"] - s1["dd"]
            if not (s1["dd"] != s1["dd"] or s2["dd"] != s2["dd"])
            else float("nan")
        )

        print(
            f"  {lbl:<10} "
            f"{_fmt(s1['ret']):>9} {_fmt(s1['dd'], '.2f'):>8} {s1['n']:>7}  "
            f"{_fmt(s2['ret']):>9} {_fmt(s2['dd'], '.2f'):>8} {s2['n']:>7}  "
            f"{_fmt(dr):>7} {_fmt(ddd, '+.2f'):>6}"
        )

    print("  " + "-" * 76 + "\n")


def _print_yearly_both(
    r1: FuturesBacktestResult,
    r2: FuturesBacktestResult,
    start: str,
    end: str,
) -> None:
    capital = 2_000_000.0  # used only as fallback base for first year
    SEP = "=" * 80
    print(f"\n{SEP}")
    print("  Yearly Performance")
    print(SEP)
    print(
        f"  {'Year':<10} "
        f"{'V1 Ret%':>8} {'V1 DD%':>7} {'V1 Tr':>6}  "
        f"{'V2 Ret%':>8} {'V2 DD%':>7} {'V2 Tr':>6}  "
        f"{'ΔRet':>7} {'ΔDD':>6}"
    )
    print("  " + "-" * 76)

    start_year = int(start[:4])
    end_year = int(end[:4])

    for year in range(start_year, end_year + 1):
        s1 = _yearly_stats(r1, year, capital)
        s2 = _yearly_stats(r2, year, capital)

        def _fmt(v: float, fmt: str = "+.2f") -> str:
            return f"{v:{fmt}}" if not (v != v) else "  —"

        label = f"{year} YTD" if year == end_year else str(year)
        dr = (
            s2["ret"] - s1["ret"]
            if not (s1["ret"] != s1["ret"] or s2["ret"] != s2["ret"])
            else float("nan")
        )
        ddd = (
            s2["dd"] - s1["dd"]
            if not (s1["dd"] != s1["dd"] or s2["dd"] != s2["dd"])
            else float("nan")
        )

        print(
            f"  {label:<10} "
            f"{_fmt(s1['ret']):>8} {_fmt(s1['dd'], '.2f'):>7} {s1['n']:>6}  "
            f"{_fmt(s2['ret']):>8} {_fmt(s2['dd'], '.2f'):>7} {s2['n']:>6}  "
            f"{_fmt(dr):>7} {_fmt(ddd, '+.2f'):>6}"
        )

    print("  " + "-" * 76 + "\n")


def _print_yearly_single(
    result: FuturesBacktestResult, label: str, capital: float = 2_000_000.0
) -> None:
    eq = result.equity_curve
    start_year = eq.index.min().year
    end_year = eq.index.max().year
    SEP = "=" * 56
    print(f"\n{SEP}")
    print(f"  Yearly Performance — {label}")
    print(SEP)
    print(f"  {'Year':<10} {'Return %':>9} {'Max DD %':>9} {'Trades':>7} {'Win%':>7}")
    print("  " + "-" * 48)
    for year in range(start_year, end_year + 1):
        s = _yearly_stats(result, year, capital)
        lbl = f"{year} YTD" if year == end_year else str(year)
        ret_s = f"{s['ret']:>+9.2f}" if s["ret"] == s["ret"] else f"{'—':>9}"
        dd_s = f"{s['dd']:>9.2f}" if s["dd"] == s["dd"] else f"{'—':>9}"
        wr_s = f"{s['wr']:>6.1f}%" if s["wr"] == s["wr"] else f"{'—':>7}"
        print(f"  {lbl:<10} {ret_s} {dd_s} {s['n']:>7} {wr_s}")
    print("  " + "-" * 48 + "\n")


# ── Chart ─────────────────────────────────────────────────────────────────────


def _plot_results(
    results: dict[str, FuturesBacktestResult],
    start: str,
    end: str,
    save_path: Path,
) -> None:
    colors = {"V1": "#1a73e8", "V2": "#e8711a", "V3": "#16a34a", "Scaled": "#9333ea"}

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    labels_info = []
    for label, result in results.items():
        m = result.metrics
        eq = result.equity_curve
        col = colors.get(label, "#333333")

        total_ret = m.get("Total Return (%)", float("nan"))
        sharpe = m.get("Sharpe Ratio", float("nan"))
        max_dd = m.get("Max Drawdown (%)", float("nan"))
        n_trades = m.get("Total Trades", 0)

        eq_m = eq / 1_000_000
        ax1.plot(
            eq.index,
            eq_m,
            color=col,
            linewidth=1.5,
            label=f"{label}  ret={total_ret:+.1f}%  SR={sharpe:.2f}  "
            f"DD={max_dd:.1f}%  n={n_trades}",
        )

        dd = (eq / eq.cummax() - 1.0) * 100.0
        ax2.fill_between(dd.index, dd, 0, where=dd < 0, color=col, alpha=0.35, label=label)
        labels_info.append((label, total_ret, sharpe, max_dd, n_trades))

    # Initial capital reference
    first_eq = next(iter(results.values())).equity_curve
    ax1.axhline(
        y=float(first_eq.iloc[0]) / 1_000_000,
        color="#888",
        linestyle="--",
        linewidth=0.8,
        label="Initial capital",
    )

    title_parts = " vs ".join(f"{lb}({tr:+.1f}%,SR={sr:.2f})" for lb, tr, sr, dd, n in labels_info)
    fig.suptitle(
        f"TX Breakout Swing  [{start} → {end}]\n{title_parts}",
        fontsize=11,
    )

    ax1.set_ylabel("Equity (TWD million)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}M"))
    ax1.grid(True, alpha=0.22)

    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylim(top=0)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.22)

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("run_tw_backtest").info("Chart saved to %s", save_path)


if __name__ == "__main__":
    main()
