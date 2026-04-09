# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`trading-agents` is a monorepo for automated trading systems supporting:
- **美股 (US Equity)** — momentum strategies via Alpaca Paper/Live API
- **台指期 (TW Futures)** — TAIFEX futures via Shioaji (永豐) or Fugle (富果) *(in development)*

Shared infrastructure (risk, monitoring, data, config) lives in `core/`.
Market-specific logic is isolated in `us_equity/` and `tw_futures/`.

## Tech Stack

| Layer | Library |
|---|---|
| Language | Python 3.11+ |
| Backtesting | vectorbt |
| 台指期 Broker | Shioaji (永豐) or Fugle (富果) |
| 美股 Broker | Alpaca |
| Data Storage | SQLite (初期), upgradeable to PostgreSQL |
| Config | pydantic-settings + `.env` |
| Scheduling | APScheduler |
| Dashboard | Streamlit or Rich (terminal) |

## Project Structure

```
trading-agents/
├── core/                          ← Shared infrastructure (market-agnostic)
│   ├── config/
│   │   └── settings.py            ← pydantic-settings, loads from .env
│   ├── risk/
│   │   ├── base.py                ← RiskAction enum, TradingHaltedError
│   │   ├── kill_switch.py         ← Hard trading halt on loss thresholds
│   │   ├── drawdown.py            ← HWM drawdown guard (HOLD/REDUCE/EXIT)
│   │   ├── anomaly.py             ← Z-score, run, volatility-spike detection
│   │   ├── concentration.py       ← Sector/position concentration limits
│   │   ├── position_sizing.py     ← Kelly, volatility-target sizing
│   │   └── slippage.py            ← Fixed + linear-impact slippage models
│   ├── monitor/
│   │   ├── pnl.py                 ← HWM, drawdown, cumulative return tracking
│   │   ├── reporter.py            ← Daily TXT report generation
│   │   ├── notifier.py            ← SMTP email alerts
│   │   ├── monitor.py             ← Integrated daily monitoring pipeline
│   │   └── anomaly.py             ← Re-export shim → core.risk.anomaly
│   └── data/
│       ├── models.py              ← SQLAlchemy table definitions (source of truth)
│       ├── database.py            ← SQLAlchemy session management
│       ├── store.py               ← SQLite OHLCV read/write
│       └── cleaner.py             ← ffill, zero-volume filter, OHLCV validation
│
├── us_equity/                     ← US equity trading system (production-ready)
│   ├── orchestrator.py            ← TradingOrchestrator: daily check + quarterly rebalance
│   ├── strategies/
│   │   ├── dual_momentum.py       ← Absolute + relative momentum (SP500_TOP50)
│   │   ├── multi_factor.py        ← Momentum + quality + low-vol factors
│   │   └── momentum_low_turnover.py ← Low-churn momentum for quarterly rebalance
│   ├── executor/
│   │   ├── alpaca_adapter.py      ← Alpaca-py SDK wrapper with retry logic
│   │   ├── order_manager.py       ← Sell-before-buy rebalance (weight-based)
│   │   └── reconciler.py          ← Position drift detection (pp-based)
│   ├── backtester/
│   │   ├── backtester.py          ← Vectorbt-style backtest engine
│   │   ├── walk_forward.py        ← Rolling OOS validation + WalkForwardWindow
│   │   ├── bias_check.py          ← Look-ahead bias assertion
│   │   └── metrics.py             ← OOS metric aggregation + pass/fail threshold
│   ├── data/
│   │   └── fetcher.py             ← AlpacaFetcher: historical bars via alpaca-py
│   └── risk/
│       └── risk_manager.py        ← RiskManager: integrates KillSwitch + DrawdownGuard
│
├── tw_futures/                    ← TW Futures system (skeletons — in development)
│   ├── orchestrator.py            ← TaifexOrchestrator: pre/intraday/post-market sessions
│   ├── strategies/
│   │   ├── intraday/              ← 當沖 (day-trading) strategies
│   │   └── swing/                 ← 波段 (swing) strategies
│   ├── executor/
│   │   ├── shioaji_adapter.py     ← 永豐 Shioaji API adapter
│   │   ├── fugle_adapter.py       ← 富果 Fugle API adapter
│   │   ├── order_manager.py       ← Futures order manager (integer contracts)
│   │   └── reconciler.py          ← Futures position reconciler
│   ├── risk/
│   │   ├── margin_manager.py      ← TAIFEX margin tracking + margin-call alerts
│   │   ├── concentration.py       ← Per-product contract limits (TX/MTX)
│   │   └── slippage.py            ← Tick-based slippage model (TWD per tick)
│   ├── data/
│   │   ├── fetcher.py             ← TaifexFetcher: daily OHLCV + minute bars
│   │   └── rollover.py            ← Auto rollover on third-Wednesday expiry
│   └── backtester/
│       └── backtester.py          ← FuturesBacktester: margin-aware simulation
│
├── agents/                        ← Backward-compat shims (re-export from core/ + us_equity/)
├── config/                        ← Backward-compat shim → core.config.settings
├── shared/                        ← Backward-compat shim → core.data
├── data/                          ← Backward-compat shims + SQLite DB files
│   └── db/                        ← trading.db, market_data.db
├── scripts/
│   ├── run_backtest.py            ← Walk-forward backtest CLI
│   ├── run_live.py                ← Live trading CLI (--mode rebalance|daily|auto)
│   └── test_fetch.py              ← Market data sanity check
├── tests/                         ← 119 tests (all green)
└── results/                       ← Saved chart images
```

## Import Paths (post-refactor)

| What | Import from |
|------|-------------|
| Settings | `core.config.settings` |
| KillSwitch, DrawdownGuard | `core.risk` |
| AnomalyDetector | `core.risk.anomaly` or `core.monitor.anomaly` |
| Monitor, PnLTracker | `core.monitor` |
| SQLite store | `core.data.store` |
| AlpacaFetcher | `us_equity.data.fetcher` |
| US strategies | `us_equity.strategies` |
| AlpacaAdapter | `us_equity.executor.alpaca_adapter` |
| RiskManager | `us_equity.risk.risk_manager` |
| Backtester, WalkForwardValidator | `us_equity.backtester` |
| TradingOrchestrator | `us_equity.orchestrator` |

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run live trading
python scripts/run_live.py --mode auto          # auto: rebalance or daily
python scripts/run_live.py --mode rebalance     # quarterly rebalance
python scripts/run_live.py --mode daily         # daily PnL check

# Run backtest
python scripts/run_backtest.py --walk-forward
python scripts/run_backtest.py --walk-forward --slippage dynamic

# Run tests
pytest

# Lint
ruff check .
ruff format .
```

## V2b Strategy Backtest Baseline (2020-01-02 → 2026-04-08, MTX/MXF, capital=350K)

Data: `data/MXF_Daily_Clean_2020_to_now.parquet` (1518 bars, fetched from TAIFEX public portal)
Source: `TaifexFetcher.fetch_daily("MTX", ...)` — weekday-only, no weekend contamination
Engine: `src/backtest/engine.py` `BacktestEngine` (fixed — MTM equity, "add" action, cost=160/side)
Ground truth: `scripts/verify_engine.py` — matches engine.py exactly on same data

| Metric        | next_day_open [A] | night_open (same_day_close) [B] | Δ (B−A) |
|---------------|:-----------------:|:-------------------------------:|:-------:|
| CAGR_%        | 58.36%            | 57.14%                          | -1.22   |
| MDD_%         | -35.20%           | **-20.86%**                     | +14.3   |
| Sharpe        | 1.450             | **1.637**                       | +0.19   |
| Calmar        | 1.658             | **2.739**                       | +1.08   |
| Win_Rate_%    | 58.89%            | 52.75%                          | -6.14   |
| Profit_Factor | 2.428             | 2.210                           | -0.22   |
| Total_Trades  | 90                | 91                              | +1      |
| Final_Equity  | 6,223,920 NTD     | 5,929,930 NTD                   | -294K   |

**Verdict**: night_open dominates on risk-adjusted metrics — Calmar 2.74 vs 1.66 (+65%), MDD 20.86% vs 35.20% (−40% drawdown). CAGR slightly lower but Sharpe higher. Use `exec_timing="same_day_close"` in production.

**Slippage robustness** (next_day_open A vs cost=250 C): CAGR Δ = -0.63%, Calmar Δ = -0.10. Strategy is cost-insensitive.

**Bug fix history** (old engine had inflated Calmar=5.53, CAGR=70.98%):
1. MDD bug: old engine recorded only realized equity → unrealized drawdowns invisible → MDD -12.83% (fake). Fixed: record MTM equity (realized + unrealized) at each bar close.
2. Pyramid bug: "add" action from V2bEngine silently fell through if/elif — pyramid never executed.
3. Cost bug: old engine used COST_PER_SIDE=105; canonical value is 160 NTD/side.
4. Margin bug: pyramid add_n and initial contracts not capped by floor(equity/119_250) → 2 trades exceeded margin in 2020. Fixed: both buy and pyramid paths capped in v2b_engine.py.
5. Data bug: old TXF_Daily_Clean.parquet had 293 weekend bars (Friday night session labeled as Saturday). Fixed: use TaifexFetcher.fetch_daily("MTX") → weekday-only TAIFEX official data.

## Key Conventions

- **Import from canonical locations**: use `core.*`, `us_equity.*`, `tw_futures.*` — not `agents.*`.
  The `agents/`, `config/`, `shared/`, `data/` packages are backward-compat shims only.
- **Backtester must `shift(1)` all signals** to prevent look-ahead bias.
- **`core/data/models.py`** is the single source of truth for SQLAlchemy schema.
- **All broker credentials go in `.env`** — never hardcode keys.
- **Sell before buy** in all order managers to prevent cash shortfalls.
- **tw_futures/** skeletons: all methods raise `NotImplementedError` until implemented.
