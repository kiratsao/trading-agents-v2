# CLAUDE.md

## Project Overview

`trading-agents-v2` — 台指期 (TW Futures) 自動交易系統。

V2b 策略: EMA(30/100) 金叉 + CD2 確認 + ADX(14)>25 進場濾網 + ATR×2 追蹤止損 + 反馬丁格爾口數。

## Project Structure

```
trading-agents-v2/
├── src/
│   ├── scheduler/
│   │   ├── main.py              ← 排程入口 (daemon + --run-once)
│   │   └── orchestrator.py      ← V2bOrchestrator: signal → execution → LINE
│   ├── strategy/
│   │   ├── v2b_engine.py        ← V2b 策略引擎 (generate_signal)
│   │   └── common/indicators.py ← EMA, ATR, ADX
│   ├── backtest/engine.py       ← BacktestEngine (MTM equity, settlement rollover)
│   ├── data/daily_updater.py    ← 14:25 Shioaji 日K 自動更新
│   ├── signals/
│   │   ├── fetcher.py           ← TSM ADR + SOX 價格抓取
│   │   └── tsmc_tracker.py      ← TSMC 夜盤方向信號
│   └── state/state_manager.py   ← 持倉狀態 JSON 持久化
├── tw_futures/executor/
│   └── shioaji_adapter.py       ← 永豐 Shioaji API adapter
├── config/accounts.yaml         ← 帳戶設定 (strategy params, ladder, sessions)
├── scripts/
│   ├── init_data.py             ← 首次部署: 重建完整日K parquet
│   ├── verify_data.py           ← parquet 健康檢查
│   └── verify_engine.py         ← 回測引擎交叉驗證
├── tests/                       ← 31 tests
└── data/
    └── MXF_Daily_Clean_2020_to_now.parquet  (not tracked by git)
```

## Development Commands

```bash
# Tests + lint
python -m pytest tests/ -x -q
ruff check .

# Backtest
python -c "
from src.backtest.engine import BacktestEngine
from src.strategy.v2b_engine import V2bEngine
import pandas as pd
df = pd.read_parquet('data/MXF_Daily_Clean_2020_to_now.parquet')
r = BacktestEngine(strategy=V2bEngine(product='MXF', ema_fast=30, ema_slow=100,
    confirm_days=2, adx_threshold=25), initial_capital=350_000,
    exec_timing='same_day_close').run(df)
print(r.metrics)
"

# Live (daemon)
python -m src.scheduler.main --live

# Dry run
python -m src.scheduler.main --run-once

# First-time data setup
python scripts/init_data.py

# Data health check
python scripts/verify_data.py
```

## V2b Backtest Baseline (2020-01-02 → 2026-04-08, MXF, capital=350K, with settlement rollover)

Strategy: EMA(30/100), CD2, ATR×2.0 trailing, ADX(14)>25, Anti-Martingale, settlement rollover

| Metric | Value |
|--------|------:|
| CAGR_% | **57.80%** |
| MDD_% | **-21.54%** |
| Sharpe | **1.768** |
| Calmar | **2.684** |
| Win_Rate_% | **64.29%** |
| Profit_Factor | **3.561** |
| Total_Trades | 70 |
| Final_Equity | 6,089,660 NTD |

## Key Conventions

- **Parquet not in git** — managed by daily_updater (14:25 cron) and init_data.py
- **All Shioaji connections** must include cert_path, cert_password, person_id from .env
- **daily_updater failures must be visible** — notify_fn sends LINE alert on 🔴
- **Settlement day** only forces close on existing positions; empty positions can enter normally
- **日K聚合** strictly 08:45-13:44, sort by ts, close = last chronological bar
