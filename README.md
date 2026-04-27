# trading-agents-v2

台指期 (MTX) 自動交易系統 — EMA Trend Following + Night Session Execution

## 回測績效（2020-2026，三方獨立驗證）

| 模式 | CAGR | MDD | Calmar | Sharpe |
|------|:----:|:---:|:------:|:------:|
| Night Open (production) | +57% | -21% | 2.74 | 1.64 |
| Next Day Open | +58% | -35% | 1.66 | 1.45 |

資料來源：TAIFEX 官方日K（MTX 小台指，1518 bars，2020-01-02 → 2026-04-08），零週末污染。
驗證：`BacktestEngine` + `verify_engine.py` 獨立 for-loop 雙引擎結果完全一致。

---

## 前置準備

### 1. 永豐金證券 Shioaji API
- 開立永豐金期貨帳戶（需存入 ≥ 350,000 NTD）
- 申請 Shioaji API：<https://sinotrade.github.io/>
- 取得：API Key, Secret Key
- 下載 CA 憑證 (.pfx) 並線上簽署下單同意書

### 2. LINE Messaging API（選用，建議開啟）
- 建立 LINE Bot：<https://developers.line.biz/>
- 建立 Messaging API Channel
- 取得：Channel Access Token, Channel Secret, 自己的 User ID（U 開頭）

### 3. 伺服器
- Ubuntu 22.04+ 或 macOS（本機測試）
- Python 3.11+
- 可連外網（Shioaji API + LINE API）

---

## 安裝

```bash
git clone https://github.com/kiratsao/trading-agents-v2.git
cd trading-agents-v2
pip install -e .
```

---

## 設定

```bash
cp .env.example .env
nano .env
```

`.env` 必填欄位：

```dotenv
# Shioaji
SHIOAJI_API_KEY=你的API_KEY
SHIOAJI_SECRET_KEY=你的SECRET_KEY
SHIOAJI_SIMULATION=true          # 先跑模擬，確認OK後改 false

# 實盤才需要（SHIOAJI_SIMULATION=false 時）
SHIOAJI_CERT_PATH=/path/to/Sinopac.pfx
SHIOAJI_CERT_PASSWORD=憑證密碼

# LINE（選用）
LINE_CHANNEL_ACCESS_TOKEN=你的TOKEN
LINE_USER_ID=你的USER_ID（U開頭）
```

---

## 驗證連線

```bash
# 測試 Shioaji 連線 + 合約 + 報價 + LINE 通知
python scripts/test_connection.py

# 跑一次完整決策流程（不下單）
python -m src.scheduler.main --run-once
```

確認輸出中：
- `✓ Shioaji OK`（登入成功，MXF 報價正常）
- `✓ LINE push sent`（收到 LINE 訊息）
- `Data:` 顯示資料筆數與日期範圍

---

## 啟動

### Paper Trading（模擬，SHIOAJI_SIMULATION=true）

```bash
# 前景執行（Ctrl+C 停止）
python -m src.scheduler.main

# 背景執行
nohup python -m src.scheduler.main > logs/trading.log 2>&1 &
```

### Live Trading（實盤）

```bash
# 1. 切換為實盤模式
nano .env   # 將 SHIOAJI_SIMULATION=true 改為 false，填入憑證路徑

# 2. 啟動
python -m src.scheduler.main --live
```

### 用 systemd 管理（推薦，GCP/伺服器）

```bash
sudo cp scripts/trading-agents-v2.service /etc/systemd/system/
```

編輯 service 檔，確認以下欄位：

```ini
User=你的使用者名稱
WorkingDirectory=/home/你的使用者名稱/trading-agents-v2
ExecStart=/home/你的使用者名稱/trading-agents-v2/.venv/bin/python \
    -m src.scheduler.main --live
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-agents-v2
sudo systemctl start trading-agents-v2
```

### 查看狀態與日誌

```bash
sudo systemctl status trading-agents-v2
journalctl -u trading-agents-v2 -f
tail -f logs/trading.log
```

### 停止

```bash
sudo systemctl stop trading-agents-v2
# 或（背景執行時）
bash scripts/stop_paper.sh
```

---

## 排程

| 時間 | 動作 |
|------|------|
| 系統啟動 | LINE 推播：啟動通知 + 台指即時報價 + 持倉狀態 |
| 14:30（週一至五） | 計算信號 + LINE 推播決策方向 |
| 15:05（週一至五） | 夜盤開盤後執行下單（如有買賣信號） |
| 每月第三個週三 | 結算日強制平倉 |

---

## 回測

```bash
# 標準回測（night_open，production 模式）
python -m src.backtest.runner

# 指定執行模式
python -m src.backtest.runner --timing next_day_open

# A/B 對比
python -m src.backtest.runner --ab

# 含年度分拆
python -m src.backtest.runner --yearly

# 獨立 ground-truth 驗證（zero dependency on engine.py）
python scripts/verify_engine.py
python scripts/verify_engine.py --exec_timing same_day_close
```

---

## 策略概要

| 參數 | 值 |
|------|-----|
| 商品 | MTX / MXF（小台指） |
| 方向 | Long-Only |
| 進場 | EMA(30) > EMA(100) 連續 3 日確認 |
| 出場 | Trailing Stop: highest_high − 2×ATR(14)；EMA 死叉；結算日 |
| 口數 | Anti-Martingale: 35萬→2口, 48萬→3口, 60萬→4口（含保證金安全上限） |
| 加碼 | 浮盈 ≥ 1×ATR 時加碼 0.5 倍（最多一次） |
| 信號 | 14:30 計算（日盤資料） |
| 執行 | 15:05 夜盤開盤下單 |

---

## 主要檔案

```
src/
├── scheduler/
│   ├── main.py            ← 入口：APScheduler daemon，14:30 信號 + 15:05 下單
│   └── orchestrator.py    ← V2bOrchestrator：run_signal() + run_execution()
├── strategy/
│   └── v2b_engine.py      ← V2bEngine：EMA/ATR 指標、信號生成、Anti-Martingale
├── backtest/
│   ├── engine.py          ← BacktestEngine：MTM equity、margin cap、"add" 支援
│   └── runner.py          ← CLI 回測入口（--ab / --yearly / --timing）
├── state/
│   └── state_manager.py   ← 持倉/equity 狀態持久化（JSON）
└── risk/
    └── night_guard.py     ← 夜盤風控：Guard1/2/3 止損檢查（05:15 執行）

tw_futures/
├── executor/
│   └── shioaji_adapter.py ← ShioajiAdapter：login / get_contract / place_order
└── data/
    └── fetcher.py         ← TaifexFetcher：TAIFEX 公開日K抓取

scripts/
├── test_connection.py     ← 連線測試（Shioaji + LINE）
├── verify_engine.py       ← Ground-truth 回測（pure for-loop）
└── trading-agents-v2.service ← systemd service 範本

data/
└── MXF_Daily_Clean_2020_to_now.parquet  ← 官方日K（1518 bars，無週末污染）
```

---

## 注意事項

- **永遠不要在程式碼中 hardcode API Key**；`.env` 已列入 `.gitignore`
- 首次上線建議先跑 `SHIOAJI_SIMULATION=true` 觀察 2-3 個交易日
- 結算日（每月第三個週三）系統自動平倉，無需人工介入
- systemd 設定 `Restart=on-failure`，程式 crash 後 30 秒自動重啟
- 保證金安全上限：口數 × 131,500 NTD ≤ 帳戶淨值（由策略引擎自動限制，TAIFEX 2026-04-27 調整）
