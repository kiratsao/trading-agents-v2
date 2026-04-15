# V2b 大改版升級指南 (2026-04-16)

## 改了什麼

- 完全重構：97 → 35 個 .py 檔案，刪除所有 dead code
- daily_updater 重寫：失敗會 LINE 告警，不再靜默
- 策略升級：新增 ADX(14)>25 濾網 + CD2
- parquet 資料不再由 git 管理（避免 git pull 覆蓋每日更新）
- Git history 已 squash 成單一 commit（orphan branch）

## 升級步驟

### 1. 備份你的設定

```bash
cd ~/trading-agents-v2
cp .env /tmp/env_backup
cp data/paper_state.json /tmp/state_backup.json 2>/dev/null
```

### 2. 重新 clone（因為 orphan branch，git pull 不行）

```bash
cd ~
mv trading-agents-v2 trading-agents-v2-old
git clone https://github.com/kiratsao/trading-agents-v2.git
cd trading-agents-v2
```

### 3. 還原你的設定

```bash
cp /tmp/env_backup .env
cp /tmp/state_backup.json data/paper_state.json 2>/dev/null
```

### 4. 安裝依賴

```bash
pip install -e .
pip install pyyaml  # 必需
```

### 5. 初始化資料（首次必做）

```bash
python scripts/init_data.py
```

這會從 Shioaji 抓取 2020 至今的日K資料。需要 .env 中有 Shioaji API Key。

### 6. 驗證

```bash
# 確認所有 import 正確
python -c "from src.scheduler.main import main; print('✅ imports OK')"

# 確認資料
python scripts/verify_data.py

# 確認策略信號
python -m src.scheduler.main --run-once
```

### 7. 啟動

```bash
# Paper trading（模擬）
python -m src.scheduler.main

# Live trading（實盤）— 確認 .env 有完整的 cert 設定
python -m src.scheduler.main --live

# 或用 systemd
sudo cp scripts/trading-agents-v2.service /etc/systemd/system/
# 編輯 service：確認路徑和 User 正確，ExecStart 加 --live（如需實盤）
sudo systemctl daemon-reload
sudo systemctl enable trading-agents-v2
sudo systemctl start trading-agents-v2
```

### 8. 確認正常運行

LINE 應收到啟動通知。每日排程：
- 14:25 — 資料更新（✅ 成功 / 🔴 失敗）
- 14:30 — 信號計算 + LINE 通知
- 15:05 — 夜盤下單

## .env 範本

```
SHIOAJI_API_KEY=你的KEY
SHIOAJI_SECRET_KEY=你的SECRET
SHIOAJI_CERT_PATH=/path/to/Sinopac.pfx
SHIOAJI_CERT_PASSWORD=你的密碼
SHIOAJI_PERSON_ID=你的身分證字號
LINE_CHANNEL_ACCESS_TOKEN=你的TOKEN
LINE_USER_ID=你的USER_ID
SIMULATION=true
```

## 常見問題

**Q: git pull 報錯 divergent branches**
A: 這次是 orphan branch，不能 pull。請重新 clone。

**Q: ModuleNotFoundError: pyyaml**
A: `pip install pyyaml`

**Q: 資料更新失敗**
A: 確認 .env 的 Shioaji 設定完整，跑 `python scripts/init_data.py` 重建資料。
