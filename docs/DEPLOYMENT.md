# Deployment Procedure

## 本機開發完成後

```bash
# 1. 測試
pytest -x -q

# 2. Lint
ruff check .

# 3. Smoke test (不需外部連線)
python scripts/production_smoke_test.py --local

# 4. 全部通過才 push
git add -A && git commit && git push origin main
```

## GCP 部署

```bash
# 1. 備份
cp .env /tmp/env_backup
cp data/paper_state.json /tmp/state_backup.json 2>/dev/null

# 2. 拉取最新 code
cd ~/trading-agents-v2
git fetch origin
git reset --hard origin/main

# 3. 還原設定
cp /tmp/env_backup .env
cp /tmp/state_backup.json data/paper_state.json 2>/dev/null

# 4. Smoke test (完整版，含 Shioaji + LINE)
python scripts/production_smoke_test.py

# 5. 確認 🟢 PRODUCTION READY 後重啟
sudo systemctl restart trading-agents-v2

# 6. 等待 LINE 啟動通知
# 7. 檢查 log
journalctl -u trading-agents-v2 --since "1 min ago" --no-pager
```

## Rollback

```bash
# 查看歷史
git log --oneline

# 回滾
git reset --hard <commit_hash>
sudo systemctl restart trading-agents-v2
```

## Daily Health Check

```bash
# 加到 cron（每天 08:00 週一到週五）
crontab -e
# 加入：
# 0 8 * * 1-5 cd ~/trading-agents-v2 && python scripts/daily_health_check.py
```

## 排程時間表

| 時間 | 任務 | 說明 |
|------|------|------|
| 08:00 | daily_health_check.py | 檢查 parquet + state 健康 |
| 14:25 | daily_updater | 更新日K parquet |
| 14:30 | run_signal | 計算信號 + LINE 通知 |
| 15:05 | run_execution | 夜盤下單 |
