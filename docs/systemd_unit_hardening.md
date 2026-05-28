# systemd Unit Hardening — trading-agents-v2.service

> Deployment doc. **Not auto-applied.** Kira applies these manually on the GCP
> e2-small box, then `sudo systemctl daemon-reload && sudo systemctl restart trading-agents-v2`.

## Why this exists (post-mortem)

The daemon crash-looped **2,769 times over 2 days** (2026-05-26 11:32 → 05-28)
with a one-line `TypeError` on startup, and **nobody noticed for 6 days** of
health-check warnings. Two unit-level defects turned a trivial bug into a
silent multi-day outage:

1. **Traceback was invisible to `journalctl`.** The unit routed
   `StandardError=append:.../logs/paper_trading.log`, so
   `journalctl -u trading-agents-v2` showed only `status=1/FAILURE` with no
   stack trace. Every diagnosis attempt that started with `journalctl` came up
   empty and the real cause stayed hidden.
2. **Unbounded restart loop.** `Restart=on-failure` + `RestartSec=60` with no
   `StartLimit*` meant systemd restarted forever (2,769×) instead of entering a
   visible `failed` state. A crash-looping unit and a healthy-but-idle unit look
   the same on a dashboard that only checks `active`.

## Required changes

### 1. Make tracebacks visible
Send stderr to journald (keep a file copy if desired, but journald is the
first place anyone looks):
```ini
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-agents-v2
```
`PYTHONUNBUFFERED=1` ensures Python flushes before the process dies.

### 2. Bound the restart loop so failure becomes audible
```ini
Restart=on-failure
RestartSec=60
StartLimitIntervalSec=600
StartLimitBurst=5
```
→ After 5 failures within 600s the unit enters `failed` (stops trying) instead
of looping silently. A crash on startup now surfaces within ~10 minutes.

> Note: on systemd ≥ 230 `StartLimitIntervalSec`/`StartLimitBurst` belong in the
> **`[Unit]`** section (not `[Service]`). Placing them in `[Service]` is silently
> ignored — which is itself a common cause of "the StartLimit didn't work".

### 3. Alert on failure
Add an `OnFailure=` hook so entering `failed` pushes a notification:
```ini
[Service]
OnFailure=trading-agents-notify@%n.service
```
Minimal notify unit (`/etc/systemd/system/trading-agents-notify@.service`),
oneshot that pushes a LINE message:
```ini
[Unit]
Description=Notify on failure of %i

[Service]
Type=oneshot
EnvironmentFile=/home/tommychau286/trading-agents-v2/.env
ExecStart=/usr/bin/python3 /home/tommychau286/trading-agents-v2/scripts/notify_unit_failure.py %i
```
(Alternatively `ExecStopPost=` on the main unit, but `OnFailure=` only fires on
the failure path and keeps the notification logic out of the trading process.)

## Full hardened unit template

```ini
# /etc/systemd/system/trading-agents-v2.service
[Unit]
Description=V2b MTX Paper Trading Daemon
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User=tommychau286
WorkingDirectory=/home/tommychau286/trading-agents-v2
EnvironmentFile=/home/tommychau286/trading-agents-v2/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -m src.scheduler.main --live
Restart=on-failure
RestartSec=60
OnFailure=trading-agents-notify@%n.service

# Tracebacks to journald — the FIRST place to look (this outage's root blind spot)
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-agents-v2

[Install]
WantedBy=multi-user.target
```

## Apply + verify
```bash
sudo cp trading-agents-v2.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart trading-agents-v2
systemctl status trading-agents-v2 --no-pager
journalctl -u trading-agents-v2 -n 50 --no-pager   # tracebacks now visible here
```

## Independent watchdog (recommended, separate from this fix)

The data update lives **inside** this daemon (14:25 APScheduler job), so when the
daemon dies the parquet also freezes. Decoupling the daily parquet update into a
standalone cron/timer with a non-zero exit code on failure is tracked in the
follow-up freshness PR — that way a daemon crash no longer silently stops data
updates too.
```
