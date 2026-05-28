# Daemon-independent parquet updater — systemd timer

> Deployment doc. **Not auto-applied.** Kira installs these on the GCP box.

## Why

The daily parquet update used to live **only** inside the trading daemon's
14:25 APScheduler job. When the daemon crash-looped (StateManager phantom
kwarg), the parquet froze for 2 days — data and trading died together. This
timer runs `src.data.daily_updater_cli` **independently** of the daemon, so a
daemon outage no longer stops data updates. Exit codes drive systemd:

| code | meaning            | systemd reaction          |
|------|--------------------|---------------------------|
| 0    | ok / noop          | success                   |
| 1    | partial            | `OnFailure=` → alert       |
| 2    | DataIntegrityError | `OnFailure=` → alert       |

## Files

`/etc/systemd/system/trading-agents-v2-updater.service`
```ini
[Unit]
Description=V2b daily parquet updater (daemon-independent)
After=network-online.target
Wants=network-online.target
OnFailure=trading-agents-notify@%n.service

[Service]
Type=oneshot
User=tommychau286
WorkingDirectory=/home/tommychau286/trading-agents-v2
EnvironmentFile=/home/tommychau286/trading-agents-v2/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -m src.data.daily_updater_cli
StandardOutput=journal
StandardError=journal
SyslogIdentifier=v2-updater
```

`/etc/systemd/system/trading-agents-v2-updater.timer`
```ini
[Unit]
Description=Run V2b parquet updater Mon-Fri 14:30 Taipei

[Timer]
# Box TZ must be Asia/Taipei (verify: timedatectl). Otherwise set OnCalendar in
# the box's local time that corresponds to 14:30 Taipei.
OnCalendar=Mon..Fri 14:30
Persistent=true

[Install]
WantedBy=timers.target
```

`Persistent=true` → if the box was off at 14:30, the update runs on next boot
(catch-up). The updater is declarative (`ensure_parquet_fresh`), so a late or
repeated run is a safe no-op when already fresh.

## Install + verify
```bash
sudo cp trading-agents-v2-updater.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-agents-v2-updater.timer
systemctl list-timers trading-agents-v2-updater.timer --no-pager
# Manual one-shot (dry-run shows freshness without writing):
python3 -m src.data.daily_updater_cli --dry-run
python3 -m src.data.daily_updater_cli          # real update
echo "exit=$?"   # 0 ok/noop, 1 partial, 2 integrity error
```

## Relationship to the daemon

- The daemon's in-process 14:25 job is now a **secondary** path; this timer is
  the authoritative one. Both call the same `validate_latest_bar` cross-check.
- Keep the daemon's 14:28 warmup / 14:30 signal / 15:05 execution jobs as-is —
  those need the live broker session and stay in the daemon.
