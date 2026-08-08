---
name: run-trading-agents-v2
description: Run, test, and drive trading-agents-v2 — check the current signal, run a backtest, verify parquet data health, or run the test suite. Use when asked to run the app, check what the strategy would do, or confirm a change works.
---

台指期 (TW futures) live-trading daemon. Agents drive it via
`.claude/skills/run-trading-agents-v2/driver.py` — pure computation on the
parquet, no broker connection, no orders, no LINE messages. All paths are
relative to the repo root; the interpreter is always `.venv/bin/python`.

**⚠️ NEVER run `python -m src.scheduler.main --run-once` to "check the
signal" — it is NOT a dry run.** `run_signal`/`run_execution` submit real
orders when broker credentials resolve. The production daemon runs on a GCP
VM under systemd; this repo clone is the dev machine. Signal verification is
always done with pure `generate_signal` — which is what the driver does.

## Prerequisites

- Pre-existing `.venv` (Python 3.11.11) at repo root. **There is no
  `pyproject.toml` / `requirements.txt` in the repo** — the venv is the only
  record of the dependency set (pandas, pyarrow, shioaji, apscheduler,
  pyyaml, yfinance, pytest, ruff). Don't delete it.
- `data/MXF_Daily_Clean_2020_to_now.parquet` (not in git). If missing,
  `python scripts/init_data.py` rebuilds it but needs Shioaji credentials —
  not runnable without them (unverified here).

## Run (agent path)

Everything goes through the driver:

```bash
.venv/bin/python .claude/skills/run-trading-agents-v2/driver.py smoke
```

Runs `signal` + `backtest` + `verify` in sequence; exit 0 = healthy. Verified
output (dev clone, parquet → 2026-07-03):

```
=== signal ===
data: 1578 bars 2020-01-02 → 2026-07-03 (signal is based on the LAST bar date — check it is what you expect)
indicators@2026-07-03: close=46,985 EMA30=45,018.0 EMA100=39,729.9 ATR=1,358.2 ADX=14.97
signal(position=0, equity=880,000): hold 0口 — ADX too low — no trend (ADX=15.0 < 25)
=== signal: OK ===
=== backtest ===
  CAGR_%: 59.64 / MDD_%: -21.85 / Sharpe: 1.76 / Total_Trades: 77 ...
=== backtest: OK ===
=== verify ===
  ISSUES (1): - STALE: ...
WARN: STALE only — expected on a dev clone; use --strict to fail on it
=== verify: OK ===
```

| command | what it does |
|---|---|
| `signal` | The daemon's 14:30 decision, computed exactly as `src/scheduler/main.py` wires it (engine built from `config/accounts.yaml` account `mxf_aggressive`), via pure `generate_signal`. Prints indicators + signal. |
| `backtest` | Full `BacktestEngine` baseline (EMA30/100, CD2, ADX>25, 350K, `same_day_close`) on the repo parquet. Fails if <10 trades. |
| `verify` | `scripts/verify_data.py` parquet health. STALE-only → warn (dev clones are always stale; updates run on GCP at 14:25). `--strict` restores hard-fail. |
| `smoke` | All three. |

Options (all subcommands): `--parquet PATH` overrides the data file (e.g. a
scratch copy with newer bars), `--equity N` and `--position N` set the
sizing inputs for `signal`. Verified example — signal on a scratch parquet at
live equity:

```bash
.venv/bin/python .claude/skills/run-trading-agents-v2/driver.py signal \
  --parquet /path/to/scratch.parquet --equity 650708
# → indicators@2026-07-29: close=40,387 ... ADX=29.79
# → signal(position=0, equity=650,708): buy 4口 — golden cross + 267-day confirmation...
```

Backtest with different params: edit nothing — use the one-liner from
`CLAUDE.md` (Development Commands) with `.venv/bin/python`.

## Deploy (GCP production)

The daemon runs on a GCP VM (system `python3`, no venv) under systemd. The
operator deploys; an agent's job is to hand over this exact sequence.
**Never leave the config in a half-updated state**: back up first, then chain
`checkout` and `pull` with `&&` so `accounts.yaml` cannot sit at stale values
between the two, then assert the live values immediately.

```bash
cd ~/trading-agents-v2
git diff config/accounts.yaml                      # inspect local drift FIRST
cp config/accounts.yaml /tmp/accounts.yaml.bak     # keep the running config
git checkout -- config/accounts.yaml && git pull   # chained — no gap
grep -E "risk_cap_pct|margin_buffer_atr|max_contracts" config/accounts.yaml
find . -name __pycache__ -type d -exec rm -rf {} + # stale bytecode caused a
                                                   # real incident — never skip
python3 -m pytest tests/ -q                        # sanity before restart
sudo systemctl restart trading-agents-v2           # AVOID the 14:10–15:10 window
sleep 120 && systemctl is-active trading-agents-v2 # a fast restart once
                                                   # ABRT'd the Shioaji C ext;
                                                   # systemd recovered in 60s
```

Timers pick up new code on their next fire without a restart (fresh process);
only the long-lived daemon needs one.

## Run (human path — production only)

`python -m src.scheduler.main --live` starts the APScheduler daemon
(14:25 data / 14:30 signal / 15:05 execution, Asia/Taipei). Operator-only,
on the GCP VM with `.env` credentials. **Do not start it on the dev Mac** —
it takes a `data/daemon.lock` flock and, with credentials, trades.

## Test

```bash
.venv/bin/python -m pytest tests/ -q   # 260 passed, 1 failed (known, see below)
.venv/bin/ruff check .                  # clean
```

Known failure: `test_regressions.py::TestRegressionSettlementRolloverContract::test_rollover_uses_next_month`
— pre-existing, date-fragile (mock uses the real current date; expected
contract month drifts, e.g. `MXFG6` vs `MXFE6`). Not caused by your change
if it's the only failure.

## Gotchas

- **`--run-once` submits orders.** Repeated because it's the trap: it looks
  like a dry-run flag and is documented as "useful for cron/dry-run" in the
  module docstring. It is a full signal+execution cycle.
- **The signal is only as fresh as the last parquet bar.** `signal` prints
  the last bar date for exactly this reason — a stale dev parquet gives you
  a historically-correct but outdated decision. Check that line first.
- **No dependency manifest.** Deleting `.venv` leaves nothing to rebuild
  from.
- **Business logic time is Asia/Taipei** via `src.utils.tw_time` — never
  bare `datetime.now()` (GCP VM timezone is not guaranteed Taipei).
- **Shioaji kbars timestamps changed semantics server-side on 2026-07-25**
  (UTC ns → Taipei-naive ns, decided by query time). Never parse them with a
  hardcoded `utc=True`; go through `shioaji_fetcher` which detects semantics
  per response and refuses when undecidable. **Historical backfills must use
  TAIFEX 一般 only — Shioaji is for the live today-bar.** The day-session
  aggregation includes the 13:45 closing-auction bar (official close).
- **The driver builds no notifier and no orchestrator**, so it can't spam
  LINE or touch `data/state_*.json`. Anything that instantiates
  `V2bOrchestrator` or `build_line_notifier()` can.

## Troubleshooting

- **`FAIL: data/MXF_Daily_Clean_2020_to_now.parquet not found`** (driver):
  parquet is gitignored; copy one from the GCP VM or run `scripts/init_data.py`
  with Shioaji credentials.
- **`verify` reports `STALE: latest=... gap=N trading days`**: normal on the
  dev clone (updater runs on GCP). Real corruption (gaps/dupes/out-of-range
  closes) still fails regardless.
