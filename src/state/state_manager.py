"""TradingState 持倉狀態 + StateManager 持久化。"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.utils.tw_time import today_taipei

logger = logging.getLogger(__name__)

_BACKUP_DIR = "data"
_BACKUP_KEEP_DAYS = 7

# Default account for the single live MXF deployment. Monitoring/verification
# scripts resolve their state path through resolve_state_path() so they always
# read the file the daemon actually writes.
_DEFAULT_ACCOUNT = "mxf_aggressive"
_DEFAULT_STATE_PATH = f"data/state_{_DEFAULT_ACCOUNT}.json"


def resolve_state_path(
    account: str = _DEFAULT_ACCOUNT,
    config_path: str = "config/accounts.yaml",
) -> Path:
    """Canonical on-disk state path for *account* — the SAME file the daemon writes.

    The daemon (``src/scheduler/main.py``) names each account's state file by
    convention: ``data/state_{account}.json`` where ``account`` is the
    accounts.yaml key. Monitoring / verification scripts MUST resolve the path
    through here instead of hard-coding a literal, so they always read the file
    the daemon actually maintains.

    (History: the old hard-coded ``data/paper_state.json`` was an orphan the
    daemon never wrote. Verify jobs compared the *live* broker position against
    that frozen file and cried wolf — "broker=0, state=20" — every day.)

    The config is consulted only to fail loudly if *account* was renamed or
    removed; the path itself follows the daemon's naming convention. A missing
    or unreadable config degrades to the convention path with a warning so a
    monitoring run never hard-crashes on a partial deployment.
    """
    path = Path(f"data/state_{account}.json")
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:  # missing/unreadable config → degrade, don't crash
        logger.warning(
            "resolve_state_path: cannot read %s (%s) — using convention path %s",
            config_path, exc, path,
        )
        return path
    accounts = cfg.get("accounts", {})
    if account not in accounts:
        available = ", ".join(sorted(accounts)) or "(none)"
        raise KeyError(
            f"resolve_state_path: account '{account}' not in {config_path} "
            f"(available: {available})"
        )
    return path


class StateCorruptionError(Exception):
    """State file unreadable AND no valid backup — trading must not proceed
    on a default (flat) state while a real position may exist."""


@dataclass
class TradingState:
    position: int = 0
    entry_price: float | None = None
    entry_date: str | None = None
    contracts: int = 0
    highest_high: float | None = None
    equity: float = 350_000.0
    pyramided: bool = False
    pending_action: str | None = None
    pending_contracts: int = 0
    pending_signal_date: str | None = None
    pending_reason: str | None = None


class StateManager:
    """Persist TradingState to a JSON file.

    Parameters
    ----------
    path :
        Path to the JSON state file.
    initial_equity :
        Equity to use ONLY when the state file does not yet exist (first run
        for an account) — should match the account's configured starting
        capital from accounts.yaml. ``None`` (default) falls back to the
        TradingState default, preserving the prior path-only behaviour. When a
        state file already exists it is authoritative and this is ignored.
    """

    def __init__(
        self,
        path: str = _DEFAULT_STATE_PATH,
        initial_equity: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.initial_equity = initial_equity

    # ------------------------------------------------------------------
    def load(self) -> TradingState:
        if not self.path.exists():
            equity = self.initial_equity if self.initial_equity is not None else 350_000.0
            return TradingState(equity=equity)
        try:
            return self._parse(self.path)
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            # NEVER silently fall back to a default (flat, 350K) state: with a
            # real position open that default would let the system buy again
            # on top of it. Try the daily backups newest-first; if none is
            # readable, refuse to run.
            logger.error("StateManager.load failed (%s) — trying backups", exc)
            for backup in sorted(
                Path(_BACKUP_DIR).glob(f"{self.path.stem}_backup_*.json"),
                reverse=True,
            ):
                try:
                    state = self._parse(backup)
                except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                    continue
                logger.error(
                    "State restored from backup %s — main file was corrupt; "
                    "verify position against the broker!", backup.name,
                )
                self.save(state)  # heal the main file
                return state
            raise StateCorruptionError(
                f"state file {self.path} corrupt ({exc}) and no readable backup "
                f"in {_BACKUP_DIR}/ — refusing to trade on a default flat state"
            ) from exc

    @staticmethod
    def _parse(path: Path) -> TradingState:
        raw = json.loads(path.read_text(encoding="utf-8"))
        s = raw["state"]  # missing key = corrupt, let it raise
        return TradingState(
            position=int(s.get("position", 0)),
            entry_price=s.get("entry_price"),
            entry_date=s.get("entry_date"),
            contracts=int(s.get("contracts", 0)),
            highest_high=s.get("highest_high"),
            equity=float(s.get("equity", 350_000.0)),
            pyramided=bool(s.get("pyramided", False)),
            pending_action=s.get("pending_action"),
            pending_contracts=int(s.get("pending_contracts", 0)),
            pending_signal_date=s.get("pending_signal_date"),
            pending_reason=s.get("pending_reason"),
        )

    def save(self, state: TradingState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing_trades: list[Any] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                existing_trades = raw.get("trades", [])
            except (json.JSONDecodeError, OSError):
                pass
        payload = {"state": asdict(state), "trades": existing_trades}
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("StateManager.save failed: %s", exc)
            return
        # Daily backup (one per day, keep last _BACKUP_KEEP_DAYS)
        self._backup_daily()

    def _backup_daily(self) -> None:
        """Copy state to data/{stem}_backup_{date}.json, prune old backups.

        The backup name is derived from the state filename so multiple
        accounts never overwrite each other's backups (the old fixed
        ``state_backup_{date}.json`` was shared by every account).
        """
        if not self.path.exists():
            return
        backup_dir = Path(_BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)
        today_str = today_taipei().strftime("%Y-%m-%d")
        backup_path = backup_dir / f"{self.path.stem}_backup_{today_str}.json"
        try:
            shutil.copy2(self.path, backup_path)
        except OSError as exc:
            logger.warning("State backup failed: %s", exc)
            return
        # Prune old backups — both the per-account pattern and the legacy
        # shared "state_backup_*" name so pre-migration files still age out.
        prune_globs = {f"{self.path.stem}_backup_*.json", "state_backup_*.json"}
        for pattern in prune_globs:
            for old in sorted(backup_dir.glob(pattern)):
                if old == backup_path:
                    continue
                try:
                    date_part = old.stem.rsplit("_backup_", 1)[-1]
                    backup_date = date.fromisoformat(date_part)
                    if (today_taipei() - backup_date).days > _BACKUP_KEEP_DAYS:
                        old.unlink()
                        logger.debug("Pruned old state backup: %s", old.name)
                except (ValueError, OSError):
                    pass

    @staticmethod
    def restore_from_backup(backup_path: str | Path, target_path: str | Path) -> bool:
        """Restore state from a backup file. Returns True if successful."""
        backup = Path(backup_path)
        target = Path(target_path)
        if not backup.exists():
            logger.error("Backup file not found: %s", backup)
            return False
        try:
            # Validate backup is valid JSON with state key
            raw = json.loads(backup.read_text(encoding="utf-8"))
            if "state" not in raw:
                logger.error("Backup file missing 'state' key: %s", backup)
                return False
            shutil.copy2(backup, target)
            logger.info("State restored from backup: %s → %s", backup, target)
            return True
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Restore from backup failed: %s", exc)
            return False

    def append_trade(self, trade: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trades: list[Any] = []
        state_dict: dict[str, Any] = asdict(TradingState())
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                trades = raw.get("trades", [])
                state_dict = raw.get("state", state_dict)
            except (json.JSONDecodeError, OSError):
                pass
        trades.append(trade)
        payload = {"state": state_dict, "trades": trades}
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("StateManager.append_trade failed: %s", exc)
