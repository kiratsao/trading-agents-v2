"""Kill Switch — hard circuit breaker for the Risk Manager.

Monitors daily portfolio returns and halts all trading the moment any of the
three loss thresholds is breached.  State survives process restarts via a
JSON file so a KILLED system stays halted until a human explicitly resets it.

Trigger conditions (any one suffices)
--------------------------------------
1. **Single-day loss**   : today's return < −3 %
2. **Consecutive losses**: last 5 trading days all negative
3. **Weekly loss**       : cumulative return over last 5 trading days < −5 %

Recovery
--------
Call ``reset(confirmation_code)`` where *confirmation_code* equals today's
date formatted as ``"YYYY-MM-DD-RESET"`` (e.g. ``"2026-03-31-RESET"``).
The explicit date prevents accidental resets from stale scripts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to repo root — overridable in tests via constructor)
# ---------------------------------------------------------------------------

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
_DEFAULT_STATE_PATH: Final[Path] = _REPO_ROOT / "data" / "kill_switch_state.json"
_DEFAULT_LOG_PATH: Final[Path] = _REPO_ROOT / "data" / "kill_switch_log.json"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SINGLE_DAY_LOSS_THRESHOLD: Final[float] = 0.03  # 3 %
CONSECUTIVE_LOSS_DAYS: Final[int] = 5
WEEKLY_LOSS_THRESHOLD: Final[float] = 0.05  # 5 %


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TradingHaltedError(RuntimeError):
    """Raised by RiskManager.pre_trade_check() when the kill switch is KILLED.

    Attributes
    ----------
    event:
        The :class:`KillSwitchEvent` that triggered the halt.
    """

    def __init__(self, event: KillSwitchEvent) -> None:
        self.event = event
        super().__init__(
            f"Trading HALTED — kill switch triggered at {event.timestamp.isoformat()}. "
            f"Reason: {event.reason}"
        )


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class KillSwitchState(str, Enum):
    ACTIVE = "ACTIVE"
    KILLED = "KILLED"


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class KillSwitchEvent:
    """Immutable record of a kill-switch trigger or reset action.

    Attributes
    ----------
    timestamp:
        UTC time when the event was recorded.
    reason:
        Human-readable description of which condition was breached.
    pnl_snapshot:
        Key metrics at the time of the event.  Keys vary by trigger type but
        always include ``"trigger"``.
    """

    timestamp: datetime
    reason: str
    pnl_snapshot: dict[str, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> KillSwitchEvent:
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            reason=d["reason"],
            pnl_snapshot=d.get("pnl_snapshot", {}),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class KillSwitch:
    """Hard circuit breaker for automated trading.

    Parameters
    ----------
    state_path:
        Path to the JSON file used to persist kill-switch state across
        restarts.  Defaults to ``data/kill_switch_state.json``.
    log_path:
        Path to the append-only JSON log of all kill / reset events.
        Defaults to ``data/kill_switch_log.json``.

    Example
    -------
    >>> ks = KillSwitch()
    >>> event = ks.check(daily_returns)
    >>> if event:
    ...     print(f"Trading halted: {event.reason}")
    """

    def __init__(
        self,
        state_path: Path | str | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        self._state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self._log_path = Path(log_path) if log_path else _DEFAULT_LOG_PATH

        self._state: KillSwitchState = KillSwitchState.ACTIVE
        self._last_event: KillSwitchEvent | None = None

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return ``True`` if trading is allowed (state == ACTIVE)."""
        return self._state == KillSwitchState.ACTIVE

    def check(self, daily_returns: pd.Series) -> KillSwitchEvent | None:
        """Evaluate the latest returns against all kill-switch conditions.

        Parameters
        ----------
        daily_returns:
            Chronologically ordered series of daily portfolio returns
            (e.g. ``[0.012, -0.005, -0.031, ...]``).  At least 1 element
            is required; 5 or more are needed for consecutive-loss and
            weekly-loss checks.

        Returns
        -------
        KillSwitchEvent | None
            A new event if a condition was just triggered, the existing
            event if the switch was already KILLED, or ``None`` if all
            conditions pass.
        """
        if self._state == KillSwitchState.KILLED:
            logger.warning(
                "KillSwitch already KILLED — skipping check. Call reset() to resume trading."
            )
            return self._last_event

        if daily_returns.empty:
            logger.debug("KillSwitch.check: empty series — nothing to check.")
            return None

        reason, snapshot = self._evaluate(daily_returns)
        if reason is None:
            return None

        return self._trigger(reason, snapshot)

    def reset(self, confirmation_code: str) -> bool:
        """Attempt to reset the kill switch back to ACTIVE.

        The *confirmation_code* must equal today's date (UTC) formatted as
        ``"YYYY-MM-DD-RESET"`` to prevent accidental or automated resets.

        Parameters
        ----------
        confirmation_code:
            Expected format: ``"2026-03-31-RESET"``.

        Returns
        -------
        bool
            ``True`` if the reset succeeded; ``False`` if the code was wrong
            or the switch was already ACTIVE.
        """
        expected = datetime.now(tz=UTC).strftime("%Y-%m-%d") + "-RESET"

        if confirmation_code != expected:
            logger.error(
                "KillSwitch reset REJECTED — wrong confirmation code. Expected '%s', got '%s'.",
                expected,
                confirmation_code,
            )
            return False

        if self._state == KillSwitchState.ACTIVE:
            logger.warning("KillSwitch reset called but state is already ACTIVE.")
            return False

        prev_event = self._last_event
        self._state = KillSwitchState.ACTIVE
        self._last_event = None

        self._persist_state()
        self._append_log(
            {
                "action": "RESET",
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "previous_event": prev_event.to_dict() if prev_event else None,
            }
        )

        logger.warning(
            "KillSwitch RESET to ACTIVE by operator. Previous trigger: %s",
            prev_event.reason if prev_event else "unknown",
        )
        return True

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, daily_returns: pd.Series) -> tuple[str | None, dict[str, float]]:
        """Check all conditions and return (reason, snapshot) or (None, {})."""

        # Condition 1: single-day loss
        latest_return = float(daily_returns.iloc[-1])
        if latest_return < -SINGLE_DAY_LOSS_THRESHOLD:
            return (
                f"Single-day loss {latest_return * 100:.2f}% exceeded threshold "
                f"-{SINGLE_DAY_LOSS_THRESHOLD * 100:.0f}%",
                {
                    "trigger": "single_day_loss",
                    "daily_return": latest_return,
                    "threshold": -SINGLE_DAY_LOSS_THRESHOLD,
                },
            )

        # Conditions 2 & 3 require at least 5 data points
        if len(daily_returns) >= CONSECUTIVE_LOSS_DAYS:
            recent = daily_returns.iloc[-CONSECUTIVE_LOSS_DAYS:]

            # Condition 2: consecutive losses
            if (recent < 0).all():
                return (
                    f"Consecutive losses: all of the last {CONSECUTIVE_LOSS_DAYS} "
                    "trading days were negative",
                    {
                        "trigger": "consecutive_loss",
                        "days": CONSECUTIVE_LOSS_DAYS,
                        "returns": {str(i): float(v) for i, v in enumerate(recent)},
                    },
                )

            # Condition 3: weekly cumulative loss
            weekly_return = float((1 + recent).prod() - 1)
            if weekly_return < -WEEKLY_LOSS_THRESHOLD:
                return (
                    f"Weekly loss {weekly_return * 100:.2f}% exceeded threshold "
                    f"-{WEEKLY_LOSS_THRESHOLD * 100:.0f}%",
                    {
                        "trigger": "weekly_loss",
                        "weekly_return": weekly_return,
                        "threshold": -WEEKLY_LOSS_THRESHOLD,
                    },
                )

        return None, {}

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def _trigger(self, reason: str, pnl_snapshot: dict[str, float]) -> KillSwitchEvent:
        """Set state to KILLED, log, persist, and return the event."""
        now = datetime.now(tz=UTC)
        event = KillSwitchEvent(
            timestamp=now,
            reason=reason,
            pnl_snapshot=pnl_snapshot,
        )

        self._state = KillSwitchState.KILLED
        self._last_event = event

        logger.critical(
            "KILL SWITCH TRIGGERED — trading halted immediately. "
            "Reason: %s | PnL snapshot: %s | Time: %s",
            reason,
            pnl_snapshot,
            now.isoformat(),
        )

        self._persist_state()
        self._append_log(
            {
                "action": "KILLED",
                **event.to_dict(),
            }
        )

        return event

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Write current state to the state JSON file (atomic-ish write)."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict = {"state": self._state.value}
        if self._last_event is not None:
            payload["last_event"] = self._last_event.to_dict()

        tmp_path = self._state_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._state_path)
        except OSError as exc:
            logger.error("KillSwitch: failed to persist state: %s", exc)
            # Don't re-raise — in-memory state is still correct.

    def _load_state(self) -> None:
        """Load persisted state on startup; silently defaults to ACTIVE if absent."""
        if not self._state_path.exists():
            logger.debug("KillSwitch: no state file at %s — starting ACTIVE.", self._state_path)
            return

        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._state = KillSwitchState(raw.get("state", KillSwitchState.ACTIVE))
            if "last_event" in raw:
                self._last_event = KillSwitchEvent.from_dict(raw["last_event"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(
                "KillSwitch: corrupt state file %s (%s) — defaulting to ACTIVE.",
                self._state_path,
                exc,
            )
            self._state = KillSwitchState.ACTIVE
            self._last_event = None
            return

        if self._state == KillSwitchState.KILLED:
            logger.critical(
                "KillSwitch loaded KILLED state from disk. "
                "Trading is HALTED. Trigger: %s | Time: %s",
                self._last_event.reason if self._last_event else "unknown",
                self._last_event.timestamp.isoformat() if self._last_event else "unknown",
            )

    def _append_log(self, entry: dict) -> None:
        """Append *entry* to the kill-switch event log (JSON array file)."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        existing: list[dict] = []
        if self._log_path.exists():
            try:
                existing = json.loads(self._log_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    logger.warning(
                        "KillSwitch log file %s is not a JSON array — resetting.",
                        self._log_path,
                    )
                    existing = []
            except json.JSONDecodeError as exc:
                logger.warning(
                    "KillSwitch: unreadable log file %s (%s) — starting fresh.",
                    self._log_path,
                    exc,
                )
                existing = []

        existing.append(entry)
        try:
            self._log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("KillSwitch: failed to write log: %s", exc)
