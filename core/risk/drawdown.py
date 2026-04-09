"""Drawdown tracking, halt logic, and DrawdownGuard circuit breaker.

DrawdownGuard
-------------
Tracks a portfolio's high-water mark (HWM) and emits a :class:`DrawdownAction`
on every ``check()`` call.  The action drives position sizing in
:class:`~agents.risk_manager.risk_manager.RiskManager`:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Drawdown from HWM   │  Action  │  Effect                      │
    ├──────────────────────┼──────────┼──────────────────────────────┤
    │  < 10 %              │  HOLD    │  trade normally              │
    │  10 – 15 %           │  REDUCE  │  halve non-SHY weights       │
    │  > 15 %              │  EXIT    │  go 100% SHY                 │
    └─────────────────────────────────────────────────────────────────┘

Recovery hysteresis: once in REDUCE or EXIT, the action reverts to HOLD
only when ``current_equity ≥ hwm × 0.95``.  This prevents thrashing around
the threshold boundaries.

State is persisted to ``data/drawdown_state.json`` so the guard survives
process restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
_DEFAULT_STATE_PATH: Final[Path] = _REPO_ROOT / "data" / "drawdown_state.json"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

REDUCE_THRESHOLD: Final[float] = 0.10  # 10 %
EXIT_THRESHOLD: Final[float] = 0.15  # 15 %
RECOVERY_PCT: Final[float] = 0.95  # 95 % of HWM to resume normal trading


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class DrawdownAction(str, Enum):
    """Recommended portfolio action based on drawdown severity."""

    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


# Severity ordering — higher number means worse situation
_SEVERITY: Final[dict[DrawdownAction, int]] = {
    DrawdownAction.HOLD: 0,
    DrawdownAction.REDUCE: 1,
    DrawdownAction.EXIT: 2,
}


# ---------------------------------------------------------------------------
# DrawdownGuard
# ---------------------------------------------------------------------------


class DrawdownGuard:
    """High-water-mark drawdown monitor with persistence.

    Parameters
    ----------
    state_path:
        JSON file for persisting HWM and current action across restarts.
    reduce_threshold:
        Drawdown fraction at which REDUCE fires (default 10 %).
    exit_threshold:
        Drawdown fraction at which EXIT fires (default 15 %).
    recovery_pct:
        Fraction of HWM that must be recovered before lifting a REDUCE/EXIT
        (default 95 %).

    Example
    -------
    >>> guard = DrawdownGuard()
    >>> action = guard.check(current_equity=950_000)
    >>> if action == DrawdownAction.EXIT:
    ...     weights = {"SHY": 1.0}
    """

    def __init__(
        self,
        state_path: Path | str | None = None,
        reduce_threshold: float = REDUCE_THRESHOLD,
        exit_threshold: float = EXIT_THRESHOLD,
        recovery_pct: float = RECOVERY_PCT,
    ) -> None:
        self._state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self._reduce_threshold = reduce_threshold
        self._exit_threshold = exit_threshold
        self._recovery_pct = recovery_pct

        self._hwm: float | None = None  # set on first observation
        self._action: DrawdownAction = DrawdownAction.HOLD

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def high_water_mark(self) -> float | None:
        """Current HWM, or None before the first observation."""
        return self._hwm

    @property
    def current_action(self) -> DrawdownAction:
        """Most recent DrawdownAction (HOLD / REDUCE / EXIT)."""
        return self._action

    def current_drawdown(self) -> float:
        """Return current drawdown as a positive fraction in [0, 1].

        Returns 0.0 if HWM is not yet established.
        """
        if self._hwm is None or self._hwm <= 0:
            return 0.0
        return 0.0  # no equity to compare against yet (use check() instead)

    def check(self, current_equity: float) -> DrawdownAction:
        """Evaluate current equity against the HWM and return the required action.

        Internally updates the HWM and persists state if the action changes.

        Parameters
        ----------
        current_equity:
            Current portfolio NAV in the same currency units as prior calls.

        Returns
        -------
        DrawdownAction
            HOLD, REDUCE, or EXIT.
        """
        if self._hwm is None:
            self._hwm = current_equity
            logger.info("DrawdownGuard: HWM initialised to %.2f.", self._hwm)
            self._persist_state()
            return self._action

        # Update HWM
        if current_equity > self._hwm:
            self._hwm = current_equity
            logger.debug("DrawdownGuard: new HWM = %.2f", self._hwm)

        dd = (self._hwm - current_equity) / self._hwm

        # Determine action purely from thresholds
        if dd > self._exit_threshold:
            threshold_action = DrawdownAction.EXIT
        elif dd >= self._reduce_threshold:
            threshold_action = DrawdownAction.REDUCE
        else:
            threshold_action = DrawdownAction.HOLD

        prev_action = self._action

        if _SEVERITY[threshold_action] < _SEVERITY[self._action]:
            # Attempting recovery (less severe) — apply hysteresis
            if current_equity >= self._hwm * self._recovery_pct:
                self._action = threshold_action
                logger.info(
                    "DrawdownGuard: recovered to %s "
                    "(equity=%.2f, hwm=%.2f, dd=%.2%%, recovery_pct=%.0f%%).",
                    self._action.value,
                    current_equity,
                    self._hwm,
                    dd,
                    self._recovery_pct * 100,
                )
            # else: not enough recovery — keep current (worse) action
        else:
            self._action = threshold_action

        if self._action != prev_action:
            logger.warning(
                "DrawdownGuard: action %s → %s (equity=%.2f, hwm=%.2f, drawdown=%.2f%%).",
                prev_action.value,
                self._action.value,
                current_equity,
                self._hwm,
                dd * 100,
            )
            self._persist_state()

        return self._action

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hwm": self._hwm,
            "action": self._action.value,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError as exc:
            logger.error("DrawdownGuard: failed to persist state: %s", exc)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._hwm = raw.get("hwm")
            self._action = DrawdownAction(raw.get("action", DrawdownAction.HOLD))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("DrawdownGuard: corrupt state file (%s) — starting fresh.", exc)


# ---------------------------------------------------------------------------
# Legacy module-level functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def current_drawdown(equity_curve: pd.Series) -> float:
    """Return current drawdown as a positive fraction (0.0 to 1.0).

    The current drawdown is measured from the running peak to the last value.
    """
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax().iloc[-1]
    if peak <= 0:
        return 0.0
    return float((peak - equity_curve.iloc[-1]) / peak)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return maximum drawdown over the series as a positive fraction."""
    if equity_curve.empty:
        return 0.0
    rolling_peak = equity_curve.cummax()
    drawdowns = (rolling_peak - equity_curve) / rolling_peak
    return float(drawdowns.max())


def should_halt(equity_curve: pd.Series, threshold: float) -> bool:
    """Return True if current drawdown exceeds the halt threshold."""
    return current_drawdown(equity_curve) >= threshold
