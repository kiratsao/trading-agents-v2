"""TradingState 持倉狀態 + StateManager 持久化。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


class StateManager:
    """Persist TradingState to a JSON file."""

    def __init__(self, path: str = "data/paper_state.json") -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    def load(self) -> TradingState:
        if not self.path.exists():
            return TradingState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            s = raw.get("state", {})
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
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
            logger.warning("StateManager.load failed (%s) — returning default state.", exc)
            return TradingState()

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
        try:
            self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.error("StateManager.append_trade failed: %s", exc)
