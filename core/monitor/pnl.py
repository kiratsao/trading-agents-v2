"""PnL tracking — daily / cumulative returns, high-water mark, drawdown.

PnLTracker is the single source of truth for portfolio performance metrics.
It reads the shared ``data/equity_history.json`` that TradingOrchestrator
maintains, so it works correctly whether called from the orchestrator or
as a standalone tool.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_EQUITY_HISTORY_PATH = Path("data/equity_history.json")


class PnLTracker:
    """Track daily and cumulative portfolio performance.

    Loads ``data/equity_history.json`` on construction.  Each call to
    :meth:`update` records the new equity value, recomputes all metrics,
    and returns a snapshot dict.

    Snapshot keys
    -------------
    equity, initial_equity, daily_return, cumulative_return,
    high_water_mark, current_drawdown, max_drawdown_ever,
    timestamp, date
    """

    def __init__(self, history_path: Path | str = _EQUITY_HISTORY_PATH) -> None:
        self._history_path = Path(history_path)
        self._history: list[dict] = self._load_history()

        equities = [e["equity"] for e in self._history if e.get("equity", 0) > 0]
        self._initial_equity: float | None = equities[0] if equities else None
        self._hwm: float = max(equities) if equities else 0.0
        self._max_dd_ever: float = self._compute_max_dd_from_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, equity: float, timestamp: dt.datetime | None = None) -> dict:
        """Record *equity* and return a full PnL snapshot.

        Parameters
        ----------
        equity :
            Current portfolio NAV in USD.
        timestamp :
            Snapshot time (defaults to UTC now).

        Returns
        -------
        dict
            ``{equity, initial_equity, daily_return, cumulative_return,
            high_water_mark, current_drawdown, max_drawdown_ever,
            timestamp, date}``
        """
        if timestamp is None:
            timestamp = dt.datetime.now(dt.UTC)

        today_str = timestamp.strftime("%Y-%m-%d")
        daily_return = self._compute_daily_return(equity)

        if self._initial_equity is None:
            self._initial_equity = equity
            logger.info("PnLTracker: initial equity set to $%.2f", equity)

        cumulative_return = (
            (equity / self._initial_equity - 1.0)
            if self._initial_equity and self._initial_equity > 0
            else 0.0
        )

        if equity > self._hwm:
            self._hwm = equity

        # Negative value; 0.0 = at high-water mark
        current_drawdown = (equity / self._hwm - 1.0) if self._hwm > 0 else 0.0
        self._max_dd_ever = min(self._max_dd_ever, current_drawdown)

        snapshot: dict[str, Any] = {
            "equity": equity,
            "initial_equity": self._initial_equity,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "high_water_mark": self._hwm,
            "current_drawdown": current_drawdown,
            "max_drawdown_ever": self._max_dd_ever,
            "timestamp": timestamp.isoformat(),
            "date": today_str,
        }

        logger.info(
            "PnL: equity=$%.2f  daily=%+.3f%%  cum=%+.3f%%  dd=%.3f%%",
            equity,
            (daily_return or 0.0) * 100,
            cumulative_return * 100,
            current_drawdown * 100,
        )
        return snapshot

    def get_daily_returns_series(self) -> pd.Series:
        """Return a pd.Series of daily portfolio returns from history."""
        if len(self._history) < 2:
            return pd.Series(dtype=float)
        equities = pd.Series(
            [e["equity"] for e in self._history],
            index=pd.to_datetime([e["date"] for e in self._history]),
        )
        return equities.pct_change().dropna()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_daily_return(self, equity: float) -> float | None:
        if not self._history:
            return None
        prev = self._history[-1].get("equity", 0.0)
        return ((equity - prev) / prev) if prev > 0 else None

    def _compute_max_dd_from_history(self) -> float:
        equities = [e["equity"] for e in self._history if e.get("equity", 0) > 0]
        if not equities:
            return 0.0
        hwm = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > hwm:
                hwm = eq
            max_dd = min(max_dd, eq / hwm - 1.0)
        return max_dd

    def _load_history(self) -> list[dict]:
        if not self._history_path.exists():
            return []
        try:
            data = json.loads(self._history_path.read_text())
            return sorted(
                (e for e in data if "date" in e and "equity" in e),
                key=lambda e: e["date"],
            )
        except Exception as exc:
            logger.warning("PnLTracker: could not load %s — %s", self._history_path, exc)
            return []


# ---------------------------------------------------------------------------
# Legacy module-level functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def realized_pnl(trades: pd.DataFrame) -> float:
    """Sum of (exit_price - entry_price) * qty for all closed trades."""
    if trades.empty:
        return 0.0
    req = {"exit_price", "entry_price", "qty"}
    if not req.issubset(trades.columns):
        raise ValueError(f"trades must contain columns: {req}")
    return float(((trades["exit_price"] - trades["entry_price"]) * trades["qty"]).sum())


def unrealized_pnl(positions: pd.DataFrame, current_prices: dict[str, float]) -> float:
    """Sum of (current_price - avg_cost) * qty for all open positions."""
    if positions.empty:
        return 0.0
    total = 0.0
    for _, row in positions.iterrows():
        sym = row.get("symbol")
        price = current_prices.get(sym, row.get("avg_cost", 0.0))
        total += (price - row.get("avg_cost", 0.0)) * row.get("qty", 0.0)
    return total


def equity_curve(trades: pd.DataFrame, initial_equity: float) -> pd.Series:
    """Build cumulative equity curve indexed by timestamp (stub)."""
    if trades.empty:
        return pd.Series(dtype=float)
    raise NotImplementedError("equity_curve() requires trade history — not yet wired to DB.")
