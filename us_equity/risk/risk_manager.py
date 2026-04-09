"""Risk Manager orchestrator.

Combines KillSwitch, DrawdownGuard, and ConcentrationGuard into a single
``validate_trade`` pipeline that any upstream component calls before placing
orders.

Pipeline (``validate_trade``)
------------------------------
1. **KillSwitch** — if triggered (single-day / consecutive / weekly loss
   thresholds breached), raise :class:`~kill_switch.TradingHaltedError`.
2. **DrawdownGuard** — if portfolio drawdown is in the danger zone:
   - EXIT  → return ``{"SHY": 1.0}`` (liquidate all equities)
   - REDUCE → halve all non-SHY weights (add freed weight to SHY)
3. **ConcentrationGuard** — cap individual positions at 10 % and sectors at
   30 %, redistribute excess proportionally.

The output is a ``dict[str, float]`` of final tradable weights summing to 1.0.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.risk.concentration import ConcentrationGuard
from core.risk.drawdown import DrawdownAction, DrawdownGuard
from core.risk.kill_switch import KillSwitch, TradingHaltedError

logger = logging.getLogger(__name__)

_SAFE_HAVEN = "SHY"


class RiskManager:
    """Reviews approved strategies, computes position sizes, enforces limits.

    Parameters
    ----------
    db_path:
        Path to the SQLite database (passed through to future persistence
        code; not yet used for KillSwitch / DrawdownGuard state which have
        their own JSON files).
    kill_switch:
        Pre-constructed :class:`~kill_switch.KillSwitch`.  A default instance
        (using standard file paths) is created if not provided.
    drawdown_guard:
        Pre-constructed :class:`~drawdown.DrawdownGuard`.  A default instance
        is created if not provided.
    concentration_guard:
        Pre-constructed :class:`~concentration.ConcentrationGuard`.  A default
        instance is created if not provided.
    """

    def __init__(
        self,
        db_path: str,
        kill_switch: KillSwitch | None = None,
        drawdown_guard: DrawdownGuard | None = None,
        concentration_guard: ConcentrationGuard | None = None,
    ) -> None:
        self.db_path = db_path
        self.kill_switch: KillSwitch = kill_switch if kill_switch is not None else KillSwitch()
        self.drawdown_guard: DrawdownGuard = (
            drawdown_guard if drawdown_guard is not None else DrawdownGuard()
        )
        self.concentration_guard: ConcentrationGuard = (
            concentration_guard if concentration_guard is not None else ConcentrationGuard()
        )

    # ------------------------------------------------------------------
    # Pre-trade gate
    # ------------------------------------------------------------------

    def pre_trade_check(self, daily_returns: pd.Series | None = None) -> None:
        """Block trading when the kill switch is KILLED.

        Evaluates new return data (if provided), then raises
        :class:`TradingHaltedError` if the switch is in the KILLED state.

        Parameters
        ----------
        daily_returns:
            Optional series of daily portfolio returns to evaluate.

        Raises
        ------
        TradingHaltedError
            If the kill switch is or becomes KILLED.
        """
        if daily_returns is not None and not daily_returns.empty:
            self.kill_switch.check(daily_returns)

        if not self.kill_switch.is_active():
            raise TradingHaltedError(self.kill_switch._last_event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Main validation pipeline
    # ------------------------------------------------------------------

    def validate_trade(
        self,
        target_weights: dict[str, float],
        current_equity: float,
        daily_returns: pd.Series,
    ) -> dict[str, float]:
        """Run the full risk pipeline and return executable weights.

        Parameters
        ----------
        target_weights:
            Proposed ``{symbol: weight}`` from the strategy signal.
            Weights must sum to ~1.0.
        current_equity:
            Current portfolio NAV used by :class:`DrawdownGuard`.
        daily_returns:
            Recent daily portfolio returns evaluated by :class:`KillSwitch`.

        Returns
        -------
        dict[str, float]
            Final weights after all risk adjustments.  Sums to 1.0.

        Raises
        ------
        TradingHaltedError
            If the kill switch triggers or is already KILLED.
        """
        # ---- Step 1: Kill Switch ----------------------------------------
        ks_event = self.kill_switch.check(daily_returns)
        if ks_event is not None or not self.kill_switch.is_active():
            raise TradingHaltedError(
                ks_event if ks_event is not None else self.kill_switch._last_event  # type: ignore[arg-type]
            )

        # ---- Step 2: Drawdown Guard -------------------------------------
        dd_action = self.drawdown_guard.check(current_equity)

        weights: dict[str, float]

        if dd_action == DrawdownAction.EXIT:
            logger.warning(
                "DrawdownGuard EXIT — liquidating all equities to SHY (equity=%.2f, hwm=%.2f).",
                current_equity,
                self.drawdown_guard.high_water_mark or 0.0,
            )
            return {_SAFE_HAVEN: 1.0}

        if dd_action == DrawdownAction.REDUCE:
            logger.warning(
                "DrawdownGuard REDUCE — halving non-SHY weights (equity=%.2f, hwm=%.2f).",
                current_equity,
                self.drawdown_guard.high_water_mark or 0.0,
            )
            weights = {}
            freed = 0.0
            for sym, w in target_weights.items():
                if sym == _SAFE_HAVEN:
                    weights[sym] = w
                else:
                    weights[sym] = w * 0.5
                    freed += w * 0.5
            weights[_SAFE_HAVEN] = weights.get(_SAFE_HAVEN, 0.0) + freed
        else:
            weights = dict(target_weights)

        # ---- Step 3: Concentration Guard --------------------------------
        final_weights = self.concentration_guard.check(weights)

        # Sanity check
        total = sum(final_weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.error(
                "validate_trade: weight sum %.8f ≠ 1.0 after concentration guard — this is a bug.",
                total,
            )

        return final_weights

    # ------------------------------------------------------------------
    # Core agent loop (placeholder)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Continuous loop: check new approved strategies + re-evaluate live portfolio."""
        raise NotImplementedError

    def evaluate_strategy(self, strategy_id: str) -> dict | None:
        """Return sized order dict if strategy passes risk checks, else None."""
        raise NotImplementedError

    def check_portfolio_risk(self) -> None:
        """Re-evaluate live portfolio against drawdown and concentration limits."""
        raise NotImplementedError

    def _halt_trading(self, reason: str) -> None:
        """Set system_state.trading_halted = True and log to risk_events."""
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Risk Manager Agent")
    parser.add_argument("--check-now", action="store_true")
    args = parser.parse_args()
    logger.info("RiskManager starting (check_now=%s)", args.check_now)
