"""Margin manager — tracks TAIFEX initial and maintenance margin requirements."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# TAIFEX margins (TWD per contract, effective 2026-04-01 — update when exchange revises)
_INITIAL_MARGIN: dict[str, float] = {
    "TX": 477_000.0,
    "MTX": 119_250.0,
}
_MAINTENANCE_MARGIN: dict[str, float] = {
    "TX": 366_000.0,
    "MTX": 91_500.0,
}


@dataclass
class MarginSnapshot:
    """Point-in-time margin state."""

    account_equity: float
    initial_margin_required: float
    maintenance_margin: float
    margin_call_threshold: float
    open_contracts: int
    excess_margin: float = field(init=False)

    def __post_init__(self) -> None:
        self.excess_margin = self.account_equity - self.initial_margin_required

    @property
    def utilisation(self) -> float:
        """Fraction of equity consumed by initial margin (0–1+).

        Returns 0.0 when equity is 0 (simulation mode — data unavailable).
        """
        if self.account_equity <= 0:
            return 0.0
        return self.initial_margin_required / self.account_equity


class MarginManager:
    """Track margin utilisation and issue pre-emptive margin-call warnings.

    Parameters
    ----------
    product :
        ``"TX"`` or ``"MTX"`` — determines per-contract margin rates.
    warn_utilisation :
        Emit a warning when initial margin / equity exceeds this fraction
        (default 0.80 = 80 %).
    """

    def __init__(
        self,
        product: str = "TX",
        warn_utilisation: float = 0.80,
    ) -> None:
        product = product.upper()
        self.product = product
        self.warn_utilisation = warn_utilisation
        self._init_margin_per = _INITIAL_MARGIN.get(product, _INITIAL_MARGIN["TX"])
        self._maint_margin_per = _MAINTENANCE_MARGIN.get(product, _MAINTENANCE_MARGIN["TX"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        account_info: dict[str, Any],
        positions: list[dict],
    ) -> MarginSnapshot:
        """Compute a margin snapshot from account and position data.

        If ``account_info["equity"]`` is 0 (simulation mode), the snapshot is
        built from broker-reported ``margin_used``/``available_margin`` only if
        those are non-zero; otherwise falls back to computing margin from the
        position list and the hardcoded exchange rates.

        Parameters
        ----------
        account_info :
            Dict from ``ShioajiAdapter.get_account()``.
        positions :
            List of dicts from ``ShioajiAdapter.get_positions()``.
        """
        equity = float(account_info.get("equity", 0))
        broker_margin = float(account_info.get("margin_used", 0))
        available = float(account_info.get("available_margin", 0))

        open_contracts = sum(int(p.get("contracts", 0)) for p in positions)
        computed_init = open_contracts * self._init_margin_per
        computed_maint = open_contracts * self._maint_margin_per

        if equity == 0 and broker_margin == 0:
            # Simulation mode — broker returns zeros; we cannot compute useful margin
            logger.warning(
                "MarginManager: broker returned equity=0 (simulation mode). "
                "Margin snapshot will show zeros — skipping utilisation check."
            )
            return MarginSnapshot(
                account_equity=0.0,
                initial_margin_required=computed_init,
                maintenance_margin=computed_maint,
                margin_call_threshold=computed_maint,
                open_contracts=open_contracts,
            )

        # Use broker-reported values when available
        init_margin = broker_margin if broker_margin > 0 else computed_init
        maint_margin = computed_maint  # broker doesn't expose maint margin directly

        snap = MarginSnapshot(
            account_equity=equity if equity > 0 else (broker_margin + available),
            initial_margin_required=init_margin,
            maintenance_margin=maint_margin,
            margin_call_threshold=maint_margin,
            open_contracts=open_contracts,
        )

        if snap.utilisation > self.warn_utilisation:
            logger.warning(
                "MarginManager: margin utilisation %.1f%% > warn threshold %.1f%%  "
                "(equity=%.0f  init_margin=%.0f  contracts=%d)",
                snap.utilisation * 100,
                self.warn_utilisation * 100,
                snap.account_equity,
                init_margin,
                open_contracts,
            )

        return snap

    def check_margin_call(self, snap: MarginSnapshot) -> bool:
        """Return True if equity is at or below maintenance margin threshold."""
        if snap.account_equity == 0:
            return False  # simulation — skip
        return snap.account_equity <= snap.margin_call_threshold

    def max_new_contracts(self, snap: MarginSnapshot, equity: float | None = None) -> int:
        """Return how many additional contracts can be opened given current margin."""
        effective_equity = equity if equity is not None else snap.account_equity
        if effective_equity <= 0 or self._init_margin_per <= 0:
            return 0
        available = effective_equity - snap.initial_margin_required
        return max(0, int(available // self._init_margin_per))
