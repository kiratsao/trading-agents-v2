"""Futures reconciler — verify open positions match expected state.

Skeleton.  Implement position drift detection specific to futures
(contract expiry, rollover positions, margin drift).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DRIFT_WARN_CONTRACTS: int = 1  # Warn if actual differs by >= 1 contract


class FuturesReconciler:
    """Reconcile expected vs actual open futures positions.

    Parameters
    ----------
    db_path :
        SQLite path for persisting reconciliation snapshots.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path

    def check(
        self,
        expected: dict[str, int],
        actual: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        """Compare expected vs actual net positions.

        Parameters
        ----------
        expected :
            ``{product_code: expected_net_contracts}``
        actual :
            ``{product_code: actual_net_contracts}`` from ShioajiAdapter.

        Returns
        -------
        dict
            ``{product_code: {expected, actual, drift, status}}``
            where status is "ok" | "warning" | "missing" | "unexpected".
        """
        raise NotImplementedError
