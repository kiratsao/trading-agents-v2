"""Reconcile expected weights against broker-reported positions.

:class:`Reconciler` provides two complementary views:

* :meth:`check` — weight-based drift report comparing *expected_weights*
  (strategy output) against *actual_positions* (live broker snapshot).
  Symbols drifting more than ``_DRIFT_WARN_THRESHOLD`` percentage points are
  logged at WARNING level.

* :meth:`reconcile` — legacy qty-based discrepancy detection (backward
  compatibility; compares broker-reported quantities against a local DB
  record).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Log a WARNING when |actual_weight - expected_weight| exceeds this many
# percentage points.
_DRIFT_WARN_THRESHOLD: float = 2.0


class Reconciler:
    """Detects weight drift and position discrepancies.

    Parameters
    ----------
    db_path : str
        SQLite path (for future persistence; not required for
        :meth:`check` functionality).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Weight-based drift report
    # ------------------------------------------------------------------

    def check(
        self,
        expected_weights: dict[str, float],
        actual_positions: dict[str, dict],
        equity: float,
    ) -> dict[str, dict]:
        """Compare expected weights against live positions and report drift.

        Parameters
        ----------
        expected_weights :
            ``{symbol: weight}`` as produced by the strategy / risk pipeline.
            Weights should sum to ~1.0.
        actual_positions :
            ``{symbol: {market_value, qty, …}}`` from
            ``adapter.get_positions()``.
        equity :
            Total portfolio NAV in USD.  Used to convert ``market_value`` to
            an actual weight.

        Returns
        -------
        dict[str, dict]
            One entry per symbol appearing in *either* dict:

            .. code-block:: python

                {
                    "AAPL": {
                        "expected_weight": 0.10,
                        "actual_weight":   0.124,
                        "drift_pct":       +2.4,   # signed, percentage points
                        "status":          "warning",  # ok | warning | missing | unexpected
                    },
                    …
                }

        Notes
        -----
        * ``drift_pct`` is *signed*: positive means over-weight, negative
          means under-weight.
        * ``status`` is:

          - ``"ok"``         — within ±2 pp of target
          - ``"warning"``    — |drift| ≥ 2 pp
          - ``"missing"``    — expected weight > 0 but no broker position found
          - ``"unexpected"`` — not in target but position exists in broker
        """
        if equity <= 0:
            logger.error("Reconciler.check: equity=%.2f ≤ 0 — cannot compute weights.", equity)
            return {}

        report: dict[str, dict] = {}
        all_symbols = sorted(set(expected_weights) | set(actual_positions))

        for symbol in all_symbols:
            expected_w = float(expected_weights.get(symbol, 0.0))
            pos = actual_positions.get(symbol)
            actual_w = (
                float(pos.get("market_value", 0.0) or 0.0) / equity if pos is not None else 0.0
            )
            drift_pct = (actual_w - expected_w) * 100.0  # signed, in pp

            if expected_w > 0.0 and actual_w == 0.0:
                # Target says hold, but broker reports no position
                status = "missing"
                logger.warning(
                    "Reconciler: %-6s expected in portfolio but NO broker position "
                    "(expected=%.2f%%).",
                    symbol,
                    expected_w * 100,
                )
            elif expected_w == 0.0 and actual_w > 0.0:
                # Not in target, but position exists — stale or rogue holding
                status = "unexpected"
                logger.warning(
                    "Reconciler: %-6s is in portfolio but NOT in target weights (actual=%.2f%%).",
                    symbol,
                    actual_w * 100,
                )
            elif abs(drift_pct) >= _DRIFT_WARN_THRESHOLD:
                status = "warning"
                logger.warning(
                    "Reconciler: %-6s drift=%+.2f pp  (expected=%.2f%%  actual=%.2f%%)",
                    symbol,
                    drift_pct,
                    expected_w * 100,
                    actual_w * 100,
                )
            else:
                status = "ok"

            report[symbol] = {
                "expected_weight": round(expected_w, 6),
                "actual_weight": round(actual_w, 6),
                "drift_pct": round(drift_pct, 4),
                "status": status,
            }

        ok_count = sum(1 for v in report.values() if v["status"] == "ok")
        warn_count = sum(1 for v in report.values() if v["status"] == "warning")
        missing_count = sum(1 for v in report.values() if v["status"] == "missing")
        unexpected_count = sum(1 for v in report.values() if v["status"] == "unexpected")
        logger.info(
            "Reconciler: %d symbol(s) checked — %d ok, %d warning(s), %d missing, %d unexpected.",
            len(report),
            ok_count,
            warn_count,
            missing_count,
            unexpected_count,
        )
        return report

    # ------------------------------------------------------------------
    # Legacy qty-based reconcile (backward compatibility)
    # ------------------------------------------------------------------

    def reconcile(self, broker_positions: list[dict]) -> list[dict]:
        """Detect discrepancies between broker-reported and local DB quantities.

        Legacy interface kept for backward compatibility.  Returns an empty
        list until the DB layer is wired; the weight-based :meth:`check` is
        the recommended interface for new code.

        Parameters
        ----------
        broker_positions :
            List of ``{symbol, qty, avg_cost, …}`` dicts from the broker.

        Returns
        -------
        list[dict]
            List of ``{symbol, local_qty, broker_qty}`` discrepancy records.
        """
        return []
