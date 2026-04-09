"""Concentration risk guard — enforces single-position and sector weight limits.

Rules
-----
1. **Single-position cap** (default 10 %): any symbol (except exempt ones like
   SHY) whose weight exceeds the cap has its excess redistributed
   proportionally to all other symbols that still have room below the cap.

2. **Sector cap** (default 30 %): if the combined weight of all symbols in a
   sector exceeds the cap, every symbol in that sector is scaled down
   proportionally, and the freed weight is redistributed to symbols in
   other sectors proportionally to their current weight.

Both rules are applied iteratively (up to ``_MAX_ITERATIONS`` rounds) until
the portfolio converges, because capping one dimension may push another over
its limit.  In practice 2–3 iterations always suffice.

Invariant: ``sum(output_weights.values()) == 1.0`` (within floating-point
tolerance).

Exempt symbols (SHY by default) are never subject to either cap because they
represent safe-haven / cash-equivalent positions that may legitimately reach
100 % of the portfolio.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector mapping  (covers SP500_TOP50 + SPY / SHY)
# ---------------------------------------------------------------------------

SECTOR_MAP: Final[dict[str, str]] = {
    # ---- Technology ----
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Technology",  # cloud / AWS treated as Tech
    "NVDA": "Technology",
    "META": "Technology",
    "AVGO": "Technology",
    "ADBE": "Technology",
    "CRM": "Technology",
    "AMD": "Technology",
    "CSCO": "Technology",
    "ACN": "Technology",
    "ORCL": "Technology",
    "INTC": "Technology",
    "IBM": "Technology",
    "TXN": "Technology",
    # ---- Healthcare ----
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "LLY": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "ABT": "Healthcare",
    "AMGN": "Healthcare",
    "TMO": "Healthcare",
    "DHR": "Healthcare",
    # ---- Finance ----
    "JPM": "Finance",
    "BRK.B": "Finance",
    "V": "Finance",
    "MA": "Finance",
    # ---- Consumer ----
    "TSLA": "Consumer",
    "WMT": "Consumer",
    "PG": "Consumer",
    "HD": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",
    "COST": "Consumer",
    "MCD": "Consumer",
    "LOW": "Consumer",
    "CMCSA": "Consumer",
    "NFLX": "Consumer",
    "PM": "Consumer",
    # ---- Energy ----
    "XOM": "Energy",
    "CVX": "Energy",
    # ---- Industrial ----
    "CAT": "Industrial",
    "GE": "Industrial",
    "HON": "Industrial",
    "UNP": "Industrial",
    "BA": "Industrial",
    # ---- Utilities ----
    "NEE": "Utilities",
    # ---- Materials ----
    "LIN": "Materials",
    # ---- Fixed Income / Special ----
    "SPY": "Fixed Income",
    "SHY": "Fixed Income",
}

# Symbols that are never subject to position or sector caps
_EXEMPT_SYMBOLS: Final[frozenset[str]] = frozenset({"SHY"})

_MAX_ITERATIONS: Final[int] = 20
_WEIGHT_TOLERANCE: Final[float] = 1e-9


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConcentrationGuard:
    """Enforces single-position and sector concentration limits.

    Parameters
    ----------
    max_position:
        Maximum weight for any single non-exempt symbol (default 0.10 = 10 %).
    max_sector:
        Maximum combined weight for any single sector (default 0.30 = 30 %).
    sector_map:
        Custom ``{symbol: sector}`` mapping.  Defaults to :data:`SECTOR_MAP`.
    exempt_symbols:
        Symbols excluded from all caps (default: ``{"SHY"}``).

    Example
    -------
    >>> guard = ConcentrationGuard()
    >>> adjusted = guard.check({"AAPL": 0.15, "MSFT": 0.10, "SHY": 0.75})
    >>> assert adjusted["AAPL"] <= 0.10
    >>> assert abs(sum(adjusted.values()) - 1.0) < 1e-9
    """

    def __init__(
        self,
        max_position: float = 0.10,
        max_sector: float = 0.30,
        sector_map: dict[str, str] | None = None,
        exempt_symbols: frozenset[str] | set[str] = _EXEMPT_SYMBOLS,
    ) -> None:
        self._max_pos = max_position
        self._max_sector = max_sector
        self._sector_map: dict[str, str] = sector_map if sector_map is not None else SECTOR_MAP
        self._exempt: frozenset[str] = frozenset(exempt_symbols)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, target_weights: dict[str, float]) -> dict[str, float]:
        """Apply concentration limits and return adjusted weights.

        The method iterates until both the position cap and the sector cap
        are satisfied simultaneously, or until ``_MAX_ITERATIONS`` is reached.

        Parameters
        ----------
        target_weights:
            ``{symbol: weight}`` — weights must sum to ~1.0.  Exempt symbols
            (SHY) are passed through unchanged.

        Returns
        -------
        dict[str, float]
            Adjusted weights.  ``sum(values()) == 1.0`` within floating-point
            tolerance.
        """
        if not target_weights:
            return {}

        weights = {k: float(v) for k, v in target_weights.items()}
        original = dict(weights)

        for iteration in range(_MAX_ITERATIONS):
            prev = dict(weights)

            weights = self._cap_individual(weights)
            weights = self._cap_sectors(weights)

            # Normalise to absorb floating-point drift
            total = sum(weights.values())
            if total > _WEIGHT_TOLERANCE:
                weights = {k: v / total for k, v in weights.items()}

            converged = all(
                abs(weights.get(k, 0.0) - prev.get(k, 0.0)) < _WEIGHT_TOLERANCE
                for k in set(weights) | set(prev)
            )
            if converged:
                logger.debug(
                    "ConcentrationGuard: converged after %d iteration(s).",
                    iteration + 1,
                )
                break
        else:
            logger.warning(
                "ConcentrationGuard: did not fully converge after %d iterations.",
                _MAX_ITERATIONS,
            )

        self._log_adjustments(original, weights)
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cap_individual(self, weights: dict[str, float]) -> dict[str, float]:
        """Cap each non-exempt symbol at ``_max_pos``, redistribute excess."""
        result = dict(weights)

        # Collect total excess from over-limit symbols
        total_excess = 0.0
        for sym in list(result):
            if sym in self._exempt:
                continue
            if result[sym] > self._max_pos + _WEIGHT_TOLERANCE:
                total_excess += result[sym] - self._max_pos
                result[sym] = self._max_pos

        if total_excess <= _WEIGHT_TOLERANCE:
            return result

        # Recipients: non-exempt symbols with room below the cap
        recipients = [
            s
            for s, w in result.items()
            if s not in self._exempt and w < self._max_pos - _WEIGHT_TOLERANCE
        ]
        recipient_total = sum(result[s] for s in recipients)

        if recipient_total > _WEIGHT_TOLERANCE:
            for sym in recipients:
                result[sym] += total_excess * (result[sym] / recipient_total)
        else:
            # No equity room — overflow goes to SHY (safe haven)
            if "SHY" in result:
                result["SHY"] += total_excess
            else:
                # Last resort: spread equally among all non-exempt
                non_exempt = [s for s in result if s not in self._exempt]
                if non_exempt:
                    per = total_excess / len(non_exempt)
                    for sym in non_exempt:
                        result[sym] += per

        return result

    def _cap_sectors(self, weights: dict[str, float]) -> dict[str, float]:
        """Cap each sector at ``_max_sector``, redistribute excess."""
        result = dict(weights)

        # Build sector → [symbols] and sector total
        sector_syms: dict[str, list[str]] = defaultdict(list)
        for sym in result:
            if sym in self._exempt:
                continue
            sector = self._sector_map.get(sym, "Other")
            sector_syms[sector].append(sym)

        for sector, syms in sector_syms.items():
            sector_total = sum(result[s] for s in syms)
            if sector_total <= self._max_sector + _WEIGHT_TOLERANCE:
                continue

            excess = sector_total - self._max_sector
            scale = self._max_sector / sector_total

            # Scale down all symbols in the violating sector
            for sym in syms:
                result[sym] *= scale

            # Redistribute excess to symbols outside this sector
            outside = [
                s
                for s in result
                if s not in self._exempt and self._sector_map.get(s, "Other") != sector
            ]
            outside_total = sum(result[s] for s in outside)

            if outside_total > _WEIGHT_TOLERANCE:
                for sym in outside:
                    result[sym] += excess * (result[sym] / outside_total)
            elif "SHY" in result:
                result["SHY"] += excess

        return result

    def _log_adjustments(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> None:
        """Log any weight changes at DEBUG level; log significant ones at INFO."""
        changes: list[str] = []
        all_syms = set(before) | set(after)
        for sym in sorted(all_syms):
            b = before.get(sym, 0.0)
            a = after.get(sym, 0.0)
            if abs(a - b) > 1e-6:
                changes.append(f"{sym}: {b:.4f} → {a:.4f}")

        if not changes:
            logger.debug("ConcentrationGuard: no adjustments needed.")
            return

        logger.info(
            "ConcentrationGuard: %d weight(s) adjusted — %s",
            len(changes),
            ", ".join(changes),
        )
