"""Shared base types for risk management across all markets."""

from __future__ import annotations

from enum import Enum


class RiskAction(str, Enum):
    """Standardised action codes returned by risk guards."""

    HOLD = "hold"  # No action required
    REDUCE = "reduce"  # Reduce position sizes
    EXIT = "exit"  # Exit all positions immediately
    HALT = "halt"  # Hard trading halt


class TradingHaltedError(RuntimeError):
    """Raised when a kill-switch or hard halt blocks trade submission."""
