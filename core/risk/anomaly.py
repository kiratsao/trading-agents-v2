"""Anomaly detection rules for portfolio returns.

AnomalyDetector.check() inspects a daily-return series and fires messages
when any of three rules triggers:

  Rule A — Extreme single-day return (|z-score| > 3σ of the series).
  Rule B — Trending run: three consecutive days all moving >1 % in the
            same direction.
  Rule C — Volatility spike: current 20-day realised vol > 2× the
            historical mean 20-day vol computed over the full series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_ZSCORE_THRESHOLD: float = 3.0  # Rule A: |z| > 3σ
_TRENDING_MIN_MOVE: float = 0.01  # Rule B: each day must move ≥ 1 %
_TRENDING_RUN_LEN: int = 3  # Rule B: consecutive days
_VOL_WINDOW: int = 20  # Rule C: rolling volatility window
_VOL_SPIKE_MULTIPLE: float = 2.0  # Rule C: current vol > 2× mean


# ---------------------------------------------------------------------------
# Legacy dataclass (kept so existing `from .anomaly import Alert` still works)
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    rule: str
    severity: str  # "warning" | "critical"
    message: str
    timestamp: pd.Timestamp


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Stateless anomaly detector for daily portfolio return series.

    All methods are pure functions of the input series — no state is kept
    between calls.  This makes the detector safe to call repeatedly without
    reset.

    Example
    -------
    >>> detector = AnomalyDetector()
    >>> messages = detector.check(daily_returns)
    >>> if messages:
    ...     print("Anomalies detected:", messages)
    """

    def check(self, daily_returns: pd.Series) -> list[str]:
        """Run all anomaly rules and return a list of alert message strings.

        Parameters
        ----------
        daily_returns :
            A pd.Series of daily fractional returns (e.g. 0.02 = +2 %).
            Index does not need to be a DatetimeIndex but must be ordered.

        Returns
        -------
        list[str]
            Empty list = no anomalies.  Each string is a human-readable
            description of a detected anomaly.
        """
        daily_returns = daily_returns.dropna()
        if daily_returns.empty:
            return []

        messages: list[str] = []
        messages.extend(self._rule_a_extreme_return(daily_returns))
        messages.extend(self._rule_b_trending_run(daily_returns))
        messages.extend(self._rule_c_volatility_spike(daily_returns))

        if messages:
            logger.warning(
                "AnomalyDetector: %d anomaly/anomalies detected: %s",
                len(messages),
                " | ".join(messages),
            )
        return messages

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_a_extreme_return(returns: pd.Series) -> list[str]:
        """Fire when the most recent return exceeds ±3 standard deviations."""
        if len(returns) < 2:
            return []
        latest = float(returns.iloc[-1])
        mean = float(returns.mean())
        std = float(returns.std())
        if std <= 0:
            return []
        z = (latest - mean) / std
        if abs(z) > _ZSCORE_THRESHOLD:
            direction = "positive" if latest > 0 else "negative"
            return [
                f"Extreme daily return: {latest * 100:+.2f}%  "
                f"(z={z:.2f}, {direction} outlier >{_ZSCORE_THRESHOLD:.0f}σ)"
            ]
        return []

    @staticmethod
    def _rule_b_trending_run(returns: pd.Series) -> list[str]:
        """Fire when the last N consecutive days all move >1 % in one direction."""
        n = _TRENDING_RUN_LEN
        if len(returns) < n:
            return []
        recent = returns.iloc[-n:].values
        threshold = _TRENDING_MIN_MOVE
        all_up = all(r > threshold for r in recent)
        all_down = all(r < -threshold for r in recent)
        if all_up:
            return [f"Trending anomaly: {n} consecutive up days (each >{threshold * 100:.0f}%)"]
        if all_down:
            return [f"Trending anomaly: {n} consecutive down days (each <-{threshold * 100:.0f}%)"]
        return []

    @staticmethod
    def _rule_c_volatility_spike(returns: pd.Series) -> list[str]:
        """Fire when current 20-day vol > 2× historical mean 20-day vol."""
        w = _VOL_WINDOW
        multiple = _VOL_SPIKE_MULTIPLE
        # Need at least 2 full windows to compute a meaningful mean
        if len(returns) < w * 2:
            return []

        # Annualised rolling volatility (fractional, not percentage)
        rolling_vol = returns.rolling(w).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        if rolling_vol.empty:
            return []

        current_vol = float(rolling_vol.iloc[-1])
        hist_mean_vol = float(rolling_vol.mean())

        if hist_mean_vol <= 0:
            return []

        if current_vol > multiple * hist_mean_vol:
            return [
                f"Volatility spike: current 20-day vol={current_vol * 100:.1f}%  "
                f"vs historical mean={hist_mean_vol * 100:.1f}%  "
                f"({current_vol / hist_mean_vol:.1f}× average)"
            ]
        return []


# ---------------------------------------------------------------------------
# Legacy module-level functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def check_drawdown(
    equity_curve: pd.Series,
    warn_threshold: float = 0.07,
    halt_threshold: float = 0.10,
) -> list[Alert]:
    """Legacy stub — use PnLTracker.update() for drawdown tracking."""
    raise NotImplementedError


def check_stale_positions(positions: pd.DataFrame, max_days: int = 5) -> list[Alert]:
    raise NotImplementedError


def check_order_error_rate(
    order_errors: pd.DataFrame, window_hours: int = 1, max_errors: int = 3
) -> list[Alert]:
    raise NotImplementedError


def check_agent_heartbeats(system_state: dict, max_silence_minutes: int = 15) -> list[Alert]:
    raise NotImplementedError
