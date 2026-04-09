"""Aggregate walk-forward OOS window results into summary metrics."""

from __future__ import annotations

import statistics


def aggregate_oos_metrics(window_results: list[dict]) -> dict:
    """Aggregate per-window metrics into a final performance summary.

    Expects each window dict to contain: sharpe, max_drawdown, win_rate,
    profit_factor, cagr.  Returns mean of each plus calmar = cagr / max_drawdown.

    Returns
    -------
    dict
        Keys: sharpe, max_drawdown, win_rate, profit_factor, cagr, calmar.
    """
    if not window_results:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "cagr": 0.0,
            "calmar": 0.0,
        }

    def _mean(key: str) -> float:
        vals = [w[key] for w in window_results if key in w]
        return statistics.mean(vals) if vals else 0.0

    sharpe = _mean("sharpe")
    max_dd = _mean("max_drawdown")
    cagr = _mean("cagr")
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": _mean("win_rate"),
        "profit_factor": _mean("profit_factor"),
        "cagr": cagr,
        "calmar": calmar,
    }


def passes_threshold(metrics: dict) -> bool:
    """Return True if metrics meet minimum approval thresholds.

    Thresholds
    ----------
    - sharpe >= 0.5
    - max_drawdown <= 0.25 (25 %)
    - win_rate >= 0.40
    """
    if metrics.get("sharpe", 0.0) < 0.5:
        return False
    if metrics.get("max_drawdown", 1.0) > 0.25:
        return False
    if metrics.get("win_rate", 0.0) < 0.40:
        return False
    return True
