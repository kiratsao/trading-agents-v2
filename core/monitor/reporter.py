"""Daily performance report generator.

DailyReporter.generate() produces a formatted plain-text report and saves it
to ``data/reports/YYYY-MM-DD.txt``.  The report is structured in three
sections:

  1. Portfolio Summary  — equity, daily PnL ($/%),  cumulative return
  2. Position Detail    — per-symbol weight, unrealized PnL, daily change
  3. Risk Status        — KillSwitch / DrawdownGuard state, drawdown %

If any WARNING or CRITICAL conditions are present the report opens with a
prominent banner so the severity is obvious at a glance.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path("data/reports")

# ANSI codes (used when writing to terminal; stripped for file output)
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"


class DailyReporter:
    """Produce and persist the end-of-day performance report.

    Parameters
    ----------
    reports_dir :
        Directory where ``YYYY-MM-DD.txt`` files are written.
        Created automatically if it does not exist.
    """

    def __init__(self, reports_dir: Path | str = _REPORTS_DIR) -> None:
        self._reports_dir = Path(reports_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        pnl_snapshot: dict[str, Any],
        positions: dict[str, dict],
        risk_status: dict[str, Any],
        anomalies: list[str] | None = None,
    ) -> str:
        """Generate and save the daily report.

        Parameters
        ----------
        pnl_snapshot :
            Output of :meth:`~agents.monitor.pnl.PnLTracker.update`.
        positions :
            Live positions from ``AlpacaAdapter.get_positions()``.
            ``{symbol: {qty, market_value, avg_entry, unrealized_pnl,
            current_price}}``
        risk_status :
            ``{kill_switch, drawdown_action, current_drawdown_pct,
            [liquidated]}``
        anomalies :
            List of anomaly messages from :class:`~agents.monitor.anomaly.AnomalyDetector`.

        Returns
        -------
        str
            Absolute path to the saved report file.
        """
        anomalies = anomalies or []
        date_str = pnl_snapshot.get("date") or dt.date.today().isoformat()
        severity = _classify_severity(risk_status, anomalies)

        lines: list[str] = []

        # ---- Alert banner (if needed) -----------------------------------
        if severity == "CRITICAL":
            lines += _banner("🚨  CRITICAL ALERT  🚨", char="!", width=72)
        elif severity == "WARNING":
            lines += _banner("⚠  WARNING  ⚠", char="~", width=72)

        # ---- Header -----------------------------------------------------
        lines += _banner(f"DAILY REPORT  {date_str}", width=72)

        # ---- Section 1: Portfolio Summary -------------------------------
        equity = pnl_snapshot.get("equity", 0.0)
        initial_equity = pnl_snapshot.get("initial_equity") or equity
        daily_ret = pnl_snapshot.get("daily_return")
        cum_ret = pnl_snapshot.get("cumulative_return", 0.0)
        hwm = pnl_snapshot.get("high_water_mark", equity)
        max_dd_ever = pnl_snapshot.get("max_drawdown_ever", 0.0)
        cur_dd = pnl_snapshot.get("current_drawdown", 0.0)

        daily_pnl_usd = (daily_ret or 0.0) * (equity / (1 + (daily_ret or 0.0)))
        daily_ret_str = f"{daily_ret * 100:+.3f}%" if daily_ret is not None else "N/A"
        daily_pnl_str = f"${daily_pnl_usd:+,.2f}" if daily_ret is not None else "N/A"

        lines += [
            "",
            "  [ 1 ]  PORTFOLIO SUMMARY",
            "  " + "-" * 50,
            f"  {'Equity':<30}  ${equity:>14,.2f}",
            f"  {'Initial Equity':<30}  ${initial_equity:>14,.2f}",
            f"  {'Daily PnL ($)':<30}  {daily_pnl_str:>15}",
            f"  {'Daily PnL (%)':<30}  {daily_ret_str:>15}",
            f"  {'Cumulative Return':<30}  {cum_ret * 100:>+14.3f}%",
            f"  {'High-Water Mark':<30}  ${hwm:>14,.2f}",
            f"  {'Current Drawdown':<30}  {cur_dd * 100:>+14.3f}%",
            f"  {'Max Drawdown (ever)':<30}  {max_dd_ever * 100:>+14.3f}%",
        ]

        # ---- Section 2: Position Detail ---------------------------------
        lines += ["", "  [ 2 ]  POSITION DETAIL", "  " + "-" * 70]
        if positions:
            hdr = (
                f"  {'Symbol':<8}  {'Weight':>7}  {'Mkt Value':>13}  "
                f"{'Unreal PnL':>12}  {'Avg Entry':>10}  {'Price':>8}"
            )
            lines.append(hdr)
            lines.append("  " + "-" * 70)
            for sym in sorted(positions):
                p = positions[sym]
                mv = p.get("market_value", 0.0)
                weight = mv / equity if equity > 0 else 0.0
                upnl = p.get("unrealized_pnl", 0.0)
                avg_e = p.get("avg_entry", 0.0)
                price = p.get("current_price", 0.0)
                daily_chg = (price / avg_e - 1.0) * 100 if avg_e > 0 else 0.0
                upnl_str = f"${upnl:>+10,.2f}"
                lines.append(
                    f"  {sym:<8}  {weight:>6.2%}  ${mv:>12,.2f}  "
                    f"{upnl_str}  ${avg_e:>9.2f}  ${price:>7.2f}"
                )
        else:
            lines.append("  (no open positions)")

        # ---- Section 3: Risk Status -------------------------------------
        ks = risk_status.get("kill_switch", "unknown")
        dd_action = risk_status.get("drawdown_action", "unknown")
        liquidated = risk_status.get("liquidated", False)

        ks_marker = "✓" if ks == "active" else ("✗" if ks == "killed" else "⚠")
        dd_marker = "✓" if dd_action == "hold" else ("✗" if dd_action == "exit" else "⚠")

        lines += [
            "",
            "  [ 3 ]  RISK STATUS",
            "  " + "-" * 50,
            f"  {'KillSwitch':<30}  {ks_marker} {ks}",
            f"  {'DrawdownGuard action':<30}  {dd_marker} {dd_action}",
            f"  {'Current Drawdown':<30}  {cur_dd * 100:>+14.3f}%",
        ]
        if liquidated:
            lines.append("  *** EMERGENCY LIQUIDATION WAS EXECUTED THIS SESSION ***")

        # ---- Anomaly section (if any) -----------------------------------
        if anomalies:
            lines += ["", "  [ ! ]  ANOMALIES DETECTED", "  " + "-" * 50]
            for msg in anomalies:
                lines.append(f"  • {msg}")

        lines += ["", "=" * 72, ""]

        report_text = "\n".join(lines)

        # ---- Save to file -----------------------------------------------
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._reports_dir / f"{date_str}.txt"
        out_path.write_text(report_text, encoding="utf-8")
        logger.info("Daily report saved → %s", out_path)

        return str(out_path.resolve())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _classify_severity(risk_status: dict[str, Any], anomalies: list[str]) -> str:
    """Return 'CRITICAL', 'WARNING', or 'OK'."""
    ks = risk_status.get("kill_switch", "active")
    dd = risk_status.get("drawdown_action", "hold")
    liquidated = risk_status.get("liquidated", False)

    if ks in ("killed", "triggered") or dd == "exit" or liquidated:
        return "CRITICAL"
    if ks != "active" or dd in ("reduce",) or anomalies:
        return "WARNING"
    return "OK"


def _banner(text: str, char: str = "=", width: int = 72) -> list[str]:
    bar = char * width
    return [bar, f"  {text:^{width - 4}}", bar]


# ---------------------------------------------------------------------------
# Legacy module-level function (kept for backward compatibility)
# ---------------------------------------------------------------------------


def generate_daily_report(*_args, **_kwargs) -> str:
    """Legacy stub — use DailyReporter.generate() instead."""
    raise NotImplementedError("Use DailyReporter.generate() instead.")
