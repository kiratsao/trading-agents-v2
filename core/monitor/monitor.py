"""Monitor orchestrator — integrates PnLTracker, AnomalyDetector,
DailyReporter, and Notifier into a single run_daily() call.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd

from .anomaly import AnomalyDetector
from .notifier import Notifier
from .pnl import PnLTracker
from .reporter import DailyReporter

logger = logging.getLogger(__name__)


class Monitor:
    """Live PnL tracking, anomaly detection, and daily report delivery.

    Parameters
    ----------
    db_path :
        SQLite path (kept for backward compatibility; not used by the
        current implementation but accepted so legacy callers continue
        to work).

    Example
    -------
    >>> mon = Monitor()
    >>> result = mon.run_daily(
    ...     equity=1_050_000,
    ...     positions=adapter.get_positions(),
    ...     risk_status={"kill_switch": "active", "drawdown_action": "hold"},
    ...     daily_returns=pnl_tracker.get_daily_returns_series(),
    ... )
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path
        self.pnl_tracker = PnLTracker()
        self.anomaly_detector = AnomalyDetector()
        self.reporter = DailyReporter()
        self.notifier = Notifier()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run_daily(
        self,
        equity: float,
        positions: dict[str, dict],
        risk_status: dict[str, Any],
        daily_returns: pd.Series | None = None,
        timestamp: dt.datetime | None = None,
    ) -> dict:
        """Execute the full daily monitoring pipeline.

        Steps
        -----
        1. Update PnL metrics from current equity.
        2. Run anomaly detection on recent daily returns.
        3. Generate and persist the daily report.
        4. Send alert if anomalies or critical risk conditions are present.
        5. Send the daily report email.

        Parameters
        ----------
        equity :
            Current portfolio NAV in USD.
        positions :
            Live positions from ``AlpacaAdapter.get_positions()``.
        risk_status :
            ``{kill_switch, drawdown_action, current_drawdown_pct,
            [liquidated]}``
        daily_returns :
            Recent portfolio daily returns (pd.Series).  If None, the
            PnLTracker's history is used.
        timestamp :
            Override for "now" (defaults to UTC now).

        Returns
        -------
        dict
            ``{report_path, anomalies, pnl_snapshot, notifications_sent}``
        """
        timestamp = timestamp or dt.datetime.now(dt.UTC)
        result: dict[str, Any] = {
            "report_path": None,
            "anomalies": [],
            "pnl_snapshot": {},
            "notifications_sent": [],
        }

        # ---- Step 1: update PnL ----------------------------------------
        pnl_snapshot = self.pnl_tracker.update(equity, timestamp)
        result["pnl_snapshot"] = pnl_snapshot
        # Augment risk_status with current drawdown from PnL tracker
        risk_status = dict(risk_status)
        risk_status.setdefault(
            "current_drawdown_pct",
            pnl_snapshot.get("current_drawdown", 0.0) * 100,
        )

        # ---- Step 2: anomaly detection ----------------------------------
        returns_series = (
            daily_returns
            if daily_returns is not None
            else self.pnl_tracker.get_daily_returns_series()
        )
        anomalies = self.anomaly_detector.check(returns_series)
        result["anomalies"] = anomalies

        # ---- Step 3: generate daily report ------------------------------
        try:
            report_path = self.reporter.generate(
                pnl_snapshot=pnl_snapshot,
                positions=positions,
                risk_status=risk_status,
                anomalies=anomalies,
            )
            result["report_path"] = report_path
        except Exception as exc:
            logger.error("DailyReporter.generate() failed: %s", exc)
            report_path = None

        # ---- Step 4: send alert if needed -------------------------------
        from .reporter import _classify_severity

        severity = _classify_severity(risk_status, anomalies)

        if severity in ("CRITICAL", "WARNING"):
            parts: list[str] = []
            if anomalies:
                parts.append("Anomalies: " + "; ".join(anomalies))
            ks = risk_status.get("kill_switch", "")
            dd = risk_status.get("drawdown_action", "")
            if ks not in ("active", ""):
                parts.append(f"KillSwitch={ks}")
            if dd not in ("hold", ""):
                parts.append(f"DrawdownGuard={dd}")

            alert_subject = "Portfolio risk alert"
            alert_body = "\n".join(parts) or "Risk threshold breached — review logs."
            sent = self.notifier.send_alert(alert_subject, alert_body, level=severity)
            if sent:
                result["notifications_sent"].append(
                    {"type": "alert", "level": severity, "subject": alert_subject}
                )

        # ---- Step 5: send daily report ----------------------------------
        if report_path:
            try:
                report_text = open(report_path, encoding="utf-8").read()
            except Exception:
                report_text = "(report unavailable)"
            date_str = pnl_snapshot.get("date")
            sent = self.notifier.send_daily_report(report_text, date_str=date_str)
            if sent:
                result["notifications_sent"].append({"type": "daily_report", "date": date_str})

        logger.info(
            "Monitor.run_daily: severity=%s  anomalies=%d  report=%s  notifications=%d",
            severity,
            len(anomalies),
            report_path or "(none)",
            len(result["notifications_sent"]),
        )
        return result

    # ------------------------------------------------------------------
    # Legacy interface (kept for backward compatibility)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Legacy stub — use run_daily() for live monitoring."""
        raise NotImplementedError("Use Monitor.run_daily() for live monitoring.")

    def run_report(self) -> str:
        """Legacy stub — use run_daily() which calls DailyReporter internally."""
        raise NotImplementedError("Use Monitor.run_daily() which calls DailyReporter internally.")

    def check_anomalies(self) -> list[dict]:
        """Legacy stub — use AnomalyDetector.check(daily_returns) directly."""
        raise NotImplementedError("Use AnomalyDetector.check(daily_returns) directly.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Monitor Agent")
    parser.add_argument("--report-today", action="store_true")
    args = parser.parse_args()
    logger.info("Monitor starting (report_today=%s)", args.report_today)
