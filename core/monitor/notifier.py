"""Alert and report notifications via LINE Messaging API and SMTP email.

Priority: LINE Push Message → SMTP email → log-only.

LINE setup
----------
Set these environment variables (or in ``.env``):

    LINE_CHANNEL_ACCESS_TOKEN   Long-lived token from LINE Developers Console
                                → Messaging API → Channel access token → Issue
    LINE_USER_ID                Recipient user ID (starts with "U")

SMTP setup (optional fallback)
------------------------------
    SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / NOTIFY_EMAIL

If neither LINE nor SMTP is configured the notifier degrades to log-only
mode — it never raises an exception, so the rest of the pipeline continues.

LINE emoji prefixes by level
----------------------------
    CRITICAL  →  🚨
    WARNING   →  ⚠️
    INFO      →  📊
"""

from __future__ import annotations

import json as _json
import logging
import smtplib
import traceback
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)

_LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"

_LEVEL_EMOJI = {
    "CRITICAL": "🚨",
    "WARNING": "⚠️",
    "INFO": "📊",
}

# ── Lazy settings loader ───────────────────────────────────────────────────────

_SETTINGS_LOADED = False
_LINE_TOKEN: str = ""
_LINE_USER: str = ""
_SMTP_HOST: str = ""
_SMTP_PORT: int = 587
_SMTP_USER: str = ""
_SMTP_PASS: str = ""
_NOTIFY_EMAIL: str = ""


def _load_settings() -> None:
    global _SETTINGS_LOADED
    global _LINE_TOKEN, _LINE_USER
    global _SMTP_HOST, _SMTP_PORT, _SMTP_USER, _SMTP_PASS, _NOTIFY_EMAIL
    if _SETTINGS_LOADED:
        return
    try:
        from core.config.settings import settings

        _LINE_TOKEN = settings.LINE_CHANNEL_ACCESS_TOKEN
        _LINE_USER = settings.LINE_USER_ID
        _SMTP_HOST = settings.SMTP_HOST
        _SMTP_PORT = settings.SMTP_PORT
        _SMTP_USER = settings.SMTP_USER
        _SMTP_PASS = settings.SMTP_PASS
        _NOTIFY_EMAIL = settings.NOTIFY_EMAIL
    except Exception as exc:
        logger.debug("Notifier: could not load settings — %s", exc)
    _SETTINGS_LOADED = True


# ── Notifier class ─────────────────────────────────────────────────────────────


class Notifier:
    """Send notifications via LINE Push Message (primary) and SMTP email (fallback).

    Parameters
    ----------
    line_token, line_user_id :
        Override LINE credentials (fall back to settings if None).
    smtp_host, smtp_port, smtp_user, smtp_pass, notify_email :
        Override SMTP parameters (fall back to settings if None).

    Example
    -------
    >>> n = Notifier()
    >>> n.send_line("策略信號：BUY 2口 TX @ 33,402", level="INFO")
    >>> n.send_alert("KillSwitch 觸發", "單日虧損超過 5%", level="CRITICAL")
    """

    def __init__(
        self,
        line_token: str | None = None,
        line_user_id: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_pass: str | None = None,
        notify_email: str | None = None,
    ) -> None:
        _load_settings()
        self._line_token = line_token if line_token is not None else _LINE_TOKEN
        self._line_user = line_user_id if line_user_id is not None else _LINE_USER
        self._host = smtp_host if smtp_host is not None else _SMTP_HOST
        self._port = smtp_port if smtp_port is not None else _SMTP_PORT
        self._smtp_user = smtp_user if smtp_user is not None else _SMTP_USER
        self._smtp_pass = smtp_pass if smtp_pass is not None else _SMTP_PASS
        self._to = notify_email if notify_email is not None else _NOTIFY_EMAIL

        self._line_configured = bool(self._line_token and self._line_user)
        self._email_configured = bool(self._host and self._smtp_user and self._to)

        if not self._line_configured and not self._email_configured:
            logger.info(
                "Notifier: LINE and SMTP not configured — notifications will be logged only."
            )
        elif self._line_configured:
            logger.debug("Notifier: LINE configured (user=%s…)", self._line_user[:8])

    # ------------------------------------------------------------------
    # LINE Push Message
    # ------------------------------------------------------------------

    def send_line(self, message: str, level: str = "INFO") -> bool:
        """Push a text message to LINE.

        Parameters
        ----------
        message :
            Plain text (max ~5,000 chars; LINE truncates beyond that).
        level :
            ``"INFO"`` / ``"WARNING"`` / ``"CRITICAL"``.
            Emoji prefix is prepended automatically.

        Returns
        -------
        bool
            ``True`` if delivered; ``False`` on failure or not configured.
        """
        emoji = _LEVEL_EMOJI.get(level.upper(), "📊")
        body = f"{emoji} {message}"

        logger.info("LINE [%s]: %s", level, message[:120])

        if not self._line_configured:
            logger.info("LINE not configured — would have sent: %r", body[:80])
            return False

        payload = _json.dumps(
            {
                "to": self._line_user,
                "messages": [{"type": "text", "text": body}],
            }
        ).encode()

        req = urllib.request.Request(
            _LINE_PUSH_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._line_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            if status == 200:
                logger.info("LINE push delivered (status=200)")
                return True
            logger.warning("LINE push unexpected status=%d", status)
            return False
        except urllib.error.HTTPError as exc:
            body_err = exc.read().decode(errors="replace")
            logger.error("LINE push HTTP %d: %s", exc.code, body_err[:200])
        except urllib.error.URLError as exc:
            logger.error("LINE push network error: %s", exc.reason)
        except Exception:
            logger.error("LINE push unexpected error:\n%s", traceback.format_exc())
        return False

    # ------------------------------------------------------------------
    # High-level alert / report methods
    # ------------------------------------------------------------------

    def send_alert(
        self,
        subject: str,
        body: str,
        level: str = "WARNING",
    ) -> bool:
        """Send an alert via LINE (primary) then email (fallback).

        Parameters
        ----------
        subject :
            Alert headline (used as email subject; prepended to LINE message).
        body :
            Detail text.
        level :
            ``"WARNING"`` or ``"CRITICAL"``.

        Returns
        -------
        bool
            ``True`` if delivered by any channel.
        """
        level_upper = level.upper()
        if level_upper == "CRITICAL":
            logger.error("ALERT [CRITICAL]: %s | %s", subject, body)
        else:
            logger.warning("ALERT [WARNING]: %s | %s", subject, body)

        # Try LINE first
        line_message = f"{subject}\n{body}"
        if self.send_line(line_message, level=level_upper):
            return True

        # Fallback to email
        prefix = "🚨 CRITICAL — " if level_upper == "CRITICAL" else "[WARNING] "
        return self._send_email(f"{prefix}{subject}", body)

    def send_daily_report(self, report: str, date_str: str | None = None) -> bool:
        """Send the daily report via LINE (primary) then email (fallback).

        Parameters
        ----------
        report :
            Full report text from DailyReporter.
        date_str :
            Date label (defaults to today).

        Returns
        -------
        bool
            ``True`` if delivered.
        """
        import datetime as dt

        if date_str is None:
            date_str = dt.date.today().isoformat()

        logger.info("Sending daily report for %s …", date_str)

        # Try LINE first (truncate if too long)
        line_msg = f"Daily Report — {date_str}\n\n{report}"
        if len(line_msg) > 4_900:
            line_msg = line_msg[:4_897] + "…"
        if self.send_line(line_msg, level="INFO"):
            return True

        # Fallback to email
        return self._send_email(f"Trading Report — {date_str}", report)

    # ------------------------------------------------------------------
    # TW Futures specific helpers
    # ------------------------------------------------------------------

    def notify_tw_daily(
        self,
        date_str: str,
        signal: dict[str, Any],
        position: dict[str, Any],
        indicators: dict[str, Any],
        kill_switch: str = "ACTIVE",
        dry_run: bool = False,
    ) -> bool:
        """Push formatted 台指期 daily summary to LINE.

        Parameters
        ----------
        date_str :   ``"2026-04-07"``
        signal :     ``{"action": "hold", "contracts": 0, "reason": "..."}``
        position :   ``{"direction": 0, "contracts": 0, "entry_price": None}``
        indicators : ``{"ema_fast": 32469, "ema_slow": 29464, "adx": 14.2, "atr": 1054, "close": 33402}``
        kill_switch : ``"ACTIVE"`` | ``"KILLED"``
        dry_run :    Whether this is a dry-run.
        """
        action = signal.get("action", "hold").upper()
        contracts = signal.get("contracts", 0)
        reason = signal.get("reason", "")

        ema_f = indicators.get("ema_fast", 0)
        ema_s = indicators.get("ema_slow", 0)
        adx = indicators.get("adx", 0.0)
        atr = indicators.get("atr", 0.0)
        close = indicators.get("close", 0)

        direction = int(position.get("direction", 0))
        pos_str = {1: "多頭持倉", -1: "空頭持倉", 0: "空倉"}.get(direction, "未知")
        if direction != 0:
            pos_str += f" {position.get('contracts', 0)}口"

        # Signal emoji
        sig_emoji = {
            "BUY": "📈",
            "SELL": "📉",
            "CLOSE": "🔴",
            "HOLD": "⏸",
        }.get(action, "📊")

        # ADX comment
        if adx >= 20:
            adx_note = f"ADX={adx:.1f} ✓"
        else:
            adx_note = f"ADX={adx:.1f} < 20（趨勢弱）"

        # ATR warning
        atr_note = f"ATR={atr:.0f}"
        if atr > 600:
            atr_note += " ⚠️ 偏高"

        # EMA trend
        ema_note = f"EMA50={ema_f:,.0f} {'>' if ema_f > ema_s else '<'} EMA150={ema_s:,.0f}"
        ema_trend = "多頭 ✓" if ema_f > ema_s else "空頭（Long-Only 觀望）"

        ks_note = "ACTIVE ✓" if kill_switch == "ACTIVE" else "⚠️ KILLED"
        mode = " [DRY-RUN]" if dry_run else ""

        lines = [
            f"📊 台指期日報 {date_str}{mode}",
            "策略：V2b EMA50/150 Long-Only 2口",
            "",
            f"信號：{sig_emoji} {action}" + (f"（{reason[:60]}）" if reason else ""),
            f"持倉：{pos_str}",
            f"收盤：{close:,.0f}",
            "",
            f"{ema_note}（{ema_trend}）",
            f"{adx_note}",
            f"{atr_note}",
            "",
            f"KillSwitch：{ks_note}",
        ]
        message = "\n".join(lines)

        level = "INFO"
        if action in ("BUY", "SELL"):
            level = "INFO"
        return self.send_line(message, level=level)

    def notify_tw_trade(
        self,
        action: str,
        contracts: int,
        price: float,
        reason: str = "",
        hard_stop: float | None = None,
    ) -> bool:
        """Push trade execution notification to LINE.

        action : ``"buy"`` | ``"sell"`` | ``"close"``
        """
        emoji = {"buy": "📈", "sell": "📉", "close": "🔴"}.get(action.lower(), "📊")
        dir_str = {"buy": "開多", "sell": "開空", "close": "平倉"}.get(action.lower(), action)

        lines = [
            f"{emoji} TX 台指期 {dir_str}",
            f"口數：{contracts} 口",
            f"價格：{price:,.0f}",
        ]
        if hard_stop:
            lines.append(f"硬停損：{hard_stop:,.0f}")
        if reason:
            lines.append(f"原因：{reason[:80]}")

        level = (
            "CRITICAL" if action.lower() == "close" and "hard stop" in reason.lower() else "INFO"
        )
        return self.send_line("\n".join(lines), level=level)

    def notify_kill_switch(self, reason: str) -> bool:
        """Push KillSwitch trigger alert to LINE."""
        message = f"KillSwitch 觸發\n{reason}\n⚠️ 所有新訂單已封鎖，請立即確認帳戶狀態。"
        return self.send_line(message, level="CRITICAL")

    # ------------------------------------------------------------------
    # SMTP fallback (internal)
    # ------------------------------------------------------------------

    def _send_email(self, subject: str, body: str) -> bool:
        if not self._email_configured:
            logger.info(
                "Email not configured — would have sent: subject=%r (body length=%d)",
                subject,
                len(body),
            )
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._smtp_user
        msg["To"] = self._to
        msg.set_content(body)

        try:
            with smtplib.SMTP(self._host, self._port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self._smtp_user, self._smtp_pass)
                server.send_message(msg)
            logger.info("Email sent → %s  subject=%r", self._to, subject)
            return True
        except smtplib.SMTPAuthenticationError as exc:
            logger.error("SMTP authentication failed: %s", exc)
        except smtplib.SMTPException as exc:
            logger.error("SMTP error sending %r: %s", subject, exc)
        except OSError as exc:
            logger.error("Network error sending email: %s", exc)
        except Exception:
            logger.error("Unexpected error sending email:\n%s", traceback.format_exc())
        return False
