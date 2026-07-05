"""共用 LINE notifier + 跨 process 去重。

此前 repo 內有 8 份複製貼上的 urllib LINE 發送函式(main.py、daily_updater_cli、
各健康檢查腳本各一份),且完全沒有去重機制——同一個資料缺口每天被 daemon(14:25)
與 systemd timer(14:30)各重發一次,crash-loop 時啟動通知連發。所有 production
發送路徑一律改走這裡。

去重設計
--------
`DedupNotifier` 以「訊息文字的 hash」為 key(可用 ``send(msg, key=..., ttl=...)``
指定語意 key),記錄最後發送時間於 ``data/notify_journal.json``。同 key 在 TTL 內
重複出現時抑制不發、只記 log。journal 檔由 daemon / updater CLI / 健康檢查等
process 共用,因此跨 process 也能去重(兩邊措辭需一致才會撞 key)。

- 預設 TTL:0(不去重)——透過 ``__call__`` 的舊介面呼叫時行為與從前一致,
  除非訊息以 🔴/⚠️ 開頭(告警類),預設套用 ``ALERT_TTL``(20 小時):
  同一告警一天最多一次,而不是每天 ×2、每次排程都發。
- ✅/📊 等一般訊息不去重(內容含日期/數字,天然唯一,且屬使用者要的 heartbeat)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

from src.utils.tw_time import now_taipei

logger = logging.getLogger(__name__)

#: 告警類訊息(🔴/⚠️ 開頭)的預設抑制窗:20h ⇒ 同一內容一天最多發一次。
ALERT_TTL_SECONDS = 20 * 3600

_JOURNAL_PATH = Path("data/notify_journal.json")
_JOURNAL_PRUNE_SECONDS = 7 * 24 * 3600


def _load_journal(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_journal(path: Path, journal: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(journal, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # journal 壞掉絕不能擋通知
        logger.warning("notify journal save failed: %s", exc)


class DedupNotifier:
    """Callable notifier with TTL-based suppression.

    ``notifier(msg)`` — drop-in replacement for the plain notify_fn everywhere.
    ``notifier.send(msg, key=..., ttl=...)`` — explicit semantic key / TTL.
    """

    def __init__(self, send_fn, journal_path: str | Path = _JOURNAL_PATH) -> None:
        self._send_fn = send_fn
        self._journal_path = Path(journal_path)

    # -- public ---------------------------------------------------------
    def __call__(self, msg: str) -> None:
        ttl = ALERT_TTL_SECONDS if msg.startswith(("🔴", "⚠️")) else 0
        self.send(msg, ttl=ttl)

    def send(self, msg: str, *, key: str | None = None, ttl: int = 0) -> None:
        if ttl > 0 and self._suppressed(key or _text_key(msg), ttl):
            logger.info("notify suppressed (dedup, ttl=%ds): %s", ttl, msg.splitlines()[0])
            return
        self._send_fn(msg)
        if ttl > 0:
            self._record(key or _text_key(msg))

    def seconds_since(self, key: str) -> float | None:
        """Seconds since *key* was last sent, or None if never/unknown."""
        journal = _load_journal(self._journal_path)
        ts = journal.get(key, {}).get("ts")
        if ts is None:
            return None
        try:
            from datetime import datetime

            last = datetime.fromisoformat(ts)
        except ValueError:
            return None
        return (now_taipei() - last).total_seconds()

    def record(self, key: str) -> None:
        """Public journal stamp for callers doing their own gating."""
        self._record(key)

    # -- internals ------------------------------------------------------
    def _suppressed(self, key: str, ttl: int) -> bool:
        age = self.seconds_since(key)
        return age is not None and 0 <= age < ttl

    def _record(self, key: str) -> None:
        journal = _load_journal(self._journal_path)
        now = now_taipei()
        journal[key] = {"ts": now.isoformat()}
        # prune stale entries so the journal never grows unbounded
        from datetime import datetime

        def _fresh(entry: dict) -> bool:
            try:
                return (now - datetime.fromisoformat(entry["ts"])).total_seconds() < (
                    _JOURNAL_PRUNE_SECONDS
                )
            except (KeyError, ValueError):
                return False

        journal = {k: v for k, v in journal.items() if _fresh(v)}
        _save_journal(self._journal_path, journal)


def _text_key(msg: str) -> str:
    return hashlib.sha1(msg.encode("utf-8")).hexdigest()[:16]


def _raw_line_sender(token: str, user_id: str):
    def _send(msg: str) -> None:
        payload = {"to": user_id, "messages": [{"type": "text", "text": msg}]}
        req = urllib.request.Request(
            "https://api.line.me/v2/bot/message/push",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("LINE notified: %s", resp.status)
        except Exception as exc:
            logger.warning("LINE notify failed: %s", exc)

    return _send


def build_line_notifier(*, dedup: bool = True) -> Any:
    """Build the shared LINE notifier from env vars (no-op when unset).

    Returns a callable ``fn(msg)``; when *dedup* is True it is a
    :class:`DedupNotifier` (alert-class messages auto-deduped).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")
    if not token or not user_id:
        logger.info(
            "LINE notifier disabled (LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID not set)"
        )
        return (lambda msg: None) if not dedup else DedupNotifier(lambda msg: None)
    sender = _raw_line_sender(token, user_id)
    return DedupNotifier(sender) if dedup else sender
