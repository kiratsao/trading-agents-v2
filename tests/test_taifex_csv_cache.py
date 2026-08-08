"""TAIFEX 月 CSV TTL 快取: 單一 update cycle 內 primary/guard/oracle/provenance
必須讀到同一份 bytes (cycle 內一致性 — 消除 TAIFEX-vs-TAIFEX 自打與 guard 在
primary 成功後的瞬時失敗; verifier 2026-08-08), 並把每 cycle 下載 4→1。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts import init_data


def _fake_urlopen_factory(counter: list):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return "交易日期,收盤價\n".encode("big5")

    def _urlopen(req, timeout=None):
        counter.append(req)
        return _Resp()

    return _urlopen


def test_same_month_served_from_cache():
    calls: list = []
    with patch("urllib.request.urlopen", _fake_urlopen_factory(calls)):
        a = init_data._download_taifex_csv(2026, 8, "MTX")
        b = init_data._download_taifex_csv(2026, 8, "MTX")
        init_data._download_taifex_csv(2026, 7, "MTX")       # 不同月 → 下載
    assert a == b
    assert len(calls) == 2          # 8月一次 + 7月一次


def test_ttl_expiry_refetches(monkeypatch):
    calls: list = []
    fake_now = [1000.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_now[0])
    with patch("urllib.request.urlopen", _fake_urlopen_factory(calls)):
        init_data._download_taifex_csv(2026, 8, "MTX")
        fake_now[0] += init_data._CSV_TTL_S + 1
        init_data._download_taifex_csv(2026, 8, "MTX")
    assert len(calls) == 2


def test_failure_not_cached():
    calls: list = []
    boom = MagicMock(side_effect=OSError("net down"))
    with patch("urllib.request.urlopen", boom):
        try:
            init_data._download_taifex_csv(2026, 8, "MTX")
        except OSError:
            pass
    with patch("urllib.request.urlopen", _fake_urlopen_factory(calls)):
        out = init_data._download_taifex_csv(2026, 8, "MTX")
    assert len(calls) == 1 and out    # 失敗未快取, 重試成功
