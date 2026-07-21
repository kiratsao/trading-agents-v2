"""The 3-source day-close resolver: spot (^TWII) is night-proof truth; the MXF
source within the basis band and closest to spot is the day close; all-far-from-
spot fails loud; spot-missing degrades to TAIFEX/Shioaji agreement (never a
blanket reject)."""
from __future__ import annotations

from datetime import date

from src.data.spot_ref import resolve_day_close

D = date(2026, 7, 17)


def test_picks_day_when_one_source_is_night():
    val, _ = resolve_day_close(D, taifex=42697, shioaji=44714, spot=42671)
    assert val == 42697  # TAIFEX day, |26| from spot; Shioaji night rejected


def test_failloud_when_all_mxf_are_night():
    val, why = resolve_day_close(D, taifex=44714, shioaji=44714, spot=42671)
    assert val is None and "fail-loud" in why


def test_rescues_via_shioaji_when_taifex_is_night():
    val, _ = resolve_day_close(D, taifex=44714, shioaji=42697, spot=42671)
    assert val == 42697  # spot anchors on the Shioaji day value


def test_no_spot_uses_agreement(monkeypatch):
    monkeypatch.setattr("src.data.spot_ref.fetch_spot_close", lambda d, **k: None)
    val, why = resolve_day_close(D, taifex=42697, shioaji=42700, spot=None)
    assert abs(val - 42698.5) < 1 and "agree" in why


def test_no_spot_disagreement_fails_loud(monkeypatch):
    monkeypatch.setattr("src.data.spot_ref.fetch_spot_close", lambda d, **k: None)
    val, why = resolve_day_close(D, taifex=42697, shioaji=44714, spot=None)
    assert val is None and "disagree" in why


def test_no_spot_single_source_accepted_unverified(monkeypatch):
    monkeypatch.setattr("src.data.spot_ref.fetch_spot_close", lambda d, **k: None)
    val, why = resolve_day_close(D, taifex=42697, spot=None)
    assert val == 42697 and "unverified" in why
