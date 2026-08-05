"""Shared test fixtures.

Network hermeticity: the 幽靈-bar calendar guard (``_calendar_guard``) downloads
a TAIFEX month CSV for every date about to be written to parquet. Tests default
to "TAIFEX confirms this trading day" so every existing updater/gap-fill test
stays offline and behavior-identical; the guard's own tests override
``_taifex_confirms_day`` explicitly.
"""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_calendar_guard(monkeypatch):
    from src.data import daily_updater

    monkeypatch.setattr(daily_updater, "_taifex_confirms_day", lambda d: True)
    # The live provenance gate (orchestrator._validate_today_bar → daily_updater
    # ._night_provenance) fetches TAIFEX month CSVs. Null the two underlying
    # fetchers so provenance degrades to False offline; tests exercising real
    # provenance semantics re-patch these themselves (they already did).
    monkeypatch.setattr(daily_updater, "_taifex_night_close", lambda d: None)
    monkeypatch.setattr(daily_updater, "_taifex_day_bar", lambda d: None)
