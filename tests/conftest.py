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
    # validate_latest_bar binds its default oracles as module-level ALIASES at
    # import time (validation.py: _default_taifex_fetch = fetch_taifex_...),
    # so patching the fetch by NAME does not redirect them — tests driving
    # update() with the default validate_fn silently hit the real TAIFEX site
    # (verifier finding 2026-08-08). Null the aliases; oracle-specific tests
    # inject their own fetchers via validate_latest_bar kwargs.
    # Only the TAIFEX alias is nulled: it needs no credentials so it ALWAYS
    # reaches the real site from tests; the Shioaji default already degrades
    # hermetically without creds (fetch_via_env raises → oracle skipped) and
    # its wiring has a dedicated test.
    from src.data import validation

    monkeypatch.setattr(validation, "_default_taifex_fetch", lambda s, e: None)
    # TAIFEX month-CSV TTL cache: never let bytes leak across tests.
    from scripts import init_data

    init_data._csv_cache_clear()
    # The live provenance gate (orchestrator._validate_today_bar → daily_updater
    # ._night_provenance) fetches TAIFEX month CSVs. Null the two underlying
    # fetchers so provenance degrades to False offline; tests exercising real
    # provenance semantics re-patch these themselves (they already did).
    monkeypatch.setattr(daily_updater, "_taifex_night_close", lambda d: None)
    monkeypatch.setattr(daily_updater, "_taifex_day_bar", lambda d: None)
