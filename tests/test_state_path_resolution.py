"""Regression: monitoring/verification scripts must resolve the daemon's
canonical per-account state file (``data/state_{account}.json``), never the
orphaned ``data/paper_state.json`` the daemon stopped writing (which produced
the daily false "broker=0, state=20 ⚠️" alert)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from src.state import state_manager
from src.state.state_manager import StateManager, resolve_state_path


def test_resolver_returns_daemon_convention_path():
    # Same convention main.py uses: data/state_{accounts.yaml-key}.json
    assert resolve_state_path("mxf_aggressive") == Path("data/state_mxf_aggressive.json")


def test_resolver_raises_on_unknown_account():
    with pytest.raises(KeyError):
        resolve_state_path("nope_not_an_account")


def test_resolver_degrades_when_config_missing(tmp_path):
    # Partial deployment (no config) → convention path, never a hard crash.
    missing = str(tmp_path / "nope.yaml")
    assert resolve_state_path("mxf_aggressive", missing) == Path(
        "data/state_mxf_aggressive.json"
    )


def test_statemanager_default_is_not_the_orphan():
    assert "paper_state" not in StateManager().path.name


def test_post_execution_verify_reads_canonical_state(tmp_path):
    """A flat broker (0) vs the canonical state (0) → the previously-false
    "持倉口數" line now passes as broker=0, state=0."""
    sp = tmp_path / "state_mxf_aggressive.json"
    sp.write_text(
        json.dumps(
            {
                "state": {
                    "position": 0,
                    "contracts": 0,
                    "equity": 1_200_938.0,
                    "pending_action": "hold",
                },
                "trades": [],
            }
        )
    )

    class _Contract:
        code = "MXF202608"

    class _FakeAdapter:
        def __init__(self, *a, **k):
            pass

        def get_positions(self):
            return []

        def get_account(self):
            return {"equity": 1_200_938.0}

        def get_contract(self, product="MXF"):
            return _Contract()

        def logout(self):
            pass

    import scripts.post_execution_verify as pev

    with mock.patch.object(state_manager, "resolve_state_path", lambda: sp), mock.patch(
        "tw_futures.executor.shioaji_adapter.ShioajiAdapter", _FakeAdapter
    ):
        res = pev.run_verify(skip_external=False, notify_fn=None)

    pos = next(r for r in res["results"] if r["name"] == "持倉口數")
    assert pos["detail"] == "broker=0, state=0"
    assert pos["passed"] is True
