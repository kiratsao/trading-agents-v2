"""Startup smoke test for the daemon's orchestrator-build path.

Regression guard for the phantom-kwarg crash-loop: main.py:_build_orchestrators
calls StateManager(path=..., initial_equity=...), but StateManager only accepted
`path` for a month. The bug survived 143 unit tests because the daemon startup
path (_build_orchestrators) had ZERO coverage and only fires under --live. This
test exercises it with a minimal config so any signature drift fails loudly.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scheduler.main import _build_orchestrators
from src.state.state_manager import StateManager

_MIN_CFG = {
    "accounts": {
        "mxf_test": {
            "product": "MXF",
            "equity": 350000,
            "strategy_params": {
                "ema_fast": 30,
                "ema_slow": 100,
                "atr_stop_mult": 2.0,
                "confirm_days": 2,
                "adx_threshold": 25,
            },
            "scale_ladder": [{"equity": 350000, "contracts": 2}],
            "margin_per_contract": 131500,
            "max_contracts": 15,
            "sessions": {"day": {"execution_timing": "night_open", "decision_time": "14:30"}},
        }
    }
}


def test_build_orchestrators_smoke():
    # Must not raise (TypeError/AttributeError on the StateManager call = the bug).
    orchestrators = _build_orchestrators(_MIN_CFG, live=False)
    assert "mxf_test" in orchestrators
    orch = orchestrators["mxf_test"]
    # The StateManager was constructed with the config's equity as initial_equity.
    assert isinstance(orch.state_mgr, StateManager)
    assert orch.state_mgr.initial_equity == 350000.0
    assert orch.execution_timing == "night_open"
    assert orch.strategy is not None


# ── StateManager initial_equity semantics (Task 1) ──────────────────────────
def test_state_manager_initial_equity_on_first_run(tmp_path):
    mgr = StateManager(path=str(tmp_path / "state.json"), initial_equity=500_000)
    assert mgr.load().equity == 500_000.0  # no file → use initial_equity


def test_state_manager_existing_file_takes_precedence(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"state": {"equity": 800_000}}), encoding="utf-8")
    mgr = StateManager(path=str(p), initial_equity=500_000)
    assert mgr.load().equity == 800_000.0  # file wins, initial_equity ignored


def test_state_manager_default_when_none(tmp_path):
    mgr = StateManager(path=str(tmp_path / "nope.json"))  # initial_equity defaults None
    assert mgr.load().equity == 350_000.0  # unchanged path-only behaviour
