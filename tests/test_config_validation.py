"""Startup config-key validation: an accounts.yaml key the engine wiring does
not consume is a silent no-op (the operator believes a knob is wired when it is
not — e.g. `atr_stop_mult` only works because main.py explicitly maps it to
`trail_atr_mult`; a typo like `risk_cap` would quietly disable the risk cap on
a live account). Unknown keys must abort startup."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.scheduler.main import _validate_account_config

_GOOD = {
    "product": "MXF",
    "equity": 1_000_000,
    "strategy_params": {
        "ema_fast": 30, "ema_slow": 100, "atr_stop_mult": 2.0,
        "confirm_days": 2, "adx_threshold": 25,
        "risk_cap_pct": 0.55, "margin_buffer_atr": 1.0,
    },
    "scale_ladder": [{"equity": 350_000, "contracts": 2}],
    "margin_per_contract": 131_500,
    "max_contracts": 3,
    "settlement_force_close": True,
    "sessions": {"day": {"execution_timing": "night_open",
                         "decision_time": "14:30", "execution_time": "15:05"}},
}


def test_production_key_set_passes():
    _validate_account_config("mxf_aggressive", _GOOD)  # must not raise


def test_repo_accounts_yaml_passes():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "accounts.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for name, acc in (cfg.get("accounts") or {}).items():
        _validate_account_config(name, acc)


def test_typoed_strategy_key_fails_loud():
    bad = {**_GOOD, "strategy_params": {**_GOOD["strategy_params"],
                                        "risk_cap": 0.2}}  # typo of risk_cap_pct
    with pytest.raises(SystemExit, match="risk_cap"):
        _validate_account_config("mxf_aggressive", bad)


def test_unknown_account_key_fails_loud():
    bad = {**_GOOD, "margin_percontract": 131_500}
    with pytest.raises(SystemExit, match="margin_percontract"):
        _validate_account_config("mxf_aggressive", bad)


def test_unknown_session_day_key_fails_loud():
    bad = {**_GOOD, "sessions": {"day": {"execution_timing": "night_open",
                                         "decission_time": "14:30"}}}
    with pytest.raises(SystemExit, match="decission_time"):
        _validate_account_config("mxf_aggressive", bad)


def test_unknown_session_group_fails_loud():
    bad = {**_GOOD, "sessions": {**_GOOD["sessions"], "night": {}}}
    with pytest.raises(SystemExit, match="night"):
        _validate_account_config("mxf_aggressive", bad)
