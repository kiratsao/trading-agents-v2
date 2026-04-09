"""Tests for StateManager — JSON persistence.

Coverage
--------
* Load from empty / missing file
* Save + load roundtrip
* Atomic write (tmp + rename)
* append_trade atomic write
* Corrupted JSON recovery
* All TradingState fields
"""

from __future__ import annotations

import json

import pytest

from src.state.state_manager import StateManager, TradingState


# ===========================================================================
# Default state
# ===========================================================================


class TestDefaultState:
    def test_default_values(self):
        s = TradingState()
        assert s.position == 0
        assert s.entry_price is None
        assert s.equity == 350_000.0
        assert s.pyramided is False
        assert s.pending_action is None
        assert s.pending_contracts == 0

    def test_load_missing_file(self, tmp_path):
        mgr = StateManager(path=str(tmp_path / "nonexistent.json"))
        state = mgr.load()
        assert state.position == 0
        assert state.equity == 350_000.0


# ===========================================================================
# Save + Load roundtrip
# ===========================================================================


class TestSaveLoad:
    def test_roundtrip_basic(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        state = TradingState(
            position=2, entry_price=20000.0, equity=400_000.0,
            highest_high=20500.0, contracts=2, pyramided=False,
        )
        mgr.save(state)
        loaded = mgr.load()
        assert loaded.position == 2
        assert loaded.entry_price == 20000.0
        assert loaded.equity == 400_000.0
        assert loaded.highest_high == 20500.0
        assert loaded.contracts == 2
        assert loaded.pyramided is False

    def test_roundtrip_pending_action(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        state = TradingState(
            pending_action="buy", pending_contracts=3,
            pending_signal_date="2025-03-19",
        )
        mgr.save(state)
        loaded = mgr.load()
        assert loaded.pending_action == "buy"
        assert loaded.pending_contracts == 3
        assert loaded.pending_signal_date == "2025-03-19"

    def test_roundtrip_pyramided(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        state = TradingState(pyramided=True, position=3, contracts=3)
        mgr.save(state)
        loaded = mgr.load()
        assert loaded.pyramided is True

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "state.json")
        mgr = StateManager(path=path)
        mgr.save(TradingState())
        loaded = mgr.load()
        assert loaded.position == 0

    def test_none_entry_price_roundtrip(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        state = TradingState(position=0, entry_price=None)
        mgr.save(state)
        loaded = mgr.load()
        assert loaded.entry_price is None


# ===========================================================================
# Atomic write
# ===========================================================================


class TestAtomicWrite:
    def test_no_tmp_file_left_after_save(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        mgr.save(TradingState(equity=500_000))
        # tmp file should not exist after successful save
        tmp_file = tmp_path / "state.json.tmp"
        assert not tmp_file.exists()

    def test_json_is_valid_after_save(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        mgr.save(TradingState(position=2, equity=450_000))
        content = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert "state" in content
        assert content["state"]["position"] == 2


# ===========================================================================
# Corrupted file recovery
# ===========================================================================


class TestCorruptedFile:
    def test_load_from_corrupted_json(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        mgr = StateManager(path=str(path))
        state = mgr.load()
        # Should return default state
        assert state.position == 0
        assert state.equity == 350_000.0

    def test_load_from_empty_file(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("", encoding="utf-8")
        mgr = StateManager(path=str(path))
        state = mgr.load()
        assert state.position == 0

    def test_load_from_missing_state_key(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text('{"trades": []}', encoding="utf-8")
        mgr = StateManager(path=str(path))
        state = mgr.load()
        assert state.position == 0


# ===========================================================================
# append_trade
# ===========================================================================


class TestAppendTrade:
    def test_append_trade_creates_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        trade = {"entry_date": "2025-01-01", "exit_date": "2025-01-10", "pnl": 5000}
        mgr.append_trade(trade)
        content = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert len(content["trades"]) == 1
        assert content["trades"][0]["pnl"] == 5000

    def test_append_preserves_existing_trades(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        mgr.append_trade({"id": 1})
        mgr.append_trade({"id": 2})
        content = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert len(content["trades"]) == 2

    def test_append_preserves_state(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        mgr.save(TradingState(position=2, equity=400_000))
        mgr.append_trade({"id": 1})
        content = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert content["state"]["position"] == 2
        assert content["state"]["equity"] == 400_000

    def test_append_trade_no_tmp_file_left(self, tmp_path):
        path = str(tmp_path / "state.json")
        mgr = StateManager(path=path)
        mgr.append_trade({"id": 1})
        tmp_file = tmp_path / "state.json.tmp"
        assert not tmp_file.exists()
