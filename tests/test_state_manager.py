"""Tests for state_manager: atomic write, roundtrip, append_trade."""

from __future__ import annotations

import json

from src.state.state_manager import StateManager, TradingState


class TestStateManager:
    def test_save_atomic_write(self, tmp_path):
        """save() writes to .tmp then renames — no partial file on crash."""
        path = tmp_path / "state.json"
        mgr = StateManager(path=str(path))
        state = TradingState(position=2, entry_price=20100.0, equity=500_000)
        mgr.save(state)

        # File exists, no .tmp leftover
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()

        # Content is valid JSON
        data = json.loads(path.read_text())
        assert data["state"]["position"] == 2
        assert data["state"]["entry_price"] == 20100.0

    def test_append_trade_atomic(self, tmp_path):
        """append_trade() uses tmp + rename, preserves existing state."""
        path = tmp_path / "state.json"
        mgr = StateManager(path=str(path))

        # Save initial state
        state = TradingState(position=2, equity=400_000)
        mgr.save(state)

        # Append a trade
        trade = {"entry": "2026-01-01", "exit": "2026-01-10", "pnl": 50000}
        mgr.append_trade(trade)

        # No .tmp leftover
        assert not path.with_suffix(".json.tmp").exists()

        # Both state and trade preserved
        data = json.loads(path.read_text())
        assert data["state"]["position"] == 2
        assert len(data["trades"]) == 1
        assert data["trades"][0]["pnl"] == 50000

    def test_load_save_roundtrip(self, tmp_path):
        """save → load produces identical state."""
        path = tmp_path / "state.json"
        mgr = StateManager(path=str(path))

        original = TradingState(
            position=3,
            entry_price=21500.0,
            entry_date="2026-04-01",
            contracts=3,
            highest_high=22000.0,
            equity=600_000.0,
            pyramided=True,
            pending_action="close",
            pending_contracts=3,
            pending_signal_date="2026-04-15",
            pending_reason="settlement-day force close",
        )
        mgr.save(original)
        loaded = mgr.load()

        assert loaded.position == original.position
        assert loaded.entry_price == original.entry_price
        assert loaded.entry_date == original.entry_date
        assert loaded.contracts == original.contracts
        assert loaded.highest_high == original.highest_high
        assert loaded.equity == original.equity
        assert loaded.pyramided == original.pyramided
        assert loaded.pending_action == original.pending_action
        assert loaded.pending_contracts == original.pending_contracts
        assert loaded.pending_signal_date == original.pending_signal_date
        assert loaded.pending_reason == original.pending_reason

    def test_load_missing_returns_default(self, tmp_path):
        """load() on missing file returns default TradingState."""
        mgr = StateManager(path=str(tmp_path / "nonexistent.json"))
        state = mgr.load()
        assert state.position == 0
        assert state.equity == 350_000.0
