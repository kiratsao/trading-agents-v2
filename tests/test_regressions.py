"""Regression tests for bugs found in production (2026-04-15~16).

Each test name documents the original bug and prevents recurrence.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.daily_updater import update


class TestRegressionFugleImportCleaned:
    """Bug: tw_futures/executor/__init__.py imported deleted fugle_adapter."""

    def test_executor_init_no_fugle(self):
        init_path = Path("tw_futures/executor/__init__.py")
        content = init_path.read_text()
        assert "fugle" not in content.lower()

    def test_executor_imports_cleanly(self):
        from tw_futures.executor import ShioajiAdapter
        assert ShioajiAdapter is not None


class TestRegressionBrokerCertParams:
    """Bug: ShioajiAdapter created without cert_path/cert_password/person_id."""

    def test_orchestrator_has_cert_in_auto_broker(self):
        """run_execution auto-broker creation must pass cert params."""
        src = Path("src/scheduler/orchestrator.py").read_text()
        # Find the auto-broker block in run_execution
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "ShioajiAdapter"
                        or (isinstance(func, ast.Name) and func.id == "ShioajiAdapter")):
                    kwarg_names = {kw.arg for kw in node.keywords}
                    assert "cert_path" in kwarg_names, (
                        f"ShioajiAdapter call missing cert_path at line {node.lineno}"
                    )

    def test_daily_updater_has_cert(self):
        # Adapter creation (incl. cert params) now lives in the unified fetcher.
        src = Path("src/data/shioaji_fetcher.py").read_text()
        assert "cert_path" in src
        assert "cert_password" in src
        assert "person_id" in src


class TestRegressionRangeIndexAfterConcat:
    """Bug: pd.concat produced RangeIndex → .normalize() crashed."""

    def test_update_preserves_datetime_index(self, tmp_path):
        pq = tmp_path / "test.parquet"
        # Create with plain RangeIndex (simulates bad parquet)
        df = pd.DataFrame(
            {"open": [100.0], "high": [110.0], "low": [90.0],
             "close": [105.0], "volume": [1000]},
        )
        df.to_parquet(pq, index=True)

        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=pd.Timestamp("2026-04-10").date()),
            patch("src.data.daily_updater._fetch_and_aggregate") as mock,
        ):
            new = pd.DataFrame(
                {"open": [200.0], "high": [210.0], "low": [190.0],
                 "close": [205.0], "volume": [2000]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-09")], name="date"),
            )
            mock.return_value = new
            result = update(parquet_path=pq)

        assert result["success"] is True
        reloaded = pd.read_parquet(pq)
        assert isinstance(reloaded.index, pd.DatetimeIndex)


class TestRegressionIOCNotROD:
    """Bug: order_type must default to IOC for night session."""

    def test_shioaji_adapter_default_ioc(self):
        src = Path("tw_futures/executor/shioaji_adapter.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "submit_order":
                for arg in node.args.defaults:
                    pass
                # Check the order_type default in the signature
                args = node.args
                kwarg_defaults = dict(
                    zip(
                        [a.arg for a in args.args[-len(args.defaults):]],
                        args.defaults,
                    )
                )
                if "order_type" in kwarg_defaults:
                    default = kwarg_defaults["order_type"]
                    assert isinstance(default, ast.Constant)
                    assert default.value == "IOC", (
                        f"order_type default is {default.value!r}, must be 'IOC'"
                    )


class TestRegressionDailyUpdaterFailureAlertsLine:
    """Bug: daily_updater failures were silently swallowed."""

    def test_failure_calls_notify(self, tmp_path):
        pq = tmp_path / "test.parquet"
        # Last bar = Apr 7 (Mon), today = Apr 10 (Thu), yesterday = Apr 9 (Wed)
        # → fetch_start = Apr 8, end = Apr 9, _fetch_and_aggregate will fire
        dates = pd.bdate_range("2026-04-03", periods=3)  # Apr 3,6,7
        df = pd.DataFrame(
            {"open": [1.0]*3, "high": [2.0]*3, "low": [0.5]*3,
             "close": [1.5]*3, "volume": [100]*3},
            index=dates,
        )
        df.index.name = "date"
        df.to_parquet(pq)

        notify = MagicMock()
        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=pd.Timestamp("2026-04-10").date()),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  side_effect=RuntimeError("connection refused")),
        ):
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["success"] is False
        notify.assert_called_once()
        assert "🔴" in notify.call_args[0][0]


class TestRegressionZeroBarsOnTradingDayWarns:
    """Bug: bars_added=0 on a trading day reported success silently."""

    def test_zero_bars_weekday_warns(self, tmp_path):
        pq = tmp_path / "test.parquet"
        # Last bar is Wed 04/15 → today=04/21(Tue), yesterday=04/20(Mon)
        # fetch_start=04/16, end=04/20, but fetch returns None → 🔴 data gap
        dates = pd.bdate_range("2026-04-13", periods=3)  # Mon/Tue/Wed
        df = pd.DataFrame(
            {"open": [1.0]*3, "high": [2.0]*3, "low": [0.5]*3,
             "close": [1.5]*3, "volume": [100]*3},
            index=dates,
        )
        df.index.name = "date"
        df.to_parquet(pq)
        notify = MagicMock()

        # Tuesday 04/21: fetch returns empty → should alert (data gap)
        with (
            patch("src.data.daily_updater._today_taipei",
                  return_value=pd.Timestamp("2026-04-21").date()),
            patch("src.data.daily_updater._fetch_and_aggregate",
                  return_value=None),
        ):
            result = update(parquet_path=pq, notify_fn=notify)

        assert result["bars_added"] == 0
        notify.assert_called_once()
        assert "🔴" in notify.call_args[0][0]


class TestRegressionFreshBrokerPerJob:
    """Bug: daemon reused stale broker → token expired after days."""

    def test_main_jobs_dont_capture_broker(self):
        """Verify scheduler jobs create fresh brokers, not a captured one."""
        src = Path("src/scheduler/main.py").read_text()
        # The old pattern: lambda: orchestrator.run_signal(broker=broker)
        # captures broker from outer scope. New pattern uses _run_signal etc.
        assert "lambda: orchestrator.run_signal(broker=broker)" not in src
        assert "lambda: orchestrator.run_execution(broker=broker)" not in src
        assert "lambda: orchestrator.run_daily(broker=broker)" not in src
        # New pattern should have _run_signal / _run_execution / _run_daily
        assert "_run_signal" in src
        assert "_run_execution" in src


class TestRegressionParquetNotInGit:
    """Bug: parquet tracked by git → git pull overwrites daily updates."""

    def test_gitignore_blocks_parquet(self):
        gitignore = Path(".gitignore").read_text()
        assert "*.parquet" in gitignore

    def test_no_parquet_tracked(self):
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "*.parquet"],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "", (
            f"Parquet files still tracked: {result.stdout.strip()}"
        )


class TestRegressionMktOrderBatchSplit:
    """Bug: Sell 13 口 MXF 被拒「超過市價單筆委託上限」— 小台市價實測上限 5 口。"""

    def _make_adapter(self):
        """Create a ShioajiAdapter with mocked internals."""
        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        adapter = object.__new__(ShioajiAdapter)
        adapter._api = MagicMock()
        adapter._api.Contracts.Futures.MXF = [
            MagicMock(delivery_date="2099/12/31", code="MXFZ9"),
        ]
        adapter._api.futopt_account = MagicMock()
        adapter._accounts = []
        return adapter

    @patch("shioaji.Order", new_callable=lambda: MagicMock)
    def test_submit_order_splits_over_5(self, mock_order_cls):
        """Market order > 5 contracts must be split into batches of 5."""
        adapter = self._make_adapter()

        call_count = 0

        def fake_place_order(contract, order):
            nonlocal call_count
            call_count += 1
            trade = MagicMock()
            trade.status.id = f"ORD-{call_count}"
            trade.status.status.value = "Filled"
            return trade

        adapter._api.place_order = fake_place_order

        result = adapter.submit_order("MXF", "Sell", 13, price_type="MKT")

        # 13 = 5 + 5 + 3 → 3 batches
        assert call_count == 3, f"Expected 3 batches, got {call_count}"
        assert result["contracts"] == 13
        assert "ORD-1" in result["order_id"]
        assert "ORD-3" in result["order_id"]

    @patch("shioaji.Order", new_callable=lambda: MagicMock)
    def test_submit_order_no_split_under_limit(self, mock_order_cls):
        """Orders <= 5 contracts should NOT be split."""
        adapter = self._make_adapter()

        trade = MagicMock()
        trade.status.id = "ORD-1"
        trade.status.status.value = "Filled"
        adapter._api.place_order.return_value = trade

        result = adapter.submit_order("MXF", "Sell", 5, price_type="MKT")

        adapter._api.place_order.assert_called_once()
        assert result["contracts"] == 5

    @patch("shioaji.Order", new_callable=lambda: MagicMock)
    def test_6_contracts_splits_into_2_batches(self, mock_order_cls):
        """6 contracts = 5 + 1 → 2 batches."""
        adapter = self._make_adapter()

        call_count = 0

        def fake_place_order(contract, order):
            nonlocal call_count
            call_count += 1
            trade = MagicMock()
            trade.status.id = f"ORD-{call_count}"
            trade.status.status.value = "Filled"
            return trade

        adapter._api.place_order = fake_place_order

        result = adapter.submit_order("MXF", "Sell", 6, price_type="MKT")

        assert call_count == 2
        assert result["contracts"] == 6

    @patch("shioaji.Order", new_callable=lambda: MagicMock)
    def test_limit_order_not_split(self, mock_order_cls):
        """Limit orders should NOT be split regardless of quantity."""
        adapter = self._make_adapter()

        trade = MagicMock()
        trade.status.id = "ORD-1"
        trade.status.status.value = "Filled"
        adapter._api.place_order.return_value = trade

        result = adapter.submit_order("MXF", "Buy", 20, price_type="LMT", price=21000.0)

        adapter._api.place_order.assert_called_once()
        assert result["contracts"] == 20

    @patch("shioaji.Order", new_callable=lambda: MagicMock)
    def test_split_exact_multiple(self, mock_order_cls):
        """10 contracts = exactly 2 batches of 5, no remainder."""
        adapter = self._make_adapter()

        call_count = 0

        def fake_place_order(contract, order):
            nonlocal call_count
            call_count += 1
            trade = MagicMock()
            trade.status.id = f"ORD-{call_count}"
            trade.status.status.value = "Filled"
            return trade

        adapter._api.place_order = fake_place_order

        result = adapter.submit_order("MXF", "Buy", 10, price_type="MKT")

        assert call_count == 2
        assert result["contracts"] == 10


class TestRegressionSettlementDayContract:
    """Bug: Buy 用了已到期合約 MXFE6 — 結算日應取新月份合約。"""

    def test_get_contract_excludes_today_expiry(self):
        """On settlement day, expiring contract (delivery_date == today) must be skipped."""
        import datetime as dt

        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        adapter = object.__new__(ShioajiAdapter)
        adapter._api = MagicMock()

        # Simulate: MXFE6 expires today (2026/05/20), MXFG6 is next month
        expiring = MagicMock(delivery_date="2026/05/20", code="MXFE6")
        next_month = MagicMock(delivery_date="2026/06/17", code="MXFG6")
        adapter._api.Contracts.Futures.MXF = [expiring, next_month]

        fake_today = dt.date(2026, 5, 20)
        with patch("datetime.date") as mock_date:
            mock_date.today.return_value = fake_today
            mock_date.side_effect = lambda *a, **kw: dt.date(*a, **kw)
            contract = adapter.get_contract("MXF")

        assert contract.code == "MXFG6"

    def test_get_contract_normal_day(self):
        """On non-settlement day, near-month contract is returned normally."""
        import datetime as dt

        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        adapter = object.__new__(ShioajiAdapter)
        adapter._api = MagicMock()

        near = MagicMock(delivery_date="2026/05/20", code="MXFE6")
        far = MagicMock(delivery_date="2026/06/17", code="MXFG6")
        adapter._api.Contracts.Futures.MXF = [near, far]

        fake_today = dt.date(2026, 5, 19)
        with patch("datetime.date") as mock_date:
            mock_date.today.return_value = fake_today
            mock_date.side_effect = lambda *a, **kw: dt.date(*a, **kw)
            contract = adapter.get_contract("MXF")

        assert contract.code == "MXFE6"


class TestRegressionSellFailAbortsBuy:
    """Bug: Sell 被拒但系統繼續執行 Buy 轉倉。"""

    def test_sell_rejected_no_rollover_buy(self):
        """If sell order fails, rollover buy must NOT execute."""
        from src.scheduler.orchestrator import V2bOrchestrator
        from src.state.state_manager import StateManager, TradingState

        state_mgr = MagicMock(spec=StateManager)
        state = TradingState(
            position=13,
            entry_price=20500.0,
            contracts=13,
            equity=1_200_000,
            pending_action="close",
            pending_contracts=13,
            pending_reason="settlement-day force close",
        )
        state_mgr.load.return_value = state

        broker = MagicMock()
        # Sell returns Failed status (exchange rejected)
        broker.place_order.return_value = {
            "order_id": "REJECTED-1",
            "status": "Failed",
        }
        broker.get_positions.return_value = []

        notify_fn = MagicMock()
        strategy = MagicMock()

        orch = V2bOrchestrator(
            strategy=strategy,
            state_mgr=state_mgr,
            notify_fn=notify_fn,
            execution_timing="night_open",
            live=False,
        )

        result = orch.run_execution(broker=broker, exec_price=22000.0)

        # Sell was called once, but Buy must NOT be called
        assert broker.place_order.call_count == 1
        assert broker.place_order.call_args.args == ("MXF", "Sell", 13)

        # Rollover must be aborted
        assert result.get("rollover") is False
        assert "sell order failed" in result.get("rollover_reason", "")

        # LINE alert must contain 🔴
        alert_msgs = [
            call.args[0] for call in notify_fn.call_args_list
            if "🔴" in call.args[0]
        ]
        assert len(alert_msgs) >= 1

    def test_sell_exception_no_rollover_buy(self):
        """If sell order raises exception, rollover buy must NOT execute."""
        from src.scheduler.orchestrator import V2bOrchestrator
        from src.state.state_manager import StateManager, TradingState

        state_mgr = MagicMock(spec=StateManager)
        state = TradingState(
            position=7,
            entry_price=21000.0,
            contracts=7,
            equity=900_000,
            pending_action="close",
            pending_contracts=7,
            pending_reason="settlement-day force close",
        )
        state_mgr.load.return_value = state

        broker = MagicMock()
        broker.place_order.side_effect = Exception("超過市價單筆委託上限")
        broker.get_positions.return_value = []

        notify_fn = MagicMock()
        strategy = MagicMock()

        orch = V2bOrchestrator(
            strategy=strategy,
            state_mgr=state_mgr,
            notify_fn=notify_fn,
            execution_timing="night_open",
            live=False,
        )

        result = orch.run_execution(broker=broker, exec_price=22000.0)

        # Buy must NOT be called (sell raised exception)
        assert broker.place_order.call_count == 1
        assert result.get("rollover") is False

    def test_sell_ok_then_buy_proceeds(self):
        """If sell succeeds, rollover buy should proceed normally."""
        import pandas as pd

        from src.scheduler.orchestrator import V2bOrchestrator
        from src.state.state_manager import StateManager, TradingState
        from src.strategy.v2b_engine import Signal

        state_mgr = MagicMock(spec=StateManager)
        state = TradingState(
            position=5,
            entry_price=21000.0,
            contracts=5,
            equity=800_000,
            pending_action="close",
            pending_contracts=5,
            pending_reason="settlement-day force close",
        )
        state_mgr.load.return_value = state

        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "OK-1",
            "status": "Filled",
            "fill_price": 22000.0,
        }
        broker.get_positions.return_value = [{"contracts": 5}]

        notify_fn = MagicMock()

        strategy = MagicMock()
        strategy.generate_signal.return_value = Signal("buy", 4, "golden cross + ADX OK")

        # Build minimal data for _load_data
        dates = pd.bdate_range("2026-01-01", periods=200)
        df = pd.DataFrame(
            {"open": [22000]*200, "high": [22100]*200, "low": [21900]*200,
             "close": [22000]*200, "volume": [100000]*200},
            index=dates,
        )

        orch = V2bOrchestrator(
            strategy=strategy,
            state_mgr=state_mgr,
            notify_fn=notify_fn,
            execution_timing="night_open",
            live=False,
        )

        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_execution(broker=broker, exec_price=22000.0)

        # Both sell and buy should be called
        assert broker.place_order.call_count == 2
        calls = broker.place_order.call_args_list
        assert calls[0].args == ("MXF", "Sell", 5)
        assert calls[1].args == ("MXF", "Buy", 4)
        assert result.get("rollover") is True
        assert result.get("rollover_contracts") == 4


class TestRegressionSettlementBatchOrder:
    """13 口 → 分批 5+5+3（結算日場景完整測試）。"""

    @patch("shioaji.Order", new_callable=lambda: MagicMock)
    def test_13_contracts_split_5_5_3(self, mock_order_cls):
        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        adapter = object.__new__(ShioajiAdapter)
        adapter._api = MagicMock()
        adapter._api.Contracts.Futures.MXF = [
            MagicMock(delivery_date="2099/12/31", code="MXFZ9"),
        ]
        adapter._api.futopt_account = MagicMock()
        adapter._accounts = []

        batch_sizes = []

        def fake_place_order(contract, order):
            batch_sizes.append(order.quantity)
            trade = MagicMock()
            trade.status.id = f"ORD-{len(batch_sizes)}"
            trade.status.status.value = "Filled"
            return trade

        adapter._api.place_order = fake_place_order
        mock_order_cls.side_effect = lambda **kw: MagicMock(quantity=kw["quantity"])

        result = adapter.submit_order("MXF", "Sell", 13, price_type="MKT")

        assert batch_sizes == [5, 5, 3]
        assert result["contracts"] == 13


class TestRegressionSettlementRolloverContract:
    """結算日轉倉使用下月合約，不用已到期合約。"""

    def test_rollover_uses_next_month(self):
        import datetime as dt

        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        adapter = object.__new__(ShioajiAdapter)
        adapter._api = MagicMock()

        # Settlement day: MXFE6 expires today
        expiring = MagicMock(delivery_date="2026/06/17", code="MXFE6")
        next_month = MagicMock(delivery_date="2026/07/15", code="MXFG6")
        adapter._api.Contracts.Futures.MXF = [expiring, next_month]

        fake_today = dt.date(2026, 6, 17)
        with patch("datetime.date") as mock_date:
            mock_date.today.return_value = fake_today
            mock_date.side_effect = lambda *a, **kw: dt.date(*a, **kw)
            contract = adapter.get_contract("MXF")

        assert contract.code == "MXFG6", f"Should use next month, got {contract.code}"


class TestRegressionLadderCoversEquity:
    """當前 equity 必須在 ladder 範圍內。"""

    def test_ladder_covers_1730k(self):
        from src.strategy.v2b_engine import _anti_martingale_contracts

        ladder = [
            {"equity": 350000, "contracts": 2},
            {"equity": 480000, "contracts": 3},
            {"equity": 600000, "contracts": 4},
            {"equity": 720000, "contracts": 5},
            {"equity": 840000, "contracts": 6},
            {"equity": 960000, "contracts": 7},
            {"equity": 1080000, "contracts": 8},
            {"equity": 1200000, "contracts": 9},
            {"equity": 1320000, "contracts": 10},
            {"equity": 1440000, "contracts": 11},
            {"equity": 1560000, "contracts": 12},
            {"equity": 1680000, "contracts": 13},
            {"equity": 1800000, "contracts": 14},
            {"equity": 1920000, "contracts": 15},
        ]
        n = _anti_martingale_contracts(1_730_000, ladder, 15, 131_500)
        assert n == 13, f"Expected 13 contracts for 1.73M equity, got {n}"

    def test_config_ladder_matches_code(self):
        """accounts.yaml ladder must cover at least 1.5M equity."""
        import re

        content = Path("config/accounts.yaml").read_text()
        # Extract all equity values from scale_ladder
        equities = [int(m) for m in re.findall(r"equity:\s*(\d+)", content)]
        assert max(equities) >= 1_500_000, f"Ladder max {max(equities)} too low"
        # Extract max_contracts
        max_c = re.search(r"max_contracts:\s*(\d+)", content)
        assert max_c and int(max_c.group(1)) >= 13, "max_contracts must be >= 13"


class TestRegressionStateBackup:
    """State 備份機制：每次 save 備份 + 保留 7 天 + 可恢復。"""

    def test_save_creates_backup(self, tmp_path):
        from src.state.state_manager import StateManager, TradingState

        state_path = tmp_path / "state.json"
        mgr = StateManager(path=str(state_path))

        with patch("src.state.state_manager._BACKUP_DIR", str(tmp_path)):
            mgr.save(TradingState(position=5, equity=800_000))

        backups = list(tmp_path.glob("state_backup_*.json"))
        assert len(backups) == 1
        # Verify backup content
        import json

        data = json.loads(backups[0].read_text())
        assert data["state"]["position"] == 5

    def test_backup_prunes_old(self, tmp_path):
        import json
        from datetime import timedelta

        from src.state.state_manager import StateManager, TradingState

        state_path = tmp_path / "state.json"
        mgr = StateManager(path=str(state_path))

        # Create fake old backups
        today = __import__("datetime").date.today()
        for i in range(10):
            old_date = today - timedelta(days=i + 5)
            old_backup = tmp_path / f"state_backup_{old_date.isoformat()}.json"
            old_backup.write_text(json.dumps({"state": {}}))

        with patch("src.state.state_manager._BACKUP_DIR", str(tmp_path)):
            mgr.save(TradingState(position=1, equity=350_000))

        backups = list(tmp_path.glob("state_backup_*.json"))
        # Should have pruned backups older than 7 days
        # Today's + within 7 days should remain
        for b in backups:
            date_part = b.stem.replace("state_backup_", "")
            d = __import__("datetime").date.fromisoformat(date_part)
            assert (today - d).days <= 7, f"Backup {b.name} should have been pruned"

    def test_restore_from_backup(self, tmp_path):
        import json

        from src.state.state_manager import StateManager

        # Create a backup
        backup = tmp_path / "state_backup_2026-05-20.json"
        backup.write_text(json.dumps({
            "state": {"position": 13, "equity": 1730000.0},
            "trades": [],
        }))

        target = tmp_path / "state.json"
        ok = StateManager.restore_from_backup(str(backup), str(target))
        assert ok is True
        assert target.exists()

        data = json.loads(target.read_text())
        assert data["state"]["position"] == 13
