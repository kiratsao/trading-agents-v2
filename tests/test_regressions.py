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
        src = Path("src/data/daily_updater.py").read_text()
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
        dates = pd.bdate_range("2026-04-07", periods=3)
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
