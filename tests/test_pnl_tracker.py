"""Tests for scripts.pnl_tracker withdrawal-aware profit splitting."""

from __future__ import annotations

from unittest.mock import patch

from scripts import pnl_tracker

# Real fund: Kira 380K / Wife 150K / Dad 350K (Σ = 880K).
_INVESTORS = [
    {"name": "Kira", "capital": 380_000},
    {"name": "Wife", "capital": 150_000},
    {"name": "Dad", "capital": 350_000},
]

# 2026-06-02 出金 250萬, 按方案B (capital 占比) 分配.
_WITHDRAWALS = [
    {"date": "2026-06-02",
     "amounts": {"Kira": 1_079_545, "Wife": 426_136, "Dad": 994_318}},
]


def _run(cfg: dict, equity: float) -> dict:
    with (
        patch.object(pnl_tracker, "_load_config", return_value=cfg),
        patch.object(pnl_tracker, "get_equity", return_value=(equity, "即時")),
        patch.object(pnl_tracker, "_PNL_CSV") as csv_path,
    ):
        # Redirect CSV append to a throwaway so the test never touches real data.
        csv_path.exists.return_value = True
        csv_path.parent.mkdir.return_value = None
        with patch("builtins.open"), patch("csv.DictWriter"):
            return pnl_tracker.track_pnl()


class TestWithdrawalAwarePnl:
    def test_total_profit_adds_back_withdrawals(self):
        equity = 5_000_000.0
        res = _run({"investors": _INVESTORS, "withdrawals": _WITHDRAWALS}, equity)

        # (equity + Σwithdrawn) − Σcapital
        assert res["total_withdrawn"] == 2_499_999
        assert abs(res["total_pnl"] - ((equity + 2_499_999) - 880_000)) < 0.01

        # Per-investor pnl sums back to the total (no leakage).
        assert abs(sum(i["pnl"] for i in res["investors"]) - res["total_pnl"]) < 0.01

    def test_per_investor_formula(self):
        equity = 5_000_000.0
        res = _run({"investors": _INVESTORS, "withdrawals": _WITHDRAWALS}, equity)
        by = {i["name"]: i for i in res["investors"]}

        kira = by["Kira"]
        share = 380_000 / 880_000
        holding = equity * share
        wd = 1_079_545
        # 總獲利 = (持分 + 已提領) − 本金
        assert abs(kira["pnl"] - ((holding + wd) - 380_000)) < 0.01
        # 淨投入 = 本金 − 已提領 (Kira took out more than capital → negative)
        assert abs(kira["net_invested"] - (380_000 - wd)) < 0.01
        assert kira["withdrawn"] == wd

    def test_withdrawal_addback_changes_result(self):
        """Guard: the withdrawn amount is genuinely added back, not ignored —
        otherwise the per-person profit understates by exactly the payout."""
        equity = 5_000_000.0
        with_wd = _run({"investors": _INVESTORS, "withdrawals": _WITHDRAWALS}, equity)
        no_wd = _run({"investors": _INVESTORS}, equity)
        by_with = {i["name"]: i for i in with_wd["investors"]}
        by_no = {i["name"]: i for i in no_wd["investors"]}
        assert abs(
            (by_with["Kira"]["pnl"] - by_no["Kira"]["pnl"]) - 1_079_545
        ) < 0.01

    def test_backward_compat_no_withdrawals(self):
        """No withdrawals key → old behaviour: pnl = holding − capital."""
        equity = 1_760_000.0  # 2× total capital
        res = _run({"investors": _INVESTORS}, equity)
        assert res["total_withdrawn"] == 0
        kira = next(i for i in res["investors"] if i["name"] == "Kira")
        share = 380_000 / 880_000
        assert abs(kira["pnl"] - (equity * share - 380_000)) < 0.01

    def test_unknown_investor_withdrawal_ignored(self):
        equity = 5_000_000.0
        bad = [{"date": "2026-06-02", "amounts": {"Ghost": 999_999}}]
        res = _run({"investors": _INVESTORS, "withdrawals": bad}, equity)
        assert res["total_withdrawn"] == 0  # ghost ignored
        assert all(i["withdrawn"] == 0 for i in res["investors"])


class TestFormatPnlLine:
    def test_shows_withdrawal_note(self):
        res = _run({"investors": _INVESTORS, "withdrawals": _WITHDRAWALS}, 5_000_000.0)
        line = pnl_tracker.format_pnl_line(res)
        assert "已提領" in line
        assert "2,499,999" in line

    def test_no_withdrawal_note_when_none(self):
        res = _run({"investors": _INVESTORS}, 1_760_000.0)
        line = pnl_tracker.format_pnl_line(res)
        assert "已提領" not in line
