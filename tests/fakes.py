"""Shared test doubles for full-day / execution-path simulation.

Two fakes, mirroring the two layers the production code talks to:

* ``FakeBroker`` — adapter-level stand-in for :class:`ShioajiAdapter`. It is
  faithful to the *broker contract the orchestrator actually uses*
  (``place_order`` / ``get_account`` / ``get_positions`` / ``get_snapshots`` /
  ``get_contract``), keeps a stateful long-only position book, records every
  (sub-)order, and can inject rejections / data failures. Batch splitting
  mirrors production by importing the real ``_MKT_ORDER_MAX_QTY``.

* ``FakeShioaji`` — raw-API stand-in (``sj.Shioaji``) for exercising the real
  ``ShioajiAdapter`` / ``daily_updater`` kbars paths without a network.

``ScriptedStrategy`` subclasses the real ``V2bEngine`` so display-indicator
computation runs on real data, but ``generate_signal`` returns a scripted
queue of :class:`Signal` objects — keeping execution-path tests deterministic.
(Indicator→signal correctness is covered by ``test_v2b_engine``.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.tw_holidays import is_trading_day
from src.strategy.v2b_engine import Signal, V2bEngine
from tw_futures.executor.shioaji_adapter import _MKT_ORDER_MAX_QTY


def write_synthetic_parquet(
    path: Path,
    *,
    n_bars: int = 200,
    end: date = date(2026, 5, 21),
    base: float = 20_000.0,
    step: float = 5.0,
) -> pd.DataFrame:
    """Write *n_bars* monotonically-rising daily OHLCV bars on real TAIFEX
    trading days ending at *end*. Enough history for EMA(100)/ATR/ADX to be
    well-defined. Returns the DataFrame (DatetimeIndex named ``date``)."""
    days: list[date] = []
    cur = end
    while len(days) < n_bars and cur >= date(2018, 1, 1):
        if is_trading_day(cur):
            days.append(cur)
        cur = date.fromordinal(cur.toordinal() - 1)
    days.reverse()

    rows = []
    for i, _d in enumerate(days):
        close = base + i * step
        rows.append({
            "open": close - 2, "high": close + 8, "low": close - 8,
            "close": close, "volume": 50_000 + i,
        })
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="date"))
    df.to_parquet(path, index=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Strategy stub
# ─────────────────────────────────────────────────────────────────────────────
class ScriptedStrategy(V2bEngine):
    """V2bEngine whose ``generate_signal`` returns preset signals in order.

    The orchestrator may call ``generate_signal`` more than once per cycle
    (settlement rollover re-checks entry after the close), so signals are
    consumed from a FIFO queue. When exhausted it returns a benign ``hold``.
    """

    def __init__(self, signals: list[Signal]):
        super().__init__(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=25,
        )
        self._signals = list(signals)
        self.calls = 0

    def generate_signal(self, **kwargs) -> Signal:  # type: ignore[override]
        self.calls += 1
        if self._signals:
            return self._signals.pop(0)
        return Signal(action="hold", contracts=0, reason="scripted: queue empty")


# ─────────────────────────────────────────────────────────────────────────────
# Raw-API fake (sj.Shioaji)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _FakeContract:
    code: str = "MXFF6"
    delivery_date: str = "2099/12/31"


class _FakeKbars:
    """Mimics shioaji kbars: attributes ts/Open/High/Low/Close/Volume."""

    def __init__(self, ts, o, h, low, c, v):
        self.ts = ts
        self.Open = o
        self.High = h
        self.Low = low
        self.Close = c
        self.Volume = v


class FakeShioaji:
    """Stateful raw-API fake. Only the surface the adapter/updater touch."""

    def __init__(self, kbars: _FakeKbars | None = None, raise_on_kbars: bool = False):
        self._kbars = kbars
        self._raise_on_kbars = raise_on_kbars
        self.placed: list = []  # recorded (action, quantity) at raw layer

    def kbars(self, contract, start=None, end=None, timeout=None):
        if self._raise_on_kbars:
            raise RuntimeError("FakeShioaji: kbars unavailable")
        return self._kbars

    def logout(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Adapter-level broker fake
# ─────────────────────────────────────────────────────────────────────────────
class FakeBroker:
    """Faithful stand-in for the broker object the orchestrator drives."""

    def __init__(
        self,
        equity: float = 2_000_000.0,
        fill_price: float = 20_000.0,
        code: str = "MXFF6",
    ):
        self._equity = float(equity)
        self._fill_price = float(fill_price)
        self._fill_queue: list[float] = []
        self._code = code
        self.position = 0
        self.avg_price = 0.0
        self.orders: list[dict] = []        # full place_order calls
        self.order_batches: list[tuple[str, int]] = []  # (side, qty) sub-orders
        self._reject_sides: set[str] = set()
        self.fail_account = False
        self.fail_data = False               # kbars + snapshot both raise
        self._api = FakeShioaji()            # for live today-bar path
        self.snapshot = {
            "code": code, "open": fill_price, "high": fill_price,
            "low": fill_price, "close": fill_price, "volume": 0,
            "total_volume": 0,
        }

    # ── injection / setup knobs ─────────────────────────────────────────────
    def reject(self, side: str) -> None:
        self._reject_sides.add(side)

    def set_fill(self, price: float) -> None:
        self._fill_price = float(price)

    def queue_fills(self, prices: list[float]) -> None:
        self._fill_queue = [float(p) for p in prices]

    def set_equity(self, equity: float) -> None:
        self._equity = float(equity)

    def seed_position(self, contracts: int, avg_price: float) -> None:
        self.position = int(contracts)
        self.avg_price = float(avg_price)

    # ── broker contract ──────────────────────────────────────────────────────
    def place_order(self, product, action, contracts, **kwargs) -> dict:
        assert action in ("Buy", "Sell"), action
        oid = f"FAKE{len(self.orders) + 1}"
        fill = self._fill_queue.pop(0) if self._fill_queue else self._fill_price

        if action in self._reject_sides:
            rec = {"product": product, "action": action, "contracts": contracts,
                   "status": "Failed", "order_id": oid, "fill_price": fill}
            self.orders.append(rec)
            return {"order_id": oid, "status": "Failed", "fill_price": fill}

        # Mirror the adapter's market-order batch split (≤ _MKT_ORDER_MAX_QTY).
        remaining = contracts
        while remaining > 0:
            q = min(remaining, _MKT_ORDER_MAX_QTY)
            self.order_batches.append((action, q))
            remaining -= q

        if action == "Buy":
            new_pos = self.position + contracts
            self.avg_price = (
                (self.avg_price * self.position + fill * contracts) / new_pos
                if new_pos else 0.0
            )
            self.position = new_pos
        else:  # Sell
            self.position = max(0, self.position - contracts)
            if self.position == 0:
                self.avg_price = 0.0

        rec = {"product": product, "action": action, "contracts": contracts,
               "status": "Filled", "order_id": oid, "fill_price": fill}
        self.orders.append(rec)
        return {"order_id": oid, "status": "Filled", "fill_price": fill}

    def get_account(self) -> dict:
        if self.fail_account:
            raise RuntimeError("FakeBroker: get_account unavailable")
        return {"equity": self._equity, "margin_used": 0.0,
                "available_margin": self._equity, "unrealized_pnl": 0.0}

    def get_positions(self) -> list[dict]:
        if self.position <= 0:
            return []
        return [{
            "code": self._code, "direction": "Buy", "contracts": self.position,
            "avg_price": self.avg_price, "last_price": self._fill_price,
            "unrealized_pnl": 0.0,
        }]

    def get_snapshots(self, product: str = "MXF") -> dict:
        if self.fail_data:
            raise RuntimeError("FakeBroker: snapshot unavailable")
        return dict(self.snapshot)

    def get_contract(self, product: str = "MXF"):
        return _FakeContract(self._code)

    def logout(self) -> None:
        pass

    # convenience for batching assertions
    def batches_for(self, side: str) -> list[int]:
        return [q for s, q in self.order_batches if s == side]
