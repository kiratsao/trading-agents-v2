"""Production market-order batch splitting (TAIFEX caps MKT at 5 lots/order).

Exercises the real ShioajiAdapter splitting logic without a network: the
adapter is built via __new__ (skipping _connect) and the recursion boundary
is patched, so the genuine _submit_batched / routing code runs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_futures.executor.shioaji_adapter import _MKT_ORDER_MAX_QTY, ShioajiAdapter


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)


def _bare_adapter() -> ShioajiAdapter:
    return ShioajiAdapter.__new__(ShioajiAdapter)


def test_submit_batched_splits_13_into_5_5_3():
    adapter = _bare_adapter()
    recorded: list[int] = []

    def fake_submit(product, action, contracts, **kw):
        recorded.append(contracts)
        return {"order_id": f"O{len(recorded)}", "status": "Filled",
                "action": action, "contracts": contracts}

    adapter.submit_order = fake_submit  # type: ignore[method-assign]
    res = adapter._submit_batched(
        product="MXF", action="Sell", contracts=13,
        price_type="MKT", price=0.0, order_type="IOC", octype="Auto",
    )

    assert recorded == [5, 5, 3]
    assert res["order_id"] == "O1,O2,O3"
    assert res["contracts"] == 13
    assert res["status"] == "Filled"


def test_submit_batched_exact_multiple():
    adapter = _bare_adapter()
    recorded: list[int] = []
    adapter.submit_order = lambda product, action, contracts, **kw: (  # type: ignore[method-assign]
        recorded.append(contracts) or {"order_id": "X", "status": "Filled"}
    )
    adapter._submit_batched(product="MXF", action="Buy", contracts=10,
                            price_type="MKT", price=0.0, order_type="IOC", octype="Auto")
    assert recorded == [5, 5]


def test_submit_order_routes_large_mkt_to_batched():
    adapter = _bare_adapter()
    seen = {}

    def fake_batched(**kw):
        seen.update(kw)
        return {"order_id": "B", "status": "Filled"}

    adapter._submit_batched = fake_batched  # type: ignore[method-assign]
    out = adapter.submit_order("MXF", "Sell", _MKT_ORDER_MAX_QTY + 8, price_type="MKT")
    assert out["order_id"] == "B"
    assert seen["contracts"] == _MKT_ORDER_MAX_QTY + 8
    assert seen["action"] == "Sell"
