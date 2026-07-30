"""scan_pollution PHANTOM: a stored bar on a date TAIFEX never traded must be
flagged (date-set validation), not silently skipped — the 2026-07-10 ghost bar
was invisible to the value-only scan precisely because no TAIFEX day row exists
to compare against."""
from __future__ import annotations

from datetime import date

import pandas as pd

from scripts import scan_pollution


def test_phantom_flagged_clean_days_pass(monkeypatch, tmp_path):
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-10"),
         pd.Timestamp("2026-07-13")], name="date")
    df = pd.DataFrame(
        [{"open": 46016.0, "high": 46160.0, "low": 45330.0, "close": 45681.0,
          "volume": 164977},
         {"open": 46000.0, "high": 46300.0, "low": 45850.0, "close": 46275.0,
          "volume": 137719},   # ← the ghost (7/09 night midnight tail)
         {"open": 46300.0, "high": 46360.0, "low": 45266.0, "close": 45598.0,
          "volume": 159495}],
        index=idx)
    pq = tmp_path / "MXF.parquet"
    df.to_parquet(pq)

    day = df.drop(pd.Timestamp("2026-07-10"))  # TAIFEX 一般 has no 7/10 row
    monkeypatch.setattr(scan_pollution, "fetch_taifex_day_session_range",
                        lambda s, e: day)
    monkeypatch.setattr(scan_pollution, "_taifex_night", lambda s, e: {})
    monkeypatch.setattr(scan_pollution, "fetch_spot_range", lambda s, e: {})

    flagged = scan_pollution.scan(pq, date(2026, 7, 9), date(2026, 7, 13))
    assert len(flagged) == 1
    f = flagged[0]
    assert f["date"] == date(2026, 7, 10)
    assert f["class"] == "PHANTOM"
    assert f["taifex_day"] is None
    assert abs(f["stored"] - 46_275.0) < 1
