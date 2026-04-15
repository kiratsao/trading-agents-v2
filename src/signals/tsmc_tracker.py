"""TSMC 夜盤信號模組。

根據 TSM ADR 及 SOX 指數的漲跌幅，計算次日台指方向性 bias。

信號邏輯
--------
* TSM ADR > +1.5%  → bullish,  confidence = 0.7 + min(change/5, 0.3)
* TSM ADR < -1.5%  → bearish,  confidence = 0.7 + min(|change|/5, 0.3)
* SOX 同向確認 → confidence += 0.05
* SOX 反向衝突 → confidence -= 0.05
* |TSM ADR| < 1.5% → neutral, confidence = 0.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_TSMC_TAIEX_WEIGHT: float = 0.3  # approx TSMC weight in TAIEX
_BULLISH_THRESHOLD: float = 1.5  # % ADR change to trigger bullish
_BEARISH_THRESHOLD: float = -1.5
_SOX_THRESHOLD: float = 1.0
_BASE_CONFIDENCE: float = 0.7
_NEUTRAL_CONFIDENCE: float = 0.5
_SOX_CONFIRM_BOOST: float = 0.05
_SOX_CONFLICT_PENALTY: float = 0.05


@dataclass
class TsmcSignal:
    tsm_adr_change_pct: float
    sox_change_pct: float
    taiex_implied_move: float
    direction_bias: str  # "bullish" | "bearish" | "neutral"
    confidence: float
    timestamp: str

    def __str__(self) -> str:
        return (
            f"TsmcSignal({self.direction_bias}, conf={self.confidence:.2f}, "
            f"TSM={self.tsm_adr_change_pct:+.2f}%, SOX={self.sox_change_pct:+.2f}%, "
            f"implied={self.taiex_implied_move:+.1f}pts)"
        )


def compute_signal(
    tsm_adr_change_pct: float,
    sox_change_pct: float,
    taiex_level: float = 20_000.0,
    timestamp: str | None = None,
) -> TsmcSignal:
    """Compute directional bias from TSM ADR and SOX changes.

    Parameters
    ----------
    tsm_adr_change_pct :
        TSM ADR change percentage for the day (e.g. +2.5 means +2.5%).
    sox_change_pct :
        SOX index change percentage for the day.
    taiex_level :
        Current TAIEX level for implied-move calculation.
    timestamp :
        ISO timestamp string. Defaults to now (UTC).
    """
    ts = timestamp or datetime.now(tz=UTC).isoformat(timespec="seconds")
    implied = tsm_adr_change_pct * _TSMC_TAIEX_WEIGHT / 100.0 * taiex_level

    if tsm_adr_change_pct >= _BULLISH_THRESHOLD:
        direction = "bullish"
        confidence = _BASE_CONFIDENCE + min(tsm_adr_change_pct / 5.0, 0.3)
        if sox_change_pct >= _SOX_THRESHOLD:
            confidence += _SOX_CONFIRM_BOOST
        elif sox_change_pct <= -_SOX_THRESHOLD:
            confidence -= _SOX_CONFLICT_PENALTY
    elif tsm_adr_change_pct <= _BEARISH_THRESHOLD:
        direction = "bearish"
        confidence = _BASE_CONFIDENCE + min(abs(tsm_adr_change_pct) / 5.0, 0.3)
        if sox_change_pct <= -_SOX_THRESHOLD:
            confidence += _SOX_CONFIRM_BOOST
        elif sox_change_pct >= _SOX_THRESHOLD:
            confidence -= _SOX_CONFLICT_PENALTY
    else:
        direction = "neutral"
        confidence = _NEUTRAL_CONFIDENCE

    return TsmcSignal(
        tsm_adr_change_pct=tsm_adr_change_pct,
        sox_change_pct=sox_change_pct,
        taiex_implied_move=implied,
        direction_bias=direction,
        confidence=min(max(confidence, 0.0), 1.0),
        timestamp=ts,
    )
