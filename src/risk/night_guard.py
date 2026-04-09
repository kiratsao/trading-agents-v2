"""Night Guard — 夜盤風控檢查（05:15 執行）。

在 15:05 建倉後，05:15 夜盤收盤時做風控檢查：
  Guard 1: 夜盤 low < entry - ATR×guard1_atr_mult → 平倉
  Guard 2: 夜盤 close < 夜盤 open - ATR×guard2_atr_mult → 平倉
  Guard 3: 夜盤 close < entry × (1 - guard3_pct) → 平倉
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NightSession:
    """夜盤 OHLCV 聚合資料（15:00-05:00）。"""

    open_price: float
    high: float
    low: float
    close: float


@dataclass
class GuardResult:
    """Night guard 檢查結果。"""

    should_close: bool
    reason: str = ""
    new_position: None = None


class NightGuard:
    """夜盤風控守衛。

    Parameters
    ----------
    guard1_atr_mult :
        Guard 1 trailing stop multiplier（夜盤 low < entry - atr×mult）。
        設為 None 可停用。
    guard2_atr_mult :
        Guard 2 reversal multiplier（夜盤 close < night_open - atr×mult）。
        設為 None 可停用。
    guard3_pct :
        Guard 3 drawdown pct（夜盤 close < entry × (1 - pct)）。
        設為 None 可停用。
    enabled_guards :
        Frozenset of guard names to enable (e.g., frozenset({"guard1", "guard2"})).
        If None, all guards with non-None multipliers are enabled.
    """

    def __init__(
        self,
        guard1_atr_mult: float | None = 2.0,
        guard2_atr_mult: float | None = 2.0,
        guard3_pct: float | None = None,
        enabled_guards: frozenset | None = None,
    ) -> None:
        self.guard1_atr_mult = guard1_atr_mult
        self.guard2_atr_mult = guard2_atr_mult
        self.guard3_pct = guard3_pct
        self.enabled_guards = enabled_guards

    def _is_enabled(self, name: str) -> bool:
        if self.enabled_guards is None:
            return True
        return name in self.enabled_guards

    def check(
        self,
        position: int,
        entry_price: float | None,
        atr: float,
        session: NightSession,
    ) -> GuardResult:
        """執行夜盤風控檢查。

        Parameters
        ----------
        position :
            目前持倉口數（0 = 空倉，> 0 = 多單）。
        entry_price :
            建倉均價（position=0 時可傳 None）。
        atr :
            當日 ATR（來自 14:30 信號計算結果）。
        session :
            夜盤 OHLCV 聚合資料。

        Returns
        -------
        GuardResult — should_close=True 表示需要平倉。
        """
        if position <= 0 or entry_price is None:
            return GuardResult(should_close=False)

        # Guard 1: night low < entry_price - guard1_atr_mult × ATR
        if self.guard1_atr_mult is not None and self._is_enabled("guard1"):
            stop = entry_price - self.guard1_atr_mult * atr
            if session.low < stop:
                return GuardResult(
                    should_close=True,
                    reason=(
                        f"guard1: night low {session.low:.0f} < stop {stop:.0f} "
                        f"(entry {entry_price:.0f} - ×ATR {atr:.0f})"
                    ),
                )

        # Guard 2: night close < night_open - guard2_atr_mult × ATR
        if self.guard2_atr_mult is not None and self._is_enabled("guard2"):
            reversal_stop = session.open_price - self.guard2_atr_mult * atr
            if session.close < reversal_stop:
                return GuardResult(
                    should_close=True,
                    reason=(
                        f"guard2: night close {session.close:.0f} < night_open "
                        f"{session.open_price:.0f} - ×ATR {atr:.0f}"
                    ),
                )

        # Guard 3: night close < entry_price × (1 - guard3_pct)
        if self.guard3_pct is not None and self._is_enabled("guard3"):
            drawdown_stop = entry_price * (1.0 - self.guard3_pct)
            if session.close < drawdown_stop:
                return GuardResult(
                    should_close=True,
                    reason=(
                        f"guard3: night close {session.close:.0f} < entry {entry_price:.0f} "
                        f"× (1-{self.guard3_pct}) = {drawdown_stop:.0f}"
                    ),
                )

        return GuardResult(should_close=False)
