import backtrader as bt
import joblib
import pandas as pd


class PyramidSwingStrategy(bt.Strategy):
    """
    階梯式金字塔 Swing 策略 v2
    核心哲學：「只加不減，只追贏家」

    三階段火箭發射：
      Level 1 (進場)：均線交叉 + AI 通過 → 用 ATR 最大口數的 33% 建倉
      Level 2 (確認)：持倉浮盈超過 1×ATR → 加碼至 66%
      Level 3 (爆發)：持倉浮盈超過 2×ATR → 加碼至 100%

    出場：均線反向交叉 / 停損 / 時間斷路器 → 一次性全部清倉
    絕不在持倉期間減碼，避免震盪盤中被來回洗造成手續費失血。
    """

    params = (
        ("fast", 10),
        ("slow", 50),
        ("trend_ma", 200),
        ("stop_loss", 100),
        ("model_path", "strategy/rf_model.pkl"),
        ("level2_atr_mult", 1.0),  # 浮盈達 1×ATR 觸發二次加碼
        ("level3_atr_mult", 2.0),  # 浮盈達 2×ATR 觸發三次加碼
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.trend_ma = bt.indicators.SMA(self.data.close, period=self.p.trend_ma)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

        self.entry_price = None
        self.direction = 0  # +1 多, -1 空, 0 觀望
        self.current_level = 0  # 目前火箭階段 (0=空倉, 1/2/3)
        self.entry_atr = 50.0  # 進場當下的 ATR 快照

        try:
            self.model = joblib.load(self.p.model_path)
        except Exception:
            self.model = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if self.current_level == 1:
                self.entry_price = order.executed.price

    # ── AI 預測 ──
    def predict_win(self, direction):
        if not self.model:
            return True
        fast_slope = (
            (self.fast_ma[0] - self.fast_ma[-1]) / self.fast_ma[-1] if self.fast_ma[-1] > 0 else 0
        )
        slow_slope = (
            (self.slow_ma[0] - self.slow_ma[-1]) / self.slow_ma[-1] if self.slow_ma[-1] > 0 else 0
        )
        trend_dist = (
            (self.data.close[0] - self.trend_ma[0]) / self.trend_ma[0]
            if self.trend_ma[0] > 0
            else 0
        )
        atr_ratio = self.atr[0] / self.data.close[0] if self.data.close[0] > 0 else 0
        features = pd.DataFrame(
            [
                {
                    "fast_slope": fast_slope,
                    "slow_slope": slow_slope,
                    "trend_dist": trend_dist,
                    "atr_ratio": atr_ratio,
                    "direction": direction,
                }
            ]
        )
        return self.model.predict(features)[0] == 1

    # ── 取得 ATR 最大口數 ──
    def get_max_size(self):
        comminfo = self.broker.getcommissioninfo(self.data)
        multiplier = comminfo.p.mult if hasattr(comminfo.p, "mult") else 50.0
        total_value = self.broker.getvalue()
        risk_cash = total_value * 0.05
        atr_val = max(self.atr[0], 10.0) if self.atr[0] == self.atr[0] else 50.0
        loss_per = atr_val * multiplier
        contracts = int(risk_cash // loss_per)
        return max(min(contracts, 60), 1)

    def full_close(self):
        """一次性全部清倉"""
        if self.position:
            self.close()
        self.direction = 0
        self.current_level = 0
        self.entry_price = None

    def next(self):
        if not self.fast_ma[-1] or not self.slow_ma[-1] or not self.trend_ma[0] or not self.atr[0]:
            return

        dt = self.data.datetime.datetime(0)
        weekday = dt.weekday()
        time_str = dt.strftime("%H:%M")

        # ── 時間斷路器 ──
        force_close = False
        if weekday == 2 and "13:15" <= time_str <= "13:45":
            force_close = True
        elif weekday == 4 and "13:25" <= time_str <= "13:45":
            force_close = True

        if force_close:
            self.full_close()
            return

        # ── 停損：全部清倉 ──
        if self.position and self.entry_price:
            if self.position.size > 0 and self.data.close[0] <= self.entry_price - self.p.stop_loss:
                self.full_close()
                return
            elif (
                self.position.size < 0 and self.data.close[0] >= self.entry_price + self.p.stop_loss
            ):
                self.full_close()
                return

        # ── 方向判定 (均線交叉觸發) ──
        uptrend = self.data.close[0] > self.trend_ma[0]
        downtrend = self.data.close[0] < self.trend_ma[0]

        # 反向交叉 → 全部清倉，然後可能反手建倉
        if self.crossover > 0 and self.direction == -1:
            self.full_close()
        elif self.crossover < 0 and self.direction == 1:
            self.full_close()

        # ── Level 1: 初始建倉 (33%) ──
        if self.direction == 0:
            max_size = self.get_max_size()
            level1_size = max(int(max_size * 0.33), 1)

            if self.crossover > 0 and uptrend:
                if self.predict_win(1):
                    self.direction = 1
                    self.current_level = 1
                    self.entry_atr = max(self.atr[0], 10.0)
                    self.buy(size=level1_size)
            elif self.crossover < 0 and downtrend:
                if self.predict_win(-1):
                    self.direction = -1
                    self.current_level = 1
                    self.entry_atr = max(self.atr[0], 10.0)
                    self.sell(size=level1_size)
            return

        # ── Level 2 & 3: 只在浮盈時加碼 ──
        if not self.position or not self.entry_price:
            return

        # 計算浮盈點數
        if self.direction == 1:
            unrealized_points = self.data.close[0] - self.entry_price
        else:
            unrealized_points = self.entry_price - self.data.close[0]

        max_size = self.get_max_size()

        # Level 2: 浮盈超過 1×ATR → 加碼至 66%
        if self.current_level == 1 and unrealized_points >= self.entry_atr * self.p.level2_atr_mult:
            level2_target = max(int(max_size * 0.66), 2)
            current_size = abs(self.position.size)
            add_size = level2_target - current_size
            if add_size > 0:
                if self.direction == 1:
                    self.buy(size=add_size)
                else:
                    self.sell(size=add_size)
                self.current_level = 2

        # Level 3: 浮盈超過 2×ATR → 加碼至 100%
        elif (
            self.current_level == 2 and unrealized_points >= self.entry_atr * self.p.level3_atr_mult
        ):
            level3_target = max_size
            current_size = abs(self.position.size)
            add_size = level3_target - current_size
            if add_size > 0:
                if self.direction == 1:
                    self.buy(size=add_size)
                else:
                    self.sell(size=add_size)
                self.current_level = 3
