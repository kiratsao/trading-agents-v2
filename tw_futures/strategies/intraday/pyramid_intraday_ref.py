import backtrader as bt


class PyramidIntradayStrategy(bt.Strategy):
    """
    階梯式金字塔日內當沖 v2
    核心哲學：「突破先試探，確認再追加，回頭就全跑」

    Level 1：突破 ORB → 30% 口數試探
    Level 2：突破後價格持續前進超過 0.5×ORB → 加碼至 70%
    Level 3：突破後價格持續前進超過 1.0×ORB → 加碼至 100%
    回跌至 ORB 區間內 → 全部清倉（不做減碼，直接全出）
    13:25 → 無條件全部清倉
    """

    params = (
        ("orb_mins", 60),
        ("max_qty", 20),
        ("init_ratio", 0.30),
        ("level2_ratio", 0.70),
        ("level2_breakout", 0.5),  # 突破 ORB 幅度的 50% 才觸發 Level 2
        ("level3_breakout", 1.0),  # 突破 ORB 幅度的 100% 才觸發 Level 3
    )

    def __init__(self):
        self.day_high = 0
        self.day_low = float("inf")
        self.current_date = None
        self.direction = 0
        self.current_level = 0
        self.orb_range = 0

    def full_close(self):
        if self.position:
            self.close()
        self.direction = 0
        self.current_level = 0

    def next(self):
        dt = self.data.datetime.time(0)
        d_date = self.data.datetime.date(0)

        # 換日重置
        if self.current_date != d_date:
            self.current_date = d_date
            self.day_high = 0
            self.day_low = float("inf")
            self.direction = 0
            self.current_level = 0
            self.orb_range = 0

        current_time = dt.hour * 60 + dt.minute
        open_time = 8 * 60 + 45

        # ── 13:25 收盤強制清倉 ──
        if current_time >= 13 * 60 + 25:
            self.full_close()
            return

        # ── ORB 取樣期 ──
        if current_time < open_time + self.p.orb_mins:
            if self.data.high[0] > self.day_high:
                self.day_high = self.data.high[0]
            if self.data.low[0] < self.day_low:
                self.day_low = self.data.low[0]
            return

        if self.orb_range == 0 and self.day_high > self.day_low:
            self.orb_range = self.day_high - self.day_low

        if self.orb_range <= 0:
            return

        # ── Level 1: 突破建倉 ──
        if not self.position and self.direction == 0:
            if open_time + self.p.orb_mins <= current_time < 13 * 60:
                init_size = max(int(self.p.max_qty * self.p.init_ratio), 1)
                if self.data.close[0] > self.day_high:
                    self.buy(size=init_size)
                    self.direction = 1
                    self.current_level = 1
                elif self.data.close[0] < self.day_low:
                    self.sell(size=init_size)
                    self.direction = -1
                    self.current_level = 1
            return

        # ── 持倉期間監控 ──
        if self.direction == 0 or not self.position:
            return

        # 計算突破距離
        if self.direction == 1:
            breakout_dist = self.data.close[0] - self.day_high
        else:
            breakout_dist = self.day_low - self.data.close[0]

        # 回跌到 ORB 區間內 → 全出
        if breakout_dist < 0:
            self.full_close()
            return

        current_size = abs(self.position.size)

        # Level 2: 突破持續前進超過 0.5×ORB
        if self.current_level == 1 and breakout_dist >= self.orb_range * self.p.level2_breakout:
            target = max(int(self.p.max_qty * self.p.level2_ratio), 2)
            add = target - current_size
            if add > 0:
                if self.direction == 1:
                    self.buy(size=add)
                else:
                    self.sell(size=add)
                self.current_level = 2

        # Level 3: 突破持續前進超過 1.0×ORB
        elif self.current_level == 2 and breakout_dist >= self.orb_range * self.p.level3_breakout:
            target = self.p.max_qty
            add = target - current_size
            if add > 0:
                if self.direction == 1:
                    self.buy(size=add)
                else:
                    self.sell(size=add)
                self.current_level = 3
