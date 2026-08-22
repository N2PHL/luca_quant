"""
Backtest Engine (Blueprint §13).

QUY ƯỚC THỜI GIAN — viết rõ ra vì đây là chỗ dễ sai nhất và repo cũ
không hề tài liệu hoá:

    t   : chốt phiên, feature tính xong, mô hình cho xác suất p_t,
          sizer cho vị thế mong muốn w_t
    t+1 : ĐẶT LỆNH và khớp (mở cửa hoặc ATC), chịu chi phí giao dịch
    t+1 : lợi suất được hưởng = close(t+1)/close(t) - 1

    => position_effective(t+1) = w_t  <=>  position = signal.shift(1)

Repo cũ có `signal.shift(1)` và đúng về mặt số học, nhưng vì không ghi rõ
quy ước nên mỗi lần đổi label horizon là lệch một phiên mà không ai phát hiện.
Ở đây `execution_lag` là tham số tường minh và được kiểm bằng unit test.

SỬA LỖI SO VỚI REPO CŨ
  1. `np.clip(position, -1, 1)` cho phép BÁN KHỐNG. TTCK Việt Nam không có
     bán khống cổ phiếu cơ sở. Mọi Sharpe sinh ra từ nhánh short là không
     thực hiện được. Mặc định ở đây: long-only [0, max_exposure].
  2. Chi phí đối xứng 0.15% cho cả mua và bán -> thiếu thuế bán 0.1%.
  3. Không mô hình hoá T+2: mua hôm nay không bán được ngay.
  4. `turnover * cost` áp một hệ số duy nhất -> ở đây tách buy_cost/sell_cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from luca_quant.config.settings import CostConfig


@dataclass
class BacktestResult:
    data: pd.DataFrame
    initial_capital: float

    @property
    def returns(self) -> pd.Series:
        return self.data["strategy_return"]

    @property
    def equity(self) -> pd.Series:
        return self.data["equity"]

    @property
    def positions(self) -> pd.Series:
        return self.data["position"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.data["gross_return"]

    @property
    def cost_drag(self) -> float:
        """Tổng chi phí đã ăn vào lợi nhuận — cần báo cáo minh bạch."""
        return float(self.data["cost"].sum())


class BacktestEngine:
    def __init__(self, cost: Optional[CostConfig] = None, execution_lag: int = 1):
        self.cost = cost or CostConfig()
        self.execution_lag = execution_lag

    # ------------------------------------------------------------------
    def _apply_settlement(self, target: pd.Series) -> pd.Series:
        """
        Ràng buộc T+n: cổ phiếu mua tại t chỉ được bán từ t+n.

        Cài đặt xấp xỉ nhưng bảo thủ: không cho phép GIẢM vị thế trong
        `settlement_days` phiên kể từ lần TĂNG gần nhất. Bỏ qua ràng buộc này
        sẽ cho phép chiến lược đảo trạng thái hằng ngày — điều không thực
        hiện được trên HOSE và là nguồn Sharpe ảo phổ biến nhất.
        """
        n = self.cost.settlement_days
        if n <= 0:
            return target

        w = target.to_numpy(dtype=float).copy()
        locked_until = -1
        for t in range(1, len(w)):
            if w[t] < w[t - 1] and t <= locked_until:
                w[t] = w[t - 1]              # chưa về tài khoản, không bán được
            elif w[t] > w[t - 1]:
                locked_until = t + n
        return pd.Series(w, index=target.index)

    # ------------------------------------------------------------------
    def run(
        self,
        prices: pd.DataFrame,
        signal: pd.Series,
        initial_capital: float = 100_000_000.0,
        apply_settlement: bool = True,
    ) -> BacktestResult:
        df = prices.loc[signal.index].copy() if not prices.index.equals(signal.index) \
            else prices.copy()
        df["signal"] = signal.astype(float)

        lo = -1.0 if self.cost.allow_short else 0.0
        target = df["signal"].shift(self.execution_lag).fillna(0.0).clip(lo, 1.0)
        if apply_settlement:
            target = self._apply_settlement(target)
        df["position"] = target

        df["market_return"] = df["close"].pct_change().fillna(0.0)
        df["gross_return"] = df["position"] * df["market_return"]

        # Chi phí bất đối xứng: tăng vị thế trả phí mua, giảm trả phí bán + thuế
        delta = df["position"].diff().fillna(df["position"])
        buys = delta.clip(lower=0)
        sells = (-delta).clip(lower=0)
        df["turnover"] = delta.abs()
        df["cost"] = buys * self.cost.buy_cost + sells * self.cost.sell_cost

        df["strategy_return"] = df["gross_return"] - df["cost"]
        df["equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()
        df["benchmark_equity"] = initial_capital * (1 + df["market_return"]).cumprod()

        eq = df["equity"]
        df["drawdown"] = eq / eq.cummax() - 1.0
        return BacktestResult(data=df, initial_capital=initial_capital)

    # ------------------------------------------------------------------
    def buy_and_hold(self, prices: pd.DataFrame,
                     initial_capital: float = 100_000_000.0) -> BacktestResult:
        """Benchmark bắt buộc. Có tính phí mua một lần cho công bằng."""
        sig = pd.Series(1.0, index=prices.index)
        return self.run(prices, sig, initial_capital, apply_settlement=False)
