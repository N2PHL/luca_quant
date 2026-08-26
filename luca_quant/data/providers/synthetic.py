"""
Synthetic provider — sinh dữ liệu giả có tính chất thống kê kiểm soát được.

Mục đích (rất quan trọng cho đồ án):
  1. Chạy unit test offline, không phụ thuộc API.
  2. NEGATIVE CONTROL: sinh chuỗi random walk thuần (không có alpha).
     Nếu pipeline vẫn báo Sharpe cao trên dữ liệu KHÔNG có tín hiệu
     -> chắc chắn có leakage hoặc overfitting. Đây là bài kiểm tra
     thuyết phục nhất để bảo vệ con số "Sharpe >= 1.8".
  3. POSITIVE CONTROL: nhúng một tín hiệu momentum đã biết trước,
     kiểm tra pipeline có phát hiện được không (test độ nhạy).
"""
from __future__ import annotations

import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from luca_quant.data.providers.base import DataProvider


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(
        self,
        mode: str = "random_walk",
        seed: int = 42,
        annual_drift: float = 0.08,
        annual_vol: float = 0.28,
        signal_strength: float = 0.0,
    ):
        """
        mode:
          "random_walk" -> GBM thuần, KHÔNG có khả năng dự báo (negative control)
          "momentum"    -> lợi suất phụ thuộc momentum quá khứ (positive control)
        signal_strength: hệ số AR trên lợi suất, chỉ dùng khi mode="momentum"
        """
        self.mode = mode
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.signal_strength = signal_strength

    @staticmethod
    def _ticker_offset(ticker: str) -> int:
        """
        Trộn mã CK vào seed một cách TẤT ĐỊNH.

        Không được dùng `hash(ticker)`: từ Python 3.3 (PEP 456), hash của chuỗi
        được muối ngẫu nhiên theo từng tiến trình, trừ khi đặt PYTHONHASHSEED.
        Hệ quả với repo này rất nặng: `SyntheticProvider` là CONTROL EXPERIMENT,
        và toàn bộ lập luận khoa học dựa trên câu "chạy random walk phải ra
        AUC ≈ 0.50 và gate REJECT". Nếu mỗi lần khởi động app lại sinh ra một
        chuỗi giá khác, con số bạn viết trong báo cáo không ai tái lập được —
        kể cả chính bạn.

        Tệ hơn: với một lần rút thăm may mắn, chuỗi nhiễu thuần vẫn có thể cho
        Sharpe cao, và người đọc sẽ tưởng hệ thống bị rò rỉ dữ liệu.

        crc32 là tất định qua mọi tiến trình và mọi phiên bản Python.
        """
        return zlib.crc32(ticker.upper().encode("utf-8")) % 1000

    def _fetch_raw(self, ticker, start, end, resolution="1D") -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + self._ticker_offset(ticker))
        idx = pd.bdate_range(start=start, end=end)
        n = len(idx)
        if n < 2:
            return pd.DataFrame()

        mu = self.annual_drift / 252.0
        sigma = self.annual_vol / np.sqrt(252.0)
        shocks = rng.normal(mu, sigma, n)

        if self.mode == "momentum" and self.signal_strength > 0:
            rets = np.zeros(n)
            for t in range(1, n):
                past = rets[max(0, t - 5):t].mean() if t >= 5 else 0.0
                rets[t] = shocks[t] + self.signal_strength * past
        else:
            rets = shocks

        close = 20_000 * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, sigma * 0.6, n))
        open_ = close * (1 + rng.normal(0, sigma * 0.3, n))
        high = np.maximum(open_, close) * (1 + intraday)
        low = np.minimum(open_, close) * (1 - intraday)
        volume = rng.lognormal(13.0, 0.6, n)

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
