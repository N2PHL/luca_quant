"""
Signal Overlay — nơi ở ĐÚNG của các quy tắc Hurst + MACD.

Blueprint §14 nói rõ: nếu muốn giữ chiến lược Hurst/MACD thì phải chuyển nó
thành Alpha Strategy / Signal Overlay và đánh giá riêng thành các cánh tay:

    AI only | AI + Hurst | AI + MACD | AI + Hurst + MACD

Chỉ khi đó mới trả lời được câu hỏi trung tâm của đồ án:
"cái gì thực sự tạo ra Sharpe?"

Toàn bộ overlay ở đây là CAUSAL: mọi tín hiệu tại t chỉ dùng dữ liệu <= t,
và được kiểm bằng LeakageDetector.check_point_in_time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


def macd_lines(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd, signal


@dataclass
class Overlay:
    name: str
    description: str

    def multiplier(self, prices: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MACDOverlay(Overlay):
    """Nhân 1.0 khi MACD > Signal, nhân `damp` khi ngược lại."""

    def __init__(self, damp: float = 0.0):
        super().__init__("macd", "Lọc theo trạng thái MACD so với đường tín hiệu")
        self.damp = damp

    def multiplier(self, prices, features) -> pd.Series:
        macd, sig = macd_lines(prices["close"])
        bull = (macd > sig).reindex(prices.index).fillna(False)
        return pd.Series(np.where(bull, 1.0, self.damp), index=prices.index)


class HurstOverlay(Overlay):
    """
    Nhân theo chế độ fractal: Hurst > 0.5 (dai dẳng) giữ nguyên vị thế,
    Hurst < ngưỡng (quay về trung bình / nhiễu) thì giảm.

    Khác biệt căn bản so với repo cũ: KHÔNG all-in, KHÔNG ghi đè AI.
    Đây là hệ số nhân trong [damp, 1.0] — nghĩa là overlay chỉ điều tiết
    độ lớn của quan điểm mà mô hình đã có.
    """

    def __init__(self, low: float = 0.45, damp: float = 0.5, col: str = "fractal__hurst_50"):
        super().__init__("hurst", "Điều tiết vị thế theo chế độ fractal (Hurst)")
        self.low = low
        self.damp = damp
        self.col = col

    def multiplier(self, prices, features) -> pd.Series:
        if self.col not in features.columns:
            # Không có feature fractal -> overlay trung tính, KHÔNG im lặng bỏ qua
            # rồi để hai kịch bản chạy hai logic khác nhau như repo cũ.
            return pd.Series(1.0, index=prices.index)
        h = features[self.col].reindex(prices.index)
        return pd.Series(np.where(h < self.low, self.damp, 1.0), index=prices.index)


class OverlayStack:
    """Ghép nhiều overlay bằng phép nhân. Kết quả luôn nằm trong [0, 1]."""

    def __init__(self, overlays: List[Overlay] | None = None):
        self.overlays = overlays or []

    def apply(self, raw_signal: pd.Series, prices: pd.DataFrame,
              features: pd.DataFrame) -> pd.Series:
        w = raw_signal.astype(float).copy()
        for ov in self.overlays:
            m = ov.multiplier(prices, features).reindex(w.index).fillna(1.0)
            if (m < 0).any() or (m > 1).any():
                raise ValueError(f"Overlay '{ov.name}' trả hệ số ngoài [0,1] — sẽ tạo alpha ẩn.")
            w = w * m
        return w.clip(0.0, 1.0)

    @property
    def label(self) -> str:
        return "+".join([o.name for o in self.overlays]) if self.overlays else "none"


def standard_arms() -> Dict[str, OverlayStack]:
    """Bốn cánh tay tiêu chuẩn để quy kết nguồn gốc của Sharpe."""
    return {
        "AI only": OverlayStack([]),
        "AI + MACD": OverlayStack([MACDOverlay()]),
        "AI + Hurst": OverlayStack([HurstOverlay()]),
        "AI + Hurst + MACD": OverlayStack([HurstOverlay(), MACDOverlay()]),
    }
