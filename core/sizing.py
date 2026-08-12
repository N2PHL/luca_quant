import pandas as pd
import numpy as np

class ProbabilitySizer:
    """Chuyển đổi xác suất AI thành tỷ trọng phân bổ vốn."""
    
    def __init__(self, thresholds: dict = None):
        # Mặc định theo yêu cầu của bạn
        self.thresholds = thresholds or {
            0.90: 0.40,  # P > 90% -> Mua 40% vốn
            0.80: 0.30,  # P > 80% -> Mua 30% vốn
            0.65: 0.15,  # P > 65% -> Mua 15% vốn
            0.00: 0.00   # P < 65% -> Không mua (Cash)
        }
        # Sắp xếp giảm dần để quét điều kiện
        self.sorted_thresh = sorted(self.thresholds.keys(), reverse=True)

    def calculate_size(self, probability: float) -> float:
        """Map 1 xác suất sang 1 tỷ trọng."""
        for t in self.sorted_thresh:
            if probability >= t:
                return self.thresholds[t]
        return 0.0

    def apply_to_series(self, prob_series: pd.Series) -> pd.Series:
        """Vectorized áp dụng lên toàn chuỗi thời gian."""
        return prob_series.apply(self.calculate_size)
    