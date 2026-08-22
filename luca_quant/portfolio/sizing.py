"""
Position Sizing (Blueprint §14).

Điểm mấu chốt: NGƯỠNG XÁC SUẤT LÀ SIÊU THAM SỐ.

Repo cũ hard-code {0.90: 0.40, 0.80: 0.30, 0.65: 0.15}. Ba con số này
không rơi từ trên trời xuống — chúng được chọn vì cho kết quả đẹp trên
chính tập dùng để báo cáo. Đó là threshold leakage (Blueprint §7),
và là loại leakage khó phát hiện nhất vì không có dòng code nào "sai".

Ở đây `ProbabilitySizer.tune()` chỉ nhận dữ liệu VALID. Tập TEST không
bao giờ được truyền vào hàm này — LeakageDetector kiểm tra điều đó.

Một vấn đề nữa: xác suất từ LightGBM/RandomForest KHÔNG được hiệu chuẩn
tốt. p = 0.90 của mô hình không có nghĩa là 90% khả năng tăng thật.
Dùng thẳng p để quyết định cấp 40% vốn là sai bản chất. Vì vậy có
`calibrate=True` (isotonic/Platt fit trên VALID) trước khi map sang size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SizingParams:
    entry_threshold: float = 0.55
    max_position: float = 1.0
    mode: str = "step"          # "step" | "linear" | "binary"
    tuned_on: str = "valid"     # phải là "valid" — LeakageDetector kiểm tra


class ProbabilitySizer:
    def __init__(self, params: Optional[SizingParams] = None):
        self.params = params or SizingParams()
        self.calibrator_ = None

    # ------------------------------------------------------------------
    def fit_calibration(self, proba_valid: np.ndarray, y_valid: np.ndarray,
                        method: str = "isotonic") -> "ProbabilitySizer":
        """Hiệu chuẩn xác suất trên VALID. Không bao giờ trên TEST."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        p = np.asarray(proba_valid, dtype=float).reshape(-1)
        y = np.asarray(y_valid, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            self.calibrator_ = None
            return self

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(p, y)
            self.calibrator_ = lambda x: cal.predict(np.asarray(x, dtype=float))
        else:                                    # Platt scaling
            lr = LogisticRegression()
            lr.fit(p.reshape(-1, 1), y)
            self.calibrator_ = lambda x: lr.predict_proba(
                np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]
        return self

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        if self.calibrator_ is None:
            return np.asarray(proba, dtype=float)
        return np.clip(self.calibrator_(proba), 0.0, 1.0)

    # ------------------------------------------------------------------
    def size(self, proba: np.ndarray) -> np.ndarray:
        p = self.calibrate(proba)
        thr, mx, mode = self.params.entry_threshold, self.params.max_position, self.params.mode

        if mode == "binary":
            return np.where(p >= thr, mx, 0.0)
        if mode == "linear":
            # Tỷ trọng tăng tuyến tính từ ngưỡng lên 1.0 — mượt hơn bậc thang,
            # giảm turnover và do đó giảm chi phí giao dịch.
            scaled = (p - thr) / max(1.0 - thr, 1e-9)
            return np.clip(scaled, 0.0, 1.0) * mx
        # "step": bậc thang trên chính ngưỡng đã tune
        out = np.zeros_like(p)
        out[p >= thr] = 0.5 * mx
        out[p >= thr + (1 - thr) * 0.4] = 0.75 * mx
        out[p >= thr + (1 - thr) * 0.7] = 1.0 * mx
        return out

    def size_series(self, proba: pd.Series) -> pd.Series:
        return pd.Series(self.size(proba.to_numpy()), index=proba.index)

    # ------------------------------------------------------------------
    def tune(
        self,
        proba_valid: pd.Series,
        prices_valid: pd.DataFrame,
        objective: Callable[[pd.Series, pd.Series], float],
        grid_entry: List[float],
        modes: Optional[List[str]] = None,
        max_position: float = 1.0,
    ) -> Dict:
        """
        Quét lưới ngưỡng TRÊN TẬP VALID.

        objective(returns, positions) -> điểm số cần cực đại (thường là Sharpe).
        Trả về bảng kết quả đầy đủ để đưa vào báo cáo — người chấm đồ án
        cần thấy ngưỡng được chọn thế nào, không chỉ thấy con số cuối.
        """
        modes = modes or ["binary", "linear", "step"]
        rows = []
        best = (-np.inf, None)

        for mode in modes:
            for thr in grid_entry:
                trial = ProbabilitySizer(
                    SizingParams(entry_threshold=thr, max_position=max_position, mode=mode)
                )
                trial.calibrator_ = self.calibrator_
                w = trial.size_series(proba_valid)
                score = objective(prices_valid, w)
                rows.append({"mode": mode, "entry_threshold": thr, "score": score})
                if np.isfinite(score) and score > best[0]:
                    best = (score, SizingParams(entry_threshold=thr,
                                                max_position=max_position, mode=mode))

        if best[1] is not None:
            self.params = best[1]
        return {
            "best_params": self.params,
            "best_score": best[0],
            "grid": pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True),
        }
