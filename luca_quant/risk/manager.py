"""
Risk Manager (Blueprint §14) — TÁI KIẾN TRÚC MẠNH NHẤT TRONG BẢN NÀY.

VẤN ĐỀ CỦA REPO CŨ
------------------
`core/risk.py::RiskManager.filter_signals()` chứa dòng:

    entry_mask = (Hurst_50 < 0.35) & MACD_Cross_Up_Below_0
    data.loc[entry_mask, signal_col] = self.max_exposure   # "Bất chấp AI, All-in 100%"

Đây không phải quản trị rủi ro. Đây là MỘT CHIẾN LƯỢC ALPHA khác được cấy
vào bên trong lớp risk, và nó GHI ĐÈ mô hình AI. Ba hệ quả:

  1. Không thể quy kết Sharpe. Nếu kết quả tốt, không ai biết là do mô hình
     học máy hay do quy tắc Hurst+MACD. Toàn bộ phần Ablation Study — vốn
     là trọng tâm của đồ án — mất ý nghĩa, vì mọi kịch bản đều bị cùng một
     lớp override đè lên.
  2. Risk manager LÀM TĂNG vị thế lên 100%. Một module mang tên "risk" mà
     hành vi là all-in thì tên gọi sai bản chất.
  3. Điều kiện dùng `Hurst_50` — trong khi nhóm feature fractal có thể bị TẮT
     ở kịch bản ablation. Lúc đó `if 'Hurst_50' in columns` là False, toàn bộ
     khối override im lặng không chạy. Kịch bản 1 và kịch bản 4 vì thế chạy
     bằng hai logic giao dịch KHÁC NHAU, không chỉ khác feature.

CÁCH SỬA
--------
Tách làm hai lớp có bất biến rõ ràng:

  RiskConstraints  — CHỈ ĐƯỢC GIẢM vị thế. Bất biến `final <= raw` được
                     assert trong code, không phải quy ước miệng.
  SignalOverlay    — quy tắc Hurst/MACD, được coi là ALPHA và đánh giá như
                     một cánh tay riêng: AI / AI+Hurst / AI+MACD / AI+cả hai.
                     (xem luca_quant/risk/overlay.py)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from luca_quant.config.settings import RiskConfig


class RiskConstraintViolation(RuntimeError):
    pass


@dataclass
class RiskReport:
    raw_exposure: float
    final_exposure: float
    reduced_by: dict


class RiskConstraints:
    """
    Bất biến: với mọi t,  0 <= final[t] <= raw[t] <= max_exposure.

    Risk manager không được tạo ra vị thế mà mô hình không đề xuất.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()

    def apply(
        self,
        raw_signal: pd.Series,
        prices: pd.DataFrame,
        realized_vol: Optional[pd.Series] = None,
    ) -> tuple[pd.Series, RiskReport]:
        w = raw_signal.astype(float).clip(0.0, self.cfg.max_exposure)
        raw = w.copy()
        reduced: dict = {}

        # --- 1. Ràng buộc thanh khoản -----------------------------------
        if "volume" in prices.columns and self.cfg.min_liquidity:
            illiquid = prices["volume"].reindex(w.index) < self.cfg.min_liquidity
            reduced["liquidity"] = float(w[illiquid].sum())
            w = w.where(~illiquid, 0.0)

        # --- 2. Vol targeting -------------------------------------------
        if self.cfg.target_volatility:
            if realized_vol is None:
                realized_vol = (
                    prices["close"].pct_change()
                    .rolling(self.cfg.vol_lookback).std() * np.sqrt(252)
                ).reindex(w.index)
            # shift(1): vol dùng để quyết định vị thế cho phiên t phải là vol
            # BIẾT ĐƯỢC tại t-1. Không shift = look-ahead.
            vol_lag = realized_vol.shift(1)
            scale = (self.cfg.target_volatility / vol_lag.replace(0, np.nan)).clip(upper=1.0)
            scale = scale.fillna(1.0)
            before = w.sum()
            w = w * scale
            reduced["vol_target"] = float(before - w.sum())

        # --- 3. Kill-switch theo drawdown -------------------------------
        if self.cfg.max_drawdown_stop:
            w = self._drawdown_stop(w, prices)

        w = w.clip(0.0, self.cfg.max_exposure)

        # --- BẤT BIẾN ---------------------------------------------------
        violations = (w > raw + 1e-9).sum()
        if violations:
            raise RiskConstraintViolation(
                f"RiskConstraints làm TĂNG vị thế tại {violations} phiên. "
                "Lớp risk chỉ được phép giảm. Nếu muốn quy tắc tăng vị thế, "
                "hãy khai báo nó là SignalOverlay (alpha) và đánh giá riêng."
            )

        return w, RiskReport(float(raw.mean()), float(w.mean()), reduced)

    # ------------------------------------------------------------------
    def _drawdown_stop(self, w: pd.Series, prices: pd.DataFrame) -> pd.Series:
        """
        Cắt vị thế khi drawdown của CHIẾN LƯỢC vượt ngưỡng.

        Phải tính tuần tự vì equity phụ thuộc vị thế, mà vị thế lại phụ thuộc
        equity. Dùng equity của phiên TRƯỚC để quyết định phiên hiện tại —
        đây chính là chỗ dễ vô tình tạo look-ahead nếu vector hoá cẩu thả.
        """
        ret = prices["close"].pct_change().reindex(w.index).fillna(0.0).to_numpy()
        wv = w.to_numpy(dtype=float).copy()
        equity, peak = 1.0, 1.0
        stopped = False
        for t in range(len(wv)):
            if stopped:
                wv[t] = 0.0
            dd = equity / peak - 1.0
            if dd <= -abs(self.cfg.max_drawdown_stop):
                stopped = True
                wv[t] = 0.0
            elif stopped and dd > -abs(self.cfg.max_drawdown_stop) * 0.5:
                stopped = False              # cho phép quay lại sau khi hồi phục
            equity *= 1 + wv[t] * ret[t]
            peak = max(peak, equity)
        return pd.Series(wv, index=w.index)
