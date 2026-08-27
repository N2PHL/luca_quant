"""
Label Engine (Blueprint §8).

Mọi label đều sinh bằng shift ÂM — đây là chỗ DUY NHẤT trong toàn hệ thống
được phép nhìn tương lai. Vì vậy label luôn được trả về kèm `horizon` để:
  - SplitEngine biết phải purge bao nhiêu phiên
  - LeakageDetector biết cột nào là target và phải bị cấm khỏi X
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass
class LabelSpec:
    name: str
    series: pd.Series
    horizon: int
    task: str                # "classification" | "regression"
    meta: dict


_LABELS: Dict[str, Callable] = {}


def register_label(name: str):
    def deco(fn):
        _LABELS[name] = fn
        return fn
    return deco


def available_labels() -> List[str]:
    return list(_LABELS.keys())


def make_label(name: str, df: pd.DataFrame, **kwargs) -> LabelSpec:
    if name not in _LABELS:
        raise KeyError(f"Label '{name}' chưa đăng ký. Có: {available_labels()}")
    return _LABELS[name](df, **kwargs)


def forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Lợi suất tương lai — dùng chung cho cả label lẫn tính PnL."""
    return df["close"].shift(-horizon) / df["close"] - 1.0


# --------------------------------------------------------------------------
@register_label("direction")
def _direction(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> LabelSpec:
    """Nhị phân: lợi suất h phiên tới > threshold."""
    fr = forward_return(df, horizon)
    y = (fr > threshold).astype("float64")
    y[fr.isna()] = np.nan
    return LabelSpec("direction", y, horizon, "classification",
                     {"threshold": threshold, "base_rate": float(y.mean(skipna=True))})


@register_label("direction_vol_adj")
def _direction_vol_adj(df: pd.DataFrame, horizon: int = 5, k: float = 0.5) -> LabelSpec:
    """
    Nhị phân có ngưỡng động theo biến động.

    Lý do tồn tại: `close.shift(-1)/close - 1 > 0` (label của repo cũ) cho
    base rate ~50.5% và phần lớn nhãn nằm trong vùng nhiễu nhỏ hơn cả chi phí
    giao dịch 0.25%. Mô hình học chủ yếu là nhiễu. Ngưỡng = k * sigma
    lọc bớt vùng vô nghĩa về kinh tế.
    """
    fr = forward_return(df, horizon)
    sigma = df["close"].pct_change().rolling(20).std() * np.sqrt(horizon)
    thr = k * sigma
    y = (fr > thr).astype("float64")
    y[fr.isna() | thr.isna()] = np.nan
    return LabelSpec("direction_vol_adj", y, horizon, "classification",
                     {"k": k, "base_rate": float(y.mean(skipna=True))})


@register_label("triple_barrier")
def _triple_barrier(df: pd.DataFrame, horizon: int = 10,
                    pt: float = 2.0, sl: float = 1.0) -> LabelSpec:
    """
    Triple-Barrier (López de Prado).

    Nhãn 1 nếu chạm rào chắn lãi trước; 0 nếu chạm rào chắn lỗ trước hoặc
    hết thời gian. Sát thực tế giao dịch hơn hẳn nhãn close-to-close vì có
    tính đến đường đi của giá trong kỳ nắm giữ, không chỉ điểm cuối.
    """
    c = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    sigma = (df["close"].pct_change().rolling(20).std()).to_numpy()
    n = len(c)
    y = np.full(n, np.nan)

    for t in range(n):
        if not np.isfinite(sigma[t]) or sigma[t] <= 0 or t + horizon >= n:
            continue
        up = c[t] * (1 + pt * sigma[t])
        dn = c[t] * (1 - sl * sigma[t])
        label = 0.0
        for j in range(t + 1, min(t + horizon + 1, n)):
            if high[j] >= up:
                label = 1.0
                break
            if low[j] <= dn:
                label = 0.0
                break
        y[t] = label

    s = pd.Series(y, index=df.index)
    return LabelSpec("triple_barrier", s, horizon, "classification",
                     {"pt": pt, "sl": sl, "base_rate": float(s.mean(skipna=True))})


@register_label("forward_return")
def _forward_return_label(df: pd.DataFrame, horizon: int = 1) -> LabelSpec:
    """Hồi quy: dự báo trực tiếp lợi suất."""
    fr = forward_return(df, horizon)
    return LabelSpec("forward_return", fr, horizon, "regression", {})


@register_label("forward_volatility")
def _forward_vol(df: pd.DataFrame, horizon: int = 20) -> LabelSpec:
    """Hồi quy: dự báo biến động tương lai (dùng cho vol targeting)."""
    ret = df["close"].pct_change()
    fv = ret.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    return LabelSpec("forward_volatility", fv, horizon, "regression", {})
