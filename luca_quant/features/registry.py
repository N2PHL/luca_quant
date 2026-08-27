"""
Feature Registry (Blueprint §9).

Thay boolean flags (use_trend=True, use_momentum=True, ...) bằng registry
khai báo được. Nhờ đó Ablation Engine có thể chạy 2^k - 1 tổ hợp mà
không phải sửa chữ ký hàm.

HAI NGUYÊN TẮC BẤT BIẾN của mọi feature trong registry:

  (1) CAUSAL — giá trị tại t chỉ được tính từ dữ liệu <= t.
      Cấm tuyệt đối: shift(-k), rolling(center=True), expanding trên full data,
      bất kỳ thống kê nào tính trên toàn bộ mẫu (mean/std toàn cục).

  (2) STATIONARY — đây là lỗi lớn nhất của repo cũ.
      Repo cũ đưa EMA_10/20/50/200 (mức GIÁ tuyệt đối) làm feature.
      Giá FPT năm 2020 là 25k, năm 2025 là 130k. StandardScaler fit trên
      train (giá thấp) rồi transform sang test (giá cao) sẽ cho z-score
      lệch hẳn ra ngoài phân phối huấn luyện -> mô hình vô nghĩa trên OOS.
      Cách sửa: chuyển mọi feature mức giá thành TỶ LỆ (close/EMA - 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    warmup: int              # số phiên cần để giá trị đầu tiên hợp lệ
    description: str


_REGISTRY: Dict[str, FeatureGroup] = {}


def register(name: str, warmup: int, description: str):
    def deco(fn):
        _REGISTRY[name] = FeatureGroup(name=name, fn=fn, warmup=warmup, description=description)
        return fn
    return deco


def available_groups() -> List[str]:
    return list(_REGISTRY.keys())


def get_group(name: str) -> FeatureGroup:
    if name not in _REGISTRY:
        raise KeyError(f"Feature group '{name}' chưa đăng ký. Có: {available_groups()}")
    return _REGISTRY[name]


def total_warmup(groups: List[str]) -> int:
    return max([get_group(g).warmup for g in groups], default=0)


# ==========================================================================
# PRICE — lợi suất và hình dạng nến (đã stationary sẵn)
# ==========================================================================
@register("price", warmup=6, description="Lợi suất trễ, gap, hình dạng nến")
def _price(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5):
        out[f"ret_{k}d"] = c.pct_change(k)
    prev_c = c.shift(1)
    out["gap"] = (o - prev_c) / prev_c
    out["hl_range"] = (h - l) / c
    out["body"] = (c - o) / o
    # Vị trí đóng cửa trong biên độ ngày: 0 = đáy, 1 = đỉnh
    out["close_loc"] = (c - l) / (h - l).replace(0, np.nan)
    return out


# ==========================================================================
# TREND — CHỈ dùng tỷ lệ, không dùng mức giá
# ==========================================================================
@register("trend", warmup=200, description="Khoảng cách giá tới EMA (tỷ lệ), độ dốc EMA")
def _trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    emas = {}
    for p in (10, 20, 50, 200):
        emas[p] = c.ewm(span=p, adjust=False, min_periods=p).mean()
        # SỬA LỖI: dùng tỷ lệ thay vì mức giá tuyệt đối
        out[f"px_vs_ema{p}"] = c / emas[p] - 1.0
    out["ema10_vs_ema20"] = emas[10] / emas[20] - 1.0
    out["ema20_vs_ema50"] = emas[20] / emas[50] - 1.0
    out["ema50_vs_ema200"] = emas[50] / emas[200] - 1.0
    out["ema20_slope"] = emas[20].pct_change(5)
    return out


# ==========================================================================
# MOMENTUM
# ==========================================================================
@register("momentum", warmup=60, description="RSI, MACD chuẩn hoá theo giá, ROC")
def _momentum(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    # RSI Wilder (dùng ewm alpha=1/14 đúng chuẩn, repo cũ dùng rolling mean)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs)) / 100.0

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    # SỬA LỖI: MACD thô tỷ lệ thuận với mức giá -> chia cho close để stationary
    out["macd_norm"] = macd / c
    out["macd_hist_norm"] = (macd - signal) / c

    for k in (10, 20, 60):
        out[f"roc_{k}"] = c.pct_change(k)
    return out


# ==========================================================================
# VOLATILITY
# ==========================================================================
@register("volatility", warmup=60, description="Realized vol, ATR/close, tỷ lệ vol ngắn/dài")
def _volatility(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)

    out["rvol_20"] = ret.rolling(20).std() * np.sqrt(252)
    out["rvol_60"] = ret.rolling(60).std() * np.sqrt(252)
    out["vol_ratio"] = out["rvol_20"] / out["rvol_60"].replace(0, np.nan)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr_norm"] = tr.rolling(14).mean() / c        # ATR/close = stationary

    # Parkinson estimator — hiệu quả hơn close-to-close
    out["parkinson_20"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2))
    ) * np.sqrt(252)
    out["downside_vol_20"] = ret.clip(upper=0).rolling(20).std() * np.sqrt(252)
    return out


# ==========================================================================
# VOLUME
# ==========================================================================
@register("volume", warmup=60, description="Volume tương đối, OBV chuẩn hoá, dollar volume")
def _volume(df: pd.DataFrame) -> pd.DataFrame:
    v, c = df["volume"], df["close"]
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)

    v_ma20 = v.rolling(20).mean()
    out["vol_ratio_20"] = v / v_ma20.replace(0, np.nan) - 1.0
    out["vol_zscore_60"] = (v - v.rolling(60).mean()) / v.rolling(60).std().replace(0, np.nan)

    obv = (np.sign(ret).fillna(0) * v).cumsum()
    # OBV thô là chuỗi tích luỹ -> non-stationary. Chuẩn hoá bằng z-score cuộn.
    out["obv_z60"] = (obv - obv.rolling(60).mean()) / obv.rolling(60).std().replace(0, np.nan)
    out["dollar_vol_z60"] = (
        (v * c) - (v * c).rolling(60).mean()
    ) / (v * c).rolling(60).std().replace(0, np.nan)
    return out


# ==========================================================================
# FRACTAL — Hurst exponent
# ==========================================================================
def _hurst_vectorized(logp: np.ndarray) -> float:
    """
    Ước lượng Hurst bằng scaling của độ lệch chuẩn theo lag.

    Sửa so với repo cũ:
      - tính trên LOG GIÁ (đúng lý thuyết fBm), không phải giá thô
      - chặn log(0) khi tau = 0 (repo cũ sẽ ném -inf vào polyfit)
      - trả NaN thay vì 0.5 khi không tính được, để dropna xử lý minh bạch
        (repo cũ trả 0.5 -> tạo giá trị giả, làm loãng phân phối feature)
    """
    n = len(logp)
    if n < 20 or not np.all(np.isfinite(logp)):
        return np.nan
    max_lag = min(20, n // 3)
    if max_lag < 4:
        return np.nan

    lags = np.arange(2, max_lag)
    taus = np.empty(len(lags))
    for i, lag in enumerate(lags):
        d = logp[lag:] - logp[:-lag]
        taus[i] = d.std()

    ok = taus > 1e-12
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log(lags[ok]), np.log(taus[ok]), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


@register("fractal", warmup=110, description="Hurst cuộn 50/100, fractal dimension, biến thiên Hurst")
def _fractal(df: pd.DataFrame) -> pd.DataFrame:
    logp = np.log(df["close"])
    out = pd.DataFrame(index=df.index)
    # Cửa sổ 20 phiên của repo cũ là QUÁ NGẮN cho ước lượng Hurst:
    # sai số chuẩn lớn hơn cả biên độ tín hiệu. Dùng 50 và 100.
    for w in (50, 100):
        out[f"hurst_{w}"] = logp.rolling(w).apply(_hurst_vectorized, raw=True)
    out["fractal_dim_50"] = 2.0 - out["hurst_50"]
    out["hurst_50_chg5"] = out["hurst_50"].diff(5)
    out["hurst_spread"] = out["hurst_50"] - out["hurst_100"]
    return out


# ==========================================================================
# MARKET / REGIME — beta và tương quan với VNINDEX (tuỳ chọn)
# ==========================================================================
@register("regime", warmup=120, description="Chế độ thị trường: xu hướng, phân vị vol, drawdown")
def _regime(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)
    out["trend_strength_60"] = c.pct_change(60) / (ret.rolling(60).std() * np.sqrt(60)).replace(0, np.nan)
    out["vol_pctile_120"] = ret.rolling(20).std().rolling(120).rank(pct=True)
    running_max = c.rolling(120, min_periods=20).max()
    out["dd_from_120d_high"] = c / running_max - 1.0
    out["up_days_20"] = (ret > 0).rolling(20).mean()
    return out
