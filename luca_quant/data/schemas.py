"""Chuẩn hoá schema OHLCV. Mọi provider PHẢI trả về đúng contract này."""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


class SchemaError(ValueError):
    pass


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đưa mọi DataFrame về contract chuẩn:
      - index: DatetimeIndex, tăng dần, không trùng, không timezone
      - cột: open/high/low/close/volume (chữ thường)
      - dtype: float64

    Lỗi ở repo cũ: dnse_client trả cột 'Close' (hoa) còn FeatureEngineer
    tự lowercase bên trong -> hai nơi cùng chịu trách nhiệm chuẩn hoá,
    dẫn đến KeyError im lặng khi đổi provider.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="date"))

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise SchemaError(f"Thiếu cột bắt buộc: {missing}. Có: {list(out.columns)}")

    if not isinstance(out.index, pd.DatetimeIndex):
        raise SchemaError("Index phải là DatetimeIndex.")

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out.index.name = "date"

    out = out[OHLCV].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def validate_ohlcv(df: pd.DataFrame, strict: bool = False) -> dict:
    """Kiểm định chất lượng dữ liệu. Trả về report thay vì im lặng nuốt lỗi."""
    report: dict = {"n_rows": len(df), "issues": []}
    if df.empty:
        report["issues"].append("Dữ liệu rỗng.")
        return report

    if not df.index.is_monotonic_increasing:
        report["issues"].append("Index không tăng dần.")
    if df.index.has_duplicates:
        report["issues"].append("Index có ngày trùng.")

    n_nan = int(df[OHLCV].isna().sum().sum())
    if n_nan:
        report["issues"].append(f"Còn {n_nan} giá trị NaN.")

    n_inf = int(np.isinf(df[OHLCV].to_numpy()).sum())
    if n_inf:
        report["issues"].append(f"Còn {n_inf} giá trị vô cực.")

    bad_hl = int((df["high"] < df["low"]).sum())
    if bad_hl:
        report["issues"].append(f"{bad_hl} phiên có high < low.")

    bad_range = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
    if bad_range:
        report["issues"].append(f"{bad_range} phiên có close nằm ngoài [low, high].")

    non_positive = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
    if non_positive:
        report["issues"].append(f"{non_positive} giá <= 0.")

    # Biên độ HOSE là +/-7%; vượt xa ngưỡng này thường là lỗi chia tách cổ phiếu
    ret = df["close"].pct_change()
    n_jump = int((ret.abs() > 0.25).sum())
    if n_jump:
        report["issues"].append(
            f"{n_jump} phiên biến động > 25% — nghi ngờ dữ liệu chưa điều chỉnh cổ tức/chia tách."
        )

    report["clean"] = len(report["issues"]) == 0
    if strict and not report["clean"]:
        raise SchemaError("; ".join(report["issues"]))
    return report


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch AN TOÀN — tuyệt đối không dùng ffill/bfill trên cột giá.

    Lỗi nghiêm trọng ở repo cũ: `df.ffill()` áp lên toàn bộ DataFrame.
    Với chuỗi thời gian tài chính, ffill giá là chấp nhận được (phiên nghỉ),
    nhưng ffill vô tội vạ sau khi đã sinh feature sẽ kéo giá trị tương lai
    lùi về quá khứ nếu thứ tự xử lý thay đổi. Ở đây ta chỉ:
      - bỏ hàng không có close
      - thay inf bằng NaN
      - volume khuyết -> 0 (không giao dịch), KHÔNG ffill
    """
    if df is None or df.empty:
        return df

    out = df.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["close"])
    out["volume"] = out["volume"].fillna(0.0)
    for col in ["open", "high", "low"]:
        out[col] = out[col].fillna(out["close"])
    return out
