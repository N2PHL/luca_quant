"""
Leakage Detection Engine (Blueprint §7).

Không phải checklist trên giấy — đây là các kiểm định CHẠY ĐƯỢC, trả về
PASS/FAIL, và được Acceptance Gate gọi trước khi chấp nhận bất kỳ mô hình nào.

Năm loại leakage được kiểm:
  1. Target leakage        — cột nhãn/hàm của nhãn lọt vào X
  2. Temporal leakage      — feature tại t dùng dữ liệu > t
  3. Preprocessing leakage — scaler/imputer fit trên dữ liệu ngoài TRAIN
  4. Split leakage         — index train/valid/test giao nhau hoặc sai thứ tự
  5. Hyperparameter/threshold leakage — chọn tham số bằng TEST

Kiểm định (2) là quan trọng nhất và cũng là cái duy nhất chứng minh được
bằng thực nghiệm: POINT-IN-TIME RECONSTRUCTION. Cắt chuỗi giá tại t, tính
lại feature chỉ với dữ liệu <= t, so với giá trị tính trên chuỗi đầy đủ.
Nếu khác nhau -> feature đó nhìn tương lai. Không có cách nào lách được.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class LeakageResult:
    passed: bool
    checks: List[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", severity: str = "critical"):
        self.checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL",
             "severity": severity, "detail": detail}
        )
        if not passed and severity == "critical":
            self.passed = False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)

    def failures(self) -> List[dict]:
        return [c for c in self.checks if c["status"] == "FAIL"]


FORBIDDEN_TOKENS = (
    "future", "label", "target", "forward", "next_", "_ahead", "y_true", "outcome",
)


class LeakageDetector:
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    # ---------------- 1. Target leakage ----------------
    def check_target_columns(self, X: pd.DataFrame, res: LeakageResult) -> None:
        bad = [c for c in X.columns if any(t in c.lower() for t in FORBIDDEN_TOKENS)]
        res.add(
            "target_column_names",
            not bad,
            f"Cột nghi ngờ chứa nhãn: {bad}" if bad else "Không có cột nào mang tên nhãn.",
        )

    def check_target_correlation(self, X: pd.DataFrame, y: pd.Series,
                                 res: LeakageResult, thresh: float = 0.95) -> None:
        """Feature nào tương quan gần tuyệt đối với nhãn = chính nhãn trá hình."""
        common = X.index.intersection(y.index)
        Xa, yv = X.loc[common], y.loc[common].astype(float)
        if len(common) < 30:
            res.add("target_correlation", True,
                    "Bỏ qua: quá ít quan sát chung giữa X và y.", severity="warning")
            return

        suspects = {}
        for c in Xa.columns:
            v = Xa[c].astype(float)
            if v.std() == 0:
                continue
            r = abs(float(np.corrcoef(v, yv)[0, 1]))
            if np.isfinite(r) and r > thresh:
                suspects[c] = round(r, 4)
        res.add(
            "target_correlation",
            not suspects,
            f"|corr| > {thresh} với nhãn: {suspects}" if suspects
            else f"Không feature nào có |corr| > {thresh} với nhãn.",
        )

    # ---------------- 2. Temporal leakage ----------------
    def check_point_in_time(
        self,
        raw_df: pd.DataFrame,
        feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
        res: LeakageResult,
        n_probes: int = 6,
        tol: float = 1e-8,
    ) -> None:
        """
        Point-in-time reconstruction.

        Với vài mốc thời gian t ngẫu nhiên: tính feature trên raw_df[:t]
        và so với giá trị tại t khi tính trên raw_df đầy đủ. Bằng nhau
        <=> feature là causal.
        """
        full = feature_fn(raw_df)
        n = len(raw_df)
        rng = np.random.default_rng(0)
        probes = sorted(rng.choice(np.arange(int(n * 0.6), n - 2), size=min(n_probes, max(1, n // 4)), replace=False))

        offenders: Dict[str, float] = {}
        for t in probes:
            truncated = feature_fn(raw_df.iloc[: t + 1])
            if len(truncated) == 0:
                continue
            a = truncated.iloc[-1]
            b = full.iloc[t]
            for col in full.columns:
                va, vb = a.get(col, np.nan), b.get(col, np.nan)
                if pd.isna(va) and pd.isna(vb):
                    continue
                denom = max(abs(float(vb)) if pd.notna(vb) else 1.0, 1e-6)
                diff = abs(float(va) - float(vb)) / denom if pd.notna(va) and pd.notna(vb) else np.inf
                if diff > tol:
                    offenders[col] = max(offenders.get(col, 0.0), float(diff))

        res.add(
            "point_in_time_causality",
            not offenders,
            (f"{len(offenders)} feature thay đổi khi cắt chuỗi -> nhìn tương lai: "
             f"{dict(sorted(offenders.items(), key=lambda kv: -kv[1])[:8])}")
            if offenders else
            f"Toàn bộ {len(full.columns)} feature tái lập đúng tại {len(probes)} mốc kiểm tra.",
        )

    # ---------------- 3. Preprocessing leakage ----------------
    def check_scaler_fit_scope(self, fit_index: pd.Index, train_index: pd.Index,
                               res: LeakageResult) -> None:
        outside = fit_index.difference(train_index)
        res.add(
            "scaler_fit_scope",
            len(outside) == 0,
            f"Scaler được fit trên {len(outside)} hàng ngoài TRAIN."
            if len(outside) else "Scaler chỉ fit trên TRAIN.",
        )

    # ---------------- 4. Split leakage ----------------
    def check_split(self, train_idx, valid_idx, test_idx, res: LeakageResult) -> None:
        tr, va, te = set(map(int, train_idx)), set(map(int, valid_idx)), set(map(int, test_idx))
        overlaps = {
            "train∩valid": len(tr & va),
            "train∩test": len(tr & te),
            "valid∩test": len(va & te),
        }
        res.add("split_disjoint", sum(overlaps.values()) == 0, f"Giao nhau: {overlaps}")

        ordered = max(tr) < min(va) and max(va) < min(te)
        res.add("split_chronological", ordered,
                "TRAIN < VALID < TEST theo thời gian." if ordered
                else "Thứ tự thời gian bị đảo — tập huấn luyện chứa dữ liệu sau tập kiểm định.")

        gap_tv = min(va) - max(tr) - 1
        gap_vt = min(te) - max(va) - 1
        ok = gap_tv >= self.horizon and gap_vt >= self.horizon
        res.add("purge_gap_vs_horizon", ok,
                f"gap(train→valid)={gap_tv}, gap(valid→test)={gap_vt}, horizon={self.horizon}. "
                + ("Đủ purge." if ok else "GAP NHỎ HƠN HORIZON — nhãn cuối train nhìn sang tập sau."))

    # ---------------- 5. Threshold leakage ----------------
    def check_threshold_source(self, tuned_on: str, res: LeakageResult) -> None:
        ok = tuned_on.lower() in ("valid", "validation", "train")
        res.add("threshold_tuned_on_valid", ok,
                f"Ngưỡng xác suất được chọn trên tập '{tuned_on}'."
                + ("" if ok else " Chọn ngưỡng trên TEST là threshold leakage."))

    # ---------------- Chạy toàn bộ ----------------
    def run_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        raw_df: pd.DataFrame | None = None,
        feature_fn: Callable | None = None,
        scaler_fit_index: pd.Index | None = None,
        threshold_tuned_on: str = "valid",
    ) -> LeakageResult:
        res = LeakageResult(passed=True)
        self.check_target_columns(X, res)
        self.check_target_correlation(X, y, res)
        if train_idx is not None and valid_idx is not None and test_idx is not None:
            self.check_split(train_idx, valid_idx, test_idx, res)
            if scaler_fit_index is not None:
                self.check_scaler_fit_scope(scaler_fit_index, X.index[train_idx], res)
        if raw_df is not None and feature_fn is not None:
            self.check_point_in_time(raw_df, feature_fn, res)
        self.check_threshold_source(threshold_tuned_on, res)
        return res
