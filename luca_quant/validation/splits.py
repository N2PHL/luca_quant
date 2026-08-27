"""
Purged Walk-Forward với ba tập TRAIN / VALID / TEST (Blueprint §6).

Repo cũ dùng `TimeSeriesSplit(n_splits=5)` -> chỉ có TRAIN / TEST.
Hệ quả: không có chỗ nào để chọn siêu tham số ngoài TEST. Mọi lựa chọn
(model, ngưỡng xác suất 0.65/0.80/0.90, feature set) đều gián tiếp
được tối ưu trên TEST. Đây là hyperparameter leakage — Sharpe báo cáo
sẽ bị thổi phồng và không tái lập được trên dữ liệu thật.

Sơ đồ một fold ở đây:

    |<---- TRAIN ---->|<gap>|<-VALID->|<gap>|<-TEST->|
                       purge           purge
                     +embargo        +embargo

  gap = purge_days + embargo_days, với purge_days >= horizon của label.

Vì sao BẮT BUỘC phải có gap: nhãn tại phiên cuối TRAIN được tính từ giá
tại t+horizon, tức là đã nằm trong TEST. Không purge = mô hình học nhãn
chứa thông tin của chính giai đoạn nó sắp bị đo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray

    def sizes(self) -> dict:
        return {
            "fold": self.fold_id,
            "train": len(self.train_idx),
            "valid": len(self.valid_idx),
            "test": len(self.test_idx),
        }


class PurgedWalkForward:
    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 5,
        valid_ratio: float = 0.25,
        expanding: bool = True,
        min_train_size: int = 250,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.valid_ratio = valid_ratio
        self.expanding = expanding
        self.min_train_size = min_train_size

    @property
    def gap(self) -> int:
        return self.purge_days + self.embargo_days

    def split(self, n_samples: int) -> Iterator[Fold]:
        gap = self.gap
        # Chia phần đuôi thành n_splits khối test bằng nhau
        usable = n_samples - self.min_train_size - 2 * gap
        if usable <= self.n_splits * 20:
            raise ValueError(
                f"Không đủ dữ liệu cho {self.n_splits} fold. "
                f"Cần tối thiểu ~{self.min_train_size + 2 * gap + self.n_splits * 30} phiên, "
                f"đang có {n_samples}."
            )
        test_size = usable // self.n_splits

        for k in range(self.n_splits):
            test_start = self.min_train_size + 2 * gap + k * test_size
            test_end = test_start + test_size if k < self.n_splits - 1 else n_samples
            if test_end <= test_start:
                continue

            valid_end = test_start - gap
            valid_size = max(int(test_size * self.valid_ratio * 4), 40)
            valid_start = max(valid_end - valid_size, 0)

            train_end = valid_start - gap
            train_start = 0 if self.expanding else max(0, train_end - self.min_train_size * 2)

            if train_end - train_start < 60 or valid_end - valid_start < 20:
                continue

            yield Fold(
                fold_id=k,
                train_idx=np.arange(train_start, train_end),
                valid_idx=np.arange(valid_start, valid_end),
                test_idx=np.arange(test_start, test_end),
            )

    def report(self, n_samples: int, index: pd.Index | None = None) -> pd.DataFrame:
        rows = []
        for f in self.split(n_samples):
            row = f.sizes()
            if index is not None:
                row["test_from"] = str(index[f.test_idx[0]].date())
                row["test_to"] = str(index[f.test_idx[-1]].date())
            rows.append(row)
        return pd.DataFrame(rows)


def holdout_split(n_samples: int, train: float = 0.6, valid: float = 0.2,
                  purge: int = 5, embargo: int = 5) -> Fold:
    """
    Chia một lần 60/20/20 theo thời gian, có purge — dùng cho 'UNTOUCHED TEST'.

    Quy ước nghiên cứu của L.U.C.A: tập TEST của holdout này chỉ được chạm
    ĐÚNG MỘT LẦN, sau khi đã chốt toàn bộ mô hình và siêu tham số trên
    walk-forward. Chạy nhiều lần trên nó = quay lại đúng lỗi cũ.
    """
    gap = purge + embargo
    tr_end = int(n_samples * train)
    va_start = tr_end + gap
    va_end = va_start + int(n_samples * valid)
    te_start = va_end + gap
    if te_start >= n_samples:
        raise ValueError("Dữ liệu quá ngắn cho holdout 60/20/20 có purge.")
    return Fold(
        fold_id=-1,
        train_idx=np.arange(0, tr_end),
        valid_idx=np.arange(va_start, va_end),
        test_idx=np.arange(te_start, n_samples),
    )
