"""
FeaturePipeline — sinh ma trận đặc trưng.

SỬA LỖI PHƯƠNG PHÁP NGHIÊM TRỌNG NHẤT CỦA REPO CŨ
--------------------------------------------------
`generate_all()` cũ kết thúc bằng `df.dropna()`. Vì mỗi nhóm feature có
warmup khác nhau (price ~5 phiên, EMA_200 ~200 phiên, Hurst_50 ~50 phiên),
số hàng bị cắt sẽ KHÁC NHAU giữa các kịch bản Ablation:

    Kịch bản 1 (Price only)  -> mất 5 phiên   -> test trên 1240 phiên
    Kịch bản 4 (Full)        -> mất 200 phiên -> test trên 1045 phiên

Hai kịch bản chạy trên hai giai đoạn thị trường KHÁC NHAU. Chênh lệch
Sharpe giữa chúng phản ánh chủ yếu sự khác biệt của thị trường, không phải
đóng góp của feature. Toàn bộ kết luận Ablation của repo cũ do đó không
có hiệu lực.

Cách sửa: dựng ma trận ĐẦY ĐỦ một lần, lấy warmup lớn nhất trong TẤT CẢ
các nhóm tham gia nghiên cứu, cắt một lần duy nhất -> mọi kịch bản dùng
CHUNG một index. Lúc này Δ Sharpe mới quy được về feature.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from luca_quant.features import registry as reg


class FeaturePipeline:
    def __init__(self, groups: Sequence[str] | None = None):
        self.groups: List[str] = list(groups) if groups else reg.available_groups()
        unknown = [g for g in self.groups if g not in reg.available_groups()]
        if unknown:
            raise KeyError(f"Nhóm feature không tồn tại: {unknown}")
        self.column_map_: Dict[str, List[str]] = {}
        self.warmup_: int = 0

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trả về ma trận feature CHƯA cắt NaN, kèm column_map_ để Ablation
        biết cột nào thuộc nhóm nào.
        """
        frames = []
        self.column_map_ = {}
        for g in self.groups:
            fg = reg.get_group(g)
            block = fg.fn(df)
            block = block.replace([np.inf, -np.inf], np.nan)
            block.columns = [f"{g}__{c}" for c in block.columns]
            self.column_map_[g] = list(block.columns)
            frames.append(block)

        self.warmup_ = reg.total_warmup(self.groups)
        return pd.concat(frames, axis=1)

    def columns_for(self, groups: Sequence[str]) -> List[str]:
        cols: List[str] = []
        for g in groups:
            if g not in self.column_map_:
                raise KeyError(f"Nhóm '{g}' chưa được build.")
            cols.extend(self.column_map_[g])
        return cols


def build_common_matrix(
    df: pd.DataFrame,
    groups: Sequence[str],
    label: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, FeaturePipeline]:
    """
    Dựng (X, y) trên một INDEX CHUNG cho toàn bộ nghiên cứu.

    Index chung = giao của:
      - các hàng mà MỌI feature trong `groups` đều hợp lệ
      - các hàng mà label hợp lệ (label horizon h cắt h phiên cuối)

    Trả về X đầy đủ mọi cột; các kịch bản Ablation chỉ việc chọn cột con.
    """
    pipe = FeaturePipeline(groups)
    X_full = pipe.build(df)

    valid = X_full.notna().all(axis=1) & label.notna()
    common_idx = X_full.index[valid]

    X = X_full.loc[common_idx]
    y = label.loc[common_idx]

    if len(X) == 0:
        raise ValueError(
            f"Index chung rỗng. Cần ít nhất ~{pipe.warmup_ + 300} phiên, "
            f"hiện có {len(df)}."
        )
    return X, y, pipe
