import pandas as pd
import numpy as np
from typing import Union

class LabelGenerator:
    """Tạo nhãn dự báo: Classification (Tăng/Giảm) hoặc Regression (Sinh lời)."""
    
    @staticmethod
    def create_classification_label(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        """Nhãn 1 nếu lợi suất tương lai > threshold, ngược lại 0."""
        df['Future_Return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['Label'] = (df['Future_Return'] > threshold).astype(int)
        return df.dropna()

    @staticmethod
    def create_regression_label(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Dự báo trực tiếp giá trị lợi suất tương lai."""
        df['Label'] = df['close'].shift(-horizon) / df['close'] - 1
        return df.dropna()
    