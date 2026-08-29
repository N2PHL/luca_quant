"""
Econometric baselines (Blueprint §10, Phase 4) — đề bài yêu cầu rõ:
"Chạy dữ liệu trên mô hình cổ điển: hồi qui, capm, apt, dạng luật, …"

Toàn bộ được bọc trong sklearn API (`fit` / `predict_proba`) để ExperimentRunner
xử lý y hệt mọi mô hình khác — cùng purged walk-forward, cùng tuning trên VALID,
cùng acceptance gate. Không có đường tắt cho baseline.

LƯU Ý PHƯƠNG PHÁP QUAN TRỌNG
----------------------------
CAPM và APT vốn là mô hình ĐỊNH GIÁ TÀI SẢN (giải thích lợi suất kỳ vọng bằng
phơi nhiễm rủi ro), không phải mô hình DỰ BÁO chuỗi thời gian. Dùng chúng làm
baseline dự báo hướng giá là một sự vay mượn — và điều đó phải được nói rõ trong
báo cáo, chứ không lặng lẽ đưa vào bảng như thể chúng là mô hình dự báo.

Cách dùng hợp lệ ở đây: ước lượng hệ số trên TRAIN, rồi dùng phần lợi suất mà
mô hình KHÔNG giải thích được (alpha + residual momentum) làm tín hiệu. Đó là
cách các quỹ thực sự dùng CAPM/APT trong quy trình nghiên cứu.

ARIMA/GARCH cần `statsmodels` / `arch`. Thiếu thư viện thì các mô hình này tự
biến mất khỏi registry thay vì làm sập app.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression


def _has(mod: str) -> bool:
    """Kiểm tra thư viện mà KHÔNG import — xem giải thích ở models/registry.py."""
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# ==========================================================================
# CAPM — single factor
# ==========================================================================
class CAPMClassifier(BaseEstimator, ClassifierMixin):
    """
    CAPM một nhân tố:  r_i − r_f = α + β(r_m − r_f) + ε

    Không có dữ liệu VNINDEX truyền vào, ta dùng lợi suất trễ của chính tài sản
    làm proxy nhân tố thị trường (một xấp xỉ thô — phải nêu rõ giới hạn này).

    Tín hiệu: dự báo hướng dựa trên alpha ước lượng cộng phần dư gần nhất.
    Ý tưởng kinh tế: alpha > 0 dai dẳng hàm ý tài sản sinh lợi vượt mức phần bù
    rủi ro hệ thống của nó.
    """

    def __init__(self, market_col: str | None = None, rf_daily: float = 0.0):
        self.market_col = market_col
        self.rf_daily = rf_daily

    def _market_proxy(self, X: pd.DataFrame) -> np.ndarray:
        if self.market_col and self.market_col in X.columns:
            return X[self.market_col].to_numpy(dtype=float)
        cands = [c for c in X.columns if c.endswith("ret_1d")] or \
                [c for c in X.columns if "ret_" in c]
        if not cands:
            return np.zeros(len(X))
        return X[cands[0]].to_numpy(dtype=float)

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        X = pd.DataFrame(X)
        # Nhân tố = lợi suất trễ; biến phụ thuộc = nhãn hướng (proxy cho r_i)
        f = self._market_proxy(X).reshape(-1, 1)
        yv = np.asarray(y, dtype=float)
        self.model_ = LinearRegression().fit(f, yv)
        self.alpha_ = float(self.model_.intercept_)
        self.beta_ = float(self.model_.coef_[0])
        self.base_rate_ = float(yv.mean())
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        f = self._market_proxy(X).reshape(-1, 1)
        p = np.clip(self.model_.predict(f), 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ==========================================================================
# APT — multi-factor
# ==========================================================================
class APTClassifier(BaseEstimator, ClassifierMixin):
    """
    Arbitrage Pricing Theory: hồi quy đa nhân tố có phạt Ridge.

    Nhân tố lấy từ chính ma trận feature (momentum, volatility, volume — tương
    ứng với các nhân tố phong cách kiểu Fama-French mở rộng). Phạt Ridge là cần
    thiết vì các nhân tố kỹ thuật có đa cộng tuyến rất cao (RSI, MACD, ROC đều
    là biến thể của cùng một thứ).
    """

    def __init__(self, n_factors: int = 8, alpha: float = 1.0):
        self.n_factors = n_factors
        self.alpha = alpha

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        from sklearn.decomposition import PCA

        self.classes_ = np.array([0, 1])
        Xv = np.asarray(pd.DataFrame(X), dtype=float)
        k = min(self.n_factors, Xv.shape[1], max(Xv.shape[0] // 20, 2))
        # PCA rút nhân tố chung — đúng tinh thần APT: một số ít nhân tố ẩn
        # giải thích phần lớn biến thiên chéo.
        self.pca_ = PCA(n_components=k, random_state=42).fit(Xv)
        F = self.pca_.transform(Xv)
        self.model_ = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(F, np.asarray(y, dtype=float))
        self.explained_ = float(self.pca_.explained_variance_ratio_.sum())
        return self

    def predict_proba(self, X):
        F = self.pca_.transform(np.asarray(pd.DataFrame(X), dtype=float))
        p = np.clip(self.model_.predict(F), 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ==========================================================================
# ARIMA / GARCH
# ==========================================================================
class ARIMAClassifier(BaseEstimator, ClassifierMixin):
    """
    ARIMA trên chuỗi lợi suất trễ.

    Fit MỘT LẦN trên TRAIN rồi áp cố định sang TEST — không refit cuốn chiếu
    từng phiên. Refit mỗi phiên bằng dữ liệu tới hiện tại là hợp lệ về mặt
    causal nhưng chậm gấp hàng nghìn lần; ở vai trò baseline thì fit-once là
    lựa chọn bảo thủ (thiên về đánh giá THẤP baseline, chấp nhận được).
    """

    def __init__(self, order=(2, 0, 1), ret_col: str | None = None):
        self.order = order
        self.ret_col = ret_col

    def _series(self, X: pd.DataFrame) -> np.ndarray:
        col = self.ret_col
        if col is None or col not in X.columns:
            cands = [c for c in X.columns if c.endswith("ret_1d")]
            col = cands[0] if cands else X.columns[0]
        self.ret_col_ = col
        return X[col].to_numpy(dtype=float)

    def fit(self, X, y):
        from statsmodels.tsa.arima.model import ARIMA

        self.classes_ = np.array([0, 1])
        X = pd.DataFrame(X)
        r = self._series(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_ = ARIMA(r, order=self.order).fit()
        self.base_rate_ = float(np.mean(y))
        # Chuẩn hoá dự báo thành xác suất bằng độ lệch chuẩn phần dư của TRAIN
        self.scale_ = float(np.std(self.res_.resid)) or 1e-6
        return self

    def predict_proba(self, X):
        from scipy.stats import norm
        X = pd.DataFrame(X)
        r = self._series(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # apply() gắn tham số đã ước lượng trên TRAIN vào chuỗi mới,
            # KHÔNG ước lượng lại -> không rò rỉ thông tin của TEST.
            fitted = self.res_.apply(r).fittedvalues
        p = norm.cdf(np.asarray(fitted, dtype=float) / self.scale_)
        p = np.clip(np.nan_to_num(p, nan=0.5), 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class GARCHVolClassifier(BaseEstimator, ClassifierMixin):
    """
    GARCH(1,1) — mô hình BIẾN ĐỘNG, không phải mô hình hướng giá.

    Vì vậy nó không dự báo lợi suất dương/âm. Vai trò hợp lệ của nó ở đây là
    baseline cho chiến lược "vol timing": giảm phơi nhiễm khi biến động dự báo
    cao, tăng khi thấp. Nhiều nghiên cứu cho thấy riêng vol timing đã cải thiện
    Sharpe đáng kể — nên đây là một baseline KHÓ VƯỢT và rất đáng có trong bảng.
    """

    def __init__(self, ret_col: str | None = None):
        self.ret_col = ret_col

    def _series(self, X: pd.DataFrame) -> np.ndarray:
        col = self.ret_col
        if col is None or col not in X.columns:
            cands = [c for c in X.columns if c.endswith("ret_1d")]
            col = cands[0] if cands else X.columns[0]
        return X[col].to_numpy(dtype=float)

    def fit(self, X, y):
        from arch import arch_model

        self.classes_ = np.array([0, 1])
        r = self._series(pd.DataFrame(X)) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_ = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        self.median_vol_ = float(np.median(self.res_.conditional_volatility))
        self.base_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        from arch import arch_model
        r = self._series(pd.DataFrame(X)) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fix(self.res_.params)
            vol = np.asarray(fixed.conditional_volatility, dtype=float)
        # Vol thấp -> tin cậy cao -> xác suất cao. Đây là vol timing thuần,
        # không chứa quan điểm về hướng giá.
        p = np.clip(self.base_rate_ * (self.median_vol_ / np.maximum(vol, 1e-6)), 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ==========================================================================
def register_econometric(register_fn, ModelSpec) -> None:
    """Đăng ký vào ModelRegistry. Thiếu thư viện thì bỏ qua đúng mô hình đó."""
    register_fn(ModelSpec("capm", "econometric", lambda **k: CAPMClassifier(**k), True))
    register_fn(ModelSpec("apt", "econometric", lambda **k: APTClassifier(**k), True))
    if _has("statsmodels"):
        register_fn(ModelSpec("arima", "econometric", lambda **k: ARIMAClassifier(**k),
                              False, note="cần statsmodels"))
    if _has("arch"):
        register_fn(ModelSpec("garch_vol", "econometric", lambda **k: GARCHVolClassifier(**k),
                              False, note="cần arch — baseline vol timing"))
