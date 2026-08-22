"""
Model Zoo / Registry (Blueprint §10).

Sửa so với repo cũ:
  - `ModelFactory.get_model` khởi tạo TẤT CẢ mô hình trong một dict rồi mới
    chọn một cái. Nghĩa là mỗi lần gọi đều dựng cả LightGBM lẫn CatBoost.
    Lãng phí, và crash toàn bộ app nếu thiếu MỘT thư viện.
  - Ở đây mỗi mô hình là một factory lazy, thiếu thư viện thì mô hình đó
    biến mất khỏi danh sách chứ không làm sập hệ thống.
  - Bổ sung BASELINE — repo cũ hoàn toàn không có. Không có baseline thì
    con số "Sharpe = 1.9" không nói lên điều gì: Buy & Hold VN30 nhiều giai
    đoạn cũng đạt Sharpe > 1.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


# ==========================================================================
# BASELINES — bắt buộc phải có để so sánh
# ==========================================================================
class AlwaysLongClassifier(BaseEstimator, ClassifierMixin):
    """Buy & Hold: luôn dự báo tăng với xác suất 1.0."""
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class RandomClassifier(BaseEstimator, ClassifierMixin):
    """
    Dự báo ngẫu nhiên theo base rate của TRAIN.

    Vai trò: nếu mô hình ML không vượt được cái này một cách có ý nghĩa
    thống kê thì không có alpha, chỉ có may rủi.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.base_rate_ = float(np.mean(y)) if len(y) else 0.5
        self._rng = np.random.default_rng(self.random_state)
        return self

    def predict_proba(self, X):
        p = self._rng.uniform(0, 1, len(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class MomentumRuleClassifier(BaseEstimator, ClassifierMixin):
    """
    Quy tắc cổ điển: giá trên EMA dài hạn -> dự báo tăng.

    Dùng cột feature `trend__px_vs_ema50` nếu có, nếu không lấy cột đầu tiên
    có 'px_vs_ema' trong tên. Đây là baseline "dạng luật" mà đề bài yêu cầu.
    """
    def __init__(self, col_hint: str = "px_vs_ema50"):
        self.col_hint = col_hint

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        cols = list(X.columns) if isinstance(X, pd.DataFrame) else []
        match = [c for c in cols if self.col_hint in c] or [c for c in cols if "px_vs_ema" in c]
        self.col_ = match[0] if match else (cols[0] if cols else None)
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and self.col_ in X.columns:
            v = X[self.col_].to_numpy()
        else:
            v = np.zeros(len(X))
        p = np.where(v > 0, 0.75, 0.25)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ==========================================================================
# REGISTRY
# ==========================================================================
@dataclass
class ModelSpec:
    name: str
    family: str
    factory: Callable[..., object]
    needs_scaling: bool = True
    available: bool = True
    note: str = ""


_MODELS: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    _MODELS[spec.name] = spec


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# --- Baseline -------------------------------------------------------------
_register(ModelSpec("buy_hold", "baseline", lambda **k: AlwaysLongClassifier(), False))
_register(ModelSpec("random", "baseline", lambda **k: RandomClassifier(**k), False))
_register(ModelSpec("momentum_rule", "baseline", lambda **k: MomentumRuleClassifier(), False))

# --- Classical ------------------------------------------------------------
def _logistic(**k):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, random_state=42, **k)


def _ridge_clf(**k):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(RidgeClassifier(random_state=42, **k), cv=3)


_register(ModelSpec("logistic", "classical", _logistic, True))
_register(ModelSpec("ridge", "classical", _ridge_clf, True))

# --- Econometric: CAPM, APT, ARIMA, GARCH (đề bài yêu cầu) ----------------
from luca_quant.models.econometric import register_econometric  # noqa: E402

register_econometric(_register, ModelSpec)

# --- Machine Learning -----------------------------------------------------
def _rf(**k):
    from sklearn.ensemble import RandomForestClassifier
    params = dict(n_estimators=300, max_depth=6, min_samples_leaf=20,
                  random_state=42, n_jobs=-1)
    params.update(k)
    return RandomForestClassifier(**params)


def _hgb(**k):
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_depth=4, learning_rate=0.05, max_iter=200,
                  l2_regularization=1.0, random_state=42)
    params.update(k)
    return HistGradientBoostingClassifier(**params)


def _mlp(**k):
    from sklearn.neural_network import MLPClassifier
    params = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                  early_stopping=True, random_state=42)
    params.update(k)
    return MLPClassifier(**params)


_register(ModelSpec("random_forest", "ml", _rf, False))
_register(ModelSpec("hist_gb", "ml", _hgb, False))
_register(ModelSpec("mlp", "deep_learning", _mlp, True))

if _has("lightgbm"):
    def _lgbm(**k):
        import lightgbm as lgb
        params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                      max_depth=4, min_child_samples=40, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.7,
                      reg_alpha=0.5, reg_lambda=1.0,
                      random_state=42, n_jobs=-1, verbose=-1)
        params.update(k)
        return lgb.LGBMClassifier(**params)
    _register(ModelSpec("lightgbm", "ml", _lgbm, False))

if _has("catboost"):
    def _catboost(**k):
        import catboost as cb
        params = dict(iterations=400, learning_rate=0.03, depth=4,
                      l2_leaf_reg=5.0, random_state=42, verbose=0,
                      allow_writing_files=False)   # tránh rác catboost_info/
        params.update(k)
        return cb.CatBoostClassifier(**params)
    _register(ModelSpec("catboost", "ml", _catboost, False))

if _has("xgboost"):
    def _xgb(**k):
        import xgboost as xgb
        params = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                      subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                      random_state=42, n_jobs=-1, eval_metric="logloss")
        params.update(k)
        return xgb.XGBClassifier(**params)
    _register(ModelSpec("xgboost", "ml", _xgb, False))

if _has("torch"):
    from luca_quant.models.deep import make_torch_model  # noqa: E402
    for _arch in ("lstm", "gru", "cnn1d", "transformer"):
        _register(ModelSpec(_arch, "deep_learning",
                            (lambda a: (lambda **k: make_torch_model(a, **k)))(_arch),
                            True, note="cần torch"))


# ==========================================================================
def available_models(family: str | None = None) -> List[str]:
    return [n for n, s in _MODELS.items() if family is None or s.family == family]


def families() -> List[str]:
    return sorted({s.family for s in _MODELS.values()})


def get_spec(name: str) -> ModelSpec:
    if name not in _MODELS:
        raise KeyError(f"Model '{name}' không có. Khả dụng: {available_models()}")
    return _MODELS[name]


def build_model(name: str, **kwargs):
    return get_spec(name).factory(**kwargs)


def catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": s.name, "family": s.family, "needs_scaling": s.needs_scaling, "note": s.note}
         for s in _MODELS.values()]
    ).sort_values(["family", "model"]).reset_index(drop=True)
