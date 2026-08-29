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
    """
    Kiểm tra thư viện có cài không mà KHÔNG import nó.

    `__import__(mod)` (cách cũ) thực thi luôn module. Với torch, riêng việc
    import tốn khoảng 250–450MB RSS — và khoản đó bị trả ngay lúc import
    registry, tức là ngay khi app khởi động, kể cả khi người dùng chỉ định
    chạy `buy_hold`. Trên hạ tầng giới hạn 1GB thì đó là gần một nửa ngân
    sách tiêu vào việc không ai yêu cầu.

    `find_spec` chỉ định vị module trong sys.path, không chạy nó. Torch chỉ
    thực sự được import khi một mô hình chuỗi được `fit` lần đầu.
    """
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
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


def _elasticnet_logistic(**k):
    """
    Logistic + phạt ElasticNet (L1 + L2).

    Vì sao đáng có: feature kỹ thuật đa cộng tuyến rất nặng (RSI, ROC, MACD
    đều đo cùng một thứ ở dạng khác). L1 chọn biến, L2 ổn định hệ số khi các
    biến tương quan cao — ElasticNet làm cả hai. Đây là baseline tuyến tính
    "có kỷ luật" đúng nghĩa, và nó vượt LightGBM thường xuyên hơn nhiều người
    tưởng trên dữ liệu tài chính tỉ lệ nhiễu/tín hiệu cao.
    """
    from sklearn.linear_model import LogisticRegression
    # sklearn <1.8 dùng penalty="elasticnet"; từ 1.8 `penalty` bị deprecate và
    # l1_ratio một mình quyết định dạng phạt. Thử cách mới trước, fallback về
    # cách cũ để repo chạy được trên cả hai (requirements chỉ pin >=1.3).
    import sklearn

    params = dict(solver="saga", l1_ratio=0.5, C=0.5, max_iter=3000, random_state=42)
    params.update(k)
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2] if x.isdigit())
    if ver < (1, 8) and "penalty" not in params:
        params["penalty"] = "elasticnet"
    return LogisticRegression(**params)


def _sgd_logistic(**k):
    """Logistic huấn luyện bằng SGD — nhanh, hợp với chuỗi dài."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    params = dict(loss="log_loss", penalty="l2", alpha=1e-4,
                  max_iter=2000, random_state=42)
    params.update(k)
    return CalibratedClassifierCV(SGDClassifier(**params), cv=TimeSeriesSplit(3))


def _gaussian_nb(**k):
    """Naive Bayes — giả định độc lập giữa feature, gần như chắc chắn SAI ở
    đây. Chính vì thế nó là baseline hữu ích: nó cho biết một mô hình bỏ qua
    hoàn toàn tương quan thì đạt tới đâu."""
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(**k)


def _qda(**k):
    """Quadratic Discriminant Analysis — ranh giới bậc hai, giả định Gaussian
    theo lớp với ma trận hiệp phương sai riêng. `reg_param` bắt buộc phải > 0,
    nếu không ma trận hiệp phương sai suy biến khi p lớn so với n."""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    params = dict(reg_param=0.1)
    params.update(k)
    return QuadraticDiscriminantAnalysis(**params)


def _lda(**k):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    params = dict(solver="lsqr", shrinkage="auto")
    params.update(k)
    return LinearDiscriminantAnalysis(**params)


def _knn(**k):
    """
    k-NN trên không gian feature đã chuẩn hoá.

    Cảnh báo phương pháp: k-NN KHÔNG có khái niệm thời gian. Nó có thể lấy
    "hàng xóm" là một phiên cách đó 5 năm. Điều đó hợp lệ về mặt causal (hàng
    xóm luôn nằm trong TRAIN, tức quá khứ) nhưng ngầm giả định thị trường
    dừng — giả định rất mạnh. Giữ lại như một baseline phi tham số.
    """
    from sklearn.neighbors import KNeighborsClassifier
    params = dict(n_neighbors=50, weights="distance", n_jobs=-1)
    params.update(k)
    return KNeighborsClassifier(**params)


def _svm_rbf(**k):
    """SVM nhân RBF, xác suất hiệu chuẩn bằng Platt scaling."""
    from sklearn.svm import SVC
    params = dict(C=1.0, gamma="scale", probability=True,
                  class_weight="balanced", random_state=42)
    params.update(k)
    return SVC(**params)


def _svm_linear(**k):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    params = dict(C=0.1, class_weight="balanced", max_iter=5000, random_state=42)
    params.update(k)
    return CalibratedClassifierCV(LinearSVC(**params), cv=TimeSeriesSplit(3))


_register(ModelSpec("logistic", "classical", _logistic, True))
_register(ModelSpec("ridge", "classical", _ridge_clf, True))
_register(ModelSpec("elasticnet", "classical", _elasticnet_logistic, True))
_register(ModelSpec("sgd_logistic", "classical", _sgd_logistic, True))
_register(ModelSpec("lda", "classical", _lda, True))
_register(ModelSpec("qda", "classical", _qda, True))
_register(ModelSpec("gaussian_nb", "classical", _gaussian_nb, True))
_register(ModelSpec("knn", "classical", _knn, True,
                    note="phi tham số — ngầm giả định thị trường dừng"))
_register(ModelSpec("svm_linear", "classical", _svm_linear, True))
_register(ModelSpec("svm_rbf", "classical", _svm_rbf, True, note="chậm khi n lớn"))

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


def _extra_trees(**k):
    """
    Extremely Randomized Trees. Ngưỡng cắt chọn NGẪU NHIÊN thay vì tối ưu,
    nên phương sai thấp hơn Random Forest — một tính chất rất hợp với dữ liệu
    tài chính, nơi ngưỡng cắt "tối ưu" trên TRAIN gần như luôn là nhiễu.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    params = dict(n_estimators=400, max_depth=6, min_samples_leaf=20,
                  random_state=42, n_jobs=-1)
    params.update(k)
    return ExtraTreesClassifier(**params)


def _decision_tree(**k):
    """Cây đơn — baseline "dạng luật" đọc được, cố ý giới hạn độ sâu để in ra được."""
    from sklearn.tree import DecisionTreeClassifier
    params = dict(max_depth=3, min_samples_leaf=30, random_state=42)
    params.update(k)
    return DecisionTreeClassifier(**params)


def _gbdt(**k):
    from sklearn.ensemble import GradientBoostingClassifier
    params = dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                  subsample=0.8, random_state=42)
    params.update(k)
    return GradientBoostingClassifier(**params)


def _adaboost(**k):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    params = dict(n_estimators=200, learning_rate=0.05, random_state=42)
    params.update(k)
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42), **params
    )


_register(ModelSpec("decision_tree", "ml", _decision_tree, False))
_register(ModelSpec("random_forest", "ml", _rf, False))
_register(ModelSpec("extra_trees", "ml", _extra_trees, False))
_register(ModelSpec("hist_gb", "ml", _hgb, False))
_register(ModelSpec("gbdt", "ml", _gbdt, False))
_register(ModelSpec("adaboost", "ml", _adaboost, False))
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
    from luca_quant.models.deep import SEQ_ARCHS, make_torch_model  # noqa: E402

    _ARCH_NOTE = {
        "rnn":         "Elman RNN — baseline hồi tiếp, dễ tiêu biến gradient",
        "lstm":        "cổng quên/vào/ra — chuẩn công nghiệp cho chuỗi",
        "gru":         "2 cổng thay vì 3 — ít tham số hơn LSTM ~25%",
        "bilstm":      "đọc 2 chiều TRONG cửa sổ — không leakage",
        "bigru":       "GRU 2 chiều",
        "cnn1d":       "conv nhân quả — bắt mẫu hình cục bộ",
        "tcn":         "conv giãn nở + residual — trường tiếp nhận theo cấp số nhân",
        "cnn_lstm":    "CNN trích đặc trưng cục bộ -> LSTM ghép dài hạn",
        "attn_lstm":   "LSTM + attention pooling — có `attention_profile()`",
        "transformer": "self-attention + positional encoding",
    }
    for _arch in SEQ_ARCHS:
        _register(ModelSpec(
            _arch, "deep_learning",
            (lambda a: (lambda **k: make_torch_model(a, **k)))(_arch),
            True, note=f"torch — {_ARCH_NOTE.get(_arch, '')}",
        ))


# --- Ensemble: kết hợp nhiều họ ------------------------------------------
# Lý do có mục này: các họ mô hình sai theo những cách KHÁC NHAU. Cây bắt
# tương tác phi tuyến, mô hình tuyến tính ổn định khi nhiễu lớn, mô hình chuỗi
# bắt phụ thuộc thời gian. Trung bình hoá các sai số ít tương quan là cách
# giảm phương sai rẻ nhất — thường là cải thiện thật, không phải trang trí.
#
# CHÚ Ý: `cv` của StackingClassifier PHẢI là TimeSeriesSplit. Mặc định KFold
# huấn luyện meta-learner trên các fold chứa dữ liệu SAU tập dự báo — chính là
# leakage mà toàn bộ repo này tồn tại để tránh.
def _voting(**k):
    from sklearn.ensemble import VotingClassifier
    ests = [("logistic", _logistic()), ("rf", _rf()), ("hgb", _hgb())]
    if _has("lightgbm"):
        ests.append(("lgbm", _MODELS["lightgbm"].factory()))
    params = dict(voting="soft", n_jobs=None)
    params.update(k)
    return VotingClassifier(estimators=ests, **params)


class TimeSeriesStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacking tôn trọng thứ tự thời gian.

    Vì sao phải tự viết thay vì dùng `sklearn.ensemble.StackingClassifier`:
    lớp đó sinh meta-feature bằng `cross_val_predict`, và hàm này YÊU CẦU cv
    phải là một PHÂN HOẠCH (mọi mẫu xuất hiện đúng một lần ở tập test).
    `TimeSeriesSplit` không phải phân hoạch — n_splits đầu tiên không bao giờ
    nằm trong tập test — nên sklearn ném `ValueError: cross_val_predict only
    works for partitions`.

    Cách duy nhất còn lại nếu muốn dùng lớp của sklearn là quay về `KFold`.
    Nhưng KFold huấn luyện base model trên các fold nằm SAU tập đang dự báo:
    meta-learner học từ những xác suất được sinh ra bởi mô hình đã thấy tương
    lai. Đó chính xác là loại leakage mà toàn bộ repo này tồn tại để chặn.

    Ở đây: meta-feature sinh bằng cửa sổ MỞ RỘNG (expanding), meta-learner chỉ
    được fit trên phần mẫu thực sự có dự báo out-of-fold, base model sau đó
    được fit lại trên toàn bộ TRAIN.
    """

    def __init__(self, estimators=None, final_estimator=None,
                 n_splits: int = 3, passthrough: bool = False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.passthrough = passthrough

    @staticmethod
    def _rows(X, idx):
        return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]

    @staticmethod
    def _p(est, X) -> np.ndarray:
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X))[:, 1]
        return 1 / (1 + np.exp(-np.asarray(est.predict(X), dtype=float)))

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit

        self.classes_ = np.array([0, 1])
        ests = list(self.estimators or [])
        final = self.final_estimator or LogisticRegression(max_iter=1000, random_state=42)
        yv = np.asarray(y).astype(int)
        n, m = len(yv), len(ests)

        meta = np.full((n, m), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=self.n_splits).split(np.arange(n)):
            if len(np.unique(yv[tr])) < 2:
                continue
            for j, (_, est) in enumerate(ests):
                try:
                    e = clone(est).fit(self._rows(X, tr), yv[tr])
                    meta[te, j] = self._p(e, self._rows(X, te))
                except Exception:                       # noqa: BLE001
                    pass                                # cột đó để NaN -> bị loại

        covered = ~np.isnan(meta).any(axis=1)
        if covered.sum() < 30 or len(np.unique(yv[covered])) < 2:
            raise ValueError(
                "Không đủ mẫu out-of-fold để huấn luyện meta-learner. "
                "Giảm n_splits hoặc tăng độ dài TRAIN."
            )

        Z = meta[covered]
        if self.passthrough:
            Xc = self._rows(X, np.where(covered)[0])
            Xc = Xc.to_numpy() if isinstance(Xc, pd.DataFrame) else np.asarray(Xc)
            Z = np.column_stack([Z, Xc])

        self.final_ = clone(final).fit(Z, yv[covered])
        # Base model fit lại trên TOÀN BỘ TRAIN (nhiều dữ liệu hơn = tốt hơn)
        self.fitted_ = [(nm, clone(e).fit(X, yv)) for nm, e in ests]
        self.oof_coverage_ = float(covered.mean())
        return self

    def predict_proba(self, X):
        Z = np.column_stack([self._p(e, X) for _, e in self.fitted_])
        if self.passthrough:
            Xa = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
            Z = np.column_stack([Z, Xa])
        p = np.clip(np.asarray(self.final_.predict_proba(Z))[:, 1], 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _stacking(**k):
    from sklearn.linear_model import LogisticRegression
    ests = [("logistic", _logistic()), ("rf", _rf()), ("hgb", _hgb())]
    if _has("lightgbm"):
        ests.append(("lgbm", _MODELS["lightgbm"].factory()))
    params = dict(
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        n_splits=3,
        passthrough=False,
    )
    params.update(k)
    return TimeSeriesStackingClassifier(estimators=ests, **params)


_register(ModelSpec("voting_ensemble", "ensemble", _voting, True,
                    note="soft voting: logistic + RF + HGB (+LGBM)"))
_register(ModelSpec("stacking_ensemble", "ensemble", _stacking, True,
                    note="meta-learner OOF theo cửa sổ mở rộng, không dùng KFold"))


# ==========================================================================
def available_models(family: str | None = None) -> List[str]:
    """
    Danh sách mô hình dùng được = (đã đăng ký) ∩ (được profile cho phép).

    Một mô hình chỉ vắng mặt vì hai lý do, và cả hai đều im lặng có chủ đích:
      - thiếu thư viện  -> factory không bao giờ được đăng ký
      - profile lọc bỏ  -> đăng ký rồi nhưng bị ẩn khỏi UI

    Muốn biết mô hình nào bị ẩn vì lý do nào, dùng `catalogue(all_profiles=True)`.
    """
    from luca_quant.config.profiles import active_profile
    prof = active_profile()
    return [n for n, s in _MODELS.items()
            if (family is None or s.family == family) and prof.allows(n)]


def registered_models(family: str | None = None) -> List[str]:
    """Mọi mô hình đăng ký được, KHÔNG lọc theo profile."""
    return [n for n, s in _MODELS.items() if family is None or s.family == family]


def families() -> List[str]:
    return sorted({_MODELS[n].family for n in available_models()})


def get_spec(name: str) -> ModelSpec:
    if name not in _MODELS:
        raise KeyError(f"Model '{name}' không có. Khả dụng: {available_models()}")
    return _MODELS[name]


def build_model(name: str, **kwargs):
    """
    Dựng mô hình.

    Trần tài nguyên của profile được áp bên trong `deep.make_torch_model`
    (sau khi đã trộn `ARCH_DEFAULTS`), không áp ở đây — nếu áp ở đây thì các
    giá trị mặc định theo kiến trúc chưa tồn tại và trần sẽ trượt qua chúng.
    """
    return get_spec(name).factory(**kwargs)


def catalogue(all_profiles: bool = False) -> pd.DataFrame:
    """all_profiles=True -> hiện cả mô hình bị profile hiện tại ẩn đi."""
    from luca_quant.config.profiles import active_profile
    prof = active_profile()
    names = registered_models() if all_profiles else available_models()
    return pd.DataFrame(
        [{"model": n,
          "family": _MODELS[n].family,
          "needs_scaling": _MODELS[n].needs_scaling,
          "in_profile": prof.allows(n),
          "note": _MODELS[n].note}
         for n in names]
    ).sort_values(["family", "model"]).reset_index(drop=True)
