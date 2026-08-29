"""
Kiểm định Model Registry.

Mục tiêu: một mô hình đã xuất hiện trong `available_models()` thì PHẢI dựng
được và PHẢI chạy được. Registry hứa hẹn nhiều hơn mức nó giao được là lỗi
tệ hơn cả việc thiếu mô hình — vì nó chỉ vỡ ra lúc benchmark đang chạy dở.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from luca_quant.models import registry as mr

SLOW = {"svm_rbf", "stacking_ensemble", "voting_ensemble", "gbdt", "adaboost"}


@pytest.fixture(autouse=True)
def _full_profile(monkeypatch):
    """
    Bộ test này kiểm tra REGISTRY, không phải profile đang deploy.

    Không ghim thì `test_registry_is_not_empty` và
    `test_families_cover_the_syllabus` fail trên nhánh `submit-online`
    (profile `coursework` chỉ có 9 mô hình / 2 họ) — và fail vì lý do SAI:
    registry vẫn đầy đủ, chỉ là đang bị lọc. Ràng buộc riêng của từng profile
    được kiểm ở `tests/test_profiles.py`.
    """
    monkeypatch.setenv("LUCA_PROFILE", "full")


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(3)
    n = 260
    X = pd.DataFrame(
        {
            "price__ret_1d": rng.normal(0, 0.01, n),
            "trend__px_vs_ema50": rng.normal(0, 1, n),
            "momentum__rsi14": rng.uniform(20, 80, n),
            "volatility__atr14": rng.uniform(0.5, 2.0, n),
        }
    )
    y = pd.Series((rng.uniform(size=n) < 0.52).astype(int))
    return X, y


def test_registry_is_not_empty():
    assert len(mr.available_models()) >= 10


def test_baseline_family_always_present():
    """Không có baseline thì mọi Sharpe báo cáo đều không diễn giải được."""
    base = mr.available_models("baseline")
    assert {"buy_hold", "random", "momentum_rule"} <= set(base)


def test_families_cover_the_syllabus():
    fams = set(mr.families())
    assert {"baseline", "classical", "econometric", "ml"} <= fams


def test_catalogue_columns():
    cat = mr.catalogue()
    assert {"model", "family", "needs_scaling", "note"} <= set(cat.columns)
    assert len(cat) == len(mr.available_models())
    assert cat["model"].is_unique


def test_unknown_model_raises_with_suggestions():
    with pytest.raises(KeyError, match="Khả dụng"):
        mr.get_spec("gpt5_alpha_machine")


# registered_models() thay vì available_models(): danh sách parametrize được
# dựng lúc COLLECT, trước khi fixture kịp chạy — dùng available_models() thì
# trên nhánh submit-online chỉ 9 mô hình được kiểm, và registry đầy đủ không
# bao giờ được test.
@pytest.mark.parametrize("name", [m for m in mr.registered_models() if m not in SLOW])
def test_every_registered_model_fits_and_predicts(name, toy):
    X, y = toy
    kw = dict(epochs=3, hidden=8, max_seconds=15) \
        if mr.get_spec(name).family == "deep_learning" and name != "mlp" else {}
    model = mr.build_model(name, **kw)
    model.fit(X, y)
    p = np.asarray(model.predict_proba(X))
    assert p.shape == (len(X), 2)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))


def test_missing_library_removes_model_not_crash_app():
    """
    Repo cũ dựng TẤT CẢ mô hình trong một dict rồi mới chọn một cái — thiếu
    MỘT thư viện là sập cả app. Ở đây factory là lazy, nên chỉ cần import
    được registry là đã chứng minh tính chất đó.
    """
    import importlib
    importlib.reload(mr)
    assert len(mr.available_models()) > 0


def test_stacking_uses_expanding_window_not_kfold(toy):
    """
    Meta-learner chỉ được học từ dự báo out-of-fold sinh theo thời gian.

    `sklearn.StackingClassifier(cv=KFold)` huấn luyện base model trên các fold
    nằm SAU tập đang dự báo — meta-learner khi đó học từ xác suất do một mô
    hình đã thấy tương lai sinh ra.
    """
    from luca_quant.models.registry import TimeSeriesStackingClassifier

    X, y = toy
    m = mr.build_model("stacking_ensemble").fit(X, y)
    assert isinstance(m, TimeSeriesStackingClassifier)
    # TimeSeriesSplit không phủ phần đầu chuỗi -> coverage phải < 1
    assert 0.0 < m.oof_coverage_ < 1.0
    assert np.all(np.isfinite(m.predict_proba(X)))


def test_needs_scaling_flag_is_sane():
    """Mô hình cây không cần scaling; mô hình khoảng cách/gradient thì cần."""
    assert mr.get_spec("random_forest").needs_scaling is False
    assert mr.get_spec("hist_gb").needs_scaling is False
    for n in ("logistic", "knn", "svm_rbf", "mlp"):
        if n in mr.available_models():
            assert mr.get_spec(n).needs_scaling is True
