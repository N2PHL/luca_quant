"""
Kiểm định nhóm mô hình chuỗi.

Ưu tiên số một của bộ test này KHÔNG phải là độ chính xác — mà là TÍNH NHÂN
QUẢ. Một mô hình sai vẫn là kết quả nghiên cứu hợp lệ; một mô hình rò rỉ dữ
liệu thì mọi con số nó sinh ra đều vô giá trị, kể cả con số đẹp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="nhóm mô hình chuỗi cần torch")

from luca_quant.models.deep import (          # noqa: E402
    ARCH_DEFAULTS,
    SEQ_ARCHS,
    TorchSequenceClassifier,
    make_torch_model,
    make_windows,
)

FAST = dict(epochs=4, hidden=8, layers=1, max_seconds=20, val_fraction=0.2)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, f = 320, 5
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"f{i}" for i in range(f)])
    sig = X["f0"].rolling(4).mean().fillna(0.0)
    y = pd.Series(((sig + rng.normal(scale=0.5, size=n)) > 0).astype(int))
    return X, y


# ======================================================================
# CỬA SỔ TRƯỢT — phần dễ sai nhất và tốn kém nhất khi sai
# ======================================================================
def test_window_ends_at_current_bar():
    """
    Cửa sổ dự báo y[t] phải KẾT THÚC tại t, tức là chứa X[t].

    Bản cũ dùng X[t-lookback:t] và bỏ mất X[t]. Khi đó mô hình chuỗi bị trễ
    một phiên so với LightGBM (vốn được dùng X[t]), và bảng benchmark so sánh
    hai thứ không cùng điều kiện.
    """
    X = np.arange(12, dtype=np.float32).reshape(-1, 1)
    Xw, _, idx = make_windows(X, None, lookback=3, pad_mode="none")
    assert idx[0] == 2 and idx[-1] == 11
    np.testing.assert_array_equal(Xw[0].ravel(), [0, 1, 2])    # dự báo y[2]
    np.testing.assert_array_equal(Xw[-1].ravel(), [9, 10, 11])  # dự báo y[11]


def test_window_is_strictly_causal():
    """Không giá trị nào trong cửa sổ của t được lớn hơn X[t]."""
    X = np.arange(40, dtype=np.float32).reshape(-1, 1)
    for pad in ("none", "edge"):
        Xw, _, idx = make_windows(X, None, lookback=6, pad_mode=pad)
        for w, t in zip(Xw, idx):
            assert w.max() <= X[t, 0], f"cửa sổ tại t={t} chứa dữ liệu tương lai"


def test_edge_padding_covers_every_row_and_stays_causal():
    X = np.arange(10, dtype=np.float32).reshape(-1, 1)
    Xw, _, idx = make_windows(X, None, lookback=4, pad_mode="edge")
    assert len(idx) == len(X)                       # mọi phiên đều có dự báo
    np.testing.assert_array_equal(Xw[0].ravel(), [0, 0, 0, 0])   # chỉ lặp X[0]
    np.testing.assert_array_equal(Xw[3].ravel(), [0, 1, 2, 3])


def test_window_raises_when_block_too_short():
    X = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Cần >="):
        make_windows(X, None, lookback=10, pad_mode="none")


def test_windows_never_cross_block_boundary(toy):
    """
    Dựng cửa sổ PHẢI xảy ra bên trong mỗi khối.

    Nếu cửa sổ được dựng trên toàn chuỗi rồi mới cắt, mẫu đầu tiên của TEST
    sẽ chứa các phiên cuối của TRAIN — mô hình đã thấy chúng lúc học.
    Test này ép ranh giới phải sạch: cửa sổ đầu của khối B chỉ chứa dữ liệu B.
    """
    X, _ = toy
    A, Bblk = X.iloc[:200].to_numpy(np.float32), X.iloc[200:].to_numpy(np.float32)
    Bw, _, idx = make_windows(Bblk, None, lookback=10, pad_mode="none")
    # mọi hàng của cửa sổ đầu tiên phải tìm thấy trong B, không phải trong A
    for row in Bw[0]:
        assert np.any(np.all(np.isclose(Bblk, row), axis=1))
        assert not np.any(np.all(np.isclose(A, row), axis=1))


# ======================================================================
# HỢP ĐỒNG SKLEARN
# ======================================================================
@pytest.mark.parametrize("arch", SEQ_ARCHS)
def test_every_arch_fits_and_predicts(arch, toy):
    X, y = toy
    m = make_torch_model(arch, **FAST).fit(X.iloc[:250], y.iloc[:250])
    p = m.predict_proba(X.iloc[250:])
    assert p.shape == (len(X) - 250, 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(m.predict(X.iloc[250:]))) <= {0, 1}


@pytest.mark.parametrize("arch", SEQ_ARCHS)
def test_arch_defaults_are_declared(arch):
    assert arch in ARCH_DEFAULTS, f"{arch} thiếu cấu hình mặc định"


def test_saliency_matches_feature_count(toy):
    X, y = toy
    m = make_torch_model("lstm", **FAST).fit(X, y)
    assert len(m.feature_importances_) == X.shape[1]
    assert np.all(np.isfinite(m.feature_importances_))


def test_attention_profile_available_only_for_attention_archs(toy):
    X, y = toy
    assert make_torch_model("attn_lstm", **FAST).fit(X, y).attention_profile() is not None
    assert make_torch_model("gru", **FAST).fit(X, y).attention_profile() is None


def test_same_seed_same_result(toy):
    X, y = toy
    a = make_torch_model("gru", random_state=7, **FAST).fit(X, y).predict_proba(X)[:, 1]
    b = make_torch_model("gru", random_state=7, **FAST).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_single_class_train_raises_clear_error(toy):
    X, _ = toy
    with pytest.raises(ValueError, match="MỘT lớp"):
        make_torch_model("lstm", **FAST).fit(X, pd.Series(np.ones(len(X), dtype=int)))


def test_unknown_arch_rejected():
    with pytest.raises(ValueError, match="không hỗ trợ"):
        make_torch_model("wavenet")


def test_no_prediction_defaults_to_no_opinion(toy):
    """Khối ngắn hơn lookback -> 0.5, KHÔNG được lấp bằng giá trị khác."""
    X, y = toy
    m = TorchSequenceClassifier(arch="lstm", lookback=30, pad_mode="none", **FAST)
    m.fit(X.iloc[:250], y.iloc[:250])
    p = m.predict_proba(X.iloc[:5])[:, 1]
    np.testing.assert_allclose(p, 0.5)


def test_early_stopping_restores_best_epoch(toy):
    X, y = toy
    m = make_torch_model("lstm", epochs=40, hidden=8, patience=3, max_seconds=30).fit(X, y)
    assert 1 <= m.best_epoch_ <= len(m.history_)
    assert {"epoch", "train_loss", "valid_loss"} <= set(m.history_.columns)


def test_nan_input_does_not_produce_nan_output(toy):
    X, y = toy
    Xn = X.copy()
    Xn.iloc[10:15, 0] = np.nan
    p = make_torch_model("cnn1d", **FAST).fit(Xn, y).predict_proba(Xn)
    assert np.all(np.isfinite(p))


def test_learns_a_real_signal_but_not_a_fake_one(toy):
    """
    Kiểm định hai chiều — quan trọng hơn kiểm định một chiều.

    Có tín hiệu  -> AUC phải > 0.5 rõ rệt.
    Nhãn ngẫu nhiên -> AUC phải quanh 0.5. Nếu mô hình vẫn "giỏi" trên nhãn
    ngẫu nhiên thì có leakage, và mọi kết quả khác đều không đáng tin.
    """
    from sklearn.metrics import roc_auc_score

    X, y = toy
    cfg = dict(epochs=40, hidden=16, max_seconds=30, random_state=0)
    m = make_torch_model("lstm", **cfg).fit(X.iloc[:250], y.iloc[:250])
    auc_real = roc_auc_score(y.iloc[250:], m.predict_proba(X.iloc[250:])[:, 1])

    rng = np.random.default_rng(1)
    y_fake = pd.Series(rng.integers(0, 2, len(X)))
    m2 = make_torch_model("lstm", **cfg).fit(X.iloc[:250], y_fake.iloc[:250])
    auc_fake = roc_auc_score(y_fake.iloc[250:], m2.predict_proba(X.iloc[250:])[:, 1])

    assert auc_real > 0.60, f"không học được tín hiệu thật (AUC={auc_real:.3f})"
    assert abs(auc_fake - 0.5) < 0.20, f"nghi ngờ leakage: AUC nhãn giả = {auc_fake:.3f}"
