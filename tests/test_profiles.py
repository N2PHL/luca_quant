"""
Kiểm định hệ thống profile.

Rủi ro lớn nhất của việc có hai bản triển khai là chúng PHÂN KỲ: sửa lỗi ở
bản local rồi quên bản nộp online. Bộ test này ép hai bản dùng chung một
codebase và chỉ khác nhau ở phần lọc mô hình + trần tài nguyên.
"""
from __future__ import annotations

import importlib

import pytest

from luca_quant.config import profiles as P
from luca_quant.models import registry as mr


@pytest.fixture
def coursework(monkeypatch):
    monkeypatch.setenv("LUCA_PROFILE", "coursework")
    return P.active_profile()


@pytest.fixture
def full(monkeypatch):
    monkeypatch.setenv("LUCA_PROFILE", "full")
    return P.active_profile()


def test_default_profile_is_full(monkeypatch):
    monkeypatch.delenv("LUCA_PROFILE", raising=False)
    monkeypatch.setattr(P, "_from_secrets", lambda: None)
    monkeypatch.setattr(P, "_from_file", lambda: None)
    assert P.resolve_profile_name() == "full"


def test_unknown_profile_name_falls_back(monkeypatch):
    """Gõ sai tên profile phải rơi về mặc định, không được ném lỗi lúc khởi động."""
    monkeypatch.setenv("LUCA_PROFILE", "khong_ton_tai")
    monkeypatch.setattr(P, "_from_secrets", lambda: None)
    monkeypatch.setattr(P, "_from_file", lambda: None)
    assert P.resolve_profile_name() == "full"


def test_coursework_exposes_exactly_the_required_architectures(coursework):
    """Đúng 6 kiến trúc đề bài yêu cầu, cộng baseline bắt buộc."""
    got = set(mr.available_models())
    assert {"mlp", "cnn1d", "rnn", "lstm", "gru", "transformer"} <= got
    # các họ ngoài phạm vi phải vắng mặt
    assert not ({"lightgbm", "random_forest", "svm_rbf", "arima",
                 "stacking_ensemble", "tcn", "bilstm"} & got)


def test_baselines_survive_every_profile():
    """
    Baseline không bao giờ bị lọc bỏ.

    `benchmark.py` dùng `buy_hold` làm mốc cho kiểm định ghép cặp và
    `acceptance_gate.py` so Sharpe với nó. Lọc mất baseline thì phần kiểm định
    ý nghĩa thống kê tắt LẶNG LẼ — bảng vẫn hiện, chỉ là không còn p-value.
    """
    for name in P.PROFILES:
        prof = P.PROFILES[name]
        for base in P.MANDATORY_BASELINES:
            assert prof.allows(base), f"profile {name} đã lọc mất baseline {base}"


def test_full_profile_is_a_superset(full):
    full_models = set(mr.available_models())
    assert set(P.PROFILES["coursework"].models) <= full_models
    assert len(full_models) > len(P.PROFILES["coursework"].models)


def test_registered_models_ignores_profile(coursework):
    """Mô hình bị profile ẩn vẫn còn đăng ký — chỉ là không hiện ra UI."""
    assert len(mr.registered_models()) > len(mr.available_models())


def test_catalogue_marks_hidden_models(coursework):
    cat = mr.catalogue(all_profiles=True)
    assert "in_profile" in cat.columns
    assert (~cat["in_profile"]).any()
    assert mr.catalogue()["in_profile"].all()


# ======================================================================
# TRẦN TÀI NGUYÊN
# ======================================================================
def test_caps_apply_even_when_params_not_passed(coursework):
    """
    Trần phải là trần THẬT.

    Nếu chỉ siết những tham số người dùng khai báo thì `build_model("lstm")`
    vẫn nhận `hidden=48` từ ARCH_DEFAULTS — vượt trần mà không ai thấy.
    """
    pytest.importorskip("torch")
    m = mr.build_model("lstm")
    assert m.hidden <= coursework.max_hidden
    assert m.epochs <= coursework.max_epochs
    assert m.lookback <= coursework.max_lookback
    assert m.max_seconds <= coursework.max_seconds


def test_caps_override_explicit_user_values(coursework):
    """Ràng buộc hạ tầng không phải gợi ý — lời gọi hàm không được vượt qua."""
    pytest.importorskip("torch")
    m = mr.build_model("transformer", hidden=512, epochs=1000, lookback=250)
    assert m.hidden == coursework.max_hidden
    assert m.epochs == coursework.max_epochs
    assert m.lookback == coursework.max_lookback


def test_caps_do_not_shrink_smaller_requests(coursework):
    pytest.importorskip("torch")
    m = mr.build_model("gru", hidden=8, epochs=5)
    assert m.hidden == 8 and m.epochs == 5


def test_full_profile_applies_no_caps(full):
    pytest.importorskip("torch")
    m = mr.build_model("lstm", hidden=256, epochs=300)
    assert m.hidden == 256 and m.epochs == 300


def test_coursework_model_still_fits_and_predicts(coursework):
    import numpy as np
    import pandas as pd

    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series((rng.uniform(size=200) < 0.5).astype(int))
    for name in ["mlp", "cnn1d", "rnn", "lstm", "gru", "transformer"]:
        kw = {} if name == "mlp" else dict(epochs=3)
        p = mr.build_model(name, **kw).fit(X, y).predict_proba(X)
        assert p.shape == (200, 2)


def test_requirements_files_both_exist():
    """
    Bản online và bản đầy đủ phải là hai file requirements TÁCH BIỆT.

    Deploy nhầm requirements-full.txt lên Streamlit Cloud là build fail chứ
    không phải chạy chậm — lightgbm + statsmodels + arch + torch vượt xa ngân
    sách free tier.
    """
    from pathlib import Path
    root = Path(P.__file__).resolve().parents[2]
    online = (root / "requirements.txt").read_text(encoding="utf-8")
    fullreq = (root / "requirements-full.txt").read_text(encoding="utf-8")

    # bản online: có torch CPU, KHÔNG có gradient boosting / econometric
    assert "download.pytorch.org/whl/cpu" in online
    for pkg in ("lightgbm", "statsmodels", "arch>="):
        assert not any(ln.strip().startswith(pkg) for ln in online.splitlines())
    # bản đầy đủ: có đủ
    for pkg in ("lightgbm", "statsmodels", "arch>=", "torch"):
        assert any(ln.strip().startswith(pkg) for ln in fullreq.splitlines()), pkg


def test_profiles_module_reimports_cleanly():
    importlib.reload(P)
    assert set(P.PROFILES) == {"coursework", "full"}
