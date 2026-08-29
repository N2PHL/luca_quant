"""
Profile — cấu hình hai phiên bản triển khai từ MỘT codebase.

Vì sao cần profile thay vì hai repo:
    Hai repo tách rời sẽ phân kỳ. Sửa một lỗi leakage ở bản offline mà quên
    port sang bản nộp online là kịch bản tệ nhất có thể xảy ra với một đồ án —
    con số trên bản giảng viên chấm không còn là con số bạn đã kiểm định.
    Ở đây chỉ có một codebase; profile chỉ LỌC danh sách mô hình và siết trần
    tài nguyên. Không có nhánh logic nào khác nhau giữa hai bản.

Hai profile
-----------
`coursework`  — bản nộp trực tuyến (Streamlit Community Cloud).
    Chỉ MLP, CNN, RNN, LSTM, GRU, Transformer + baseline.
    Siết trần `hidden`/`epochs`/`n_splits` cho vừa ~1GB RAM.

`full`        — bản chạy máy local. Toàn bộ 34 mô hình, 6 họ.

Cách chọn profile (ưu tiên từ trên xuống)
-----------------------------------------
1. Biến môi trường `LUCA_PROFILE`
2. `st.secrets["LUCA_PROFILE"]`  ← cách đặt trên Streamlit Community Cloud
3. File `.luca_profile` ở gốc repo
4. Mặc định: `full`

VÌ SAO BASELINE KHÔNG BAO GIỜ BỊ LỌC BỎ
---------------------------------------
Đề bài chỉ yêu cầu MLP/CNN/RNN/LSTM/GRU/Transformer, nhưng `buy_hold` và
`random` vẫn nằm trong profile `coursework`. Đây không phải là thêm thắt cho
đủ số:

  - `experiments/benchmark.py` dùng `buy_hold` làm mốc cho kiểm định ghép cặp
    (`paired_test_vs_benchmark`). Bỏ nó đi thì phần kiểm định ý nghĩa thống kê
    tắt lặng lẽ — bảng vẫn hiện ra, chỉ là không còn p-value nào cả.
  - `evaluation/acceptance_gate.py` so Sharpe của chiến lược với Buy & Hold.
  - Về mặt học thuật: "LSTM đạt Sharpe 1.8" là câu không diễn giải được. Nếu
    Buy & Hold cùng giai đoạn đạt 1.7 thì mô hình đóng góp 0.1 và không bù nổi
    chi phí giao dịch.

Ba mô hình baseline này là numpy thuần, không phụ thuộc thư viện nào, tốn
khoảng vài KB RAM. Chúng không phải là thứ làm app hết bộ nhớ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Ba baseline luôn có mặt ở MỌI profile — xem giải thích ở docstring.
MANDATORY_BASELINES = ["buy_hold", "random", "momentum_rule"]

# Đúng danh sách đề bài yêu cầu.
COURSEWORK_MODELS = [
    "mlp",            # MLP — mạng truyền thẳng, không có khái niệm thời gian
    "cnn1d",          # CNN 1 chiều nhân quả
    "rnn",            # RNN Elman
    "lstm",           # LSTM
    "gru",            # GRU
    "transformer",    # Transformer + positional encoding
]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    # None = không lọc (dùng tất cả những gì registry có)
    models: Optional[List[str]] = None
    # Trần tài nguyên — áp cho mô hình chuỗi để vừa RAM của môi trường
    max_hidden: Optional[int] = None
    max_epochs: Optional[int] = None
    max_lookback: Optional[int] = None
    max_seconds: Optional[float] = None
    n_splits: Optional[int] = None
    torch_threads: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def allows(self, model_name: str) -> bool:
        return self.models is None or model_name in self.models

    def clamp_model_kwargs(self, kwargs: dict) -> dict:
        """
        Siết siêu tham số của mô hình chuỗi về trong trần của profile.

        Trần áp cả khi tham số KHÔNG được truyền vào. Nếu chỉ siết những tham
        số người dùng khai báo thì `build_model("lstm")` vẫn nhận `hidden=48`
        từ `ARCH_DEFAULTS` — vượt trần mà không ai thấy. Trần phải là trần
        thật, không phải gợi ý.
        """
        out = dict(kwargs)
        for key, cap in (("hidden", self.max_hidden),
                         ("epochs", self.max_epochs),
                         ("lookback", self.max_lookback),
                         ("max_seconds", self.max_seconds)):
            if cap is None:
                continue
            cur = out.get(key)
            out[key] = cap if cur is None else min(cur, cap)
        return out


PROFILES = {
    "coursework": ProfileSpec(
        name="coursework",
        description="Bản nộp trực tuyến — MLP, CNN, RNN, LSTM, GRU, Transformer",
        models=MANDATORY_BASELINES + COURSEWORK_MODELS,
        max_hidden=32,
        max_epochs=40,
        max_lookback=30,
        max_seconds=45.0,
        n_splits=4,
        torch_threads=1,      # 1 luồng: giảm RAM và tránh tranh CPU dùng chung
        notes=[
            "Streamlit Community Cloud giới hạn khoảng 1GB RAM và cho app ngủ "
            "sau 12 giờ không có truy cập.",
            "Siêu tham số bị siết trần để vừa bộ nhớ — con số ở đây sẽ KHÁC "
            "bản chạy local, và thường là kém hơn.",
            "Baseline (buy_hold / random / momentum_rule) không bị lọc bỏ vì "
            "kiểm định thống kê và acceptance gate phụ thuộc vào chúng.",
        ],
    ),
    "full": ProfileSpec(
        name="full",
        description="Bản đầy đủ chạy local — toàn bộ 6 họ mô hình",
        models=None,
        notes=["Cài `pip install -r requirements-full.txt` để mở khoá đủ 34 mô hình."],
    ),
}

DEFAULT_PROFILE = "full"


def _from_secrets() -> Optional[str]:
    """Đọc st.secrets — cách đặt biến cấu hình trên Streamlit Community Cloud."""
    try:
        import streamlit as st
        val = st.secrets.get("LUCA_PROFILE")        # type: ignore[union-attr]
        return str(val) if val else None
    except Exception:                                # noqa: BLE001
        # Không có streamlit, hoặc chưa cấu hình secrets -> bỏ qua im lặng.
        return None


def _from_file() -> Optional[str]:
    f = _REPO_ROOT / ".luca_profile"
    try:
        if f.is_file():
            val = f.read_text(encoding="utf-8").strip()
            return val or None
    except OSError:
        pass
    return None


def resolve_profile_name() -> str:
    for candidate in (os.environ.get("LUCA_PROFILE"), _from_secrets(), _from_file()):
        if candidate and candidate.strip() in PROFILES:
            return candidate.strip()
    return DEFAULT_PROFILE


def active_profile() -> ProfileSpec:
    """
    KHÔNG cache. Test cần đổi profile qua monkeypatch env, và chi phí đọc lại
    là một lần stat file — không đáng để đánh đổi lấy một cái bẫy khó gỡ.
    """
    return PROFILES[resolve_profile_name()]


def apply_torch_threads() -> None:
    """
    Giới hạn số luồng của torch theo profile.

    Trên hạ tầng dùng chung, mặc định torch mở luồng bằng số core NHÌN THẤY
    (thường là của cả máy chủ, không phải phần được cấp). Mỗi luồng tốn bộ nhớ
    và chúng tranh nhau CPU — kết quả là vừa chậm hơn vừa dễ chạm trần RAM.
    """
    p = active_profile()
    if p.torch_threads is None:
        return
    try:
        import torch
        torch.set_num_threads(int(p.torch_threads))
    except Exception:                                # noqa: BLE001
        pass


def memory_usage_mb() -> Optional[float]:
    """
    RSS hiện tại của tiến trình, tính bằng MB. `None` nếu không đọc được.

    Dùng để hiện đồng hồ bộ nhớ trên UI. Streamlit Community Cloud kill app
    khi vượt trần chứ không cảnh báo trước, nên biết mình đang ở đâu so với
    trần là thông tin thực sự hữu ích chứ không phải trang trí.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:                                    # macOS / không có procfs
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024.0
    except Exception:                       # noqa: BLE001
        return None


def release_memory() -> None:
    """
    Trả bộ nhớ đã giải phóng về hệ điều hành.

    `gc.collect()` một mình là chưa đủ. Bộ cấp phát của glibc giữ lại các
    khối đã free trong arena của nó, nên RSS không giảm và giám sát của
    Streamlit Cloud vẫn thấy tiến trình chiếm gần trần. `malloc_trim(0)` ép
    trả phần đầu heap về kernel.

    Đo trên benchmark 4 mô hình chuỗi: thu lại khoảng 25MB mỗi lần. Không
    nhiều, nhưng miễn phí và tích luỹ qua nhiều mô hình. Phần còn lại do bộ
    cấp phát nội bộ của torch giữ và không có API nào trả về được trên CPU.

    An toàn khi không phải glibc (macOS, musl/Alpine): lỗi được nuốt.
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:                                # noqa: BLE001
        pass
