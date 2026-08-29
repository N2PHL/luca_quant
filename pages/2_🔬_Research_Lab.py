"""
Research Lab — trang chạy toàn bộ chuỗi nghiên cứu.

Trang này KHÔNG chứa logic quant. Nó chỉ:
  1. thu thập cấu hình từ người dùng
  2. gọi ResearchService.run_full_study()
  3. hiển thị kết quả và lưu vào st.session_state
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luca_quant.config.settings import Settings                    # noqa: E402
from luca_quant.data.providers.dnse import DNSEProvider            # noqa: E402
from luca_quant.data.providers.synthetic import SyntheticProvider  # noqa: E402
from luca_quant.features import registry as freg                   # noqa: E402
from luca_quant.labels.registry import available_labels            # noqa: E402
from luca_quant.models import registry as mreg                     # noqa: E402
from luca_quant.services.research_service import ResearchService   # noqa: E402

st.title("🔬 Research Lab")

# ---------------------------------------------------------------- Sidebar
sb = st.sidebar
sb.header("1. Bài toán")
ticker = sb.text_input("Mã chứng khoán", value=st.session_state.get("ticker", "FPT")).upper()
years = sb.slider("Số năm dữ liệu", 3, 15, 8,
                  help="Cần >= 6 năm để purged walk-forward có đủ fold có ý nghĩa.")

source = sb.radio(
    "Nguồn dữ liệu", ["DNSE (thật)", "Synthetic — random walk", "Synthetic — có momentum"],
    help=("Hai chế độ Synthetic là control experiment. Random walk KHÔNG có alpha: "
          "nếu hệ thống vẫn báo Sharpe cao trên đó thì chắc chắn có rò rỉ dữ liệu."),
)

sb.header("2. Nhãn")
# CHỈ hiện nhãn phân loại: runner gọi y.astype(int) nên nhãn hồi quy
# (forward_return / forward_volatility) sẽ làm mọi fold chết.
_labels = available_labels("classification")
label_name = sb.selectbox("Loại nhãn", _labels, index=_labels.index("direction_vol_adj"))
horizon = sb.slider("Horizon (phiên)", 1, 20, 5)
label_k = sb.slider("Ngưỡng nhãn (k × σ)", 0.0, 1.0, 0.0, 0.1,
                    help="k>0 loại bỏ vùng nhiễu nhỏ hơn chi phí giao dịch.")

sb.header("3. Feature")
groups = sb.multiselect("Nhóm feature", freg.available_groups(),
                        default=["price", "trend", "momentum", "volatility", "fractal"])

sb.header("4. Mô hình")
from luca_quant.config.profiles import active_profile      # noqa: E402

_prof = active_profile()
all_models = mreg.available_models()
if _prof.name != "full":
    sb.info(
        f"Profile **{_prof.name}** — chỉ {len(all_models)} mô hình khả dụng "
        f"và siêu tham số bị siết trần cho vừa bộ nhớ. Kết quả sẽ khác (và "
        f"thường kém hơn) bản chạy local đầy đủ."
    )

_fams = mreg.families()
fam_pick = sb.multiselect(
    "Lọc theo họ mô hình", _fams, default=_fams,
    help="baseline luôn nên có mặt — không có nó thì Sharpe không diễn giải được.",
)
pool = [m for m in all_models if mreg.get_spec(m).family in fam_pick] or all_models

_default = [m for m in ["buy_hold", "random", "momentum_rule", "logistic",
                        "elasticnet", "random_forest", "hist_gb", "lightgbm",
                        "mlp", "cnn1d", "rnn", "lstm", "gru", "transformer"]
            if m in pool]
models = sb.multiselect(
    "Benchmark lineup", pool, default=_default,
    help="Mỗi mô hình thêm vào là MỘT lần thử và sẽ làm tăng n_trials của "
         "Deflated Sharpe Ratio. Chọn có chủ đích, đừng chọn hết.",
)
if models:
    sb.caption(f"n_trials sẽ tính từ {len(models)} mô hình đã chọn.")

_prim_pool = pool if pool else all_models
_prim_default = next((m for m in ["lightgbm", "hist_gb", "random_forest",
                                  "gru", "lstm", "logistic"] if m in _prim_pool),
                     _prim_pool[0])
primary = sb.selectbox("Mô hình chính cho Ablation", _prim_pool,
                       index=_prim_pool.index(_prim_default))
with sb.expander("Ghi chú mô hình"):
    st.dataframe(mreg.catalogue(), use_container_width=True, hide_index=True)
ablation_mode = sb.selectbox("Chế độ Ablation", ["cumulative", "leave_one_out", "powerset"])

sb.header("5. Cấu hình nâng cao")
_max_splits = _prof.n_splits or 8
n_splits = sb.slider("Số fold walk-forward", 3, _max_splits,
                     min(5, _max_splits),
                     help=("Mỗi fold huấn luyện lại toàn bộ lineup. Profile "
                           f"{_prof.name} giới hạn tối đa {_max_splits} fold "
                           "để không chạm trần bộ nhớ."))
rf = sb.number_input("Lãi suất phi rủi ro (%/năm)", 0.0, 15.0, 4.5, 0.5) / 100
target_sharpe = sb.number_input("Ngưỡng chấp nhận Sharpe OOS", 0.0, 4.0, 1.80, 0.1)

run = sb.button("🚀 Chạy nghiên cứu", type="primary", use_container_width=True)

# ---------------------------------------------------------------- Run
if run:
    if not groups:
        st.error("Chọn ít nhất một nhóm feature.")
        st.stop()

    # AblationEngine chặn ở 64 tổ hợp. Không kiểm tra ở đây thì lỗi chỉ nổ ra
    # SAU KHI benchmark đã chạy xong vài phút -> người dùng mất trắng phiên chạy.
    if ablation_mode == "powerset" and 2 ** len(groups) - 1 > 64:
        st.error(
            f"Chế độ `powerset` với {len(groups)} nhóm feature sinh "
            f"{2 ** len(groups) - 1} kịch bản, vượt giới hạn 64. "
            "Giảm còn tối đa 6 nhóm, hoặc dùng `cumulative` / `leave_one_out`."
        )
        st.stop()

    s = Settings()
    s.split.n_splits = n_splits
    s.split.purge_days = max(s.split.purge_days, horizon)
    s.risk_free_rate = rf
    s.acceptance.min_oos_sharpe = target_sharpe

    if source.startswith("DNSE"):
        provider = DNSEProvider()
    elif "random walk" in source:
        provider = SyntheticProvider(mode="random_walk", seed=11)
    else:
        provider = SyntheticProvider(mode="momentum", seed=11, signal_strength=0.3)

    svc = ResearchService(s, provider)
    bar = st.progress(0.0, "Bắt đầu…")
    try:
        session = svc.run_full_study(
            ticker=ticker, groups=groups, models=models, primary_model=primary,
            ablation_mode=ablation_mode, label_name=label_name,
            label_kwargs={"horizon": horizon, **({"k": label_k} if label_name == "direction_vol_adj" else {})},
            years=years,
            progress=lambda p, m: bar.progress(min(p, 1.0), m),
        )
    except Exception as exc:                                       # noqa: BLE001
        bar.empty()
        st.error(f"Nghiên cứu dừng: {type(exc).__name__}: {exc}")
        st.stop()
    bar.empty()

    st.session_state["session"] = session
    st.session_state["ticker"] = ticker

    from luca_quant.config.profiles import memory_usage_mb, release_memory  # noqa: E402

    release_memory()
    _rss = memory_usage_mb()
    if _rss is not None:
        _cap = 1024.0        # trần xấp xỉ của Streamlit Community Cloud
        _msg = f"Bộ nhớ tiến trình: **{_rss:.0f} MB**"
        if _rss > 0.80 * _cap:
            st.warning(
                f"{_msg} — đã vượt 80% trần ~1GB của Streamlit Community Cloud. "
                "Lần chạy sau nên giảm số fold, số mô hình trong lineup, hoặc "
                "số nhóm feature. Vượt trần thì app bị kill giữa chừng chứ "
                "không có cảnh báo nào khác."
            )
        else:
            st.caption(f"{_msg} / trần ~{_cap:.0f} MB.")

# ---------------------------------------------------------------- Display
session = st.session_state.get("session")
if session is None:
    st.info("👈 Cấu hình ở thanh bên và bấm **Chạy nghiên cứu**.")
    st.stop()

st.success(f"Phiên nghiên cứu: **{session.ticker}** — {len(session.X):,} phiên, "
           f"{session.X.shape[1]} feature, horizon {session.horizon}.")

# --- Chất lượng dữ liệu ---------------------------------------------------
with st.expander("Chất lượng dữ liệu", expanded=not session.data_quality.get("clean", True)):
    q = session.data_quality
    if q.get("clean"):
        st.success(f"Dữ liệu sạch — {q.get('n_rows', 0):,} phiên.")
    else:
        for issue in q.get("issues", []):
            st.warning(issue)

# --- Leakage --------------------------------------------------------------
st.subheader("Kiểm định rò rỉ dữ liệu")
if session.leakage is not None:
    df = session.leakage.to_frame()
    (st.success if session.leakage.passed else st.error)(
        "Không phát hiện rò rỉ." if session.leakage.passed
        else "PHÁT HIỆN RÒ RỈ — mọi chỉ số phía dưới đều không đáng tin."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

if session.best is None:
    st.stop()

# --- Kết quả tốt nhất ------------------------------------------------------
st.subheader("Cấu hình tốt nhất")
m = session.best.trading_metrics
c = st.columns(5)
c[0].metric("Sharpe (OOS)", f"{m.get('Sharpe', float('nan')):.2f}")
c[1].metric("CAGR", f"{m.get('CAGR', float('nan')):.1%}")
c[2].metric("Max Drawdown", f"{m.get('Max Drawdown', float('nan')):.1%}")
c[3].metric("Profit Factor", f"{m.get('Profit Factor', float('nan')):.2f}")
c[4].metric("Số lần thử", session.n_trials,
            help="Càng thử nhiều thì Sharpe cao càng dễ do may rủi — xem Deflated Sharpe Ratio.")
st.caption(f"Cấu hình: `{session.best.name}`")

# --- Acceptance Gate -------------------------------------------------------
st.subheader("Acceptance Gate")
d = session.gate.decision
(st.success if d == "ACCEPT" else st.warning if "CONDITIONS" in d else st.error)(f"**{d}**")
st.dataframe(session.gate.to_frame(), use_container_width=True, hide_index=True)

# --- Overlay arms ----------------------------------------------------------
st.subheader("Sharpe đến từ đâu?")
st.caption("Bóc tách đóng góp của mô hình học máy so với các overlay kỹ thuật Hurst/MACD.")
st.dataframe(session.overlay_arms.round(3), use_container_width=True, hide_index=True)

# --- Ổn định theo fold -----------------------------------------------------
if not session.best.fold_metrics.empty:
    st.subheader("Ổn định qua các fold")
    st.caption("Sharpe tổng thể có thể do một fold duy nhất kéo lên. Bảng này cho thấy điều đó.")
    st.dataframe(session.best.fold_metrics.round(3), use_container_width=True, hide_index=True)

# --- Investment Thesis -----------------------------------------------------
st.subheader("Investment Thesis")
st.code(session.thesis, language=None)
st.download_button("Tải Investment Thesis (.txt)", session.thesis,
                   file_name=f"luca_thesis_{session.ticker}.txt")
