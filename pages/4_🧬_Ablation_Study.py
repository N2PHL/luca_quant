"""Ablation Study — đóng góp của từng nhóm feature, có hiệu chỉnh đa kiểm định."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.title("🧬 Ablation Study")

session = st.session_state.get("session")
if session is None or not session.ablation:
    st.info("Chạy trang **Research Lab** trước.")
    st.stop()

table = session.ablation.get("table", pd.DataFrame())
contrib = session.ablation.get("contribution", pd.DataFrame())
n_trials = session.ablation.get("n_trials", 0)

st.info(
    f"**Đọc bảng này đúng cách.** Đã chạy {n_trials} kịch bản. Chọn kịch bản có "
    "Sharpe cao nhất trong nhiều lần thử là một dạng dò tìm dữ liệu. Cột **DSR** "
    "(Deflated Sharpe Ratio) đã hiệu chỉnh cho điều đó — chỉ những kịch bản có "
    "DSR > 0.95 mới được coi là có bằng chứng thống kê."
)

show = [c for c in ["Experiment", "n_features", "Accuracy", "AUC", "Sharpe", "Δ Sharpe",
                    "CAGR", "Δ CAGR", "Max Drawdown", "Profit Factor",
                    "DSR", "Sharpe CI low", "Sharpe CI high", "error"] if c in table]
st.dataframe(table[show].round(3), use_container_width=True, hide_index=True)

if "Δ Sharpe" in table:
    st.subheader("Δ Sharpe theo từng bước thêm feature")
    d = table.dropna(subset=["Δ Sharpe"])
    fig = px.bar(d, x="Experiment", y="Δ Sharpe", height=420,
                 color=d["Δ Sharpe"] > 0,
                 color_discrete_map={True: "#2ecc71", False: "#e74c3c"})
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

if not contrib.empty:
    st.subheader("Đóng góp biên của từng nhóm feature")
    st.caption("Trung bình mức thay đổi Sharpe khi thêm nhóm đó vào các tập con "
               "không chứa nó. `always_positive = True` nghĩa là nhóm đó cải thiện "
               "kết quả trong MỌI tổ hợp đã thử — bằng chứng mạnh hơn nhiều so với "
               "một lần cải thiện đơn lẻ.")
    st.dataframe(contrib.round(3), use_container_width=True, hide_index=True)
    neg = contrib[contrib["mean Δ Sharpe"] < 0]
    if not neg.empty:
        st.warning("Nhóm làm GIẢM Sharpe, nên loại khỏi mô hình cuối: "
                   + ", ".join(neg["feature_group"]))

# --- Feature importance của mô hình tốt nhất -------------------------------
if session.best is not None and session.best.folds:
    imps = [f.feature_importance for f in session.best.folds
            if f.feature_importance is not None and not f.feature_importance.empty]
    if imps:
        st.subheader("Feature Importance (trung bình qua các fold)")
        st.caption("Repo cũ chỉ lấy importance của fold CUỐI CÙNG, không đại diện "
                   "cho mô hình được báo cáo. Ở đây là trung bình có độ lệch chuẩn.")
        agg = (pd.concat(imps).groupby("feature")["importance_pct"]
               .agg(["mean", "std"]).sort_values("mean", ascending=False).head(20)
               .reset_index())
        fig = px.bar(agg.sort_values("mean"), x="mean", y="feature", error_x="std",
                     orientation="h", height=600, labels={"mean": "Đóng góp (%)"})
        st.plotly_chart(fig, use_container_width=True)
