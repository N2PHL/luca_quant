"""Model Benchmark — bảng so sánh toàn bộ mô hình, tách ML metrics và Trading metrics."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.title("🏆 Model Benchmark")

session = st.session_state.get("session")
if session is None or not session.benchmark:
    st.info("Chạy trang **Research Lab** trước.")
    st.stop()

table = session.benchmark["table"]

st.subheader("Bảng benchmark")
st.caption("Baseline (buy_hold / random / momentum_rule) là hàng tham chiếu bắt buộc. "
           "Một mô hình không vượt được baseline một cách có ý nghĩa thống kê "
           "(p < 0.05) thì chưa chứng minh được alpha.")

ml_cols = [c for c in ["Experiment", "Family", "Accuracy", "AUC", "Brier", "Base Rate"] if c in table]
tr_cols = [c for c in ["Experiment", "CAGR", "Sharpe", "Sortino", "Calmar", "Max Drawdown",
                       "Profit Factor", "Win Rate", "Turnover (ann.)", "Exposure",
                       "Excess vs B&H (ann.)", "t-stat", "p-value"] if c in table]

t1, t2, t3 = st.tabs(["Trading metrics", "ML metrics", "Toàn bộ"])
with t1:
    st.dataframe(table[tr_cols].round(3), use_container_width=True, hide_index=True)
with t2:
    st.dataframe(table[ml_cols].round(4), use_container_width=True, hide_index=True)
with t3:
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)

if session.benchmark.get("skipped"):
    st.warning("Bỏ qua (thiếu thư viện): " + ", ".join(session.benchmark["skipped"]))

# --- Accuracy vs Sharpe ----------------------------------------------------
if {"Accuracy", "Sharpe"} <= set(table.columns):
    st.subheader("Accuracy KHÔNG phải Sharpe")
    st.caption("Blueprint §19: chất lượng dự báo ≠ chất lượng đầu tư. "
               "Nếu hai trục dưới đây không thẳng hàng, đó chính là bằng chứng.")
    d = table.dropna(subset=["Accuracy", "Sharpe"])
    fig = px.scatter(d, x="Accuracy", y="Sharpe", text="Experiment",
                     color=d.get("Family"), height=450)
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    if len(d) > 2:
        corr = d["Accuracy"].corr(d["Sharpe"])
        st.metric("Tương quan Accuracy ↔ Sharpe", f"{corr:.2f}")

# --- Equity curves ---------------------------------------------------------
st.subheader("Đường cong tài sản (out-of-sample)")
eq = pd.DataFrame({r.name: (1 + r.oos_returns).cumprod()
                   for r in session.benchmark.get("results", [])})
if not eq.empty:
    st.plotly_chart(px.line(eq, height=450, labels={"value": "Tăng trưởng 1 đồng"}),
                    use_container_width=True)
