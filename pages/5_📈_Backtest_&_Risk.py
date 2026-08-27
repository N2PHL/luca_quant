"""Backtest & Risk — equity curve, drawdown, phân rã chi phí, stress test."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luca_quant.evaluation.statistical_tests import bootstrap_sharpe_ci  # noqa: E402

st.title("📈 Backtest & Risk")

session = st.session_state.get("session")
if session is None or session.best is None:
    st.info("Chạy trang **Research Lab** trước.")
    st.stop()

bt = session.best.backtest
data = bt.data

st.caption(f"Cấu hình đang xem: `{session.best.name}` — "
           f"{len(data):,} phiên out-of-sample.")

# --- Equity vs benchmark ---------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["equity"], name="L.U.C.A", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data["benchmark_equity"], name="Buy & Hold",
                         line=dict(width=1, dash="dash")))
fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0),
                  legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, use_container_width=True)

# --- Drawdown --------------------------------------------------------------
st.subheader("Drawdown")
fig_dd = px.area(data, y="drawdown", height=280)
fig_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%")
st.plotly_chart(fig_dd, use_container_width=True)

# --- Chi phí giao dịch -----------------------------------------------------
st.subheader("Chi phí giao dịch có ăn hết alpha không?")
gross_ann = float(data["gross_return"].mean() * 252)
net_ann = float(data["strategy_return"].mean() * 252)
cost_ann = float(data["cost"].mean() * 252)
c = st.columns(4)
c[0].metric("Lợi suất gộp (năm)", f"{gross_ann:.2%}")
c[1].metric("Chi phí (năm)", f"-{cost_ann:.2%}")
c[2].metric("Lợi suất ròng (năm)", f"{net_ann:.2%}")
c[3].metric("Chi phí / lợi suất gộp",
            f"{cost_ann/gross_ann:.0%}" if gross_ann > 0 else "n/a")
if gross_ann > 0 and cost_ann / gross_ann > 0.5:
    st.warning("Chi phí giao dịch ăn hơn nửa lợi suất gộp. Cần giảm turnover "
               "(dùng sizing mode 'linear', tăng horizon nhãn, hoặc thêm ngưỡng "
               "chỉ giao dịch khi thay đổi vị thế đủ lớn).")

# --- Stress test theo năm --------------------------------------------------
st.subheader("Stress test theo năm")
st.caption("Một Sharpe tổng thể đẹp có thể do một năm duy nhất kéo lên. "
           "Bảng này bóc tách theo từng năm.")
yearly = data.groupby(data.index.year).agg(
    strategy=("strategy_return", lambda s: (1 + s).prod() - 1),
    benchmark=("market_return", lambda s: (1 + s).prod() - 1),
    vol=("strategy_return", lambda s: s.std() * np.sqrt(252)),
    worst_dd=("drawdown", "min"),
    exposure=("position", lambda s: (s > 0).mean()),
)
yearly["excess"] = yearly["strategy"] - yearly["benchmark"]
st.dataframe(yearly.round(3), use_container_width=True)

# --- Robustness ------------------------------------------------------------
st.subheader("Robustness: Sharpe có khác 0 một cách có ý nghĩa không?")
ci = bootstrap_sharpe_ci(bt.returns, n_boot=2000)
c = st.columns(3)
c[0].metric("Sharpe", f"{ci['sharpe']:.2f}")
c[1].metric("CI 95%", f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
c[2].metric("P(Sharpe ≤ 0)", f"{ci['p_value']:.3f}")
st.caption("Stationary block bootstrap (khối 20 phiên) — giữ nguyên tự tương quan "
           "và hiện tượng cụm biến động của lợi suất tài chính.")

# --- Độ nhạy với chi phí ---------------------------------------------------
st.subheader("Độ nhạy với giả định chi phí")
from luca_quant.backtest.engine import BacktestEngine   # noqa: E402
from luca_quant.config.settings import CostConfig       # noqa: E402
from luca_quant.evaluation.metrics import MetricsEngine # noqa: E402

me = MetricsEngine()
rows = []
for mult in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    cc = CostConfig(commission_buy=0.0015 * mult, commission_sell=0.0015 * mult,
                    sell_tax=0.001 * mult, slippage_bps=5.0 * mult)
    r = BacktestEngine(cc).run(session.prices.loc[session.best.oos_positions.index],
                               session.best.oos_positions)
    m = me.compute(r.returns, r.positions)
    rows.append({"Hệ số chi phí": f"{mult:.1f}×", "Sharpe": m["Sharpe"],
                 "CAGR": m["CAGR"], "Max DD": m["Max Drawdown"]})
sens = pd.DataFrame(rows)
st.dataframe(sens.round(3), use_container_width=True, hide_index=True)
st.caption("Chiến lược mà Sharpe sụp đổ khi chi phí tăng 1.5× là chiến lược "
           "sống nhờ giả định chi phí, không phải nhờ alpha.")
