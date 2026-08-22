"""Market Data — trực quan hoá giá và ma trận feature. Không có logic quant."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luca_quant.data.providers.dnse import DNSEProvider              # noqa: E402
from luca_quant.data.providers.synthetic import SyntheticProvider    # noqa: E402
from luca_quant.data.schemas import validate_ohlcv                   # noqa: E402
from luca_quant.features import registry as freg                     # noqa: E402
from luca_quant.features.pipeline import FeaturePipeline             # noqa: E402

st.title("📊 Market Data")

sb = st.sidebar
ticker = sb.text_input("Mã chứng khoán", st.session_state.get("ticker", "FPT")).upper()
years = sb.slider("Số năm", 1, 15, 5)
use_synth = sb.checkbox("Dùng dữ liệu synthetic (offline)", value=False)
groups = sb.multiselect("Nhóm feature", freg.available_groups(),
                        default=["price", "trend", "momentum", "volatility", "fractal"])
st.session_state["ticker"] = ticker


@st.cache_data(ttl=600, show_spinner=False)
def load(tk: str, yrs: int, synth: bool) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=int(365.25 * yrs))
    prov = SyntheticProvider(seed=11) if synth else DNSEProvider()
    return prov.get_ohlcv(tk, start, end)


with st.spinner("Đang tải dữ liệu…"):
    px = load(ticker, years, use_synth)

if px.empty:
    st.error(f"Không lấy được dữ liệu cho {ticker}.")
    st.stop()

q = validate_ohlcv(px)
if not q["clean"]:
    with st.expander("⚠️ Vấn đề chất lượng dữ liệu", expanded=True):
        for i in q["issues"]:
            st.warning(i)

pipe = FeaturePipeline(groups)
feats = pipe.build(px)
merged = px.join(feats).dropna()

if merged.empty:
    st.error(f"Chưa đủ dữ liệu: nhóm feature đã chọn cần warmup {pipe.warmup_} phiên.")
    st.stop()

last = merged.iloc[-1]
c = st.columns(4)
c[0].metric("Đóng cửa", f"{last['close']:,.0f}", f"{last.get('price__ret_1d', 0):.2%}")
c[1].metric("Khối lượng", f"{last['volume']:,.0f}")
if "momentum__rsi_14" in merged:
    c[2].metric("RSI 14", f"{last['momentum__rsi_14']*100:.1f}")
if "fractal__hurst_50" in merged:
    h = last["fractal__hurst_50"]
    c[3].metric("Hurst 50", f"{h:.3f}", "Dai dẳng" if h > 0.5 else "Quay về trung bình")

rows = 2 + int("momentum__macd_norm" in merged) + int("fractal__hurst_50" in merged)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=merged.index, open=merged["open"], high=merged["high"],
                             low=merged["low"], close=merged["close"], name="Giá"), row=1, col=1)
fig.add_trace(go.Bar(x=merged.index, y=merged["volume"], name="KLGD",
                     marker_color="#7f8c8d"), row=2, col=1)
r = 3
if "momentum__macd_norm" in merged:
    fig.add_trace(go.Scatter(x=merged.index, y=merged["momentum__macd_norm"], name="MACD/close"), row=r, col=1)
    r += 1
if "fractal__hurst_50" in merged:
    fig.add_trace(go.Scatter(x=merged.index, y=merged["fractal__hurst_50"], name="Hurst 50"), row=r, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=r, col=1)
fig.update_layout(height=250 * rows, xaxis_rangeslider_visible=False, showlegend=False,
                  margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Ma trận đặc trưng")
st.caption(f"Warmup {pipe.warmup_} phiên; {merged.shape[0]:,} phiên dùng được, "
           f"{feats.shape[1]} feature. Toàn bộ đều là causal và stationary.")
st.dataframe(merged[feats.columns].tail(30).round(4), use_container_width=True)
