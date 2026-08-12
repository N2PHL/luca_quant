import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Cấu hình đường dẫn tuyệt đối để nhận diện thư mục luca_quant
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dnse_client import fetch_historical_data
from core.features import FeatureEngineer

# --- CACHE DATA ---
@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(ticker: str, days: int) -> pd.DataFrame:
    """Tải dữ liệu thô từ API."""
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=days)).timestamp())
    df = fetch_historical_data(ticker, start_time, end_time)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def prepare_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gọi lõi Feature Engineering để sinh đặc trưng (Bao gồm cả Fractal)."""
    if df.empty: return df
    return FeatureEngineer.generate_all(df, use_fractal=True)

def render_market_data_page():
    # 1. QUẢN LÝ TRẠNG THÁI TỔNG TÀI SẢN (UNIVERSAL STATE)
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'FPT'

    # 2. SIDEBAR - ĐIỀU KHIỂN
    st.sidebar.markdown("### ⚙️ L.U.C.A Terminal")
    st.sidebar.divider()
    
    input_ticker = st.sidebar.text_input(
        "Mã chứng khoán (VN30):", 
        value=st.session_state['ticker'],
        key="market_data_ticker"
    ).upper()
    
    timeframe = st.sidebar.selectbox(
        "Khung thời gian phân tích:",
        options=[
            ("3 Tháng", 90), 
            ("6 Tháng", 180), 
            ("1 Năm", 365), 
            ("2 Năm", 730), 
            ("5 Năm", 1825)
        ],
        index=4,  
        format_func=lambda x: x[0]
    )
    
    if input_ticker != st.session_state['ticker']:
        st.session_state['ticker'] = input_ticker
        st.rerun()

    current_ticker = st.session_state['ticker']

    # 3. HEADER
    st.markdown("""
        <style>
        .luca-header { font-family: 'Courier New', monospace; color: #00ffcc; }
        .luca-sub { color: #888888; font-size: 0.9em; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<h1 class="luca-header">VŨ TRỤ DỮ LIỆU: {current_ticker}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="luca-sub">Trực quan hóa chuỗi thời gian & Sinh đặc trưng học máy (Feature Generation).</p>', unsafe_allow_html=True)
    
    # 4. KÉO VÀ XỬ LÝ DỮ LIỆU
    with st.spinner(f"L.U.C.A đang đồng bộ khối dữ liệu {current_ticker}..."):
        raw_df = load_market_data(current_ticker, days=timeframe[1])
        
    if raw_df.empty:
        st.error(f"Không tìm thấy tín hiệu dữ liệu cho mã {current_ticker}.")
        return
        
    # Chạy qua Core Engine để sinh biến
    featured_df = prepare_quant_features(raw_df)
    
    # 5. METRICS NHANH
    st.markdown("### 1. Phân tích Hiện trạng (Snapshot)")
    latest = featured_df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Giá Đóng Cửa", f"{latest['close']:,.2f}", f"{latest.get('Ret_1', 0)*100:.2f}%")
    with col2:
        st.metric("Khối lượng (Vol)", f"{latest['volume']:,.0f}", f"{latest.get('Vol_Change', 0)*100:.1f}%")
    with col3:
        st.metric("Momentum (RSI 14)", f"{latest.get('RSI_14', 0):.2f}")
    with col4:
        # Thay thế bằng Hurst Exponent nếu có
        hurst_val = latest.get('Hurst_20', 0)
        st.metric("Fractal (Hurst 20)", f"{hurst_val:.3f}", "Trending" if hurst_val > 0.5 else "Mean-reverting")

    # 6. BIỂU ĐỒ QUANT CHUYÊN SÂU (MULTI-CHART)
    st.markdown("### 2. Trực quan hóa Đa phân dạng (Multifractal Terminal)")
    
    # Tạo Subplots: 4 hàng (Giá, Khối lượng, MACD, Hurst)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.20, 0.20]
    )

    # Nến giá & EMA
    fig.add_trace(go.Candlestick(
        x=featured_df.index, open=featured_df['open'], high=featured_df['high'],
        low=featured_df['low'], close=featured_df['close'], name="Price"
    ), row=1, col=1)
    
    if 'EMA_20' in featured_df.columns:
        fig.add_trace(go.Scattergl(x=featured_df.index, y=featured_df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scattergl(x=featured_df.index, y=featured_df['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)

    # Khối lượng
    colors = ['#00ffcc' if row['open'] - row['close'] >= 0 else '#ff3333' for index, row in featured_df.iterrows()]
    fig.add_trace(go.Bar(x=featured_df.index, y=featured_df['volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # MACD 
    if 'MACD' in featured_df.columns:
        fig.add_trace(go.Scattergl(x=featured_df.index, y=featured_df['MACD'], line=dict(color='cyan', width=1.5), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scattergl(x=featured_df.index, y=featured_df['MACD_Signal'], line=dict(color='magenta', width=1), name='Signal'), row=3, col=1)
        fig.add_trace(go.Bar(x=featured_df.index, y=featured_df['MACD_Hist'], marker_color='gray', name='Histogram'), row=3, col=1)

    # Hurst Exponent (Risk Barrier)
    if 'Hurst_20' in featured_df.columns:
        fig.add_trace(go.Scattergl(x=featured_df.index, y=featured_df['Hurst_20'], line=dict(color='#bf00ff', width=2), name='Hurst (20)'), row=4, col=1)
        
        # Thêm các đường ranh giới rủi ro
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random Walk (0.5)", row=4, col=1)
        fig.add_hline(y=0.45, line_dash="dot", line_color="red", annotation_text="Risk: Noise (<0.45)", row=4, col=1)
        fig.add_hline(y=0.6, line_dash="dot", line_color="green", annotation_text="Alpha: Trend (>0.6)", row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=850, margin=dict(l=0, r=0, t=20, b=0),
        xaxis_rangeslider_visible=False, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, width='stretch')

    # 7. KIỂM ĐỊNH MATRIX ĐẶC TRƯNG (FEATURE MATRIX)
    st.markdown("### 3. Ma trận Đặc trưng (Machine Learning Input)")
    with st.expander("Bấm để xem dữ liệu đã được Vector hóa chuẩn bị đưa vào AI"):
        st.dataframe(
            featured_df.tail(20).style.background_gradient(cmap='viridis', subset=['Ret_1', 'RSI_14', 'Hurst_20']),
            width='stretch'
        )

if __name__ == "__main__":
    render_market_data_page()
    