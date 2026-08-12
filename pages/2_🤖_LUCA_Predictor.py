import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Định tuyến thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dnse_client import fetch_historical_data
from core.machine_learning import AIQuantPipeline

# --- CACHE DATA ---
@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(ticker: str, days: int = 1825) -> pd.DataFrame:
    """Tải dữ liệu 5 năm mặc định để đủ train Walk-Forward."""
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=days)).timestamp())
    df = fetch_historical_data(ticker, start_time, end_time)
    return df

def render_predictor_page():
    # 1. QUẢN LÝ TRẠNG THÁI (State)
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'FPT'
    current_ticker = st.session_state['ticker']

    # 2. SIDEBAR - CẤU HÌNH AI
    st.sidebar.markdown("### 🧠 L.U.C.A AI Engine")
    st.sidebar.divider()
    
    st.sidebar.markdown(f"**Mã đang phân tích:** `{current_ticker}`")
    st.sidebar.caption("*(Đổi mã tại trang 1_Market_Data)*")
    
    model_choice = st.sidebar.selectbox(
        "Thuật toán cốt lõi (Core Model):",
        options=["lightgbm", "catboost", "random_forest", "hist_gb", "logistic"],
        format_func=lambda x: x.upper() if x != "hist_gb" else "Hist Gradient Boosting",
        index=0
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("**Quản trị rủi ro (Risk Limits)**")
    max_exposure = st.sidebar.slider("Tỷ trọng vốn tối đa (Max Exposure):", min_value=0.5, max_value=1.0, value=1.0, step=0.1)
    
    run_btn = st.sidebar.button("🚀 KHỞI ĐỘNG L.U.C.A QUANT", width='stretch', type="primary")

    # 3. HEADER GIAO DIỆN CHÍNH
    st.markdown("""
        <style>
        .luca-header { font-family: 'Courier New', monospace; color: #ff00ff; }
        .luca-sub { color: #888888; font-size: 0.9em; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<h1 class="luca-header">L.U.C.A PREDICTOR: {current_ticker}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="luca-sub">Institutional-grade Walk-Forward Validation & Ablation Studies.</p>', unsafe_allow_html=True)
    st.divider()

    # 4. THỰC THI PIPELINE KHI BẤM NÚT
    if run_btn:
        with st.spinner("Đang tải dữ liệu 5 năm..."):
            raw_df = load_market_data(current_ticker, days=1825)
            
        if raw_df.empty or len(raw_df) < 500:
            st.error("Dữ liệu quá ngắn. L.U.C.A cần ít nhất 500 phiên giao dịch để học Walk-Forward.")
            return

        # Khởi tạo Pipeline
        pipeline = AIQuantPipeline(
            model_name=model_choice,
            risk_params={"max_exposure": max_exposure}
        )
        
        # CHẠY 1: ABLATION STUDY (Nghiên cứu bóc tách)
        with st.spinner("Đang chạy Ablation Study qua 5 Fold Walk-Forward... (Quá trình này mất khoảng 5-10 giây)"):
            ablation_results = pipeline.run_ablation_study(raw_df)
            
        # CHẠY 2: FULL FEATURES (Phân tích sâu kịch bản mạnh nhất)
        with st.spinner("Đang trích xuất Equity Curve & Feature Importance cho kịch bản Full Feature..."):
            full_flags = {"use_trend": True, "use_momentum": True, "use_volatility": True, "use_fractal": True}
            full_run_results = pipeline.run_walk_forward_backtest(raw_df, feature_flags=full_flags)
            metrics = full_run_results['metrics']
            feat_imp = full_run_results['feature_importance']
            equity = full_run_results['equity_curve']

        # --- HIỂN THỊ KẾT QUẢ ---
        
        st.markdown("### 1. Nghiên cứu Bóc tách (Ablation Study)")
        st.caption("Bảng dưới đây chứng minh hiệu quả của việc thêm các nhóm đặc trưng (Features) vào mô hình. Nếu Sharpe Ratio tăng lên, có nghĩa là Feature đó chứa Alpha thực sự.")
        
        # Bôi màu cột Sharpe Ratio để dễ nhìn
        st.dataframe(
            ablation_results.style.background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio', 'Win Rate']),
            width='stretch'
        )
        
        st.divider()
        st.markdown(f"### 2. Phân tích Chuyên sâu (Full Features - {model_choice.upper()})")
        
        # Hiển thị Metrics chính
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
        c2.metric("Win Rate", f"{metrics.get('Win Rate', 0)*100:.1f}%")
        c3.metric("Ann. Return (CAGR)", f"{metrics.get('Ann. Return (CAGR)', 0)*100:.2f}%")
        c4.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0)*100:.2f}%")

        # Vẽ biểu đồ Equity Curve & Feature Importance
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            st.markdown("**Đường cong Tài sản (Out-of-sample Equity Curve)**")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity.index, y=equity.values, mode='lines', line=dict(color='#00ffcc', width=2), fill='tozeroy', name="L.U.C.A Strategy"))
            fig_eq.update_layout(
                template="plotly_dark", 
                height=400, 
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_eq, width='stretch')
            
        with col_chart2:
            st.markdown("**Độ quan trọng của Đặc trưng (Feature Importance)**")
            if not feat_imp.empty:
                top_10 = feat_imp.head(10).sort_values(by='Importance', ascending=True)
                fig_feat = px.bar(top_10, x='Importance (%)', y='Feature', orientation='h')
                fig_feat.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_feat, width='stretch')
            else:
                st.info("Mô hình này không hỗ trợ trích xuất Feature Importance.")
                
    else:
        st.info("👈 Bấm 'KHỞI ĐỘNG L.U.C.A QUANT' ở thanh bên trái để bắt đầu quá trình huấn luyện và kiểm định Walk-Forward.")

if __name__ == "__main__":
    render_predictor_page()