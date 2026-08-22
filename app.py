"""
L.U.C.A Quant — entry point.

app.py CHỈ làm nhiệm vụ điều hướng. Không có một dòng logic quant nào ở đây
(Blueprint §1: "Streamlit không được chứa logic Quant").
"""
import streamlit as st

st.set_page_config(page_title="L.U.C.A Quant", page_icon="📈", layout="wide")

st.title("L.U.C.A Quant Research Platform")
st.caption("Light Upon Capital Allocation — Algorithmic Trading Research & Capital Allocation")

st.markdown(
    """
Nền tảng nghiên cứu định lượng theo quy trình bất biến:

`PROBLEM → DATA → FEATURE → LABEL → SPLIT → BASELINE → MODEL SELECTION →
VALIDATION → UNTOUCHED TEST → BACKTEST → RISK → ROBUSTNESS → ABLATION →
INVESTMENT THESIS`

**Bắt đầu ở trang `Research Lab`** — trang đó chạy toàn bộ chuỗi trên và
lưu kết quả vào session để các trang còn lại đọc lại.
"""
)

c1, c2, c3 = st.columns(3)
c1.info("**Nguyên tắc 1**\n\nMọi siêu tham số (kể cả ngưỡng xác suất) chỉ được chọn trên tập VALID.")
c2.info("**Nguyên tắc 2**\n\nRisk Manager chỉ được GIẢM vị thế. Quy tắc làm tăng vị thế là alpha, phải đánh giá riêng.")
c3.info("**Nguyên tắc 3**\n\nBáo cáo Sharpe luôn kèm số lần thử và Deflated Sharpe Ratio.")

with st.expander("Kiểm tra môi trường"):
    from luca_quant.models import registry as mr
    st.dataframe(mr.catalogue(), use_container_width=True)
    from luca_quant.features import registry as fr
    st.write("Nhóm feature khả dụng:", fr.available_groups())
