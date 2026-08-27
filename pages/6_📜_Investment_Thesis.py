"""Investment Thesis — output cuối cùng, phần 'biện luận mô hình' của đồ án."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.title("📜 Investment Thesis")

session = st.session_state.get("session")
if session is None or not session.thesis:
    st.info("Chạy trang **Research Lab** trước.")
    st.stop()

if session.gate is not None:
    d = session.gate.decision
    (st.success if d == "ACCEPT" else st.warning if "CONDITIONS" in d else st.error)(
        f"### {d}"
    )
    if d == "REJECT":
        st.caption("REJECT là một kết quả nghiên cứu hợp lệ. Hệ thống vừa làm đúng "
                   "việc của nó: ngăn một chiến lược chưa được chứng minh đi vào "
                   "phân bổ vốn thật.")

st.code(session.thesis, language=None)
st.download_button("Tải về (.txt)", session.thesis,
                   file_name=f"luca_thesis_{session.ticker}.txt",
                   use_container_width=True)

with st.expander("Bảng dữ liệu thô để đưa vào báo cáo"):
    if session.benchmark:
        st.write("**Benchmark**")
        st.dataframe(session.benchmark["table"].round(4), use_container_width=True)
    if session.ablation:
        st.write("**Ablation**")
        st.dataframe(session.ablation["table"].round(4), use_container_width=True)
        st.write("**Đóng góp biên**")
        st.dataframe(session.ablation["contribution"].round(4), use_container_width=True)
    st.write("**Overlay arms**")
    st.dataframe(session.overlay_arms.round(4), use_container_width=True)
