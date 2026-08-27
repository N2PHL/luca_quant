"""
Trang chuyển hướng — giữ cho link cũ /LUCA_Predictor không chết.

Bản v0.1 (cũ) có trang `2_🤖_LUCA_Predictor.py` gộp toàn bộ logic quant vào UI.
Bản v0.2 tách nó thành 6 trang, và mọi logic chuyển xuống `luca_quant/services/`.
File này KHÔNG chứa logic, chỉ điều hướng.

Có thể xoá an toàn khi không còn ai dùng link cũ.
"""
import streamlit as st

st.title("↩️ Trang này đã chuyển")

st.warning(
    "**`LUCA_Predictor` không còn tồn tại.** Trang cũ gộp feature engineering, "
    "huấn luyện mô hình, backtest và risk vào một chỗ — đó chính là kiến trúc "
    "đã bị thay thế ở v0.2."
)

st.markdown(
    """
Chức năng cũ giờ nằm ở các trang sau:

| Bạn muốn làm gì | Trang mới |
|---|---|
| Xem giá và ma trận đặc trưng | **📊 Market Data** |
| Chạy toàn bộ chuỗi nghiên cứu | **🔬 Research Lab** ← bắt đầu ở đây |
| So sánh mô hình với baseline | **🏆 Model Benchmark** |
| Xem đóng góp của từng nhóm feature | **🧬 Ablation Study** |
| Equity curve, drawdown, chi phí | **📈 Backtest and Risk** |
| Kết luận và biện luận đầu tư | **📜 Investment Thesis** |

---

**Quy trình xác minh trước khi tin bất kỳ con số nào:** vào **🔬 Research Lab**,
chọn nguồn dữ liệu `Synthetic — random walk`, bấm Chạy. Hệ thống phải cho
AUC ≈ 0.50 và Acceptance Gate phải trả về **REJECT**.

Random walk theo cấu tạo không chứa tín hiệu nào. Nếu app vẫn báo Sharpe cao
trên đó thì có rò rỉ dữ liệu ở đâu đó — và mọi kết quả khác đều không đáng tin.
Chỉ khi bước này đạt mới chuyển sang nguồn `DNSE (thật)`.
"""
)

st.page_link("pages/2_🔬_Research_Lab.py", label="Đi tới Research Lab", icon="🔬")
