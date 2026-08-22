# L.U.C.A Quant v0.2

**Light Upon Capital Allocation** — Algorithmic Trading Research & Capital Allocation Platform.

Không phải app dự đoán giá cổ phiếu. Đây là một framework nghiên cứu định lượng
với chuỗi quy trình bất biến:

```
PROBLEM → DATA → FEATURE → LABEL → SPLIT → BASELINE → MODEL SELECTION
→ VALIDATION → UNTOUCHED TEST → BACKTEST → RISK → ROBUSTNESS → ABLATION
→ INVESTMENT THESIS
```

---

## Chạy thử

```bash
pip install -r requirements.txt
pytest                      # 17 test — chạy trước khi tin bất kỳ con số nào
streamlit run app.py
```

Mở trang **🔬 Research Lab**, chọn nguồn dữ liệu `Synthetic — random walk`,
bấm Chạy. Hệ thống phải cho AUC ≈ 0.50 và gate REJECT. Nếu nó báo Sharpe cao
trên dữ liệu không có tín hiệu thì có rò rỉ ở đâu đó — đừng tin kết quả nào khác.
Sau đó đổi sang `DNSE (thật)`.

Dùng trong notebook, không cần Streamlit:

```python
from luca_quant.services.research_service import ResearchService

session = ResearchService().run_full_study(
    ticker="FPT",
    groups=["price", "trend", "momentum", "volatility", "fractal"],
    primary_model="lightgbm",
    ablation_mode="cumulative",
)
print(session.thesis)
print(session.gate.to_frame())
```

---

## Ba nguyên tắc được ép trong code

**1. Siêu tham số chỉ được chọn trên VALID.**
Bao gồm cả ngưỡng xác suất position sizing — `0.65 / 0.80 / 0.90` không phải
hằng số, chúng là ba siêu tham số và được tune trong `ProbabilitySizer.tune()`
với dữ liệu VALID. `LeakageDetector.check_threshold_source()` kiểm tra điều này.

**2. Risk Manager chỉ được GIẢM vị thế.**
`RiskConstraints.apply()` ném `RiskConstraintViolation` nếu vị thế cuối lớn hơn
vị thế thô. Quy tắc nào làm TĂNG vị thế đều là alpha, phải khai báo trong
`risk/overlay.py` và đánh giá thành cánh tay riêng.

**3. Sharpe luôn đi kèm số lần thử.**
Thử 63 tổ hợp rồi báo cáo cái tốt nhất thì kỳ vọng Sharpe cực đại ~0.8–1.2
ngay cả khi không có alpha nào. Mọi bảng kết quả đều có cột **DSR**
(Deflated Sharpe Ratio) đã hiệu chỉnh cho điều đó.

---

## Kiến trúc

```
app.py                    entry point — KHÔNG chứa logic quant
pages/                    UI — chỉ gọi services/
luca_quant/
├── config/settings.py    mọi ngưỡng, không hard-code trong logic
├── data/
│   ├── providers/        dnse | yfinance | synthetic (control experiment)
│   └── schemas.py        contract OHLCV + kiểm định chất lượng
├── features/
│   ├── registry.py       7 nhóm, mọi feature causal + stationary
│   └── pipeline.py       ma trận chung → ablation so sánh được
├── labels/registry.py    direction | vol_adj | triple_barrier | regression
├── models/
│   ├── registry.py       baseline | classical | econometric | ml | deep
│   ├── econometric.py    CAPM | APT | ARIMA | GARCH
│   └── deep.py           LSTM | GRU | CNN | Transformer (cần torch)
├── validation/
│   ├── splits.py         Purged Walk-Forward, TRAIN/VALID/TEST
│   └── leakage.py        6 kiểm định chạy được
├── backtest/engine.py    chi phí VN, long-only, T+2
├── portfolio/sizing.py   ngưỡng tune trên VALID + hiệu chuẩn xác suất
├── risk/
│   ├── manager.py        RÀNG BUỘC — chỉ giảm vị thế
│   └── overlay.py        ALPHA — Hurst/MACD, đánh giá riêng
├── evaluation/
│   ├── metrics.py        MỘT định nghĩa Sharpe duy nhất
│   ├── statistical_tests.py  DSR, bootstrap, Newey-West, BH
│   └── acceptance_gate.py    9 cổng ACCEPT/REJECT
├── experiments/          runner | ablation | benchmark
├── explainability/       investment_thesis
└── services/             ResearchService — API duy nhất cho UI
```

---

## Giao thức nghiên cứu (kỷ luật, không phải code)

1. **Chốt câu hỏi trước khi chạy.** Viết ra giả thuyết và ngưỡng chấp nhận
   *trước*, không phải sau khi nhìn kết quả.
2. **Chạy negative control đầu tiên.** Random walk phải cho AUC ≈ 0.50.
3. **Đếm số lần thử.** Mọi cấu hình đã chạy đều tính, kể cả những cái bị loại bỏ.
   Con số này vào `n_trials` của Deflated Sharpe Ratio.
4. **Holdout cuối cùng chạy ĐÚNG MỘT LẦN.** `validation/splits.py::holdout_split()`.
   Chạy lần thứ hai là quay lại đúng lỗi cũ.
5. **Báo cáo cả kết quả xấu.** REJECT là kết quả nghiên cứu hợp lệ.

---

## Giới hạn đã biết

- Nghiên cứu trên **một mã**. Sharpe 1.8 trên một mã có thể là ngẫu nhiên —
  bước tiếp theo có giá trị nhất là chạy trên rổ VN30 và báo cáo **phân phối**
  Sharpe thay vì một con số.
- Chưa mô hình hoá tác động giá khi quy mô vốn lớn.
- Ràng buộc T+2 là xấp xỉ bảo thủ (chặn giảm vị thế trong 2 phiên sau khi mua),
  không mô phỏng từng lô cổ phiếu riêng.
- CAPM/APT vốn là mô hình định giá, không phải mô hình dự báo — dùng làm baseline
  dự báo hướng là một sự vay mượn, phải nêu rõ trong báo cáo.

Chi tiết audit repo cũ và toàn bộ lỗi đã sửa: [`AUDIT.md`](AUDIT.md).
Hướng dẫn chuyển đổi: [`MIGRATION.md`](MIGRATION.md).
