# Hướng dẫn chuyển đổi v0.1 → v0.2

## Nguyên tắc: Phase 0 — đóng băng bản cũ trước

Đúng như Blueprint §20. Làm việc này **trước tiên**, trên máy bạn:

```bash
git tag v0.1-current
git push origin v0.1-current
git checkout -b v0.2-refactor
```

Bản cũ vẫn chạy được ở tag `v0.1-current`. Nếu người chấm muốn xem
"trước và sau", bạn có cả hai.

---

## Cách đưa code mới vào repo

Giải nén `luca_quant_v2.zip` vào thư mục repo. Các file/thư mục sau **thay thế**
bản cũ:

| Xoá | Thay bằng |
|---|---|
| `core/` (toàn bộ) | `luca_quant/` |
| `data/dnse_client.py` | `luca_quant/data/providers/dnse.py` |
| `data/yfinance_client.py` | `luca_quant/data/providers/yfinance_provider.py` |
| `pages/1_📊_Market_Data.py` | `pages/1_📊_Market_Data.py` (bản mới) |
| `pages/2_🤖_LUCA_Predictor.py` | `pages/2_🔬_Research_Lab.py` + 4 trang khác |
| `ui/charts.py` (rỗng 0 byte) | — xoá |
| `catboost_info/` | — xoá, đã thêm vào `.gitignore` |
| `app.py` | `app.py` (bản mới) |
| `requirements.txt` | `requirements.txt` (bản mới) |

Sau khi giải nén:

```bash
pytest                      # phải 17 passed
streamlit run app.py
```

---

## Bảng ánh xạ module (Blueprint §3)

| Cũ | Mới | Ghi chú |
|---|---|---|
| `core/features.py` | `features/registry.py` + `features/pipeline.py` | boolean flags → registry; feature chuyển sang dạng tỷ lệ |
| `core/labels.py` | `labels/registry.py` | 1 nhãn → 5 nhãn, đa horizon |
| `core/models.py` | `models/registry.py` + `models/econometric.py` + `models/deep.py` | thêm baseline, CAPM/APT/ARIMA/GARCH, LSTM/GRU/CNN/Transformer |
| `core/validation.py` | `validation/splits.py` + `validation/leakage.py` | `TimeSeriesSplit` → Purged Walk-Forward 3 tập |
| `core/metrics.py` + phần metrics trong `alpha_engine.py` | `evaluation/metrics.py` | hai công thức Sharpe → một |
| `core/alpha_engine.py` | `backtest/engine.py` | thêm chi phí VN bất đối xứng, long-only, T+2 |
| `core/risk.py` | `risk/manager.py` **+** `risk/overlay.py` | tách ràng buộc khỏi alpha |
| `core/sizing.py` | `portfolio/sizing.py` | ngưỡng hard-code → tune trên VALID + hiệu chuẩn |
| `core/explain.py` | `explainability/investment_thesis.py` | importance → thesis đầy đủ |
| `core/machine_learning.py` (God Object) | `experiments/runner.py` + `services/research_service.py` | tách theo trách nhiệm |

---

## Thứ tự triển khai đề xuất

Bạn không cần làm hết một lượt. Thứ tự này giữ cho app luôn chạy được:

**Bước 1 — nền tảng (làm ngay).**
Copy `luca_quant/` vào repo, chạy `pytest`. Chưa cần đụng UI.
Kiểm chứng bằng notebook: chạy `ResearchService` trên `SyntheticProvider`
random walk, xác nhận AUC ≈ 0.50.

**Bước 2 — UI.**
Thay `app.py` và `pages/`. Chạy `streamlit run app.py`.

**Bước 3 — chạy trên mã thật.**
Đổi nguồn sang DNSE, chạy 5–8 năm dữ liệu, xem Acceptance Gate nói gì.
Ghi lại `n_trials` — bạn sẽ cần con số này trong báo cáo.

**Bước 4 — mở rộng cho báo cáo.**
- `pip install statsmodels arch` → mở khoá ARIMA + GARCH baseline
- `pip install torch` → mở khoá LSTM/GRU/CNN/Transformer (Blueprint Phase 6)
- Chạy trên rổ VN30 thay vì một mã — đây là việc làm tăng độ thuyết phục
  của đồ án nhiều nhất

---

## Nếu Sharpe tụt xuống sau khi refactor

**Đây là kết quả dự kiến, không phải hồi quy chất lượng.** Bản cũ báo Sharpe cao
hơn vì:

1. Risk Manager all-in bằng quy tắc Hurst+MACD — Sharpe đó là của quy tắc kỹ
   thuật, không phải của mô hình AI. Bản mới tách ra, xem bảng "Sharpe đến từ đâu?"
2. Ngưỡng sizing chọn bằng cách nhìn kết quả cuối → thổi phồng.
3. Thiếu thuế bán 0.1% và không có T+2 → chi phí thực tế cao hơn.
4. Không có `Rf` → Sharpe cao hơn khoảng 0.15–0.30.
5. Cho phép bán khống — nhánh không thực hiện được trên TTCK Việt Nam.

Con số thấp hơn nhưng **tái lập được** có giá trị hơn nhiều so với con số cao
không bảo vệ được trước câu hỏi "làm sao chứng minh không phải data leakage?".

Nếu muốn chứng minh điều này bằng số liệu cho phần báo cáo: chạy repo cũ
(`v0.1-current`) trên dữ liệu random walk. Nếu nó vẫn ra Sharpe cao trên chuỗi
không có tín hiệu, đó là bằng chứng trực tiếp cho các lỗi A1–A6 trong `AUDIT.md`.
