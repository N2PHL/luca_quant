# L.U.C.A Quant — Báo cáo Audit & Refactor

Rà soát toàn bộ repo `N2PHL/luca_quant` (1.122 dòng Python, 22 file) đối chiếu
với Blueprint v1.0 và yêu cầu đề bài.

Kết luận ngắn: **repo hiện tại chạy được nhưng kết quả nghiên cứu không có
hiệu lực khoa học.** Có 6 lỗi nghiêm trọng ở tầng phương pháp — mỗi lỗi đủ để
làm vô hiệu con số Sharpe báo cáo. Chúng không phải lỗi cú pháp nên không
biểu hiện thành crash; app vẫn hiện ra một bảng kết quả trông rất thuyết phục.

---

## A. Lỗi nghiêm trọng — làm vô hiệu kết quả

### A1. Ablation Study so sánh hai giai đoạn thị trường khác nhau
**Vị trí:** `core/features.py::generate_all()` — dòng cuối `return df.dropna()`

Mỗi nhóm feature có warmup khác nhau. `dropna()` chạy sau khi sinh feature nên
số hàng bị cắt phụ thuộc vào nhóm nào được bật:

| Kịch bản | Feature nặng nhất | Phiên bị cắt | Giai đoạn test |
|---|---|---|---|
| 1. Price Only | `Ret_5` | ~5 | 2020-06 → 2025-08 |
| 2. + Trend & Momentum | `EMA_200` | ~200 | 2021-03 → 2025-08 |
| 4. + Fractal (Full) | `EMA_200`, `Hurst_50` | ~200 | 2021-03 → 2025-08 |

Kịch bản 1 chạy trên một giai đoạn thị trường dài hơn 195 phiên so với các
kịch bản còn lại. Chênh lệch Sharpe giữa chúng phản ánh chủ yếu sự khác biệt
của thị trường, không phải đóng góp của feature. **Toàn bộ kết luận Ablation —
vốn là trọng tâm của đồ án — không suy ra được điều gì.**

**Đã sửa:** `features/pipeline.py::build_common_matrix()` dựng ma trận đầy đủ
một lần, lấy giao của mọi nhóm, cắt một lần duy nhất. Mọi kịch bản dùng chung
một index. Có test khoá lại: `test_common_index_across_ablation_scenarios`.

---

### A2. Không có tập Validation → toàn bộ siêu tham số được chọn trên Test
**Vị trí:** `core/validation.py` (chỉ có `TimeSeriesSplit`), `core/sizing.py`

`ProbabilitySizer` hard-code `{0.90: 0.40, 0.80: 0.30, 0.65: 0.15}`. Ba con số
này là siêu tham số. Không có tập VALID nên không có chỗ hợp lệ nào để chọn
chúng — trên thực tế chúng được chọn vì cho kết quả đẹp trên chính tập dùng để
báo cáo. Blueprint §7 gọi đúng tên: **threshold leakage**. Đây là loại rò rỉ
khó phát hiện nhất vì không dòng code nào "sai".

Cùng lỗi này áp cho: chọn model, chọn feature set, chọn `max_exposure`.

**Đã sửa:**
- `validation/splits.py::PurgedWalkForward` — ba tập TRAIN / VALID / TEST.
- `portfolio/sizing.py::ProbabilitySizer.tune()` — quét lưới ngưỡng **chỉ trên
  VALID**, trả về cả bảng lưới để đưa vào báo cáo.
- Thêm hiệu chuẩn xác suất (isotonic) trên VALID: xác suất thô của
  LightGBM/RandomForest không được hiệu chuẩn, `p = 0.90` của mô hình không có
  nghĩa là 90% khả năng tăng thật, nên dùng thẳng nó để cấp 40% vốn là sai bản chất.

---

### A3. Không có Purge/Embargo — nhãn của tập train nhìn sang tập test
`TimeSeriesSplit` cắt liền kề, không có khoảng trống. Nhãn tại phiên cuối
của TRAIN được tính từ giá tại `t + horizon`, tức là đã nằm trong TEST.

**Đã sửa:** `purge_days >= horizon` được ép ở `ExperimentRunner.run()`, cộng
thêm `embargo_days`. Test `test_purge_gap_covers_label_horizon` khoá bất biến này.

---

### A4. Risk Manager tạo alpha và ghi đè mô hình AI
**Vị trí:** `core/risk.py::filter_signals()` dòng ~95

```python
entry_mask = (Hurst_50 < 0.35) & MACD_Cross_Up_Below_0
data.loc[entry_mask, signal_col] = self.max_exposure   # "Bất chấp AI, All-in 100%"
```

Ba hệ quả:

1. **Không quy kết được Sharpe.** Nếu kết quả tốt, không ai biết là do mô hình
   học máy hay do quy tắc Hurst+MACD. Câu hỏi trung tâm của đồ án không trả lời được.
2. **Module tên "risk" nhưng hành vi là all-in.** Tên gọi sai bản chất.
3. **Lỗi âm thầm nguy hiểm nhất:** điều kiện dùng `Hurst_50`, nhưng nhóm fractal
   có thể bị TẮT ở kịch bản ablation. Khi đó `if 'Hurst_50' in data.columns` là
   False và toàn bộ khối override im lặng không chạy. **Kịch bản 1 và kịch bản 4
   chạy bằng hai logic giao dịch khác nhau**, không chỉ khác feature. Điều này
   cộng dồn với A1 làm bảng ablation hoàn toàn không đọc được.

**Đã sửa — tách làm hai lớp có bất biến rõ ràng:**
- `risk/manager.py::RiskConstraints` — chỉ được GIẢM vị thế. Bất biến
  `final <= raw` được `assert` trong code (`RiskConstraintViolation`), có test.
- `risk/overlay.py::OverlayStack` — Hurst/MACD được khai báo là **alpha**, hệ số
  nhân trong `[0, 1]`, và đánh giá thành 4 cánh tay riêng:
  `AI only | AI+MACD | AI+Hurst | AI+Hurst+MACD`. Đây chính là bảng trả lời
  "Sharpe đến từ đâu?".

---

### A5. Hai định nghĩa Sharpe khác nhau, cả hai đều bỏ qua lãi suất phi rủi ro

| File | Công thức |
|---|---|
| `core/metrics.py` | `CAGR_hình_học / vol_năm_hoá` |
| `core/alpha_engine.py` | `mean(daily) / std(daily) × √252` |

Hai công thức cho kết quả lệch 10–30% trên cùng chuỗi lợi suất (tử số một bên
là trung bình hình học, một bên là số học). Cả hai đều đặt `Rf = 0`. Với
`rf ≈ 4.5%/năm` ở Việt Nam, bỏ qua rf thổi Sharpe lên khoảng **+0.15 đến +0.30** —
đáng kể khi ngưỡng chấp nhận là 1.80.

**Đã sửa:** `evaluation/metrics.py::MetricsEngine` là nguồn duy nhất.
`Sharpe = mean(r − rf_daily) / std(r − rf_daily) × √252`. Test
`test_single_sharpe_definition` khoá công thức.

---

### A6. Feature phi dừng khiến mô hình vô nghĩa ngoài mẫu
**Vị trí:** `core/features.py::add_trend_features()`

```python
df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()   # MỨC GIÁ tuyệt đối
```

`EMA_10/20/50/200` và `MACD` thô là mức giá. Giá FPT năm 2020 khoảng 25k, năm
2025 khoảng 130k. `StandardScaler` fit trên TRAIN (giá thấp) rồi transform sang
TEST (giá cao) cho z-score nằm hẳn ngoài phân phối huấn luyện. Cây quyết định
học ngưỡng kiểu `EMA_50 < 47.300` — ngưỡng đó vô nghĩa ở giai đoạn sau.

**Đã sửa:** mọi feature mức giá chuyển thành tỷ lệ — `close/EMA − 1`,
`MACD/close`, `ATR/close`, OBV chuẩn hoá z-score cuộn.

---

## B. Lỗi tính đúng đắn

| # | Vị trí | Vấn đề | Đã sửa |
|---|---|---|---|
| B1 | `alpha_engine.py` | `np.clip(position, -1, 1)` cho phép **bán khống**. TTCK Việt Nam không có bán khống cổ phiếu cơ sở — mọi Sharpe từ nhánh short không thực hiện được | `backtest/engine.py` mặc định long-only `[0, max_exposure]`, `allow_short` là cấu hình tường minh |
| B2 | `alpha_engine.py` | Chi phí đối xứng `0.0015` cho cả mua và bán → thiếu thuế bán 0.1% (under-estimate ~40% chi phí bán) | `CostConfig` tách `buy_cost` / `sell_cost` (+ thuế + slippage) |
| B3 | toàn bộ | Không mô hình hoá **T+2**. Chiến lược có thể đảo vị thế mỗi phiên — không thực hiện được trên HOSE, và là nguồn Sharpe ảo phổ biến nhất | `BacktestEngine._apply_settlement()`, test `test_settlement_blocks_immediate_sell` |
| B4 | `machine_learning.py` | `self.model` / `self.scaler` là state dùng chung, fit lại qua từng fold và từng kịch bản. `feature_importance` trả về là của **fold cuối của kịch bản cuối** → biểu đồ trên UI mô tả sai mô hình | Mỗi fold tạo model mới; importance lấy trung bình có độ lệch chuẩn qua các fold |
| B5 | `machine_learning.py` | `run_ablation_study` bắt `Exception` rồi `print()` → kịch bản lỗi biến mất im lặng khỏi bảng, người đọc tưởng nó không được chạy | Lỗi ghi vào cột `error` của bảng kết quả |
| B6 | `features.py` | `_calculate_hurst` trả `0.5` khi không tính được → tạo giá trị giả làm loãng phân phối. `np.polyfit(log(tau))` ném `-inf` nếu `tau = 0` | Trả `NaN` (để `dropna` xử lý minh bạch), lọc `tau > 1e-12`, tính trên **log giá** đúng lý thuyết fBm |
| B7 | `features.py` | Hurst cửa sổ **20 phiên** — sai số chuẩn của ước lượng lớn hơn cả biên độ tín hiệu | Cửa sổ 50 và 100 |
| B8 | `features.py` | RSI dùng `rolling(14).mean()` thay vì làm mượt Wilder | `ewm(alpha=1/14)` đúng chuẩn |
| B9 | `dnse_client.py` | `df.ffill()` áp lên toàn bộ DataFrame không điều kiện | `clean_ohlcv()` chỉ điền `volume → 0`, không ffill giá |
| B10 | `features.py` | `Vol_Change = volume.pct_change()` sinh `inf` khi `volume = 0` → `StandardScaler` crash | Mọi block feature `replace([inf,-inf], nan)` |
| B11 | `models.py` | `get_model()` khởi tạo **tất cả** mô hình trong dict rồi mới chọn một → thiếu một thư viện là sập toàn app | Factory lazy; thiếu thư viện thì mô hình đó biến mất khỏi registry |
| B12 | repo | `catboost_info/` được commit vào git; `ui/charts.py` rỗng 0 byte | Thêm `.gitignore`, `allow_writing_files=False` |
| B13 | `requirements.txt` | Ghi `streamlit>=1.30` nhưng code dùng `width='stretch'` (cần ≥ 1.49) → crash trên Streamlit Cloud | Nâng lên `>=1.40` và dùng `use_container_width` |
| B14 | `pages/2_*.py` | Chạy `run_ablation_study` rồi chạy lại `run_walk_forward_backtest` với cùng full flags → **tính trùng** kịch bản 4 | Chỉ chạy một lần, tái dùng kết quả |

---

## C. Thiếu so với Blueprint / đề bài

| Blueprint | Trạng thái cũ | Đã bổ sung |
|---|---|---|
| §7 Leakage Detector | không có | `validation/leakage.py` — 6 kiểm định chạy được, gồm **point-in-time reconstruction** |
| §8 Label Engine đa horizon | 1 nhãn duy nhất | `direction`, `direction_vol_adj`, `triple_barrier`, `forward_return`, `forward_volatility` |
| §9 Feature Registry | boolean flags | registry khai báo, 7 nhóm |
| §10 Model Zoo + **Baseline** | 6 mô hình ML, **0 baseline** | `buy_hold`, `random`, `momentum_rule` + classical + ML + DL (torch tuỳ chọn) |
| §11 Model Benchmarking | không có | `experiments/benchmark.py`, tách ML metrics / Trading metrics |
| §12 Ablation Engine | 4 kịch bản cứng | `cumulative` / `leave_one_out` / `powerset` + đóng góp biên kiểu Shapley |
| §16 Acceptance Gate | không có | `evaluation/acceptance_gate.py` — 9 cổng |
| §17 Investment Thesis | không có | `explainability/investment_thesis.py` |
| Đề bài: tránh data leakage | không kiểm | test suite 17 test |
| Đề bài: baseline CAPM/APT/hồi quy/luật | không có | `logistic`, `ridge`, `momentum_rule` có sẵn; CAPM/ARIMA/GARCH cần `statsmodels` (đã chừa chỗ trong registry) |

---

## D. Vấn đề thống kê chưa được nêu trong Blueprint

Blueprint §12 đề xuất chạy `2^6 − 1 = 63` tổ hợp, §16 đặt cổng `Sharpe >= 1.80`.
Ghép hai điều đó tạo ra một lỗi thống kê nghiêm trọng: **nếu thử 63 chiến lược
và báo cáo cái tốt nhất, thì ngay cả khi không tồn tại alpha nào, kỳ vọng Sharpe
cực đại vẫn khoảng 0.8–1.2 thuần do may rủi.** Đạt 1.8 sau 63 lần thử không
chứng minh được gì nếu không hiệu chỉnh.

Đã bổ sung `evaluation/statistical_tests.py`:
- **Deflated Sharpe Ratio** (Bailey & López de Prado) — hiệu chỉnh cho số lần thử,
  skew, kurtosis, độ dài mẫu.
- **Stationary block bootstrap** cho khoảng tin cậy Sharpe (khối 20 phiên, giữ
  tự tương quan và cụm biến động).
- **Kiểm định cặp Newey-West** so với Buy & Hold.
- **Benjamini-Hochberg** cho bảng ablation nhiều kịch bản.

Khi người chấm hỏi *"làm sao biết Sharpe 1.8 không phải do dò tìm dữ liệu?"* —
đây là câu trả lời có cơ sở toán học.

---

## E. Bằng chứng thực nghiệm rằng bản mới đã sạch

Chạy trên hai bộ dữ liệu control (`SyntheticProvider`), 3 fold:

**NEGATIVE CONTROL — random walk, theo cấu tạo KHÔNG có alpha:**

```
Model            Accuracy   AUC    CAGR   Sharpe
buy_hold            0.522  0.500   0.055   0.140
random              0.522  0.522   0.054   0.137
logistic            0.496  0.492   0.097   0.340
hist_gb             0.520  0.511   0.020  -0.059

GATE: REJECT   (Sharpe 0.34 < 1.8; DSR 0.80 < 0.95; CI 95% = [-0.20, 1.24])
```

AUC ≈ 0.50 và gate REJECT. **Hệ thống không tìm thấy tín hiệu ở nơi không có
tín hiệu** — đây là bằng chứng mạnh nhất rằng không còn rò rỉ.

**POSITIVE CONTROL — có momentum cấy sẵn:**

```
momentum_rule       0.529  0.526   0.202   0.919
AI + MACD                          0.224   1.122     ← DSR 0.96 PASS, CI [0.14, 2.13]
```

Hệ thống phát hiện đúng tín hiệu đã cấy. Độ nhạy còn nguyên.

Nếu chạy bản repo cũ trên cùng negative control mà vẫn ra Sharpe cao, đó là
xác nhận trực tiếp cho các lỗi A1–A6.

---

## F. Việc còn lại (không chặn, xếp theo giá trị/công sức)

1. **Đa mã cổ phiếu.** Hiện nghiên cứu trên một mã. Sharpe 1.8 trên một mã có
   thể là ngẫu nhiên; chạy trên rổ VN30 và báo cáo phân phối Sharpe thuyết phục
   hơn nhiều so với một con số đơn lẻ. Đây là bước tiếp theo có giá trị nhất.
2. **Econometric baselines** (CAPM, APT, ARIMA, GARCH) — cần `statsmodels`/`arch`,
   đề bài có yêu cầu, registry đã chừa chỗ.
3. **SHAP** cho explainability (Blueprint Phase 9).
4. **Untouched holdout cuối cùng** — `validation/splits.py::holdout_split()` đã
   có, cần kỷ luật chỉ chạy **đúng một lần** sau khi chốt mọi thứ.
5. **Portfolio allocation đa tài sản** (Blueprint Phase 8) — hiện là single-asset sizing.
