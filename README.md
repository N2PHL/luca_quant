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

## Hai phiên bản

| | **Bản nộp trực tuyến** | **Bản đầy đủ** |
|---|---|---|
| Profile | `coursework` | `full` (mặc định) |
| Chạy ở đâu | Streamlit Community Cloud | Máy local |
| Mô hình | MLP, CNN, RNN, LSTM, GRU, Transformer + 3 baseline | 34 mô hình / 6 họ |
| requirements | `requirements.txt` | `requirements-full.txt` |

Chỉ có **một codebase**. Profile lọc danh sách mô hình và siết trần tài nguyên;
không có nhánh logic nào khác nhau giữa hai bản — hai repo tách rời sẽ phân kỳ,
và sửa lỗi ở bản này rồi quên bản kia là kịch bản tệ nhất cho một đồ án.

```bash
# bản đầy đủ (mặc định)
pip install -r requirements-full.txt
pytest                                    # 87 test
streamlit run app.py

# thử bản nộp ngay trên máy local
LUCA_PROFILE=coursework streamlit run app.py
```

Chọn profile theo thứ tự ưu tiên: biến môi trường `LUCA_PROFILE` → 
`st.secrets["LUCA_PROFILE"]` → file `.luca_profile` ở gốc repo → mặc định `full`.

Hướng dẫn deploy đầy đủ, kể cả cách cài torch bản CPU và xử lý khi hết bộ nhớ:
[`DEPLOY.md`](DEPLOY.md).

---

## Chạy thử

```bash
pip install -r requirements-full.txt
pytest                      # 87 test — chạy TRƯỚC khi tin bất kỳ con số nào
streamlit run app.py
```

Cả hai file requirements đều cài `torch` bản CPU (~200MB) qua
`--extra-index-url https://download.pytorch.org/whl/cpu`. Dòng đó **phải đứng
trước** `torch`, nếu không pip lấy wheel PyPI mặc định kèm toàn bộ runtime
CUDA (~2GB).

Thiếu `torch` thì các kiến trúc chuỗi chỉ đơn giản biến mất khỏi dropdown, app
vẫn chạy bình thường — registry dùng `find_spec` nên không import thư viện chỉ
để kiểm tra sự tồn tại của nó.

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
│   ├── registry.py       34 mô hình / 6 họ, factory lazy + stacking theo thời gian
│   ├── econometric.py    CAPM | APT | ARIMA | GARCH
│   └── deep.py           RNN | LSTM | GRU | BiLSTM | BiGRU | CNN | TCN
│                         | CNN-LSTM | Attention-LSTM | Transformer (cần torch)
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

## Model Zoo — 34 mô hình, 6 họ

Registry dùng **factory lazy**: thiếu thư viện thì đúng mô hình đó biến mất
khỏi danh sách, không làm sập app. Xem danh sách thực tế đang khả dụng:

```python
from luca_quant.models import registry as mr
print(mr.catalogue())
```

| Họ | Mô hình | Vai trò trong nghiên cứu |
|---|---|---|
| `baseline` | `buy_hold`, `random`, `momentum_rule` | Mốc so sánh **bắt buộc**. Không vượt được nhóm này = không có alpha. |
| `classical` | `logistic`, `ridge`, `elasticnet`, `sgd_logistic`, `lda`, `qda`, `gaussian_nb`, `knn`, `svm_linear`, `svm_rbf` | Mô hình ít tham số, ổn định khi nhiễu lớn. Thường mạnh hơn kỳ vọng. |
| `econometric` | `capm`, `apt`, `arima`, `garch_vol` | Lý thuyết tài chính cổ điển. `garch_vol` là baseline vol-timing rất khó vượt. |
| `ml` | `decision_tree`, `random_forest`, `extra_trees`, `hist_gb`, `gbdt`, `adaboost`, `lightgbm`, `catboost`, `xgboost` | Bắt tương tác phi tuyến giữa feature. |
| `deep_learning` | `mlp`, `rnn`, `lstm`, `gru`, `bilstm`, `bigru`, `cnn1d`, `tcn`, `cnn_lstm`, `attn_lstm`, `transformer` | Mô hình hoá phụ thuộc theo thời gian. Cần `torch`. Sáu mô hình in đậm — `mlp`, `cnn1d`, `rnn`, `lstm`, `gru`, `transformer` — là toàn bộ nội dung của profile `coursework`. |
| `ensemble` | `voting_ensemble`, `stacking_ensemble` | Trung bình hoá sai số ít tương quan giữa các họ. |

**Thang bậc phức tạp.** Đọc bảng benchmark từ trên xuống dưới phải trả lời
được đúng một câu hỏi: *thêm phức tạp có đổi lấy hiệu năng không?* Nếu `lstm`
không vượt `momentum_rule`, thì toàn bộ phần deep learning là một **kết quả
âm** — và đó vẫn là kết quả nghiên cứu hợp lệ, phải báo cáo chứ không giấu đi.

> ⚠️ Mỗi mô hình đưa vào lineup là **một lần thử**, và con số đó đi thẳng
> vào `n_trials` của Deflated Sharpe Ratio. Chạy cả 34 mô hình rồi báo cáo cái
> tốt nhất chính là data snooping — DSR sẽ trừng phạt đúng hành vi đó.

---

## Kiến thức nền: họ mô hình chuỗi

Phần này giải thích *vì sao* mỗi kiến trúc tồn tại, không chỉ *cách gọi* nó.

### Vì sao dữ liệu tài chính là bài toán khó nhất cho deep learning

Ba tính chất khiến kinh nghiệm từ thị giác máy tính hay NLP **không chuyển
giao được** sang đây:

1. **Tỉ lệ tín hiệu/nhiễu cực thấp.** Ảnh mèo là mèo với gần 100% chắc chắn.
   Một mô hình dự báo hướng giá đạt AUC 0.55 đã là rất tốt. Phần lớn biến
   thiên giá là nhiễu không thể dự báo.
2. **Không dừng (non-stationary).** Quan hệ giữa feature và nhãn thay đổi
   theo chế độ thị trường. Mô hình học từ 2018–2020 có thể vô dụng năm 2024.
   Đây là lý do phải dùng **walk-forward** thay vì một lần chia train/test.
3. **Rất ít mẫu.** 10 năm dữ liệu ngày ≈ 2.500 phiên. GPT huấn luyện trên
   hàng nghìn tỉ token. Một Transformer 2 triệu tham số trên 2.500 mẫu sẽ
   thuộc lòng dữ liệu chứ không học được quy luật.

**Hệ quả thực hành:** ở đây mô hình nhỏ + regularization mạnh + early stopping
gần như luôn thắng mô hình lớn. Cấu hình mặc định trong `ARCH_DEFAULTS` cố ý
để `hidden` ở mức 32–48, không phải 256.

### Cửa sổ trượt — chỗ dễ sai nhất

Mô hình chuỗi cần input dạng `(n_samples, lookback, n_features)`. Quy ước
trong repo này:

```
window[t] = X[t-lookback+1 : t+1]     ->  dự báo y[t]
                              ^^^^
                    BAO GỒM phiên t
```

`X[t]` là feature causal đã biết tại lúc đóng cửa phiên `t`; `y[t]` là lợi
suất từ `t` đến `t+h`. Vì vậy dùng `X[t]` là hợp lệ — và **bắt buộc** phải
dùng, nếu không mô hình chuỗi bị trễ một phiên so với LightGBM và bảng
benchmark so sánh hai thứ không cùng điều kiện.

Hai cạm bẫy:

- **Cửa sổ vắt qua ranh giới split.** Nếu dựng cửa sổ một lần cho toàn chuỗi
  rồi mới chia TRAIN/TEST, mẫu đầu tiên của TEST sẽ chứa `lookback-1` phiên
  cuối của TRAIN — mô hình đã thấy chúng lúc học. Đây là nguồn leakage phổ
  biến nhất trong đồ án DL tài chính và rất khó phát hiện vì không có cột nào
  "nhìn tương lai", chỉ là ranh giới bị nhoè. Ở đây `fit`/`predict` dựng cửa
  sổ **độc lập trên từng khối**.
- **Xáo trộn khi tách valid.** `train_test_split(shuffle=True)` trên chuỗi
  thời gian là leakage. Valid nội bộ cho early stopping ở đây luôn là **đuôi
  theo thời gian** của TRAIN.

### RNN — và vì sao nó thất bại

Mạng hồi tiếp Elman cập nhật trạng thái ẩn tuần tự:

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

Đạo hàm qua `T` bước chứa tích `∏ W_h·diag(tanh')`. Vì `|tanh'| ≤ 1`:

- `‖W_h‖ < 1` → gradient **tiêu biến**: mô hình không học được phụ thuộc xa
- `‖W_h‖ > 1` → gradient **bùng nổ**: loss thành `nan` sau vài epoch

Repo xử lý vế thứ hai bằng `clip_grad_norm_` (mặc định `grad_clip=1.0`). Vế
thứ nhất thì không sửa được bằng thủ thuật — nó cần đổi kiến trúc, và đó
chính là lý do LSTM ra đời. `rnn` được giữ lại làm **baseline hồi tiếp**: nó
cho biết một mô hình chuỗi "ngây thơ" đạt tới đâu.

### LSTM — bộ nhớ có cổng

LSTM (Hochreiter & Schmidhuber, 1997) thêm một **cell state** `c_t` với đường
truyền gradient gần như tuyến tính, điều khiển bởi ba cổng:

```
f_t = σ(W_f · [h_{t-1}, x_t])        cổng quên   — giữ lại bao nhiêu quá khứ
i_t = σ(W_i · [h_{t-1}, x_t])        cổng vào    — nhận vào bao nhiêu thông tin mới
o_t = σ(W_o · [h_{t-1}, x_t])        cổng ra     — lộ ra bao nhiêu cho lớp trên
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t])
h_t = o_t ⊙ tanh(c_t)
```

Điểm mấu chốt là `c_t = f_t ⊙ c_{t-1} + ...`: khi `f_t ≈ 1`, gradient chảy
ngược qua phép **cộng** chứ không phải phép nhân lặp, nên không tiêu biến.

*Diễn giải tài chính:* cổng quên học được "khi nào nên xoá bộ nhớ chế độ thị
trường cũ" — đúng loại hành vi ta cần khi thị trường chuyển chế độ.

### GRU — rẻ hơn, thường tương đương

GRU gộp cổng quên và cổng vào thành một **update gate**, bỏ hẳn cell state
riêng. Kết quả: ít hơn LSTM khoảng 25% tham số.

```
z_t = σ(W_z · [h_{t-1}, x_t])                 update gate
r_t = σ(W_r · [h_{t-1}, x_t])                 reset  gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

Trên chuỗi ngắn và ít dữ liệu — đúng tình huống của repo này — **GRU thường
bằng hoặc hơn LSTM** vì ít tham số hơn nghĩa là ít overfit hơn. Nếu bài báo
cáo chỉ được chọn một, GRU là lựa chọn mặc định hợp lý hơn LSTM.

### Bidirectional có phải leakage không? — Không

Câu hỏi hay gặp khi phản biện. BiLSTM đọc chuỗi theo cả hai chiều, nghe như
"nhìn tương lai". Nhưng phạm vi của nó là **cửa sổ**, và mọi phiên trong
`X[t-19 : t+1]` đều đã xảy ra tại thời điểm ra quyết định `t`. Chiều ngược
chỉ là đọc lại quá khứ từ hiện tại lùi về. Leakage sẽ xảy ra nếu cửa sổ chứa
`X[t+1]` — điều đó không bao giờ xảy ra ở đây.

### CNN 1D — mẫu hình cục bộ, và tại sao phải nhân quả

Convolution 1 chiều trượt một kernel dọc trục thời gian, học các **mẫu hình
cục bộ bất biến theo vị trí**: một cú đảo chiều 3 phiên trông giống nhau dù
xảy ra tháng 3 hay tháng 9. Rẻ hơn RNN nhiều vì tính song song được.

Repo dùng `CausalConv1d` — chỉ **pad bên trái**:

```
conv thường  (padding = k//2):   [x_{t-1}, x_t, x_{t+1}]  -> out_t
conv nhân quả (pad trái):        [x_{t-2}, x_{t-1}, x_t]  -> out_t
```

Trong một cửa sổ thì `x_{t+1}` vẫn ≤ thời điểm ra quyết định nên chưa phải
leakage thật, nhưng nó phá vỡ ngữ nghĩa *"biểu diễn tại t tóm tắt quá khứ tới
t"* — vốn là điều ta dựa vào khi đọc bước cuối cùng ra quyết định.

### TCN — trường tiếp nhận theo cấp số nhân

Conv thường tăng trường tiếp nhận **tuyến tính**: `L` lớp kernel `k` chỉ phủ
`L(k-1)+1` phiên. TCN (Bai, Kolter & Koltun, 2018) dùng **dilation** nhân đôi
mỗi khối:

```
khối 1: dilation 1   ●●●
khối 2: dilation 2   ●·●·●
khối 3: dilation 4   ●···●···●
khối 4: dilation 8   ●·······●·······●
trường tiếp nhận với k=3: 1 + 2·(2⁴−1) = 31 phiên chỉ với 4 khối
```

Ít lớp hơn = ít tham số hơn = ít overfit hơn. Cộng thêm **residual
connection** để mạng sâu vẫn huấn luyện được. Trên nhiều benchmark chuỗi
thời gian, TCN ngang hoặc vượt LSTM trong khi huấn luyện nhanh hơn hẳn.

### Transformer — và lỗi khiến nó vô nghĩa nếu làm sai

Self-attention cho mỗi phiên "nhìn" mọi phiên khác trong cửa sổ với trọng số
học được:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Ưu điểm: phụ thuộc xa chỉ cách nhau **một bước** (RNN cần `T` bước), và tính
được song song hoàn toàn.

> **Cạm bẫy chí mạng: self-attention là phép toán hoán vị bất biến.**
> Không có positional encoding, đảo ngược thứ tự 20 phiên trong cửa sổ thì
> output **không đổi một chút nào**. Transformer khi đó chỉ là một "bag of
> days" — nó vứt bỏ đúng cái duy nhất làm nên chuỗi thời gian. Lấy
> `z[:, -1, :]` sau đó cũng không cứu được, vì bản thân biểu diễn tại vị trí
> cuối đã được trộn từ một tập hợp không thứ tự.

Repo dùng **sinusoidal positional encoding** (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Không có tham số học → không tốn dữ liệu quý hiếm, và tổng quát hoá sang độ
dài chuỗi khác lúc suy luận. Với `lookback` 20–60 phiên, đây là lựa chọn an
toàn hơn learned embedding (vốn cần nhiều mẫu để học).

Hai chi tiết cấu hình khác cũng quan trọng:

- `norm_first=True` (**pre-LN**): ổn định hơn hẳn post-LN khi ít dữ liệu và
  không có warmup scheduler.
- `causal_mask=False` **mặc định**: mọi phiên trong cửa sổ đều là quá khứ nên
  full attention hợp lệ. Bật mask chỉ để regularize, **không phải** để chống
  leakage.

*Kỳ vọng thực tế:* Transformer thường **không** đứng đầu bảng ở quy mô dữ
liệu này. Nó cần nhiều mẫu hơn hẳn để phát huy. Nếu nó thắng, hãy nghi ngờ và
kiểm tra leakage trước khi ăn mừng.

### Pooling — gộp `(B, T, d)` thành `(B, d)`

| `pooling` | Cách gộp | Khi nào dùng |
|---|---|---|
| `last` | lấy bước cuối `h[:, -1, :]` | Mặc định cho RNN/CNN — phiên gần nhất quan trọng nhất |
| `mean` | trung bình theo thời gian | Mặc định cho Transformer — tránh phụ thuộc quá mức vào một vị trí |
| `attention` | trọng số học được (Bahdanau) | Khi muốn biết mô hình nhìn phiên nào |

Với `attention`, repo **nối** context vector với hidden state cuối. Lý do:
context vector thuần rất hay suy biến về phân phối đều (`a_t ≈ 1/T`), khi đó
mô hình mất luôn tín hiệu mạnh nhất là phiên hiện tại. Nối thêm `h[-1]` đảm
bảo `attn_lstm` không bao giờ tệ hơn `lstm` thường.

```python
m = mr.build_model("attn_lstm").fit(X_train, y_train)
m.attention_profile()      # lag 0 = phiên hiện tại
```

> **Đọc trọng số attention một cách thận trọng.** Trong thực nghiệm trên repo
> này, phân phối thường khá phẳng. Đó là kết quả trung thực, không phải lỗi —
> đừng tinh chỉnh cho tới khi biểu đồ trông đẹp rồi diễn giải nó như bằng
> chứng. Attention là trọng số, **không phải giải thích nhân quả**.

### Feature importance cho mô hình chuỗi

Mô hình sâu không có `feature_importances_` như cây. Repo tính **saliency**:

```
importance_j = mean | ∂ logit / ∂ x_{t,j} |     (gộp theo trục thời gian)
```

Đây là **độ nhạy cục bộ**, không phải SHAP và cũng không phải nhân quả. Nó đủ
để trả lời *"mô hình phản ứng mạnh với feature nào"*, không đủ để kết luận
*"feature nào tạo ra lợi nhuận"*. Phải nêu rõ giới hạn này khi đưa vào báo cáo.

### Bảng tham chiếu siêu tham số

| Tham số | Mặc định | Ảnh hưởng |
|---|---|---|
| `lookback` | 20–40 | Dài hơn = nhiều ngữ cảnh, nhưng ít mẫu hơn và dễ overfit hơn |
| `hidden` | 32–48 | Cố ý nhỏ. Tăng lên 128+ gần như luôn làm tệ đi ở quy mô dữ liệu này |
| `layers` | 1–3 | 1 lớp là đủ cho RNN; TCN cần 3 để đủ trường tiếp nhận |
| `dropout` | 0.2 | Regularization chính |
| `patience` | 8 | Early stopping trên valid nội bộ (đuôi TRAIN) |
| `val_fraction` | 0.2 | Cắt theo **thời gian**, không xáo trộn |
| `grad_clip` | 1.0 | Bắt buộc với RNN — không có nó thì loss thành `nan` |
| `class_weight` | `"balanced"` | Nhãn tài chính hiếm khi cân bằng 50/50 |
| `pad_mode` | `"edge"` | Lặp `X[0]` để mọi phiên đều có dự báo (vẫn causal) |
| `max_seconds` | 120 | Trần thời gian mỗi fold — chặn benchmark chạy hàng giờ rồi bị kill |

Toàn bộ được ghi đè bình thường:

```python
res = ExperimentRunner().run(
    prices, X, y,
    model_name="gru",
    model_kwargs=dict(lookback=40, hidden=64, epochs=100, patience=12),
)
```

---

## Ensemble — và một cái bẫy trong sklearn

Các họ mô hình sai theo những cách **khác nhau**: cây bắt tương tác phi tuyến,
mô hình tuyến tính ổn định khi nhiễu lớn, mô hình chuỗi bắt phụ thuộc thời
gian. Trung bình hoá các sai số ít tương quan là cách giảm phương sai rẻ nhất.

`stacking_ensemble` **không** dùng `sklearn.ensemble.StackingClassifier`. Lý do:

- Lớp đó sinh meta-feature bằng `cross_val_predict`, vốn **yêu cầu cv phải là
  một phân hoạch**. `TimeSeriesSplit` không phải phân hoạch → sklearn ném
  `ValueError: cross_val_predict only works for partitions`.
- Cách duy nhất còn lại để dùng lớp của sklearn là quay về `KFold`. Nhưng
  KFold huấn luyện base model trên các fold nằm **sau** tập đang dự báo:
  meta-learner học từ những xác suất do một mô hình đã thấy tương lai sinh ra.
  Đó chính xác là loại leakage mà toàn bộ repo này tồn tại để chặn.

Repo dùng `TimeSeriesStackingClassifier` tự viết: meta-feature sinh theo **cửa
sổ mở rộng**, meta-learner chỉ fit trên phần mẫu thực sự có dự báo out-of-fold
(`oof_coverage_ < 1` là điều đúng, không phải lỗi), base model sau đó fit lại
trên toàn bộ TRAIN.

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
- **Mô hình chuỗi đang chạy dưới mức tiềm năng, và đó là lựa chọn có ý thức.**
  Siêu tham số của chúng (`lookback`, `hidden`, `layers`) hiện lấy từ
  `ARCH_DEFAULTS` chứ không được tune trên VALID như ngưỡng sizing. Tune chúng
  đúng cách sẽ làm `n_trials` tăng vọt và DSR phạt nặng — nên nếu làm, phải
  đếm đủ số tổ hợp đã thử, đừng chỉ báo cáo cấu hình thắng.
- **Một fold TEST ~ vài trăm phiên là quá ít cho deep learning.** Chênh lệch
  AUC 0.52 so với 0.54 giữa LSTM và GRU nằm trong sai số lấy mẫu. Muốn kết
  luận kiến trúc nào tốt hơn thì phải chạy nhiều seed và báo cáo **phân phối**,
  không phải một con số.
- Saliency (`feature_importances_` của mô hình chuỗi) là độ nhạy cục bộ, không
  phải SHAP và không phải bằng chứng nhân quả.
- `pad_mode="edge"` khiến vài phiên đầu mỗi khối có ngữ cảnh nghèo (lặp lại
  `X[0]`). Vẫn causal, nhưng dự báo ở những phiên đó kém tin cậy hơn.

Triển khai hai phiên bản: [`DEPLOY.md`](DEPLOY.md).
Chi tiết audit repo cũ và toàn bộ lỗi đã sửa: [`AUDIT.md`](AUDIT.md).
Hướng dẫn chuyển đổi: [`MIGRATION.md`](MIGRATION.md).
