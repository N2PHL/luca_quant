# Triển khai — hai phiên bản, một codebase

| | **Bản nộp trực tuyến** | **Bản đầy đủ** |
|---|---|---|
| Profile | `coursework` | `full` |
| Chạy ở đâu | Streamlit Community Cloud | Máy local |
| Mô hình | 6 kiến trúc + 3 baseline | 34 mô hình / 6 họ |
| requirements | `requirements.txt` | `requirements-full.txt` |
| Bộ nhớ | siết trần cho vừa ~1GB | không giới hạn |

**Chỉ có một codebase.** Hai repo tách rời sẽ phân kỳ — sửa một lỗi leakage ở
bản local rồi quên port sang bản nộp là kịch bản tệ nhất có thể xảy ra với một
đồ án: con số trên bản giảng viên chấm không còn là con số bạn đã kiểm định.
Profile chỉ **lọc danh sách mô hình** và **siết trần tài nguyên**. Không có
nhánh logic nào khác nhau giữa hai bản.

---

## Bản nộp trực tuyến (`coursework`)

### Mô hình có trong bản này

| Mô hình | Kiến trúc |
|---|---|
| `mlp` | Mạng truyền thẳng — không có khái niệm thời gian, dùng làm mốc so với mô hình chuỗi |
| `cnn1d` | CNN 1 chiều nhân quả |
| `rnn` | RNN Elman |
| `lstm` | LSTM |
| `gru` | GRU |
| `transformer` | Transformer + sinusoidal positional encoding |
| `buy_hold`, `random`, `momentum_rule` | Baseline — **không lọc bỏ được**, xem bên dưới |

### Vì sao baseline không bị lọc bỏ

Đề bài chỉ yêu cầu 6 kiến trúc trên, nhưng ba baseline vẫn ở lại. Đây không
phải thêm thắt cho đủ số:

- `experiments/benchmark.py` dùng `buy_hold` làm mốc cho kiểm định ghép cặp
  (`paired_test_vs_benchmark`). Bỏ nó đi thì phần kiểm định ý nghĩa thống kê
  **tắt lặng lẽ** — bảng vẫn hiện ra, chỉ là không còn p-value nào cả.
- `evaluation/acceptance_gate.py` so Sharpe của chiến lược với Buy & Hold.
- Về học thuật: "LSTM đạt Sharpe 1.8" là câu không diễn giải được. Nếu Buy &
  Hold cùng giai đoạn đạt 1.7 thì mô hình đóng góp 0.1, không bù nổi chi phí
  giao dịch.

Ba baseline này là numpy thuần, không phụ thuộc thư viện nào, tốn vài KB RAM.
Chúng không phải thứ làm app hết bộ nhớ.

### Các bước deploy

1. **Push code lên GitHub** (repo phải public nếu dùng free tier — free tier
   chỉ cho một private app).

2. **Tạo file `.luca_profile`** ở gốc repo, nội dung đúng một dòng:
   ```
   coursework
   ```
   Hoặc bỏ file này và đặt qua Streamlit secrets (bước 4) — chọn một trong hai.

3. **Kiểm tra `requirements.txt`** đang là bản online. Dòng
   `--extra-index-url https://download.pytorch.org/whl/cpu` **phải đứng trước**
   `torch`. Không có nó, pip lấy wheel mặc định trên PyPI kèm toàn bộ runtime
   CUDA (~2GB) và build sẽ **fail**, không phải chạy chậm.

4. **Deploy**: share.streamlit.io → New app → chọn repo, branch, `app.py`.
   Nếu muốn đặt profile qua secrets thay vì file, vào Settings → Secrets:
   ```toml
   LUCA_PROFILE = "coursework"
   ```

5. **Xác minh sau khi deploy**: mở trang chủ → mục "Kiểm tra môi trường". Phải
   thấy `Profile đang chạy: coursework` và đúng 9 mô hình.

### Giới hạn của Streamlit Community Cloud

- **~1GB RAM.** Vượt trần thì app bị kill giữa chừng, không có cảnh báo. Trang
  Research Lab hiện đồng hồ bộ nhớ sau mỗi lần chạy và cảnh báo khi vượt 80%.
- **App ngủ sau 12 giờ không có truy cập.** Người truy cập tiếp theo thấy màn
  hình "waking up". Nếu giảng viên chấm sau vài ngày, lần mở đầu sẽ chậm —
  nên báo trước, hoặc tự mở app một lần trước khi nộp.
- **Không có custom domain**, URL dạng `<tên>.streamlit.app`.

### Nếu vẫn hết bộ nhớ

Theo thứ tự đánh đổi từ ít đau nhất:

1. Giảm số fold walk-forward (5 → 3). Ảnh hưởng: ít điểm ước lượng hơn, khoảng
   tin cậy rộng hơn.
2. Bớt mô hình trong lineup mỗi lần chạy. Ảnh hưởng: **không có** — thậm chí
   tốt hơn, vì `n_trials` của Deflated Sharpe Ratio giảm theo.
3. Bớt nhóm feature. Ảnh hưởng: ablation kém đầy đủ.
4. Giảm `years` dữ liệu tải về.

Nếu vẫn không đủ: đây là lúc chuyển hạ tầng, không phải lúc cắt tiếp phương
pháp luận. Hugging Face Spaces (free, 16GB RAM) là lựa chọn thay thế gần nhất
và cũng chạy Streamlit.

---

## Bản đầy đủ (`full`)

```bash
git clone <repo> && cd luca_quant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-full.txt
pytest                      # 87 test
streamlit run app.py
```

Không có file `.luca_profile` thì profile mặc định là `full` — toàn bộ 34 mô
hình. Muốn thử bản online ngay trên máy local:

```bash
LUCA_PROFILE=coursework streamlit run app.py
```

Dùng trong notebook:

```python
from luca_quant.services.research_service import ResearchService

session = ResearchService().run_full_study(
    ticker="FPT",
    groups=["price", "trend", "momentum", "volatility", "fractal"],
    models=["buy_hold", "momentum_rule", "lightgbm", "gru", "tcn"],
    primary_model="gru",
)
print(session.thesis)
```

---

## Ghi chú về bộ nhớ

Ba thay đổi đã làm để bản online vừa trần 1GB:

1. **Hoãn import torch.** `_has()` trong registry trước đây dùng `__import__`,
   nên riêng việc import registry đã trả ~250–450MB — ngay khi app khởi động,
   kể cả khi người dùng chỉ chạy `buy_hold`. Giờ dùng `find_spec`, torch chỉ
   được nạp khi một mô hình chuỗi thực sự được `fit`. App lúc đứng yên: **194MB**.
2. **`torch.set_num_threads(1)`** ở profile `coursework`. Mặc định torch mở
   luồng theo số core *nhìn thấy* (của cả máy chủ, không phải phần được cấp);
   mỗi luồng tốn bộ nhớ và chúng tranh nhau CPU.
3. **`malloc_trim(0)` giữa các mô hình** trong vòng lặp benchmark. `gc.collect()`
   một mình không đủ: glibc giữ khối đã free trong arena nên RSS không giảm.

Con số đo được (dữ liệu synthetic 2.148 phiên, 28 feature, 4 fold, 9 mô hình):

| | RSS |
|---|---|
| App đứng yên, chưa chạy gì | 194 MB |
| Sau khi nạp torch | +450 MB *(wheel kèm CUDA — bản CPU nhẹ hơn đáng kể)* |
| Đỉnh khi chạy hết lineup 9 mô hình | 941 MB |

> **Con số 941MB đo trên wheel torch mặc định của PyPI (kèm CUDA), không phải
> wheel CPU mà bản deploy dùng.** Bản CPU nhẹ hơn nhiều ở bước import, nên
> đỉnh thực tế trên Cloud sẽ thấp hơn — nhưng tôi **chưa kiểm chứng được** con
> số đó. Hãy xem đồng hồ bộ nhớ trên trang Research Lab ở lần chạy thật đầu
> tiên, và nếu nó báo trên 80% thì áp dụng danh sách cắt giảm ở trên **trước
> khi** gửi link cho giảng viên.
