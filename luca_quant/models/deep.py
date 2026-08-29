"""
Deep Learning models (Blueprint Phase 6) — RNN / LSTM / GRU / CNN / TCN /
Transformer, bọc trong sklearn API để ExperimentRunner không cần biết bên
trong là kiến trúc gì.

Chỉ import torch khi thật sự cần. Không có torch thì registry tự bỏ nhóm này,
app KHÔNG sập.

===========================================================================
BỐN LỖI CỦA PHIÊN BẢN TRƯỚC ĐÃ SỬA Ở ĐÂY
===========================================================================

**(1) Transformer không có positional encoding.**
    Self-attention là phép toán HOÁN VỊ BẤT BIẾN: đảo ngược thứ tự 20 phiên
    trong cửa sổ thì output không đổi một chút nào. Một Transformer không
    positional encoding chạy trên chuỗi thời gian thực chất chỉ là một
    "bag of days" — nó vứt bỏ đúng cái thứ duy nhất làm nên chuỗi thời gian.
    Việc lấy `z[:, -1, :]` sau đó cũng không cứu được, vì bản thân biểu diễn
    tại vị trí cuối đã được trộn từ một tập hợp không thứ tự.
    -> Đã thêm sinusoidal positional encoding (Vaswani et al., 2017).

**(2) Cửa sổ trượt lệch một phiên, làm DL thua oan.**
    Cũ: dự báo `y[t]` bằng cửa sổ `X[t-lookback : t]` — tức là BỎ ĐI hàng
    `X[t]`. Nhưng `y[t] = close[t+h]/close[t] - 1` (xem `labels/registry.py`),
    còn `X[t]` là feature causal đã biết tại lúc đóng cửa phiên t. LightGBM
    được dùng `X[t]`, LSTM thì không. Bảng benchmark vì thế so sánh không
    công bằng: mọi mô hình chuỗi bị trễ đúng một phiên.
    -> Cửa sổ bây giờ là `X[t-lookback+1 : t+1]`, bao gồm phiên t.
    Vẫn tuyệt đối causal: mọi hàng trong cửa sổ đều <= t.

**(3) Không có early stopping.** Chạy cứng 30 epoch trên vài trăm mẫu là
    công thức overfit. Giờ có tách VALID nội bộ theo thời gian (đuôi của
    TRAIN, không đụng vào VALID của runner) + patience + khôi phục trọng số
    tốt nhất.

**(4) `predict_proba` trả 0.5 cho `lookback` phiên đầu mỗi khối.**
    Với lookback=20 và fold test 120 phiên thì 17% dự báo là "không ý kiến" —
    và quan trọng hơn, nó làm hỏng bước hiệu chuẩn xác suất trên VALID.
    -> Thêm `pad_mode="edge"`: lặp lại hàng đầu tiên của chính khối đó để
    lấp cửa sổ. Vẫn causal (chỉ dùng dữ liệu tại thời điểm <= t), chỉ là
    những phiên đầu có ngữ cảnh nghèo hơn.

===========================================================================
VÌ SAO CỬA SỔ ĐƯỢC DỰNG BÊN TRONG fit/predict
===========================================================================
Nếu dựng cửa sổ một lần cho toàn bộ chuỗi rồi mới chia TRAIN/TEST thì sẽ có
những cửa sổ VẮT QUA ranh giới: mẫu đầu tiên của TEST chứa 19 phiên cuối của
TRAIN. Mô hình đã nhìn thấy các hàng đó lúc học. Đây là nguồn leakage phổ
biến nhất trong đồ án DL tài chính, và nó rất khó phát hiện vì không có cột
nào "nhìn tương lai" — chỉ là ranh giới bị nhoè.
Ở đây `fit` và `predict` dựng cửa sổ ĐỘC LẬP trên đúng khối được truyền vào,
nên ranh giới luôn sạch.

===========================================================================
BIDIRECTIONAL CÓ PHẢI LEAKAGE KHÔNG? — KHÔNG.
===========================================================================
Câu hỏi hay gặp khi phản biện. BiLSTM đọc chuỗi theo cả hai chiều, nghe như
"nhìn tương lai". Nhưng phạm vi của nó là CỬA SỔ, và mọi phiên trong cửa sổ
`X[t-19 : t+1]` đều đã xảy ra tại thời điểm ra quyết định t. Chiều ngược chỉ
là đọc lại quá khứ từ hiện tại về trước. Leakage sẽ xảy ra nếu cửa sổ chứa
`X[t+1]` — điều đó không bao giờ xảy ra ở đây.
Với Transformer cũng vậy: full attention trong cửa sổ là hợp lệ, nên
`causal_mask` mặc định TẮT. Bật lên chỉ để regularize, không phải để chống
leakage.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

# Danh sách kiến trúc — registry.py đọc trực tiếp biến này.
SEQ_ARCHS: List[str] = [
    "rnn",           # Elman RNN — baseline hồi tiếp đơn giản nhất
    "lstm",
    "gru",
    "bilstm",
    "bigru",
    "cnn1d",         # CNN nhân quả 1 chiều
    "tcn",           # Temporal Convolutional Network (dilated + residual)
    "cnn_lstm",      # CNN trích đặc trưng cục bộ -> LSTM mô hình hoá dài hạn
    "attn_lstm",     # LSTM + additive attention pooling
    "transformer",
]

_RNN_ARCHS = {"rnn", "lstm", "gru", "bilstm", "bigru"}


# ==========================================================================
# CỬA SỔ TRƯỢT
# ==========================================================================
def _as_matrix(X) -> np.ndarray:
    """DataFrame/array -> float32 2 chiều, NaN/Inf được dọn sạch."""
    arr = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def make_windows(
    X: np.ndarray,
    y: Optional[np.ndarray],
    lookback: int,
    pad_mode: str = "none",
):
    """
    Dựng cửa sổ trượt KẾT THÚC TẠI t (bao gồm t).

        window[t] = X[t - lookback + 1 : t + 1]      -> dự báo y[t]

    pad_mode:
      "none" — chỉ tạo cửa sổ cho t >= lookback-1. Các phiên đầu không có
               dự báo (runner sẽ điền 0.5 = không quan điểm).
      "edge" — lặp hàng X[0] để lấp phần thiếu, mọi t đều có dự báo.
               Vẫn causal: X[0] đã tồn tại tại mọi t >= 0.

    Trả về (Xw, yw, idx) với idx là chỉ số hàng gốc mà mỗi cửa sổ dự báo.
    """
    n = len(X)
    if lookback < 1:
        raise ValueError("lookback phải >= 1.")

    if pad_mode == "edge":
        pad = np.repeat(X[:1], lookback - 1, axis=0) if lookback > 1 else X[:0]
        Xp = np.concatenate([pad, X], axis=0) if len(pad) else X
        idx = np.arange(n)
    elif pad_mode == "none":
        if n < lookback:
            raise ValueError(
                f"Cần >= {lookback} mẫu để tạo cửa sổ, chỉ có {n}. "
                f"Giảm `lookback` hoặc dùng pad_mode='edge'."
            )
        Xp = X
        idx = np.arange(lookback - 1, n)
    else:
        raise ValueError(f"pad_mode không hợp lệ: {pad_mode!r} (dùng 'none' | 'edge')")

    # sliding_window_view: O(1) bộ nhớ, không copy n*lookback lần như np.stack
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(Xp, lookback, axis=0)      # (m, n_feat, lookback)
    win = np.ascontiguousarray(win.transpose(0, 2, 1))   # (m, lookback, n_feat)
    win = win[: len(idx)]

    yw = np.asarray(y, dtype=np.float32)[idx] if y is not None else None
    return win, yw, idx


# ==========================================================================
# CÁC KHỐI MẠNG — định nghĩa một lần, cache lại
# ==========================================================================
@lru_cache(maxsize=1)
def _blocks():
    """Định nghĩa nn.Module bên trong hàm để file này import được khi KHÔNG có torch."""
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding (Vaswani et al., 2017, §3.5).

            PE(pos, 2i)   = sin(pos / 10000^(2i/d))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

        Không có tham số học -> không tốn dữ liệu, và tổng quát hoá được sang
        độ dài chuỗi khác lúc suy luận. Với lookback ngắn (20-60 phiên) đây là
        lựa chọn an toàn hơn learned embedding, vốn cần nhiều mẫu để học.
        """

        def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

        def forward(self, x):                             # (B, T, d)
            return self.dropout(x + self.pe[:, : x.size(1), :])

    class CausalConv1d(nn.Module):
        """
        Convolution 1 chiều NHÂN QUẢ: chỉ pad bên trái.

        `nn.Conv1d(padding=k//2)` của bản cũ pad ĐỀU HAI PHÍA. Nghĩa là output
        tại vị trí t được tính từ cả x[t+1], x[t+2]... Trong cửa sổ thì các
        phiên đó vẫn <= t nên chưa phải leakage thật sự, nhưng nó phá vỡ ngữ
        nghĩa "biểu diễn tại t tóm tắt quá khứ tới t" mà ta cần khi đọc bước
        cuối. Pad trái giữ đúng ngữ nghĩa đó.
        """

        def __init__(self, c_in, c_out, kernel_size, dilation=1):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)

        def forward(self, x):                             # (B, C, T)
            return self.conv(nn.functional.pad(x, (self.pad, 0)))

    class TemporalBlock(nn.Module):
        """
        Khối TCN (Bai, Kolter & Koltun, 2018): 2 lớp conv nhân quả giãn nở +
        kết nối tắt.

        Vì sao dilation: trường tiếp nhận tăng theo CẤP SỐ NHÂN. Với
        kernel=3 và dilation 1,2,4,8 thì 4 khối đã phủ 31 phiên, trong khi
        conv thường cần 9 lớp. Ít lớp hơn = ít tham số hơn = ít overfit hơn,
        điều rất đáng giá khi chỉ có vài nghìn phiên giao dịch.
        """

        def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
            super().__init__()
            self.net = nn.Sequential(
                CausalConv1d(c_in, c_out, kernel_size, dilation),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Dropout(dropout),
                CausalConv1d(c_out, c_out, kernel_size, dilation),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.net(x) + self.down(x))

    class AdditiveAttentionPool(nn.Module):
        """
        Attention pooling (Bahdanau): học trọng số cho từng phiên trong cửa sổ
        thay vì chỉ lấy bước cuối.

            e_t = v^T tanh(W h_t) ;  a = softmax(e) ;  z = sum_t a_t h_t

        Lợi ích thực tế: `attn_weights_` cho biết mô hình đang nhìn phiên nào —
        một dạng giải thích được, hiếm khi có ở mô hình chuỗi.
        """

        def __init__(self, d_model):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, 1, bias=False)

        def forward(self, h, return_weights: bool = False):   # (B, T, d)
            e = self.v(torch.tanh(self.proj(h))).squeeze(-1)   # (B, T)
            a = torch.softmax(e, dim=1)
            z = torch.bmm(a.unsqueeze(1), h).squeeze(1)        # (B, d)
            return (z, a) if return_weights else z

    class SequenceNet(nn.Module):
        """Bộ khung chung: encoder (tuỳ arch) -> pooling -> head nhị phân."""

        def __init__(self, arch, n_features, hidden, layers, dropout,
                     n_heads, kernel_size, pooling, lookback, causal_mask):
            super().__init__()
            self.arch = arch
            self.pooling = pooling
            self.causal_mask = causal_mask
            self._attn = None

            if arch in _RNN_ARCHS:
                bi = arch in ("bilstm", "bigru")
                cell = {
                    "rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU,
                    "bilstm": nn.LSTM, "bigru": nn.GRU,
                }[arch]
                kw = dict(
                    input_size=n_features, hidden_size=hidden, num_layers=layers,
                    batch_first=True, bidirectional=bi,
                    dropout=dropout if layers > 1 else 0.0,
                )
                if cell is nn.RNN:
                    kw["nonlinearity"] = "tanh"
                self.rnn = cell(**kw)
                d_out = hidden * (2 if bi else 1)

            elif arch == "cnn1d":
                ch, blocks = hidden, []
                c_in = n_features
                for _ in range(max(layers, 1)):
                    blocks += [
                        CausalConv1d(c_in, ch, kernel_size),
                        nn.BatchNorm1d(ch),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                    c_in = ch
                self.conv = nn.Sequential(*blocks)
                d_out = ch

            elif arch == "tcn":
                blocks, c_in = [], n_features
                for i in range(max(layers, 1)):
                    blocks.append(TemporalBlock(c_in, hidden, kernel_size, 2 ** i, dropout))
                    c_in = hidden
                self.conv = nn.Sequential(*blocks)
                d_out = hidden

            elif arch == "cnn_lstm":
                self.conv = nn.Sequential(
                    CausalConv1d(n_features, hidden, kernel_size),
                    nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                )
                self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
                d_out = hidden

            elif arch == "attn_lstm":
                self.rnn = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
                d_out = hidden

            elif arch == "transformer":
                # d_model phải chia hết cho n_heads
                heads = max(1, n_heads)
                while heads > 1 and hidden % heads != 0:
                    heads -= 1
                self.heads_ = heads
                self.proj = nn.Linear(n_features, hidden)
                self.pos = PositionalEncoding(hidden, max_len=max(lookback + 1, 64),
                                              dropout=dropout)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
                    dropout=dropout, batch_first=True,
                    activation="gelu",
                    norm_first=True,      # pre-LN: ổn định hơn hẳn khi ít dữ liệu
                )
                self.enc = nn.TransformerEncoder(
                    layer, num_layers=layers, norm=nn.LayerNorm(hidden),
                    enable_nested_tensor=False,   # norm_first=True -> nested tensor vô hiệu
                )
                d_out = hidden
            else:
                raise ValueError(f"arch không hỗ trợ: {arch!r}. Có: {SEQ_ARCHS}")

            use_attn = (pooling == "attention") or (arch == "attn_lstm")
            self.pool_attn = AdditiveAttentionPool(d_out) if use_attn else None
            # Attention pooling NỐI context vector với hidden state cuối.
            # Chỉ dùng context thuần thì attention hay bị suy biến về phân phối
            # đều (a_t ~ 1/T) và mô hình mất luôn tín hiệu mạnh nhất là phiên t.
            # Nối thêm h[-1] đảm bảo attn_lstm không bao giờ tệ hơn lstm thường.
            d_head = d_out * 2 if use_attn else d_out
            self.head = nn.Sequential(
                nn.LayerNorm(d_head),
                nn.Dropout(dropout),
                nn.Linear(d_head, max(d_head // 2, 8)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(d_head // 2, 8), 1),
            )

        # -- encoder -> (B, T, d) --------------------------------------
        def encode(self, x):
            if self.arch in _RNN_ARCHS:
                out, _ = self.rnn(x)
                return out
            if self.arch in ("cnn1d", "tcn"):
                return self.conv(x.transpose(1, 2)).transpose(1, 2)
            if self.arch == "cnn_lstm":
                z = self.conv(x.transpose(1, 2)).transpose(1, 2)
                out, _ = self.rnn(z)
                return out
            if self.arch == "attn_lstm":
                out, _ = self.rnn(x)
                return out
            # transformer
            h = self.pos(self.proj(x))
            mask = None
            if self.causal_mask:
                T = x.size(1)
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            return self.enc(h, mask=mask)

        def pool(self, h, keep_weights: bool = False):
            if self.pool_attn is not None:
                z, a = self.pool_attn(h, return_weights=True)
                if keep_weights:
                    self._attn = a.detach()
                return torch.cat([z, h[:, -1, :]], dim=-1)
            if self.pooling == "mean":
                return h.mean(dim=1)
            return h[:, -1, :]          # "last" — biểu diễn tại phiên t

        def forward(self, x, keep_weights: bool = False):
            return self.head(self.pool(self.encode(x), keep_weights)).squeeze(-1)

    return SimpleNamespace(
        torch=torch, nn=nn,
        PositionalEncoding=PositionalEncoding,
        CausalConv1d=CausalConv1d,
        TemporalBlock=TemporalBlock,
        AdditiveAttentionPool=AdditiveAttentionPool,
        SequenceNet=SequenceNet,
    )


# ==========================================================================
# ESTIMATOR
# ==========================================================================
class TorchSequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Bộ phân loại chuỗi tương thích sklearn cho toàn bộ họ mô hình sâu.

    Tham số quan trọng
    ------------------
    arch        : một trong `SEQ_ARCHS`.
    lookback    : số phiên trong cửa sổ (bao gồm phiên hiện tại).
    hidden      : chiều ẩn / số kênh.
    layers      : số lớp RNN, số khối conv, hoặc số lớp encoder.
    pooling     : "last" | "mean" | "attention" — cách gộp (B,T,d) -> (B,d).
    val_fraction: tỉ lệ ĐUÔI của TRAIN dùng làm valid nội bộ cho early stopping.
                  Cắt theo THỜI GIAN, không xáo trộn — xáo trộn ở đây là
                  leakage kinh điển.
    patience    : số epoch không cải thiện thì dừng.
    pad_mode    : "edge" (mặc định) | "none". Xem `make_windows`.
    class_weight: "balanced" -> pos_weight = n_neg/n_pos, hoặc None.
    max_seconds : trần thời gian huấn luyện mỗi fold. Streamlit Cloud free
                  tier rất chậm; không có trần này thì benchmark 10 mô hình
                  x 5 fold có thể chạy hàng giờ rồi bị kill.

    Thuộc tính sau khi fit
    ----------------------
    history_            : DataFrame loss theo epoch (train / valid).
    best_epoch_         : epoch có valid loss thấp nhất (trọng số được khôi phục).
    feature_importances_: saliency trung bình |d logit / d x| theo từng feature.
    attn_weights_       : (chỉ arch có attention) trọng số chú ý trung bình
                          theo vị trí trong cửa sổ — mô hình đang nhìn phiên nào.
    """

    def __init__(
        self,
        arch: str = "lstm",
        lookback: int = 20,
        hidden: int = 32,
        layers: int = 1,
        dropout: float = 0.2,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        patience: int = 8,
        val_fraction: float = 0.2,
        grad_clip: float = 1.0,
        n_heads: int = 4,
        kernel_size: int = 3,
        pooling: str = "last",
        pad_mode: str = "edge",
        causal_mask: bool = False,
        class_weight: Optional[str] = "balanced",
        max_seconds: float = 120.0,
        device: Optional[str] = None,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.arch = arch
        self.lookback = lookback
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.grad_clip = grad_clip
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.pad_mode = pad_mode
        self.causal_mask = causal_mask
        self.class_weight = class_weight
        self.max_seconds = max_seconds
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _resolve_device(self):
        B = _blocks()
        if self.device:
            return B.torch.device(self.device)
        return B.torch.device("cuda" if B.torch.cuda.is_available() else "cpu")

    def _build(self, n_features: int):
        B = _blocks()
        return B.SequenceNet(
            arch=self.arch, n_features=n_features, hidden=self.hidden,
            layers=max(1, self.layers), dropout=self.dropout,
            n_heads=self.n_heads, kernel_size=self.kernel_size,
            pooling=self.pooling, lookback=self.lookback,
            causal_mask=self.causal_mask,
        )

    # ------------------------------------------------------------------
    def fit(self, X, y):
        from luca_quant.config.profiles import apply_torch_threads

        apply_torch_threads()
        B = _blocks()
        torch, nn = B.torch, B.nn

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xa = _as_matrix(X)
        ya = np.asarray(y, dtype=np.float32).ravel()
        if len(Xa) != len(ya):
            raise ValueError(f"X ({len(Xa)}) và y ({len(ya)}) lệch độ dài.")

        self.n_features_in_ = Xa.shape[1]
        self.feature_names_in_ = (
            np.asarray(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        self.classes_ = np.array([0, 1])

        Xw, yw, _ = make_windows(Xa, ya, self.lookback, self.pad_mode)
        if len(np.unique(yw)) < 2:
            raise ValueError(
                "Tập huấn luyện chỉ có MỘT lớp — không thể học phân loại nhị phân. "
                "Kiểm tra label/threshold hoặc độ dài fold."
            )

        # --- Tách valid nội bộ theo THỜI GIAN (đuôi của TRAIN) -----------
        n = len(Xw)
        n_val = int(n * self.val_fraction)
        # cần đủ mẫu và cả hai lớp ở mỗi bên, nếu không thì bỏ early stopping
        use_es = n_val >= 20 and (n - n_val) >= 30
        if use_es:
            cut = n - n_val
            if len(np.unique(yw[:cut])) < 2 or len(np.unique(yw[cut:])) < 2:
                use_es = False
        if use_es:
            cut = n - n_val
            Xtr, ytr, Xva, yva = Xw[:cut], yw[:cut], Xw[cut:], yw[cut:]
        else:
            Xtr, ytr, Xva, yva = Xw, yw, None, None

        dev = self._resolve_device()
        self.device_ = str(dev)
        self.model_ = self._build(self.n_features_in_).to(dev)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(2, self.patience // 3)
        )

        # --- Lệch lớp: label tài chính hiếm khi 50/50 --------------------
        if self.class_weight == "balanced":
            pos = float(max(ytr.sum(), 1.0))
            neg = float(max(len(ytr) - pos, 1.0))
            pw = torch.tensor(neg / pos, dtype=torch.float32, device=dev)
        else:
            pw = None
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        lossf_va = nn.BCEWithLogitsLoss(pos_weight=pw)

        Xt = torch.from_numpy(Xtr).to(dev)
        yt = torch.from_numpy(ytr).to(dev)
        if use_es:
            Xv = torch.from_numpy(Xva).to(dev)
            yv = torch.from_numpy(yva).to(dev)

        best = float("inf")
        best_state, best_epoch, bad = None, 0, 0
        hist = []
        t0 = time.time()
        bs = max(8, int(self.batch_size))

        for ep in range(1, int(self.epochs) + 1):
            self.model_.train()
            perm = torch.randperm(len(Xt), device=dev)
            tot, nb = 0.0, 0
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]
                if len(b) < 2:            # BatchNorm cần >= 2 mẫu
                    continue
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.model_(Xt[b]), yt[b])
                loss.backward()
                # Clip gradient: RNN không có bước này rất dễ nổ gradient,
                # biểu hiện là loss = nan sau vài epoch.
                if self.grad_clip and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                opt.step()
                tot += float(loss.item())
                nb += 1
            tr_loss = tot / max(nb, 1)

            if use_es:
                self.model_.eval()
                with torch.no_grad():
                    va_loss = float(lossf_va(self.model_(Xv), yv).item())
                sched.step(va_loss)
                monitor = va_loss
            else:
                monitor = tr_loss
                va_loss = float("nan")

            hist.append({"epoch": ep, "train_loss": tr_loss, "valid_loss": va_loss})
            if self.verbose:
                print(f"[{self.arch}] ep{ep:03d} train={tr_loss:.4f} valid={va_loss:.4f}")

            if monitor < best - 1e-5:
                best, best_epoch, bad = monitor, ep, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if use_es and bad >= self.patience:
                    break

            if self.max_seconds and (time.time() - t0) > self.max_seconds:
                if self.verbose:
                    print(f"[{self.arch}] dừng vì vượt max_seconds={self.max_seconds}s")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.history_ = pd.DataFrame(hist)
        self.best_epoch_ = best_epoch
        self.best_score_ = best
        self.stopped_early_ = use_es and best_epoch < len(hist)
        self.train_seconds_ = time.time() - t0
        self._compute_attributions(Xtr)
        return self

    # ------------------------------------------------------------------
    def _compute_attributions(self, Xtr: np.ndarray) -> None:
        """
        Saliency: |d logit / d x| trung bình, gộp theo chiều thời gian.

        Không phải SHAP, và không nên gọi là "feature importance" theo nghĩa
        cây quyết định. Đây là độ nhạy cục bộ của output theo từng input —
        đủ để trả lời "mô hình phản ứng mạnh với feature nào", không đủ để
        kết luận nhân quả. Nêu rõ điều này khi đưa vào báo cáo.
        """
        B = _blocks()
        torch = B.torch
        try:
            dev = next(self.model_.parameters()).device
            sub = Xtr[-min(len(Xtr), 512):]
            x = torch.from_numpy(np.ascontiguousarray(sub)).to(dev).requires_grad_(True)
            self.model_.eval()
            out = self.model_(x, keep_weights=True).sum()
            g, = torch.autograd.grad(out, x)
            sal = g.abs().mean(dim=(0, 1)).detach().cpu().numpy()
            self.feature_importances_ = np.nan_to_num(sal, nan=0.0).astype(float)
            a = getattr(self.model_, "_attn", None)
            self.attn_weights_ = a.mean(dim=0).cpu().numpy() if a is not None else None
        except Exception:                                    # noqa: BLE001
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
            self.attn_weights_ = None

    # ------------------------------------------------------------------
    def _forward_batched(self, Xw: np.ndarray) -> np.ndarray:
        B = _blocks()
        torch = B.torch
        dev = next(self.model_.parameters()).device
        self.model_.eval()
        outs = []
        step = max(256, int(self.batch_size))
        with torch.no_grad():
            for i in range(0, len(Xw), step):
                chunk = torch.from_numpy(np.ascontiguousarray(Xw[i:i + step])).to(dev)
                outs.append(self.model_(chunk).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Gọi fit() trước predict_proba().")
        Xa = _as_matrix(X)
        n = len(Xa)
        # Mặc định 0.5 = KHÔNG có quan điểm. Tuyệt đối không lấp bằng giá trị
        # tương lai hay bằng trung bình toàn chuỗi (cả hai đều là leakage).
        p = np.full(n, 0.5, dtype=float)
        try:
            Xw, _, idx = make_windows(Xa, None, self.lookback, self.pad_mode)
        except ValueError:
            return np.column_stack([1 - p, p])
        logits = self._forward_batched(Xw)
        # sigmoid ổn định số học
        p[idx] = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))),
            np.exp(np.clip(logits, -50, 50)) / (1.0 + np.exp(np.clip(logits, -50, 50))),
        )
        p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    # ------------------------------------------------------------------
    def attention_profile(self) -> Optional[pd.DataFrame]:
        """Trọng số chú ý theo vị trí trong cửa sổ (lag 0 = phiên hiện tại)."""
        a = getattr(self, "attn_weights_", None)
        if a is None:
            return None
        lag = np.arange(len(a))[::-1]
        return pd.DataFrame({"lag": lag, "weight": a}).sort_values("lag").reset_index(drop=True)


# ==========================================================================
# CẤU HÌNH MẶC ĐỊNH THEO KIẾN TRÚC
# ==========================================================================
# Mỗi kiến trúc có "vùng hoạt động" khác nhau. Transformer cần d_model lớn hơn
# và nhiều epoch hơn; TCN chịu được lookback dài; CNN thuần thì lookback ngắn
# là đủ. Đặt mặc định hợp lý ở đây để benchmark không so sánh một Transformer
# cấu hình sai với một LSTM cấu hình đúng.
ARCH_DEFAULTS = {
    "rnn":         dict(lookback=20, hidden=32, layers=1, dropout=0.15, lr=1e-3, pooling="last"),
    "lstm":        dict(lookback=20, hidden=48, layers=1, dropout=0.20, lr=1e-3, pooling="last"),
    "gru":         dict(lookback=20, hidden=48, layers=1, dropout=0.20, lr=1e-3, pooling="last"),
    "bilstm":      dict(lookback=20, hidden=32, layers=1, dropout=0.25, lr=1e-3, pooling="last"),
    "bigru":       dict(lookback=20, hidden=32, layers=1, dropout=0.25, lr=1e-3, pooling="last"),
    "cnn1d":       dict(lookback=20, hidden=32, layers=2, dropout=0.20, lr=2e-3, pooling="last",
                        kernel_size=3),
    "tcn":         dict(lookback=40, hidden=32, layers=3, dropout=0.20, lr=2e-3, pooling="last",
                        kernel_size=3),
    "cnn_lstm":    dict(lookback=30, hidden=32, layers=1, dropout=0.20, lr=1e-3, pooling="last",
                        kernel_size=3),
    "attn_lstm":   dict(lookback=30, hidden=32, layers=1, dropout=0.20, lr=1e-3,
                        pooling="attention"),
    "transformer": dict(lookback=30, hidden=32, layers=2, dropout=0.20, lr=5e-4,
                        pooling="mean", n_heads=4),
}


def make_torch_model(arch: str, **kwargs) -> TorchSequenceClassifier:
    """
    Factory dùng bởi `models/registry.py`.

    Thứ tự áp: `ARCH_DEFAULTS[arch]` -> kwargs người dùng -> TRẦN của profile.
    Trần đứng cuối cùng vì nó là ràng buộc hạ tầng (RAM của Streamlit Cloud),
    không phải một gợi ý mà lời gọi hàm được phép ghi đè.
    """
    from luca_quant.config.profiles import active_profile

    if arch not in SEQ_ARCHS:
        raise ValueError(f"arch không hỗ trợ: {arch!r}. Có: {SEQ_ARCHS}")
    params = dict(ARCH_DEFAULTS.get(arch, {}))
    params.update(kwargs)
    params = active_profile().clamp_model_kwargs(params)
    return TorchSequenceClassifier(arch=arch, **params)
