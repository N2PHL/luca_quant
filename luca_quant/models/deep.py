"""
Deep Learning models (Blueprint Phase 6) — bọc trong sklearn API để
Experiment Runner không cần biết mô hình bên trong là gì.

Chỉ import khi có torch. Nếu môi trường Streamlit Cloud không có torch,
registry tự bỏ qua nhóm này.

Lưu ý về cửa sổ trượt: LSTM/CNN/Transformer cần chuỗi (n, lookback, n_feat).
Việc dựng cửa sổ được làm BÊN TRONG fit/predict và chỉ dùng dữ liệu của
chính tập đó — không bao giờ ghép cửa sổ vắt qua ranh giới train/test
(đây là một nguồn leakage rất phổ biến trong các đồ án DL tài chính).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _make_windows(X: np.ndarray, y: np.ndarray | None, lookback: int):
    n = len(X)
    if n <= lookback:
        raise ValueError(f"Cần > {lookback} mẫu để tạo cửa sổ, chỉ có {n}.")
    idx = np.arange(lookback, n)
    Xw = np.stack([X[i - lookback:i] for i in idx])
    yw = y[idx] if y is not None else None
    return Xw, yw, idx


class TorchSequenceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, arch="lstm", lookback=20, hidden=32, epochs=30,
                 lr=1e-3, batch_size=64, dropout=0.2, random_state=42):
        self.arch = arch
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        h, d = self.hidden, self.dropout

        class Net(nn.Module):
            def __init__(self, arch):
                super().__init__()
                self.arch = arch
                if arch == "lstm":
                    self.rnn = nn.LSTM(n_features, h, batch_first=True)
                elif arch == "gru":
                    self.rnn = nn.GRU(n_features, h, batch_first=True)
                elif arch == "cnn1d":
                    self.conv = nn.Sequential(
                        nn.Conv1d(n_features, h, 3, padding=1), nn.ReLU(),
                        nn.Conv1d(h, h, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1),
                    )
                else:  # transformer
                    self.proj = nn.Linear(n_features, h)
                    layer = nn.TransformerEncoderLayer(
                        d_model=h, nhead=4, dim_feedforward=h * 2,
                        dropout=d, batch_first=True)
                    self.enc = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(nn.Dropout(d), nn.Linear(h, 1))

            def forward(self, x):
                if self.arch in ("lstm", "gru"):
                    out, _ = self.rnn(x)
                    z = out[:, -1, :]
                elif self.arch == "cnn1d":
                    z = self.conv(x.transpose(1, 2)).squeeze(-1)
                else:
                    z = self.enc(self.proj(x))[:, -1, :]
                return self.head(z).squeeze(-1)

        return Net(self.arch)

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        Xw, yw, _ = _make_windows(Xa, ya, self.lookback)

        self.classes_ = np.array([0, 1])
        self.model_ = self._build(Xa.shape[1])
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        # pos_weight xử lý lệch lớp (label tài chính hiếm khi cân bằng 50/50)
        pos = max(yw.sum(), 1.0)
        neg = max(len(yw) - pos, 1.0)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32))

        Xt = torch.tensor(Xw)
        yt = torch.tensor(yw)
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = lossf(self.model_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch

        Xa = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        n = len(Xa)
        p = np.full(n, 0.5)
        if n > self.lookback:
            Xw, _, idx = _make_windows(Xa, None, self.lookback)
            with torch.no_grad():
                logits = self.model_(torch.tensor(Xw)).numpy()
            p[idx] = 1 / (1 + np.exp(-logits))
        # Các phiên đầu chưa đủ lookback -> 0.5 (không có quan điểm), KHÔNG
        # được lấp bằng giá trị tương lai.
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def make_torch_model(arch: str, **kwargs) -> TorchSequenceClassifier:
    return TorchSequenceClassifier(arch=arch, **kwargs)
