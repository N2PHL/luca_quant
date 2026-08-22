"""
MetricsEngine — NGUỒN CHÂN LÝ DUY NHẤT cho mọi chỉ số (Blueprint §15).

Repo cũ có HAI công thức Sharpe khác nhau cho cùng một chiến lược:

  core/metrics.py     : sharpe = CAGR_hình_học / vol_năm_hoá
  core/alpha_engine.py: sharpe = mean(daily) / std(daily) * sqrt(252)

Hai công thức này cho kết quả lệch nhau 10–30% trên cùng chuỗi lợi suất
(tử số một bên là trung bình hình học, một bên là trung bình số học).
Cả hai đều BỎ QUA lãi suất phi rủi ro. Với rf ≈ 4.5%/năm ở Việt Nam,
bỏ qua rf làm Sharpe bị thổi lên khoảng +0.15 đến +0.30.

Ở đây chỉ còn một định nghĩa:

    Sharpe = mean(r_t - rf_daily) / std(r_t - rf_daily) * sqrt(252)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class MetricsEngine:
    def __init__(self, risk_free_rate: float = 0.045, periods_per_year: int = TRADING_DAYS):
        self.rf = risk_free_rate
        self.ppy = periods_per_year

    @property
    def rf_daily(self) -> float:
        return (1 + self.rf) ** (1 / self.ppy) - 1

    # ---------------- Trading / Risk metrics ----------------
    def compute(self, returns: pd.Series, positions: pd.Series | None = None) -> Dict[str, float]:
        r = pd.Series(returns).dropna().astype(float)
        if len(r) < 2:
            return {k: np.nan for k in self.metric_names()}

        excess = r - self.rf_daily
        equity = (1 + r).cumprod()
        n = len(r)
        years = n / self.ppy

        total_return = float(equity.iloc[-1] - 1)
        cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else np.nan
        vol = float(r.std(ddof=1) * np.sqrt(self.ppy))

        sd = excess.std(ddof=1)
        sharpe = float(excess.mean() / sd * np.sqrt(self.ppy)) if sd > 0 else np.nan

        downside = excess[excess < 0]
        dsd = downside.std(ddof=1) if len(downside) > 1 else np.nan
        sortino = float(excess.mean() / dsd * np.sqrt(self.ppy)) if dsd and dsd > 0 else np.nan

        dd = equity / equity.cummax() - 1
        mdd = float(dd.min())
        calmar = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else np.nan

        wins, losses = r[r > 0], r[r < 0]
        active = r[r != 0]
        win_rate = float(len(wins) / len(active)) if len(active) else np.nan
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf

        out = {
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "Max Drawdown": mdd,
            "Win Rate": win_rate,
            "Profit Factor": pf,
            "Skew": float(r.skew()),
            "Kurtosis": float(r.kurtosis()),
            "VaR 95% (daily)": float(r.quantile(0.05)),
            "CVaR 95% (daily)": float(r[r <= r.quantile(0.05)].mean()),
            "Best Day": float(r.max()),
            "Worst Day": float(r.min()),
            "N Periods": float(n),
        }

        if positions is not None:
            p = pd.Series(positions).reindex(r.index).fillna(0.0)
            turn = p.diff().abs().fillna(0.0)
            out["Turnover (ann.)"] = float(turn.sum() / years) if years > 0 else np.nan
            out["Exposure"] = float((p != 0).mean())
            out["Avg Position"] = float(p.mean())
            out["N Trades"] = float((p.diff().fillna(0) != 0).sum())
        return out

    @staticmethod
    def metric_names() -> list:
        return ["Total Return", "CAGR", "Volatility", "Sharpe", "Sortino", "Calmar",
                "Max Drawdown", "Win Rate", "Profit Factor", "Skew", "Kurtosis",
                "VaR 95% (daily)", "CVaR 95% (daily)", "Best Day", "Worst Day", "N Periods"]

    # ---------------- ML metrics ----------------
    @staticmethod
    def ml_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Chỉ số học máy — TÁCH BIỆT khỏi chỉ số giao dịch (Blueprint §19).

        Accuracy cao không đồng nghĩa với Sharpe cao. Giữ hai nhóm chỉ số
        riêng để có thể chỉ ra chính xác điều đó trong phần biện luận.
        """
        from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                                     log_loss, precision_score, recall_score,
                                     roc_auc_score)
        y_true = np.asarray(y_true).astype(int)
        proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
        pred = (proba > threshold).astype(int)
        out: Dict[str, float] = {}
        try:
            out["Accuracy"] = float(accuracy_score(y_true, pred))
            out["Precision"] = float(precision_score(y_true, pred, zero_division=0))
            out["Recall"] = float(recall_score(y_true, pred, zero_division=0))
            out["F1"] = float(f1_score(y_true, pred, zero_division=0))
            out["AUC"] = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan
            out["LogLoss"] = float(log_loss(y_true, proba, labels=[0, 1]))
            # Brier score đo hiệu chuẩn xác suất — quan trọng vì position sizing
            # dùng trực tiếp giá trị xác suất, không chỉ dùng dấu của nó.
            out["Brier"] = float(brier_score_loss(y_true, proba))
            out["Base Rate"] = float(y_true.mean())
        except Exception:                                  # noqa: BLE001
            pass
        return out
