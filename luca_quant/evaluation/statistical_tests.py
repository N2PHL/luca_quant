"""
Kiểm định thống kê cho kết quả backtest.

VÌ SAO PHẦN NÀY LÀ BẮT BUỘC CHO ĐỒ ÁN NÀY
------------------------------------------
Blueprint §12 đề xuất chạy 2^6 - 1 = 63 tổ hợp feature, rồi §16 đặt cổng
"Sharpe >= 1.80". Ghép hai điều đó lại sẽ tạo ra một vấn đề thống kê rất
nghiêm trọng: nếu thử 63 chiến lược và báo cáo cái tốt nhất, thì ngay cả
khi KHÔNG có alpha nào, kỳ vọng Sharpe cực đại vẫn khoảng 0.8–1.2 thuần
do may rủi. Đạt 1.8 sau 63 lần thử không chứng minh được điều gì nếu không
hiệu chỉnh cho đa kiểm định.

Deflated Sharpe Ratio (Bailey & López de Prado, 2014) hiệu chỉnh cho:
  - số lần thử N
  - độ lệch (skew) và độ nhọn (kurtosis) của lợi suất
  - độ dài mẫu

Nếu người chấm hỏi "làm sao biết Sharpe 1.8 không phải do dò tìm?",
đây là câu trả lời có cơ sở toán học.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, rf_daily: float = 0.0, ppy: int = 252) -> float:
    r = pd.Series(returns).dropna().astype(float) - rf_daily
    if len(r) < 3 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


# --------------------------------------------------------------------------
def sharpe_standard_error(returns: pd.Series, sr: Optional[float] = None,
                          ppy: int = 252) -> float:
    """
    Sai số chuẩn của Sharpe có hiệu chỉnh moment bậc 3, 4 (Lo, 2002).

    Với lợi suất tài chính lệch trái và đuôi dày, công thức ngây thơ
    sqrt((1+SR^2/2)/n) đánh giá thấp sai số khá nhiều.
    """
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    if n < 10:
        return np.nan
    sr = sr if sr is not None else sharpe_ratio(r, ppy=ppy)
    sr_d = sr / np.sqrt(ppy)                       # đưa về tần suất ngày
    g3, g4 = float(r.skew()), float(r.kurtosis()) + 3.0
    var = (1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2) / (n - 1)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(ppy))


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0,
                               ppy: int = 252) -> float:
    """
    PSR = P(SR thật > benchmark_sr) — xác suất Sharpe quan sát được không
    phải do ngẫu nhiên, có tính đến skew/kurtosis.
    """
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    if n < 10:
        return np.nan
    sr = sharpe_ratio(r, ppy=ppy)
    sr_d, bench_d = sr / np.sqrt(ppy), benchmark_sr / np.sqrt(ppy)
    g3, g4 = float(r.skew()), float(r.kurtosis()) + 3.0
    denom = np.sqrt(max(1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2, 1e-12))
    z = (sr_d - bench_d) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sharpes: Optional[Sequence[float]] = None,
                          ppy: int = 252) -> Dict[str, float]:
    """
    Deflated Sharpe Ratio.

    n_trials      : tổng số chiến lược đã thử (kể cả những cái bị loại bỏ!)
    trial_sharpes : Sharpe của tất cả các lần thử, dùng để ước lượng
                    phương sai giữa các thử nghiệm. Nếu không có, giả định var=1.

    DSR > 0.95 nghĩa là: sau khi trừ đi lợi thế do thử nhiều lần, vẫn còn
    95% khả năng Sharpe thật > 0.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 10 or n_trials < 1:
        return {"DSR": np.nan, "SR_threshold": np.nan, "n_trials": n_trials}

    var_trials = float(np.var(trial_sharpes, ddof=1)) if (
        trial_sharpes is not None and len(trial_sharpes) > 2) else 1.0

    e = np.euler_gamma
    n = max(n_trials, 2)
    # Kỳ vọng Sharpe cực đại của n phép thử độc lập không có kỹ năng
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * np.e))
    sr_threshold = np.sqrt(var_trials) * ((1 - e) * z1 + e * z2)

    dsr = probabilistic_sharpe_ratio(r, benchmark_sr=sr_threshold, ppy=ppy)
    return {
        "DSR": dsr,
        "SR_threshold": float(sr_threshold),
        "n_trials": int(n_trials),
        "observed_SR": sharpe_ratio(r, ppy=ppy),
    }


# --------------------------------------------------------------------------
def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 20,
                        alpha: float = 0.05, ppy: int = 252,
                        seed: int = 42) -> Dict[str, float]:
    """
    Khoảng tin cậy Sharpe bằng STATIONARY BLOCK BOOTSTRAP.

    Bootstrap thường (lấy mẫu từng ngày độc lập) phá vỡ tự tương quan và
    volatility clustering — hai tính chất luôn có trong lợi suất tài chính.
    Lấy mẫu theo khối độ dài ~20 phiên giữ lại cấu trúc đó.
    """
    r = pd.Series(returns).dropna().astype(float).to_numpy()
    n = len(r)
    if n < block * 3:
        return {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        boots[b] = sample.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0

    return {
        "sharpe": sharpe_ratio(pd.Series(r), ppy=ppy),
        "ci_low": float(np.quantile(boots, alpha / 2)),
        "ci_high": float(np.quantile(boots, 1 - alpha / 2)),
        "p_value": float((boots <= 0).mean()),      # H0: Sharpe <= 0
        "n_boot": n_boot,
    }


def paired_test_vs_benchmark(strategy: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    """
    Kiểm định chiến lược có thực sự hơn benchmark hay không.

    Không so hai con số Sharpe rời rạc mà kiểm định trên chuỗi CHÊNH LỆCH
    lợi suất theo cặp ngày, dùng Newey-West để xử lý tự tương quan.
    """
    a, b = strategy.align(benchmark, join="inner")
    d = (a - b).dropna()
    n = len(d)
    if n < 30:
        return {"mean_diff_ann": np.nan, "t_stat": np.nan, "p_value": np.nan, "n": n}

    mean = float(d.mean())
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    gamma0 = float(np.var(d, ddof=1))
    s = gamma0
    for k in range(1, max(lag, 1) + 1):
        cov = float(np.cov(d[:-k], d[k:])[0, 1])
        s += 2 * (1 - k / (lag + 1)) * cov
    se = np.sqrt(max(s, 1e-16) / n)
    t = mean / se if se > 0 else np.nan
    return {
        "mean_diff_ann": mean * 252,
        "t_stat": float(t),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
        "n": n,
    }


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """Hiệu chỉnh FDR cho bảng Ablation nhiều kịch bản."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    crit = (np.arange(1, m + 1) / m) * alpha
    passed = p[order] <= crit
    cutoff = np.max(np.where(passed)[0]) if passed.any() else -1
    reject = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        reject[order[:cutoff + 1]] = True
    return pd.DataFrame({"p_value": p, "bh_critical": crit[np.argsort(order)], "significant": reject})
