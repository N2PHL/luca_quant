"""
Model Acceptance Gate (Blueprint §16).

Chuyển checklist thành mã chạy được. Mọi ngưỡng lấy từ AcceptanceConfig,
không hard-code.

Gate được thiết kế BẢO THỦ có chủ đích: mặc định là REJECT, mô hình phải
vượt qua từng cổng mới được ACCEPT. Điều kiện "Sharpe >= 1.80" mà đề bài
yêu cầu chỉ là MỘT trong tám cổng — vì một mình nó rất dễ đạt được bằng
cách dò tìm dữ liệu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from luca_quant.config.settings import AcceptanceConfig
from luca_quant.evaluation.statistical_tests import (bootstrap_sharpe_ci,
                                                     deflated_sharpe_ratio,
                                                     paired_test_vs_benchmark)


@dataclass
class GateResult:
    decision: str                      # ACCEPT | ACCEPT WITH CONDITIONS | REJECT
    checks: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "FAIL")

    @property
    def n_warned(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "WARN")


class AcceptanceGate:
    def __init__(self, config: Optional[AcceptanceConfig] = None):
        self.cfg = config or AcceptanceConfig()

    def _check(self, res: GateResult, name: str, value, target, ok: bool,
               blocking: bool = True, note: str = "") -> None:
        res.checks.append({
            "gate": name,
            "value": round(float(value), 4) if isinstance(value, (int, float, np.floating))
                     and np.isfinite(value) else value,
            "target": target,
            "status": "PASS" if ok else ("FAIL" if blocking else "WARN"),
            "note": note,
        })

    def evaluate(
        self,
        trading_metrics: Dict[str, float],
        oos_returns: pd.Series,
        leakage_passed: bool,
        n_trials: int = 1,
        trial_sharpes: Optional[List[float]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        fold_metrics: Optional[pd.DataFrame] = None,
    ) -> GateResult:
        res = GateResult(decision="REJECT")
        c = self.cfg

        # --- Cổng 1: Leakage ---------------------------------------------
        self._check(res, "Leakage check", "CLEAN" if leakage_passed else "DETECTED",
                    "CLEAN", leakage_passed or not c.require_leakage_clean,
                    note="Không đạt cổng này thì mọi chỉ số phía sau đều vô nghĩa.")

        # --- Cổng 2-5: Ngưỡng hiệu năng OOS ------------------------------
        sharpe = trading_metrics.get("Sharpe", np.nan)
        sortino = trading_metrics.get("Sortino", np.nan)
        mdd = abs(trading_metrics.get("Max Drawdown", np.nan))
        pf = trading_metrics.get("Profit Factor", np.nan)

        self._check(res, "OOS Sharpe", sharpe, f">= {c.min_oos_sharpe}",
                    np.isfinite(sharpe) and sharpe >= c.min_oos_sharpe)
        self._check(res, "Sortino", sortino, f">= {c.min_sortino}",
                    np.isfinite(sortino) and sortino >= c.min_sortino)
        self._check(res, "Max Drawdown", mdd, f"<= {c.max_drawdown:.0%}",
                    np.isfinite(mdd) and mdd <= c.max_drawdown)
        self._check(res, "Profit Factor", pf, f">= {c.min_profit_factor}",
                    np.isfinite(pf) and pf >= c.min_profit_factor)

        # --- Cổng 6: Đa kiểm định ----------------------------------------
        d = deflated_sharpe_ratio(oos_returns, n_trials=n_trials, trial_sharpes=trial_sharpes)
        self._check(
            res, "Deflated Sharpe Ratio", d["DSR"], f">= {c.min_deflated_sharpe_pvalue}",
            np.isfinite(d["DSR"]) and d["DSR"] >= c.min_deflated_sharpe_pvalue,
            note=(f"Đã thử {n_trials} chiến lược; ngưỡng Sharpe do may rủi "
                  f"là {d['SR_threshold']:.2f}."),
        )

        # --- Cổng 7: Khoảng tin cậy --------------------------------------
        ci = bootstrap_sharpe_ci(oos_returns, n_boot=1500)
        self._check(res, "Sharpe CI dưới > 0", ci["ci_low"], "> 0",
                    np.isfinite(ci["ci_low"]) and ci["ci_low"] > 0,
                    note=f"CI 95%: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")

        # --- Cổng 8: Vượt Buy & Hold -------------------------------------
        if benchmark_returns is not None and c.require_beat_buyhold:
            t = paired_test_vs_benchmark(oos_returns, benchmark_returns)
            ok = np.isfinite(t["p_value"]) and t["p_value"] < 0.05 and t["mean_diff_ann"] > 0
            self._check(res, "Vượt Buy & Hold (p<0.05)", t["p_value"], "< 0.05", ok,
                        note=f"Chênh lệch năm hoá {t['mean_diff_ann']:.2%}, t={t['t_stat']:.2f}")

        # --- Cổng 9: Ổn định qua các fold (cảnh báo, không chặn) ---------
        if fold_metrics is not None and not fold_metrics.empty and "Sharpe" in fold_metrics:
            s = fold_metrics["Sharpe"].dropna()
            if len(s) >= 3:
                pos = float((s > 0).mean())
                self._check(res, "Tỷ lệ fold có Sharpe > 0", pos, ">= 0.6",
                            pos >= 0.6, blocking=False,
                            note=f"Sharpe theo fold: {[round(v,2) for v in s.tolist()]}")
                self._check(res, "Độ phân tán Sharpe giữa các fold", float(s.std()), "<= 1.5",
                            float(s.std()) <= 1.5, blocking=False,
                            note="Phân tán lớn = hiệu năng phụ thuộc chế độ thị trường.")

        # --- Kết luận -----------------------------------------------------
        if res.n_failed == 0 and res.n_warned == 0:
            res.decision = "ACCEPT"
        elif res.n_failed == 0:
            res.decision = "ACCEPT WITH CONDITIONS"
        else:
            res.decision = "REJECT"
            res.notes.append(
                "Không đạt: " + ", ".join(c["gate"] for c in res.checks if c["status"] == "FAIL")
            )
        return res
