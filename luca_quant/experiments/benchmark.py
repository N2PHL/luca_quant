"""
Model Benchmark (Blueprint §11).

Bảng so sánh bắt buộc phải có BASELINE ở trên cùng. Không có Buy & Hold
trong bảng thì "Sharpe = 1.9" là con số không diễn giải được: nếu Buy & Hold
cùng giai đoạn đạt 1.7 thì mô hình chỉ đóng góp 0.2 và không bù nổi chi phí
giao dịch cùng rủi ro mô hình.

Bảng gồm hai khối tách bạch (Blueprint §19):
  - ML Metrics    : Accuracy, AUC, Brier
  - Trading Metrics: CAGR, Sharpe, Sortino, MDD, PF, Turnover
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from luca_quant.backtest.engine import BacktestEngine
from luca_quant.config.profiles import release_memory
from luca_quant.config.settings import Settings
from luca_quant.evaluation.metrics import MetricsEngine
from luca_quant.evaluation.statistical_tests import paired_test_vs_benchmark
from luca_quant.experiments.runner import ExperimentResult, ExperimentRunner
from luca_quant.models import registry as model_registry

# Lineup mặc định: MỘT đại diện cho mỗi họ, theo đúng thứ tự thang bậc
# phức tạp. Đọc bảng từ trên xuống phải trả lời được: "thêm phức tạp có
# đổi lấy hiệu năng không?" Nếu `lstm` không vượt `momentum_rule` thì toàn
# bộ phần deep learning của báo cáo là một kết quả ÂM — và đó vẫn là kết
# quả nghiên cứu hợp lệ, phải báo cáo chứ không được giấu.
#
# Không đưa cả 34 mô hình vào mặc định: mỗi mô hình chạy thêm là một lần
# thử, và số lần thử đi thẳng vào `n_trials` của Deflated Sharpe Ratio.
# Chạy bừa 34 mô hình rồi báo cáo cái tốt nhất là data snooping.
DEFAULT_LINEUP = ["buy_hold", "momentum_rule", "random",       # baseline
                  "logistic", "elasticnet",                     # classical
                  "arima", "garch_vol",                         # econometric
                  "random_forest", "hist_gb", "lightgbm",       # ml
                  "lstm", "gru", "cnn1d", "transformer"]        # deep (nếu có torch)


class BenchmarkEngine:
    def __init__(self, runner: Optional[ExperimentRunner] = None,
                 settings: Optional[Settings] = None):
        self.s = settings or Settings()
        self.runner = runner or ExperimentRunner(self.s)
        self.metrics = MetricsEngine(risk_free_rate=self.s.risk_free_rate)
        self.bt = BacktestEngine(cost=self.s.cost)

    def run(
        self,
        prices: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        models: Optional[Sequence[str]] = None,
        feature_groups: Optional[Sequence[str]] = None,
        horizon: int = 1,
        progress=None,
    ) -> Dict:
        available = set(model_registry.available_models())
        lineup = [m for m in (models or DEFAULT_LINEUP) if m in available]
        skipped = [m for m in (models or DEFAULT_LINEUP) if m not in available]

        results: List[ExperimentResult] = []
        rows: List[Dict] = []

        for i, name in enumerate(lineup):
            if progress:
                progress(i / len(lineup), f"Benchmark {i+1}/{len(lineup)}: {name}")
            try:
                res = self.runner.run(prices=prices, X=X, y=y, model_name=name,
                                      feature_groups=feature_groups, horizon=horizon,
                                      name=name)
            except Exception as exc:                       # noqa: BLE001
                rows.append({"Experiment": name, "Model": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            finally:
                # Mô hình chuỗi để lại vài chục MB sau mỗi lần chạy. Với lineup
                # 9-14 mô hình trên hạ tầng 1GB, không dọn là chạm trần ở
                # khoảng mô hình thứ tám và app bị kill giữa chừng.
                release_memory()
            results.append(res)
            row = res.summary_row()
            row["Family"] = model_registry.get_spec(name).family
            rows.append(row)

        table = pd.DataFrame(rows)

        # --- So sánh có kiểm định với Buy & Hold ------------------------
        bh = next((r for r in results if r.model_name == "buy_hold"), None)
        if bh is not None and len(results) > 1:
            tests = []
            for r in results:
                t = paired_test_vs_benchmark(r.oos_returns, bh.oos_returns)
                tests.append({
                    "Experiment": r.name,
                    "Excess vs B&H (ann.)": t["mean_diff_ann"],
                    "t-stat": t["t_stat"],
                    "p-value": t["p_value"],
                })
            table = table.merge(pd.DataFrame(tests), on="Experiment", how="left")

        front = ["Experiment", "Family", "Model", "Accuracy", "AUC", "Brier",
                 "CAGR", "Sharpe", "Sortino", "Calmar", "Max Drawdown",
                 "Profit Factor", "Win Rate", "Turnover (ann.)", "Exposure",
                 "Excess vs B&H (ann.)", "t-stat", "p-value"]
        cols = [c for c in front if c in table.columns] + \
               [c for c in table.columns if c not in front]
        return {"table": table[cols], "results": results, "skipped": skipped}
