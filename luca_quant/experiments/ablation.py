"""
Ablation Engine (Blueprint §12).

Repo cũ có 4 kịch bản cứng và — như đã nêu trong features/pipeline.py —
chúng chạy trên các INDEX KHÁC NHAU nên không so sánh được với nhau.

Engine này:
  1. Dựng ma trận feature MỘT LẦN trên index chung (mọi kịch bản cùng
     giai đoạn thị trường, cùng số phiên test).
  2. Chạy toàn bộ tổ hợp (2^k - 1) hoặc chuỗi cộng dồn tuỳ chọn.
  3. Tính Δ metric so với baseline VÀ hiệu chỉnh đa kiểm định bằng
     Deflated Sharpe Ratio — vì chạy 63 tổ hợp rồi lấy cái tốt nhất
     là dò tìm dữ liệu nếu không hiệu chỉnh.
  4. Xếp hạng đóng góp biên của từng nhóm feature (marginal contribution
     kiểu Shapley xấp xỉ: trung bình mức tăng Sharpe khi thêm nhóm đó vào
     tất cả các tập con không chứa nó).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from luca_quant.config.settings import Settings
from luca_quant.evaluation.statistical_tests import (bootstrap_sharpe_ci,
                                                     deflated_sharpe_ratio)
from luca_quant.experiments.runner import ExperimentResult, ExperimentRunner
from luca_quant.features.pipeline import FeaturePipeline


class AblationEngine:
    def __init__(self, runner: Optional[ExperimentRunner] = None,
                 settings: Optional[Settings] = None):
        self.s = settings or Settings()
        self.runner = runner or ExperimentRunner(self.s)

    # ------------------------------------------------------------------
    @staticmethod
    def build_scenarios(groups: Sequence[str], mode: str = "cumulative",
                        max_combos: int = 64) -> List[List[str]]:
        """
        mode="cumulative" : [A], [A,B], [A,B,C], ...   (k kịch bản, nhanh)
        mode="powerset"   : mọi tập con khác rỗng      (2^k - 1 kịch bản)
        mode="leave_one_out": full, và full trừ từng nhóm (k+1 kịch bản)
        """
        g = list(groups)
        if mode == "cumulative":
            return [g[: i + 1] for i in range(len(g))]
        if mode == "leave_one_out":
            return [g] + [[x for x in g if x != drop] for drop in g]
        subsets = []
        for r in range(1, len(g) + 1):
            subsets.extend([list(c) for c in combinations(g, r)])
        if len(subsets) > max_combos:
            raise ValueError(
                f"{len(subsets)} tổ hợp vượt giới hạn {max_combos}. "
                "Giảm số nhóm feature hoặc dùng mode='cumulative'."
            )
        return subsets

    # ------------------------------------------------------------------
    def run(
        self,
        prices: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: FeaturePipeline,
        model_name: str = "lightgbm",
        groups: Optional[Sequence[str]] = None,
        mode: str = "cumulative",
        horizon: int = 1,
        progress=None,
    ) -> Dict:
        groups = list(groups or pipeline.groups)
        scenarios = self.build_scenarios(groups, mode)

        results: List[ExperimentResult] = []
        rows: List[Dict] = []

        for i, subset in enumerate(scenarios):
            label = "+".join(subset)
            if progress:
                progress(i / len(scenarios), f"Ablation {i+1}/{len(scenarios)}: {label}")
            cols = pipeline.columns_for(subset)
            try:
                res = self.runner.run(
                    prices=prices, X=X, y=y, model_name=model_name,
                    feature_groups=subset, columns=cols, horizon=horizon,
                    name=label,
                )
            except Exception as exc:                       # noqa: BLE001
                # Ghi nhận lỗi vào bảng thay vì nuốt im lặng như repo cũ
                rows.append({"Experiment": label, "n_features": len(cols),
                             "Sharpe": np.nan, "error": f"{type(exc).__name__}: {exc}"})
                continue

            results.append(res)
            row = res.summary_row()
            row["n_features"] = len(cols)
            row["error"] = "; ".join(res.errors) if res.errors else ""
            rows.append(row)

        table = pd.DataFrame(rows)
        if table.empty or "Sharpe" not in table.columns:
            return {"table": table, "results": results, "contribution": pd.DataFrame()}

        # --- Δ so với kịch bản đầu tiên (baseline) -----------------------
        base_sharpe = table["Sharpe"].iloc[0]
        for m in ("Sharpe", "CAGR", "Sortino", "Max Drawdown", "Profit Factor"):
            if m in table.columns:
                table[f"Δ {m}"] = table[m] - table[m].iloc[0]

        # --- Hiệu chỉnh đa kiểm định ------------------------------------
        trial_sharpes = table["Sharpe"].dropna().tolist()
        n_trials = len(trial_sharpes)
        dsr_rows = []
        for res in results:
            d = deflated_sharpe_ratio(res.oos_returns, n_trials=n_trials,
                                      trial_sharpes=trial_sharpes)
            ci = bootstrap_sharpe_ci(res.oos_returns, n_boot=800)
            dsr_rows.append({
                "Experiment": res.name,
                "DSR": d["DSR"],
                "SR threshold (multiple testing)": d["SR_threshold"],
                "Sharpe CI low": ci["ci_low"],
                "Sharpe CI high": ci["ci_high"],
            })
        if dsr_rows:
            table = table.merge(pd.DataFrame(dsr_rows), on="Experiment", how="left")

        return {
            "table": table,
            "results": results,
            "contribution": self._marginal_contribution(table, groups, mode),
            "n_trials": n_trials,
            "baseline_sharpe": base_sharpe,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _marginal_contribution(table: pd.DataFrame, groups: List[str], mode: str) -> pd.DataFrame:
        """
        Đóng góp biên của từng nhóm feature.

        powerset -> xấp xỉ Shapley: trung bình Δ Sharpe khi thêm nhóm g vào
                    mọi tập con không chứa g.
        cumulative -> Δ Sharpe của bước thêm nhóm đó.
        """
        if "Sharpe" not in table.columns:
            return pd.DataFrame()

        sharpe_by_set = {
            frozenset(str(r["Experiment"]).split("+")): r["Sharpe"]
            for _, r in table.iterrows() if pd.notna(r.get("Sharpe"))
        }

        rows = []
        for g in groups:
            deltas = []
            for s, sh in sharpe_by_set.items():
                if g in s:
                    without = s - {g}
                    if without and without in sharpe_by_set:
                        deltas.append(sh - sharpe_by_set[without])
            if deltas:
                rows.append({
                    "feature_group": g,
                    "mean Δ Sharpe": float(np.mean(deltas)),
                    "median Δ Sharpe": float(np.median(deltas)),
                    "n_comparisons": len(deltas),
                    "always_positive": bool(np.all(np.array(deltas) > 0)),
                })
        return (pd.DataFrame(rows).sort_values("mean Δ Sharpe", ascending=False)
                .reset_index(drop=True)) if rows else pd.DataFrame()
