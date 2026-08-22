"""
ResearchService — Application Layer (Blueprint §1).

Nguyên tắc kiến trúc: STREAMLIT KHÔNG ĐƯỢC CHỨA LOGIC QUANT.
Các trang UI chỉ được gọi service này. Nhờ vậy toàn bộ nghiên cứu có thể
chạy trong notebook, trong CI, hoặc từ dòng lệnh mà không cần Streamlit.

Service chạy đúng pipeline chuẩn của Blueprint §5:
  PROBLEM → DATA → FEATURE → LABEL → SPLIT → BASELINE → MODEL SELECTION
  → VALIDATION → BACKTEST → RISK → ROBUSTNESS → ABLATION → INVESTMENT THESIS
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from luca_quant.config.settings import Settings
from luca_quant.data.providers.base import DataProvider
from luca_quant.data.providers.dnse import DNSEProvider
from luca_quant.data.schemas import validate_ohlcv
from luca_quant.evaluation.acceptance_gate import AcceptanceGate
from luca_quant.evaluation.metrics import MetricsEngine
from luca_quant.experiments.ablation import AblationEngine
from luca_quant.experiments.benchmark import BenchmarkEngine
from luca_quant.experiments.runner import ExperimentRunner
from luca_quant.explainability.investment_thesis import InvestmentThesisEngine
from luca_quant.features.pipeline import FeaturePipeline, build_common_matrix
from luca_quant.labels.registry import make_label
from luca_quant.risk.overlay import standard_arms
from luca_quant.validation.leakage import LeakageDetector


@dataclass
class ResearchSession:
    """Toàn bộ output của một phiên nghiên cứu — có thể serialize để báo cáo."""
    ticker: str
    prices: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    pipeline: FeaturePipeline
    horizon: int
    data_quality: dict = field(default_factory=dict)
    leakage: object = None
    benchmark: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    overlay_arms: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: object = None
    gate: object = None
    thesis: str = ""
    n_trials: int = 0


class ResearchService:
    def __init__(self, settings: Optional[Settings] = None,
                 provider: Optional[DataProvider] = None):
        self.s = settings or Settings()
        self.provider = provider or DNSEProvider()
        self.runner = ExperimentRunner(self.s)
        self.metrics = MetricsEngine(risk_free_rate=self.s.risk_free_rate)

    # ------------------------------------------------------------------
    def load_data(self, ticker: str, years: int = 8) -> tuple[pd.DataFrame, dict]:
        end = datetime.now()
        start = end - timedelta(days=int(365.25 * years))
        df = self.provider.get_ohlcv(ticker, start, end)
        return df, validate_ohlcv(df)

    # ------------------------------------------------------------------
    def prepare(
        self,
        prices: pd.DataFrame,
        groups: Sequence[str],
        label_name: str = "direction_vol_adj",
        label_kwargs: Optional[dict] = None,
        run_leakage_check: bool = True,
    ) -> ResearchSession:
        label_kwargs = label_kwargs or {"horizon": 5, "k": 0.0}
        spec = make_label(label_name, prices, **label_kwargs)

        X, y, pipe = build_common_matrix(prices, groups, spec.series)

        session = ResearchSession(
            ticker="", prices=prices.loc[X.index], X=X, y=y,
            pipeline=pipe, horizon=spec.horizon,
        )

        if run_leakage_check:
            det = LeakageDetector(horizon=spec.horizon)
            fp = FeaturePipeline(list(groups))
            session.leakage = det.run_all(
                X=X, y=y,
                raw_df=prices,
                feature_fn=lambda d: fp.build(d),
                threshold_tuned_on="valid",
            )
        return session

    # ------------------------------------------------------------------
    def run_full_study(
        self,
        ticker: str,
        groups: Sequence[str],
        models: Optional[Sequence[str]] = None,
        primary_model: str = "lightgbm",
        ablation_mode: str = "cumulative",
        label_name: str = "direction_vol_adj",
        label_kwargs: Optional[dict] = None,
        years: int = 8,
        prices: Optional[pd.DataFrame] = None,
        progress: Optional[Callable] = None,
    ) -> ResearchSession:
        def step(p, msg):
            if progress:
                progress(p, msg)

        # 1. DATA -------------------------------------------------------
        step(0.02, "Tải dữ liệu…")
        if prices is None:
            prices, quality = self.load_data(ticker, years)
        else:
            quality = validate_ohlcv(prices)
        if prices.empty:
            raise ValueError(f"Không lấy được dữ liệu cho {ticker}.")

        # 2. FEATURE + LABEL + kiểm tra leakage -------------------------
        step(0.08, "Sinh feature, gán nhãn, kiểm tra rò rỉ dữ liệu…")
        session = self.prepare(prices, groups, label_name, label_kwargs)
        session.ticker = ticker
        session.data_quality = quality

        if session.leakage is not None and not session.leakage.passed:
            # Dừng sớm: chạy tiếp khi đã phát hiện leakage chỉ tạo ra
            # những con số đẹp nhưng vô giá trị.
            session.thesis = (
                "DỪNG NGHIÊN CỨU: phát hiện rò rỉ dữ liệu.\n"
                + session.leakage.to_frame().to_string(index=False)
            )
            return session

        # 3. BASELINE + MODEL SELECTION ---------------------------------
        step(0.15, "Chạy benchmark mô hình (gồm baseline bắt buộc)…")
        bench = BenchmarkEngine(self.runner, self.s)
        session.benchmark = bench.run(
            session.prices, session.X, session.y, models=models,
            feature_groups=list(groups), horizon=session.horizon,
            progress=lambda p, m: step(0.15 + p * 0.30, m),
        )

        # 4. ABLATION ---------------------------------------------------
        step(0.48, "Chạy Ablation Study…")
        abl = AblationEngine(self.runner, self.s)
        session.ablation = abl.run(
            session.prices, session.X, session.y, session.pipeline,
            model_name=primary_model, groups=list(groups), mode=ablation_mode,
            horizon=session.horizon,
            progress=lambda p, m: step(0.48 + p * 0.32, m),
        )

        # 5. QUY KẾT NGUỒN GỐC SHARPE (overlay arms) --------------------
        step(0.82, "Bóc tách đóng góp của overlay Hurst/MACD…")
        session.overlay_arms = self._run_overlay_arms(session, primary_model, list(groups))

        # 6. Chọn cấu hình tốt nhất theo Sharpe -------------------------
        all_results = list(session.benchmark.get("results", [])) + \
                      list(session.ablation.get("results", []))
        scored = [r for r in all_results
                  if np.isfinite(r.trading_metrics.get("Sharpe", np.nan))
                  and r.model_name != "buy_hold"]
        session.n_trials = len(all_results) + len(session.overlay_arms)
        if not scored:
            session.thesis = "Không có cấu hình nào cho kết quả hợp lệ."
            return session
        best = max(scored, key=lambda r: r.trading_metrics["Sharpe"])
        session.best = best

        # 7. ACCEPTANCE GATE --------------------------------------------
        step(0.90, "Chạy Acceptance Gate…")
        bh = next((r for r in session.benchmark.get("results", [])
                   if r.model_name == "buy_hold"), None)
        trial_sharpes = [r.trading_metrics.get("Sharpe") for r in all_results
                         if np.isfinite(r.trading_metrics.get("Sharpe", np.nan))]
        session.gate = AcceptanceGate(self.s.acceptance).evaluate(
            trading_metrics=best.trading_metrics,
            oos_returns=best.oos_returns,
            leakage_passed=bool(session.leakage.passed) if session.leakage else False,
            n_trials=session.n_trials,
            trial_sharpes=trial_sharpes,
            benchmark_returns=bh.oos_returns if bh is not None else None,
            fold_metrics=best.fold_metrics,
        )

        # 8. INVESTMENT THESIS ------------------------------------------
        step(0.96, "Sinh Investment Thesis…")
        session.thesis = InvestmentThesisEngine().generate(
            ticker=ticker,
            ablation_table=session.ablation.get("table", pd.DataFrame()),
            contribution=session.ablation.get("contribution", pd.DataFrame()),
            benchmark_table=session.benchmark.get("table", pd.DataFrame()),
            best_metrics=best.trading_metrics,
            gate_result=session.gate,
            leakage_result=session.leakage,
            n_trials=session.n_trials,
            fold_metrics=best.fold_metrics,
        )
        step(1.0, "Hoàn tất.")
        return session

    # ------------------------------------------------------------------
    def _run_overlay_arms(self, session: ResearchSession, model: str,
                          groups: List[str]) -> pd.DataFrame:
        """
        Bốn cánh tay: AI / AI+MACD / AI+Hurst / AI+cả hai.

        Đây là câu trả lời cho câu hỏi trung tâm: Sharpe đến từ mô hình học máy
        hay từ quy tắc kỹ thuật? Repo cũ không thể trả lời vì overlay được nhúng
        cứng bên trong RiskManager.
        """
        rows = []
        for label, stack in standard_arms().items():
            try:
                res = self.runner.run(
                    session.prices, session.X, session.y, model_name=model,
                    feature_groups=groups, overlay=stack, horizon=session.horizon,
                    name=label,
                )
                row = {"Arm": label}
                row.update({k: res.trading_metrics.get(k) for k in
                            ("Sharpe", "CAGR", "Max Drawdown", "Profit Factor",
                             "Turnover (ann.)", "Exposure")})
                rows.append(row)
            except Exception as exc:                       # noqa: BLE001
                rows.append({"Arm": label, "Sharpe": np.nan,
                             "error": f"{type(exc).__name__}: {exc}"})
        df = pd.DataFrame(rows)
        if "Sharpe" in df.columns and df["Sharpe"].notna().any():
            base = df.loc[df["Arm"] == "AI only", "Sharpe"]
            if len(base):
                df["Δ Sharpe vs AI only"] = df["Sharpe"] - base.iloc[0]
        return df
