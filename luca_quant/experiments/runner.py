"""
ExperimentRunner — thay thế God Object `AIQuantPipeline` (Blueprint §4).

Repo cũ gom FeatureEngineer + LabelGenerator + Model + Scaler + Sizer +
RiskManager + AlphaEngine + Metrics vào một lớp. Ba lỗi cụ thể sinh ra từ đó:

  (a) `self.model` và `self.scaler` là STATE dùng chung, bị fit lại qua từng
      fold và qua từng kịch bản ablation. `feature_importance` trả về cuối
      cùng là của FOLD CUỐI của KỊCH BẢN CUỐI — không phải của mô hình được
      báo cáo. Biểu đồ Feature Importance trên UI đang mô tả sai mô hình.
  (b) Không có tập VALID, nên không có chỗ hợp lệ để chọn siêu tham số.
  (c) `run_ablation_study` bắt Exception rồi `print` — kịch bản lỗi biến mất
      im lặng khỏi bảng kết quả, người đọc tưởng nó không được chạy.

Ở đây mỗi fold tự tạo mô hình MỚI, kết quả từng fold được lưu riêng, và lỗi
được ghi nhận vào output thay vì nuốt mất.

LUỒNG MỘT FOLD
    TRAIN -> fit scaler, fit model
    VALID -> hiệu chuẩn xác suất + chọn ngưỡng sizing (chỉ ở đây!)
    TEST  -> đóng băng mọi thứ, chỉ predict -> size -> risk -> backtest
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from luca_quant.backtest.engine import BacktestEngine
from luca_quant.config.settings import Settings
from luca_quant.evaluation.metrics import MetricsEngine
from luca_quant.models import registry as model_registry
from luca_quant.portfolio.sizing import ProbabilitySizer, SizingParams
from luca_quant.risk.manager import RiskConstraints
from luca_quant.risk.overlay import OverlayStack
from luca_quant.validation.splits import Fold, PurgedWalkForward


@dataclass
class FoldResult:
    fold_id: int
    test_index: pd.Index
    proba: pd.Series
    position: pd.Series
    y_true: pd.Series
    ml_metrics: Dict[str, float]
    tuned_params: SizingParams
    tuning_grid: pd.DataFrame
    feature_importance: pd.DataFrame
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    name: str
    model_name: str
    feature_groups: List[str]
    overlay: str
    oos_returns: pd.Series
    oos_positions: pd.Series
    oos_proba: pd.Series
    oos_y: pd.Series
    backtest: object
    trading_metrics: Dict[str, float]
    ml_metrics: Dict[str, float]
    folds: List[FoldResult] = field(default_factory=list)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: List[str] = field(default_factory=list)

    def summary_row(self) -> Dict:
        row = {"Experiment": self.name, "Model": self.model_name,
               "Features": "+".join(self.feature_groups), "Overlay": self.overlay}
        row.update({k: v for k, v in self.ml_metrics.items()
                    if k in ("Accuracy", "AUC", "Brier", "Base Rate")})
        row.update({k: v for k, v in self.trading_metrics.items()
                    if k in ("CAGR", "Sharpe", "Sortino", "Calmar", "Max Drawdown",
                             "Profit Factor", "Win Rate", "Turnover (ann.)", "Exposure")})
        return row


class ExperimentRunner:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or Settings()
        self.metrics = MetricsEngine(risk_free_rate=self.s.risk_free_rate)
        self.bt = BacktestEngine(cost=self.s.cost)

    # ------------------------------------------------------------------
    def _valid_objective(self, prices: pd.DataFrame) -> Callable:
        """Hàm mục tiêu tune ngưỡng: Sharpe SAU CHI PHÍ trên tập VALID."""
        def obj(prices_valid: pd.DataFrame, weights: pd.Series) -> float:
            if weights.abs().sum() == 0:
                return -np.inf
            res = self.bt.run(prices_valid, weights, apply_settlement=True)
            m = self.metrics.compute(res.returns, res.positions)
            score = m.get(
                {"sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar"}
                .get(self.s.sizing.tune_objective, "Sharpe"), np.nan
            )
            return float(score) if np.isfinite(score) else -np.inf
        return obj

    # ------------------------------------------------------------------
    def run(
        self,
        prices: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "lightgbm",
        feature_groups: Optional[Sequence[str]] = None,
        columns: Optional[Sequence[str]] = None,
        overlay: Optional[OverlayStack] = None,
        horizon: int = 1,
        name: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ) -> ExperimentResult:
        overlay = overlay or OverlayStack([])
        cols = list(columns) if columns is not None else list(X.columns)
        Xs = X[cols]
        spec = model_registry.get_spec(model_name)

        cv = PurgedWalkForward(
            n_splits=self.s.split.n_splits,
            purge_days=max(self.s.split.purge_days, horizon),
            embargo_days=self.s.split.embargo_days,
            expanding=self.s.split.expanding,
            min_train_size=self.s.split.min_train_size,
        )
        risk = RiskConstraints(self.s.risk)

        folds: List[FoldResult] = []
        errors: List[str] = []
        proba_parts, pos_parts, y_parts = [], [], []

        for fold in cv.split(len(Xs)):
            try:
                fr = self._run_fold(fold, prices, Xs, y, spec, overlay, risk, model_kwargs or {})
                folds.append(fr)
                proba_parts.append(fr.proba)
                pos_parts.append(fr.position)
                y_parts.append(fr.y_true)
            except Exception as exc:                       # noqa: BLE001
                msg = f"Fold {fold.fold_id}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                folds.append(FoldResult(fold.fold_id, pd.Index([]), pd.Series(dtype=float),
                                        pd.Series(dtype=float), pd.Series(dtype=float),
                                        {}, SizingParams(), pd.DataFrame(), pd.DataFrame(), msg))

        if not proba_parts:
            raise RuntimeError(
                f"Toàn bộ fold đều lỗi cho '{name or model_name}'. Chi tiết: {errors}"
            )

        oos_proba = pd.concat(proba_parts).sort_index()
        oos_pos = pd.concat(pos_parts).sort_index()
        oos_y = pd.concat(y_parts).sort_index()

        bt_res = self.bt.run(prices.loc[oos_pos.index], oos_pos)
        trading = self.metrics.compute(bt_res.returns, bt_res.positions)
        ml = MetricsEngine.ml_metrics(oos_y.to_numpy(), oos_proba.to_numpy())

        fold_rows = []
        for fr in folds:
            if fr.error or len(fr.position) == 0:
                continue
            sub = self.bt.run(prices.loc[fr.position.index], fr.position)
            m = self.metrics.compute(sub.returns, sub.positions)
            fold_rows.append({
                "fold": fr.fold_id,
                "test_from": str(fr.test_index[0].date()),
                "test_to": str(fr.test_index[-1].date()),
                "n": len(fr.test_index),
                "entry_thr": fr.tuned_params.entry_threshold,
                "mode": fr.tuned_params.mode,
                "Sharpe": m.get("Sharpe"),
                "CAGR": m.get("CAGR"),
                "Max Drawdown": m.get("Max Drawdown"),
                "AUC": fr.ml_metrics.get("AUC"),
                "Accuracy": fr.ml_metrics.get("Accuracy"),
            })

        return ExperimentResult(
            name=name or f"{model_name}|{'+'.join(feature_groups or [])}|{overlay.label}",
            model_name=model_name,
            feature_groups=list(feature_groups or []),
            overlay=overlay.label,
            oos_returns=bt_res.returns,
            oos_positions=bt_res.positions,
            oos_proba=oos_proba,
            oos_y=oos_y,
            backtest=bt_res,
            trading_metrics=trading,
            ml_metrics=ml,
            folds=folds,
            fold_metrics=pd.DataFrame(fold_rows),
            errors=errors,
        )

    # ------------------------------------------------------------------
    def _run_fold(self, fold: Fold, prices, Xs, y, spec, overlay, risk, model_kwargs) -> FoldResult:
        idx = Xs.index
        tr, va, te = idx[fold.train_idx], idx[fold.valid_idx], idx[fold.test_idx]

        X_tr, X_va, X_te = Xs.loc[tr], Xs.loc[va], Xs.loc[te]
        y_tr, y_va, y_te = y.loc[tr], y.loc[va], y.loc[te]

        # --- Scaling: fit CHỈ trên TRAIN ---------------------------------
        if spec.needs_scaling:
            scaler = StandardScaler().fit(X_tr)
            A_tr = pd.DataFrame(scaler.transform(X_tr), index=tr, columns=Xs.columns)
            A_va = pd.DataFrame(scaler.transform(X_va), index=va, columns=Xs.columns)
            A_te = pd.DataFrame(scaler.transform(X_te), index=te, columns=Xs.columns)
        else:
            A_tr, A_va, A_te = X_tr, X_va, X_te

        # --- Model MỚI cho mỗi fold (không tái dùng state) ---------------
        model = spec.factory(**model_kwargs)
        model.fit(A_tr, y_tr.astype(int))

        p_va = self._proba(model, A_va)
        p_te = self._proba(model, A_te)

        # --- Hiệu chuẩn + tune ngưỡng: CHỈ dùng VALID --------------------
        sizer = ProbabilitySizer(SizingParams(max_position=self.s.sizing.max_position))
        sizer.fit_calibration(p_va, y_va.to_numpy().astype(int))

        tune_out = sizer.tune(
            proba_valid=pd.Series(p_va, index=va),
            prices_valid=prices.loc[va],
            objective=self._valid_objective(prices),
            grid_entry=self.s.sizing.grid_entry,
            max_position=self.s.sizing.max_position,
        )

        # --- TEST: đóng băng, chỉ áp dụng --------------------------------
        raw = sizer.size_series(pd.Series(p_te, index=te))
        raw = overlay.apply(raw, prices.loc[te], Xs.loc[te])
        final, _ = risk.apply(raw, prices.loc[te])

        return FoldResult(
            fold_id=fold.fold_id,
            test_index=te,
            proba=pd.Series(p_te, index=te),
            position=final,
            y_true=y_te,
            ml_metrics=MetricsEngine.ml_metrics(y_te.to_numpy(), p_te),
            tuned_params=sizer.params,
            tuning_grid=tune_out["grid"],
            feature_importance=self._importance(model, list(Xs.columns)),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _proba(model, X) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        raw = np.asarray(model.predict(X), dtype=float)
        return 1 / (1 + np.exp(-raw))

    @staticmethod
    def _importance(model, names: List[str]) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        else:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
        if len(imp) != len(names):
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
        df = pd.DataFrame({"feature": names, "importance": imp})
        total = df["importance"].sum()
        df["importance_pct"] = df["importance"] / total * 100 if total > 0 else 0.0
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
