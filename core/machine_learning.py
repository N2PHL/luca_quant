import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

# Nhập các module nội bộ
from core.features import FeatureEngineer
from core.labels import LabelGenerator
from core.models import ModelFactory
from core.sizing import ProbabilitySizer
from core.risk import RiskManager
from core.validation import WalkForwardValidator
from core.metrics import QuantMetrics
from core.explain import ModelExplainer
from core.alpha_engine import AlphaEngine

class AIQuantPipeline:
    """
    Trái tim của hệ thống AI Algorithmic Trading.
    Áp dụng Dependency Injection để điều phối các module: ML, Sizing, Risk Manager.
    """
    def __init__(self, 
                 model_name: str = "lightgbm",
                 thresholds: dict = None,
                 risk_params: dict = None):
        
        # Khởi tạo các Component cốt lõi
        self.model = ModelFactory.get_model(model_name)
        self.scaler = StandardScaler()
        self.sizer = ProbabilitySizer(thresholds)
        self.risk_manager = RiskManager(**(risk_params or {}))
        self.metrics = QuantMetrics()
        
    def process_data(self, df: pd.DataFrame, feature_flags: dict = None) -> tuple:
        """Đường ống sinh Data (ETL) có hỗ trợ tắt bật feature (Dùng cho Ablation Study)."""
        data = df.copy()
        flags = feature_flags or {}
        
        # 1. Truyền cờ điều khiển vào FeatureEngineer (sinh RSI, MACD, Hurst...)
        data = FeatureEngineer.generate_all(data, **flags)
        
        # 2. Sinh Label (Mục tiêu dự báo: Tăng giá ở cây nến tiếp theo)
        data = LabelGenerator.create_classification_label(data, horizon=1, threshold=0.0)
        
        # 3. Lọc bỏ NA do các đường MA/Rolling (bao gồm Hurst) gây ra để tránh lỗi máy học
        data.dropna(inplace=True)
        
        # 4. Tách X (Features) và Y (Label)
        # Giữ lại các biến kỹ thuật để đưa vào Model, loại bỏ giá và các biến Target
        exclude_cols = ['Label', 'Future_Return', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        return data, feature_cols
        
    def run_walk_forward_backtest(self, df: pd.DataFrame, feature_flags: dict = None) -> Dict[str, Any]:
        """Thực thi Walk-Forward Backtest với bộ tính năng tùy chỉnh."""
        
        # Truyền feature_flags xuống process_data
        data, feature_cols = self.process_data(df, feature_flags)
        
        if len(data) < 200:
            raise ValueError("Dữ liệu quá ngắn để chạy Walk-Forward Backtest (Cần tối thiểu 200 nến).")
            
        wf = WalkForwardValidator(n_splits=5)
        all_test_data = []
        
        for train_idx, test_idx in wf.get_splits(data):
            train_df = data.iloc[train_idx]
            test_df = data.iloc[test_idx]
            
            # Chuẩn hóa Dữ liệu (Scaling)
            X_train = self.scaler.fit_transform(train_df[feature_cols])
            X_test = self.scaler.transform(test_df[feature_cols])
            y_train = train_df['Label']
            
            # Huấn luyện mô hình
            self.model.fit(X_train, y_train)
            
            # Dự báo xác suất (Probability)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_test)[:, 1]
            else:
                probs = self.model.predict(X_test)
                
            # Ghi nhận kết quả vào Test DataFrame
            test_df = test_df.copy()
            test_df['Prob_Up'] = probs
            
            # --- LUỒNG XỬ LÝ LỆNH GIAO DỊCH (TRADE EXECUTION FLOW) ---
            
            # Bước 1: POSITION SIZING SƠ BỘ - Cấp vốn dựa trên độ tự tin của AI
            test_df['Raw_Signal'] = self.sizer.apply_to_series(test_df['Prob_Up'])
            
            # Bước 2: QUẢN TRỊ RỦI RO (RISK MANAGER) - Chốt chặn cuối cùng
            # Ghi đè tín hiệu Raw_Signal nếu vướng rào cản thanh khoản, ATR, hoặc Fractal (Hurst)
            test_df = self.risk_manager.filter_signals(test_df, signal_col='Raw_Signal')
            
            # ---------------------------------------------------------
            
            all_test_data.append(test_df)
            
        # Gộp toàn bộ kết quả Out-of-sample
        final_test_df = pd.concat(all_test_data)
        
        # Backtest & Đánh giá PnL qua Alpha Engine
        bt_results = AlphaEngine.backtest_signal(
            df=final_test_df, 
            signal=final_test_df['Raw_Signal'], 
            transaction_cost=0.0015  # Phí giao dịch 0.15%
        )
        
        # Tính toán Metrics & Feature Importance
        quant_metrics = self.metrics.calculate_all(bt_results['data']['strategy_return'])
        feat_importance = ModelExplainer.get_feature_importance(self.model, feature_cols)
        
        return {
            "metrics": quant_metrics,
            "feature_importance": feat_importance,
            "equity_curve": bt_results['data']['equity'],
            "signals": final_test_df[['close', 'Prob_Up', 'Raw_Signal']]
        }

    def run_ablation_study(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Khuôn khổ nghiên cứu bóc tách (Ablation Study).
        Chứng minh bằng số liệu sức mạnh của Fractal Risk Manager so với các bộ Feature khác.
        """
        # Thiết kế các kịch bản tắt/bật Feature
        experiments = [
            {"name": "1. Price Only (Baseline)", "flags": {"use_trend": False, "use_momentum": False, "use_volatility": False, "use_fractal": False}},
            {"name": "2. + Trend & Momentum", "flags": {"use_trend": True, "use_momentum": True, "use_volatility": False, "use_fractal": False}},
            {"name": "3. + Volatility", "flags": {"use_trend": True, "use_momentum": True, "use_volatility": True, "use_fractal": False}},
            {"name": "4. + Fractal Analysis (Full)", "flags": {"use_trend": True, "use_momentum": True, "use_volatility": True, "use_fractal": True}},
        ]
        
        results = []
        for exp in experiments:
            try:
                # Chạy Backtest với bộ cờ tương ứng
                res = self.run_walk_forward_backtest(df, feature_flags=exp["flags"])
                metrics = res['metrics']
                metrics['Feature Set'] = exp['name']
                results.append(metrics)
            except Exception as e:
                print(f"Lỗi kịch bản {exp['name']}: {e}")
                
        df_res = pd.DataFrame(results).set_index('Feature Set')
        return df_res
    