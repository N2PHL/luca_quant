from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import catboost as cb

class ModelFactory:
    """Factory Pattern: Khởi tạo mô hình học máy tiêu chuẩn."""
    
    @staticmethod
    def get_model(model_name: str, **kwargs):
        models = {
            "logistic": LogisticRegression(random_state=42, **kwargs),
            "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs),
            "hist_gb": HistGradientBoostingClassifier(random_state=42, **kwargs),
            "lightgbm": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **kwargs),
            "catboost": cb.CatBoostClassifier(random_state=42, verbose=0, **kwargs),
            "mlp": MLPClassifier(random_state=42, early_stopping=True, **kwargs)
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} không được hỗ trợ.")
        return models[model_name]
    