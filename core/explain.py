import pandas as pd
import numpy as np

class ModelExplainer:
    """Trích xuất Feature Importance từ Model."""
    
    @staticmethod
    def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'): # Dành cho Logistic
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Chuẩn hóa về 100%
        df['Importance (%)'] = (df['Importance'] / df['Importance'].sum()) * 100
        return df