from sklearn.model_selection import TimeSeriesSplit

class WalkForwardValidator:
    """Tạo các Folds cắt theo thời gian cho Walk-Forward Analysis."""
    
    def __init__(self, n_splits: int = 5, expanding: bool = True):
        self.n_splits = n_splits
        self.expanding = expanding
        
    def get_splits(self, df):
        # Mặc định của TimeSeriesSplit là Expanding Window
        # (Train to dần, Test cuốn về phía trước)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return tscv.split(df)
    