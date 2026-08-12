import pandas as pd
import numpy as np

class FeatureEngineer:
    """Strategy Pattern: Tạo lập các nhóm đặc trưng (Features) tĩnh, động và phi tuyến."""
    
    # ... [Giữ nguyên các hàm add_price_features, add_trend_features đã có] ...
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        df['Ret_1'] = df['close'].pct_change(1)
        df['Ret_3'] = df['close'].pct_change(3)
        df['Ret_5'] = df['close'].pct_change(5)
        df['Gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['High_Low'] = (df['high'] - df['low']) / df['close']
        df['Open_Close'] = (df['close'] - df['open']) / df['open']
        df['Vol_Change'] = df['volume'].pct_change().fillna(0)
        return df

    @staticmethod
    def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        for period in [10, 20, 50, 200]:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df['EMA_Cross_10_20'] = (df['EMA_10'] > df['EMA_20']).astype(int)
        return df

    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        df['Rolling_Std_20'] = df['close'].pct_change().rolling(20).std()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        return df

    # --- BẮT ĐẦU BỔ SUNG MODULE FRACTAL TẠI ĐÂY ---
    @staticmethod
    def _calculate_hurst(ts: np.ndarray) -> float:
        """
        Ước lượng Local Hurst Exponent bằng phương pháp Volatility Scaling (R/S Proxy).
        O(N) complexity giúp chạy siêu tốc trong cửa sổ cuộn của Pandas.
        """
        # Nếu chuỗi toàn giá trị giống nhau (ví dụ: cổ phiếu ngưng GD), trả về 0.5 (Random Walk)
        if len(ts) < 10 or np.std(ts) == 0:
            return 0.5
            
        lags = range(2, 10)
        tau = []
        for lag in lags:
            # Tính độ lệch chuẩn của chênh lệch giá theo độ trễ (lag)
            diff = np.subtract(ts[lag:], ts[:-lag])
            tau.append(np.std(diff))
            
        # Hồi quy tuyến tính log-log: log(tau) = H * log(lag) + c
        # Hệ số góc (slope) chính là Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]
        
        # Chặn giá trị H nằm trong biên độ lý thuyết [0, 1]
        return max(0.0, min(1.0, hurst))

    @staticmethod
    def add_fractal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng phân tích đa phân dạng vào không gian đặc trưng.
        """
        # Tính Hurst Exponent cục bộ trong cửa sổ 20 phiên và 50 phiên
        # Cửa sổ 20: Bắt nhịp chu kỳ ngắn hạn
        df['Hurst_20'] = df['close'].rolling(window=20).apply(FeatureEngineer._calculate_hurst, raw=True)
        # Cửa sổ 50: Bắt nhịp chu kỳ trung hạn
        df['Hurst_50'] = df['close'].rolling(window=50).apply(FeatureEngineer._calculate_hurst, raw=True)
        
        # Fractal Dimension (D = 2 - H): Đo lường độ nhiễu loạn/độ nhám của đường giá
        df['Fractal_Dim_20'] = 2.0 - df['Hurst_20']
        
        return df
    # --- KẾT THÚC BỔ SUNG ---

    @staticmethod
    def generate_all(df: pd.DataFrame, 
                     use_trend=True, 
                     use_momentum=True, 
                     use_volatility=True, 
                     use_fractal=True) -> pd.DataFrame:
        """
        Khởi tạo Pipeline sinh Feature có hỗ trợ cờ bật/tắt cho Ablation Study.
        """
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        df = df.sort_index()
        
        # Nhóm Price & Volume luôn là bắt buộc (Baseline)
        df = FeatureEngineer.add_price_features(df)
        
        # Các nhóm còn lại có thể bật/tắt
        if use_trend:
            df = FeatureEngineer.add_trend_features(df)
        if use_momentum:
            df = FeatureEngineer.add_momentum_features(df)
        if use_volatility:
            df = FeatureEngineer.add_volatility_features(df)
        if use_fractal:
            # GỌI HÀM FRACTAL BẠN ĐÃ THÊM Ở BƯỚC TRƯỚC
            df = FeatureEngineer.add_fractal_features(df)
            
        return df.dropna()
    
    