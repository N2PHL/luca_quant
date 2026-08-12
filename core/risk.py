import pandas as pd
import numpy as np

class RiskManager:
    """
    Bộ lọc rủi ro & Ghi đè Tín hiệu (Alpha Override) của L.U.C.A Quant.
    Chiến lược: Bắt đáy bằng MACD cắt lên + Hurst cực tiểu; Thoát lệnh bằng MACD cắt xuống + Hurst tạo Vai Đầu Vai.
    """
    
    def __init__(self, 
                 max_exposure: float = 1.0, 
                 min_liquidity: float = 50000,
                 hurst_extreme_low: float = 0.35,  # Đại diện cho "Hurst = 0" (Vùng nén cực đại)
                 lookback_window: int = 20):
        self.max_exposure = max_exposure
        self.min_liquidity = min_liquidity
        self.hurst_extreme_low = hurst_extreme_low
        self.lookback = lookback_window
        
    def _detect_macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nhận diện các điểm giao cắt của MACD."""
        data = df.copy()
        if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
            data['MACD_Cross_Up_Below_0'] = False
            data['MACD_Cross_Down'] = False
            return data
            
        macd = data['MACD']
        sig = data['MACD_Signal']
        
        # MACD cắt LÊN (Cross Up)
        cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        # Điều kiện: Cắt lên từ bên dưới đường Zero (Biên dưới)
        data['MACD_Cross_Up_Below_0'] = cross_up & (macd < 0)
        
        # MACD cắt XUỐNG (Cross Down) ở bất kỳ đâu
        data['MACD_Cross_Down'] = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        
        return data

    def _detect_hurst_head_shoulders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận diện Mô hình Vai Đầu Vai (Bearish H&S) TRÊN CHÍNH ĐƯỜNG HURST.
        Giúp phát hiện sự kiệt sức của cấu trúc thị trường.
        """
        data = df.copy()
        if 'Hurst_50' not in data.columns:
            data['Hurst_Bearish_HS'] = False
            return data
            
        hurst = data['Hurst_50']
        
        # 1. Tìm cái Đầu (Head) - Điểm Hurst cao nhất trong 20 ngày qua
        highest_hurst = hurst.rolling(self.lookback).max()
        
        # 2. Tìm Vai Phải (Right Shoulder)
        # - Hôm qua là một đỉnh cục bộ (Đỉnh vai)
        is_local_peak_yesterday = (hurst.shift(1) > hurst) & (hurst.shift(1) > hurst.shift(2))
        
        # - Đỉnh vai này THẤP HƠN cái Đầu (Tạo Lower High)
        lower_high = hurst.shift(1) < highest_hurst.shift(1)
        
        # - Phân kỳ giảm: Đường Hurst hiện tại đang cắm đầu xuống
        hurst_dropping = hurst < hurst.shift(1)
        
        # Hợp lưu: Vai Đầu Vai trên Hurst
        data['Hurst_Bearish_HS'] = is_local_peak_yesterday & lower_high & hurst_dropping
        
        return data

    def filter_signals(self, df: pd.DataFrame, signal_col: str = 'Target_Size') -> pd.DataFrame:
        """
        Thực thi quyền lực tối cao của Risk Manager.
        """
        data = df.copy()
        
        if signal_col not in data.columns:
            return data
            
        # --- 0. CHUẨN BỊ TÍN HIỆU TOÁN HỌC ---
        data = self._detect_macd_signals(data)
        data = self._detect_hurst_head_shoulders(data)
            
        # --- 1. LỌC THANH KHOẢN (Bảo vệ cơ bản) ---
        if 'volume' in data.columns:
            liquidity_mask = data['volume'] < self.min_liquidity
            data.loc[liquidity_mask, signal_col] = 0.0
            
        # --- 2. CHIẾN LƯỢC GHI ĐÈ TÍN HIỆU (ALPHA OVERRIDE) ---
        if 'Hurst_50' in data.columns and 'MACD' in data.columns:
            
            # A. LỆNH TẤN CÔNG (ALL-IN LONG)
            # Điều kiện: Hurst tiệm cận 0 (bị nén chặt) + MACD cắt lên từ dưới 0
            entry_mask = (data['Hurst_50'] < self.hurst_extreme_low) & (data['MACD_Cross_Up_Below_0'] == True)
            data.loc[entry_mask, signal_col] = self.max_exposure  # Bất chấp AI, All-in 100%
            
            # B. LỆNH RÚT LUI (THOÁT VỊ THẾ)
            # Điều kiện: MACD cắt xuống + Hurst tạo Vai Đầu Vai giảm (Cấu trúc phân rã)
            exit_mask = (data['MACD_Cross_Down'] == True) & (data['Hurst_Bearish_HS'] == True)
            data.loc[exit_mask, signal_col] = 0.0  # Chém sạch vị thế về 0
            
            # C. QUẢN LÝ NHIỄU THÔNG THƯỜNG
            # Nếu giá đang trong vùng nhiễu (Hurst < 0.45) nhưng không có tín hiệu setup đẹp như trên
            # -> Bóp nghẹt 50% khối lượng giao dịch mà AI đề xuất để bảo toàn vốn
            normal_noise = (data['Hurst_50'] >= self.hurst_extreme_low) & (data['Hurst_50'] < 0.45) & (~entry_mask) & (~exit_mask)
            data.loc[normal_noise, signal_col] = data.loc[normal_noise, signal_col] * 0.5
            
        # --- 3. ĐẢM BẢO KHÔNG VƯỢT QUÁ MARGIN ---
        data[signal_col] = data[signal_col].clip(lower=0.0, upper=self.max_exposure)
            
        return data
    