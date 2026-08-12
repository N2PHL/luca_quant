# core/alpha_engine.py
import numpy as np
import pandas as pd

class AlphaEngine:
    """
    Động cơ tính toán tín hiệu Alpha & Backtest định lượng (Vectorized).
    """
    @staticmethod
    def calculate_alpha_signal(df: pd.DataFrame, expression_type: str = "Momentum_RSI") -> pd.Series:
        """
        Tính toán tín hiệu Alpha dựa trên biểu thức toán học.
        df yêu cầu các cột: 'close', 'open', 'high', 'low', 'volume'
        """
        data = df.copy()
        
        if expression_type == "Momentum_RSI":
            # Tín hiệu Momentum dựa trên RSI đảo chiều
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            signal = (rsi - 50) / 50.0  # Normalize về khoảng [-1, 1]
            
        elif expression_type == "Mean_Reversion_ZScore":
            # Tín hiệu Mean Reversion dựa trên Z-score (20 ngày)
            sma20 = data['close'].rolling(20).mean()
            std20 = data['close'].rolling(20).std()
            z_score = (data['close'] - sma20) / (std20 + 1e-9)
            signal = -z_score  # Mua khi quá bán (Z < 0), Bán khi quá mua (Z > 0)
            
        elif expression_type == "Volume_Price_Trend":
            # Tín hiệu kết hợp Giá x Khối lượng (Volume-Weighted Momentum)
            ret = data['close'].pct_change()
            vol_sma = data['volume'].rolling(20).mean()
            vol_ratio = data['volume'] / (vol_sma + 1e-9)
            signal = ret * vol_ratio
            signal = signal.clip(-1.0, 1.0)
            
        else:
            # Tín hiệu Momentum đơn giản (Return 5 ngày)
            signal = data['close'].pct_change(5)
            
        return signal.fillna(0)

    @staticmethod
    def backtest_signal(
        df: pd.DataFrame, 
        signal: pd.Series, 
        initial_capital: float = 100_000_000.0,
        transaction_cost: float = 0.0015  # Phí giao dịch + Thuế + Slippage mặc định = 0.15% mỗi lượt
    ) -> dict:
        """
        Giả lập Backtest giao dịch theo tín hiệu Alpha (Long/Short/Cash).
        Đã bổ sung chi phí giao dịch & trượt giá (Transaction Costs & Slippage) chuẩn Quant.
        """
        if df.empty:
            return {
                'data': pd.DataFrame(),
                'total_return': 0.0,
                'ann_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'final_equity': initial_capital
            }

        data = df.copy()
        data['signal'] = signal
        
        # Shift vị thế 1 ngày để tránh Look-ahead Bias (Lỗi nhìn trước tương lai)
        data['position'] = data['signal'].shift(1).fillna(0)
        data['position'] = np.clip(data['position'], -1.0, 1.0)
        
        # Tính mức độ xoay chuyển vị thế (Turnover)
        # Ví dụ: Đang 0 -> Mua (1) = Thay đổi 1 | Đang Mua (1) -> Bán (-1) = Thay đổi 2
        data['turnover'] = data['position'].diff().abs().fillna(0)
        
        # Lợi nhuận thị trường hàng ngày
        data['market_return'] = data['close'].pct_change().fillna(0)
        
        # Lợi nhuận chiến lược = (Vị thế * Biến động giá) - (Tần suất đảo vị thế * Phí giao dịch)
        data['strategy_return'] = (data['position'] * data['market_return']) - (data['turnover'] * transaction_cost)
        
        # Đường cong tài sản (Equity Curve)
        data['cum_market_return'] = (1 + data['market_return']).cumprod()
        data['cum_strategy_return'] = (1 + data['strategy_return']).cumprod()
        data['equity'] = initial_capital * data['cum_strategy_return']
        
        # Thống kê chỉ số Quant
        total_return = data['cum_strategy_return'].iloc[-1] - 1
        ann_return = (1 + total_return) ** (252 / max(len(data), 1)) - 1
        
        daily_std = data['strategy_return'].std()
        sharpe_ratio = (data['strategy_return'].mean() / (daily_std + 1e-9)) * np.sqrt(252) if daily_std > 0 else 0
        
        # Max Drawdown
        cum_max = data['equity'].cummax()
        drawdown = (data['equity'] - cum_max) / (cum_max + 1e-9)
        max_drawdown = drawdown.min()
        
        # Win Rate
        active_trades = data[data['position'] != 0]['strategy_return']
        win_rate = (active_trades > 0).mean() if len(active_trades) > 0 else 0
        
        return {
            'data': data,
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_equity': data['equity'].iloc[-1]
        }
    