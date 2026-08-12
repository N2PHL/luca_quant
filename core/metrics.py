import numpy as np
import pandas as pd

class QuantMetrics:
    """Bộ công cụ đo lường tiêu chuẩn của Quỹ Định lượng."""
    
    @staticmethod
    def calculate_all(strategy_returns: pd.Series) -> dict:
        if strategy_returns.empty: return {}
        
        cum_returns = (1 + strategy_returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        
        # Annualized
        trading_days = 252
        ann_return = (1 + total_return) ** (trading_days / max(len(strategy_returns), 1)) - 1
        ann_vol = strategy_returns.std() * np.sqrt(trading_days)
        
        # Ratios
        sharpe = (ann_return) / (ann_vol + 1e-9)
        negative_returns = strategy_returns[strategy_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(trading_days)
        sortino = ann_return / (downside_vol + 1e-9)
        
        # Drawdown
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar = ann_return / (abs(max_drawdown) + 1e-9)
        
        # Win Rate & Profit Factor
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        win_rate = len(wins) / max(len(strategy_returns[strategy_returns != 0]), 1)
        profit_factor = wins.sum() / (abs(losses.sum()) + 1e-9)
        
        return {
            "Total Return": total_return,
            "Ann. Return (CAGR)": ann_return,
            "Ann. Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor
        }
    