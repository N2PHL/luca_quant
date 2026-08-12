# data/yfinance_client.py
import yfinance as yf
import pandas as pd
import numpy as np

def clean_yfinance_data(df: pd.DataFrame, is_financial_statement: bool = False) -> pd.DataFrame:
    """
    Hàm vệ sinh dữ liệu chung cho Yahoo Finance.
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    # 1. Dọn dẹp các lỗi vô cực sinh ra do lỗi chia số liệu
    df = df.replace([np.inf, -np.inf], np.nan)
    
    if is_financial_statement:
        # 2A. Vệ sinh Báo cáo tài chính (Income Statement, Balance Sheet)
        # Xóa các khoản mục (hàng) mà TOÀN BỘ các quý/năm đều bị trống (NaN)
        df = df.dropna(how='all')
        
        # Với BCTC, nếu một khoản mục ở 1 quý bị khuyết, về mặt tài chính ta thường coi nó = 0 
        # (VD: Quý này không có chi phí lãi vay -> 0)
        df = df.fillna(0.0)
    else:
        # 2B. Vệ sinh bảng Ratios (Chỉ số định giá)
        # Không điền 0 cho P/E hay P/B vì chỉ số bằng 0 là sai bản chất tài chính.
        # Ta giữ nguyên np.nan. Pandas và Plotly sẽ tự động bỏ qua np.nan khi vẽ/tính toán.
        pass
        
    return df

def fetch_financial_ratios(ticker: str) -> pd.DataFrame:
    """
    Kéo các chỉ số định giá cơ bản từ Yahoo Finance.
    Đã xử lý nghiêm ngặt kiểu dữ liệu: Dùng np.nan thay vì chuỗi "N/A".
    """
    symbol = f"{ticker}.VN"
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # BƯỚC SỬA LỖI CHÍ MẠNG: Thay chuỗi "N/A" bằng np.nan để bảo toàn tính toán toán học
        ratios = {
            "P/E": info.get("trailingPE", np.nan),
            "P/B": info.get("priceToBook", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "ROA": info.get("returnOnAssets", np.nan),
            "Biên LN Gộp": info.get("grossMargins", np.nan),
            "Biên LN Ròng": info.get("profitMargins", np.nan),
            "Vốn hóa (Tỷ)": info.get("marketCap", np.nan) / 1e9 if info.get("marketCap") else np.nan
        }
        
        raw_df = pd.DataFrame([ratios])
        
        # Đưa qua máy lọc
        clean_df = clean_yfinance_data(raw_df, is_financial_statement=False)
        return clean_df
        
    except Exception as e:
        print(f"Lỗi yfinance (Ratios) - Mã {ticker}: {e}")
        return pd.DataFrame()

def fetch_income_statement(ticker: str, is_yearly: bool = False) -> pd.DataFrame:
    """
    Kéo Báo cáo kết quả kinh doanh từ Yahoo Finance.
    Đã tích hợp màng lọc dữ liệu rác.
    """
    symbol = f"{ticker}.VN"
    try:
        stock = yf.Ticker(symbol)
        
        # Kéo dữ liệu gốc
        if is_yearly:
            raw_df = stock.financials
        else:
            raw_df = stock.quarterly_financials
            
        # Đưa qua máy lọc chuyên dụng cho Báo cáo tài chính
        clean_df = clean_yfinance_data(raw_df, is_financial_statement=True)
        return clean_df
        
    except Exception as e:
        print(f"Lỗi yfinance (Income) - Mã {ticker}: {e}")
        return pd.DataFrame()