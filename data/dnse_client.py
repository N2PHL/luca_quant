import requests
import pandas as pd
import numpy as np

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm vệ sinh dữ liệu: Xử lý các giá trị khuyết (NaN) và vô cực (Inf)
    để tránh làm sập các mô hình Quant và đồ thị Plotly.
    """
    # 1. Nếu DataFrame rỗng, trả về nguyên bản để UI tự xử lý
    if df.empty:
        return df
        
    # 2. Xóa các giá trị vô cực (Infinity) sinh ra do lỗi hệ thống API
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 3. Loại bỏ hoàn toàn các hàng không có giá đóng cửa (Close)
    # Vì giá Close là cốt lõi của mọi tính toán Quant, không có nó thì không thể nội suy
    if 'Close' in df.columns:
        df = df.dropna(subset=['Close'])
        
    # 4. Nội suy (Forward Fill) cho các cột bị khuyết nhẹ (như Khối lượng - Volume)
    # Nếu hụt dữ liệu 1 ngày, lấy giá trị của ngày hôm trước đắp vào
    df = df.ffill()
    
    return df

def fetch_historical_data(ticker: str, start_timestamp: int, end_timestamp: int, resolution: str = '1D', is_index: bool = False) -> pd.DataFrame:
    """
    Lấy dữ liệu giá từ API DNSE.
    Bản hợp nhất: Hỗ trợ is_index (cho mô hình Quant) và trả về đủ OHLCV (cho giao diện Chart/Summary).
    Đã tích hợp lớp màng lọc vệ sinh dữ liệu (Data Sanitization).
    """
    # 1. Định tuyến linh hoạt giữa cổ phiếu và chỉ số
    endpoint = "index" if is_index else "stock"
    url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{endpoint}"
    
    params = {
        "symbol": ticker,
        "from": start_timestamp,
        "to": end_timestamp,
        "resolution": resolution
    }
    
    try:
        response = requests.get(url, params=params, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        data = response.json()
        
        if not data or 't' not in data or len(data['t']) == 0:
            return pd.DataFrame()
            
        # 2. Giữ lại toàn bộ cấu trúc OHLCV để trang Summary không bị lỗi KeyError
        raw_df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data.get('v', [0] * len(data['t'])) # Dùng get() để chống lỗi nếu API không có Volume
        })
        
        # 3. Chuẩn hóa Index để ghép nối ma trận ở các trang Quant không bị lệch dòng
        raw_df['Date'] = raw_df['Date'].dt.tz_localize(None).dt.normalize()
        raw_df.set_index('Date', inplace=True)
        
        # BƯỚC QUAN TRỌNG: Làm sạch dữ liệu trước khi trả về
        clean_df = clean_financial_data(raw_df)
        
        return clean_df
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu cho mã {ticker}: {e}")
        return pd.DataFrame()
    