"""Yahoo Finance provider — dùng cho benchmark quốc tế và fallback."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from luca_quant.data.providers.base import DataProvider

log = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, suffix: str = ".VN"):
        self.suffix = suffix

    def _fetch_raw(self, ticker, start, end, resolution="1D") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            log.error("Chưa cài yfinance.")
            return pd.DataFrame()

        symbol = ticker if "." in ticker or "^" in ticker else f"{ticker}{self.suffix}"
        try:
            df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        except Exception as exc:                        # noqa: BLE001
            log.error("yfinance lỗi %s: %s", symbol, exc)
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
