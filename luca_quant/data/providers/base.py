"""Provider interface — mọi nguồn dữ liệu implement chung một contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from luca_quant.data.schemas import clean_ohlcv, normalize_ohlcv


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def _fetch_raw(self, ticker: str, start: datetime, end: datetime, resolution: str) -> pd.DataFrame:
        ...

    def get_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        resolution: str = "1D",
    ) -> pd.DataFrame:
        raw = self._fetch_raw(ticker, start, end, resolution)
        return clean_ohlcv(normalize_ohlcv(raw))
