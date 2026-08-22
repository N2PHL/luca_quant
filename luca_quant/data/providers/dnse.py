"""DNSE / Entrade provider (TTCK Việt Nam)."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

from luca_quant.data.providers.base import DataProvider

log = logging.getLogger(__name__)
BASE_URL = "https://services.entrade.com.vn/chart-api/v2/ohlcs"


class DNSEProvider(DataProvider):
    name = "dnse"

    def __init__(self, is_index: bool = False, timeout: int = 15):
        self.is_index = is_index
        self.timeout = timeout

    def _fetch_raw(self, ticker, start, end, resolution="1D") -> pd.DataFrame:
        endpoint = "index" if self.is_index else "stock"
        params = {
            "symbol": ticker.upper(),
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "resolution": resolution,
        }
        try:
            r = requests.get(
                f"{BASE_URL}/{endpoint}",
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:                       # noqa: BLE001
            log.error("DNSE fetch lỗi cho %s: %s", ticker, exc)
            return pd.DataFrame()

        if not data or "t" not in data or not data["t"]:
            log.warning("DNSE không trả dữ liệu cho %s", ticker)
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data.get("v", [0] * len(data["t"])),
            },
            index=pd.to_datetime(data["t"], unit="s"),
        )
