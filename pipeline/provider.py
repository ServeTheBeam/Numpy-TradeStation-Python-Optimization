"""TradeStation data provider - fetches OHLCV bars via the v3 API.

Handles bar normalization (multiple field-name variants), JSON streaming,
and auto-chunking for minute bars (57,600-bar API limit).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd
import requests

from pipeline.auth import TradeStationAuth
from pipeline.config import TradeStationConfig

logger = logging.getLogger(__name__)

MAX_BARS_PER_REQUEST = 57_600
MINUTES_PER_TRADING_DAY = 390  # US equities RTH


class TradeStationProvider:
    """Fetch OHLCV bar data from the TradeStation v3 API."""

    def __init__(self, config: TradeStationConfig) -> None:
        self._config = config
        self._auth = TradeStationAuth(config)
        self._session = requests.Session()

    def fetch_daily_bars(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        bars = self._get_bars(
            symbol, interval=1, unit="Daily",
            firstdate=start.strftime("%Y-%m-%d"),
            lastdate=end.strftime("%Y-%m-%d"),
        )
        return self._to_dataframe(bars)

    def fetch_minute_bars(
        self, symbol: str, start: datetime, end: datetime, interval: int = 1
    ) -> pd.DataFrame:
        bars_per_day = MINUTES_PER_TRADING_DAY / interval
        total_days = (end - start).days
        estimated_bars = int(total_days * bars_per_day * 0.7)

        if estimated_bars <= MAX_BARS_PER_REQUEST:
            bars = self._get_bars(
                symbol, interval=interval, unit="Minute",
                firstdate=start.strftime("%Y-%m-%d"),
                lastdate=end.strftime("%Y-%m-%d"),
            )
            return self._to_dataframe(bars)

        # Chunk the date range to stay under the limit
        days_per_chunk = int((MAX_BARS_PER_REQUEST / bars_per_day) * 0.25)
        days_per_chunk = max(days_per_chunk, 1)
        all_bars: list[dict] = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=days_per_chunk), end)
            logger.info(
                "Fetching %s %d-min bars: %s -> %s",
                symbol, interval,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            chunk_bars = self._get_bars(
                symbol, interval=interval, unit="Minute",
                firstdate=chunk_start.strftime("%Y-%m-%d"),
                lastdate=chunk_end.strftime("%Y-%m-%d"),
            )
            all_bars.extend(chunk_bars)
            chunk_start = chunk_end

        df = self._to_dataframe(all_bars)
        return df[~df.index.duplicated(keep="first")]

    def _get_bars(
        self,
        symbol: str,
        interval: int,
        unit: str,
        firstdate: str | None = None,
        lastdate: str | None = None,
        barsback: int | None = None,
    ) -> list[dict]:
        """Call GET /v3/marketdata/barcharts/{symbol} and parse the JSON stream."""
        url = f"{self._config.api_base}/marketdata/barcharts/{symbol}"
        params: dict = {"interval": str(interval), "unit": unit}
        if firstdate:
            params["firstdate"] = firstdate
        if lastdate:
            params["lastdate"] = lastdate
        if barsback is not None:
            params["barsback"] = str(barsback)

        max_retries = 3
        for attempt in range(max_retries):
            token = self._auth.get_access_token()
            try:
                resp = self._session.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                    params=params,
                    stream=True,
                    timeout=(10, 120),
                )
                resp.raise_for_status()

                bars: list[dict] = []
                for obj in _iter_json_stream(resp):
                    if isinstance(obj, dict):
                        if "Bars" in obj and isinstance(obj["Bars"], list):
                            bars.extend(b for b in obj["Bars"] if isinstance(b, dict))
                        elif any(k in obj for k in ("Open", "OpenPrice", "O", "High")):
                            bars.append(obj)
                        if str(obj.get("IsEndOfHistory", "")).lower() == "true":
                            break
                return bars

            except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt < max_retries - 1 and status in (429, 500, 502, 503, 504, None):
                    wait = 10 * (2 ** attempt)
                    logger.warning(
                        "Request failed (%s) - retrying in %ds (attempt %d/%d)",
                        status or type(exc).__name__, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                raise

        return []

    @staticmethod
    def _normalize_bar(bar: dict) -> dict:
        """Map TradeStation's varied field names to standard OHLCV."""
        open_price = bar.get("Open") or bar.get("OpenPrice") or bar.get("O")
        high_price = bar.get("High") or bar.get("HighPrice") or bar.get("H")
        low_price = bar.get("Low") or bar.get("LowPrice") or bar.get("L")
        close_price = bar.get("Close") or bar.get("ClosePrice") or bar.get("C")
        volume = bar.get("TotalVolume") or bar.get("Volume") or bar.get("V") or 0
        timestamp = bar.get("TimeStamp") or bar.get("EpochTime")
        return {
            "open": float(open_price) if open_price is not None else float("nan"),
            "high": float(high_price) if high_price is not None else float("nan"),
            "low": float(low_price) if low_price is not None else float("nan"),
            "close": float(close_price) if close_price is not None else float("nan"),
            "volume": float(volume),
            "timestamp": timestamp,
        }

    def _to_dataframe(self, bars: list[dict]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows = [self._normalize_bar(b) for b in bars]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[~df.index.duplicated(keep="first")]
        return df[["open", "high", "low", "close", "volume"]]


def _iter_json_stream(resp: requests.Response) -> Iterator[dict]:
    """Yield complete JSON objects from a streaming response."""
    decoder = json.JSONDecoder()
    buf = ""
    enc = resp.encoding or "utf-8"

    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        if isinstance(chunk, bytes):
            chunk = chunk.decode(enc, errors="replace")
        buf += chunk

        while True:
            stripped = buf.lstrip()
            if not stripped:
                buf = ""
                break
            try:
                obj, idx = decoder.raw_decode(stripped)
            except ValueError:
                break
            consumed = len(buf) - len(stripped) + idx
            buf = buf[consumed:]
            yield obj
