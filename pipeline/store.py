"""Parquet cache with transparent fetch-on-miss."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import StorageConfig
from pipeline.provider import TradeStationProvider

logger = logging.getLogger(__name__)


class ParquetStore:
    """Read/write Parquet cache for bar data.

    File naming: ``{SYMBOL}_{timeframe}.parquet``
    Each file has a UTC DatetimeIndex named ``timestamp`` with lowercase
    OHLCV columns.
    """

    def __init__(self, config: StorageConfig) -> None:
        self._dir = Path(config.parquet_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._compression = config.compression

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        provider: TradeStationProvider,
        interval: int = 1,
    ) -> pd.DataFrame:
        """Return bar data, fetching from the provider on cache miss.

        Only fetches missing date ranges (before cached start or after
        cached end) rather than re-downloading everything.
        """
        path = self._path_for(symbol, timeframe)
        cached = self._read(path)

        if cached is not None and not cached.empty and self._covers(symbol, timeframe, start, end):
            logger.info("Cache HIT: %s %s", symbol, timeframe)
            return self._slice(cached, start, end)

        meta = self._read_meta(self._meta_path_for(symbol, timeframe))
        has_cache = cached is not None and not cached.empty and meta

        if has_cache:
            # Determine which gaps need fetching
            cached_start = datetime.fromisoformat(meta["start"])
            cached_end = datetime.fromisoformat(meta["end"])
            gaps = []

            if start < cached_start:
                gaps.append((start, cached_start, "pre-gap"))
            if end > cached_end:
                gaps.append((cached_end, end, "post-gap"))

            if not gaps:
                # Meta says covered but _covers failed — refetch all
                logger.info("Cache MISS: %s %s - full refetch", symbol, timeframe)
                fresh = self._fetch(provider, symbol, timeframe, start, end, interval)
                merged = pd.concat([cached, fresh])
            else:
                parts = [cached]
                for gap_start, gap_end, label in gaps:
                    logger.info("Fetching %s %s %s: %s -> %s",
                                symbol, timeframe, label,
                                gap_start.strftime("%Y-%m-%d"),
                                gap_end.strftime("%Y-%m-%d"))
                    parts.append(self._fetch(provider, symbol, timeframe,
                                             gap_start, gap_end, interval))
                merged = pd.concat(parts)
        else:
            logger.info("Cache MISS: %s %s - no cached data", symbol, timeframe)
            merged = self._fetch(provider, symbol, timeframe, start, end, interval)

        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self._write(path, merged, start, end)
        return self._slice(merged, start, end)

    def list_cached(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.parquet")]

    def _path_for(self, symbol: str, timeframe: str) -> Path:
        return self._dir / f"{symbol.upper()}_{timeframe}.parquet"

    def _meta_path_for(self, symbol: str, timeframe: str) -> Path:
        return self._dir / f"{symbol.upper()}_{timeframe}.meta.json"

    def _read(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
            return df
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return None

    def _write(
        self, path: Path, df: pd.DataFrame, start: datetime, end: datetime
    ) -> None:
        df.to_parquet(path, compression=self._compression)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Wrote %s (%.2f MB, %d rows)", path.name, size_mb, len(df))
        meta_path = path.with_suffix(".meta.json")
        meta = self._read_meta(meta_path)
        existing_start = meta.get("start", start.isoformat())
        existing_end = meta.get("end", end.isoformat())
        meta_path.write_text(
            json.dumps({
                "start": min(existing_start, start.isoformat()),
                "end": max(existing_end, end.isoformat()),
            }),
            encoding="utf-8",
        )

    @staticmethod
    def _read_meta(meta_path: Path) -> dict:
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _covers(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> bool:
        meta = self._read_meta(self._meta_path_for(symbol, timeframe))
        if not meta:
            return False
        cached_start = meta.get("start", "")
        cached_end = meta.get("end", "")
        return cached_start <= start.isoformat() and cached_end >= end.isoformat()

    @staticmethod
    def _slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        return df.loc[s:e]

    @staticmethod
    def _fetch(
        provider: TradeStationProvider,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        interval: int,
    ) -> pd.DataFrame:
        if timeframe.lower() in ("1day", "daily", "d"):
            return provider.fetch_daily_bars(symbol, start, end)
        return provider.fetch_minute_bars(symbol, start, end, interval=interval)
