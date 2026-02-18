"""YAML configuration for data download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TradeStationConfig:
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8080"
    scopes: str = "openid offline_access profile MarketData"
    api_base: str = "https://api.tradestation.com/v3"
    token_file: str = "data/tokens/ts_token.json"


@dataclass(frozen=True)
class StorageConfig:
    parquet_dir: str = "data/parquet"
    compression: str = "snappy"


def load_download_config(path: str | Path = "config.yaml") -> dict:
    """Load download-related config sections from the project YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return {
        "tradestation": TradeStationConfig(**raw.get("tradestation", {})),
        "storage": StorageConfig(**raw.get("storage", {})),
    }
