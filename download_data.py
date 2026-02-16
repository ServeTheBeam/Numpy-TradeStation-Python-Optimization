"""Download & cache bar data from TradeStation.

Usage
-----
python download_data.py --symbols SMH --start 2016-01-01 --end 2026-02-14
python download_data.py --symbols SMH QQQ --timeframe 1min --start 2016-01-01 --end 2026-02-14
python download_data.py --symbols SMH --timeframe 1day --start 2016-01-01 --end 2026-02-14
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from pipeline.config import load_download_config
from pipeline.provider import TradeStationProvider
from pipeline.store import ParquetStore


def _parse_timeframe(tf: str) -> tuple[str, int]:
    tf = tf.lower().strip()
    if tf in ("1day", "daily", "d"):
        return "1day", 1
    if tf.endswith("min"):
        interval = int(tf.replace("min", ""))
        return f"{interval}min", interval
    raise ValueError(f"Unsupported timeframe: {tf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & cache bar data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="1min", help="1min, 5min, 1day, etc.")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    log = logging.getLogger("download_data")

    cfg = load_download_config(args.config)
    provider = TradeStationProvider(cfg["tradestation"])
    store = ParquetStore(cfg["storage"])

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    tf_key, interval = _parse_timeframe(args.timeframe)

    for symbol in args.symbols:
        symbol = symbol.upper()
        log.info("Downloading %s %s [%s -> %s]",
                 symbol, tf_key, args.start, args.end)
        df = store.get(symbol, tf_key, start, end, provider, interval=interval)
        log.info("  -> %d bars for %s", len(df), symbol)

    log.info("Done. Cached files: %s", store.list_cached())


if __name__ == "__main__":
    main()
