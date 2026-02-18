#!/usr/bin/env python3
"""Run Numba WFO grid search.

Usage:
    python run_wfo.py --symbol SMH --strategy atr_d_l --grid phase1_slim
    python run_wfo.py --symbol SMH --rescore --wfo-dir results/SMH/wfo
    python run_wfo.py --symbol TQQQ --symbol2 QQQ --strategy atr_d_duals_ls --grid phase1_slim

Multi-strategy expansion (default):
    atr_d_duals_ls → runs 4 variants: duals_ls, duals_l, ls, l
    Use --no-expand to run only the specified strategy.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

from strategies.expansion import expand_strategy, STRATEGY_TS_MAP
from wfo.runner import _is_dual_symbol


def load_data(symbol: str, data_dir: str = "data/parquet",
              start_date: str | None = None):
    """Load daily OHLC data from Parquet."""
    path = os.path.join(data_dir, f"{symbol}_1day.parquet")
    if not os.path.exists(path):
        print(f"ERROR: Data file not found: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]

    print(f"Loaded {symbol}: {len(df)} daily bars "
          f"({df.index[0].date()} to {df.index[-1].date()})")

    return (df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df.index)


def ensure_data(symbol: str, data_dir: str, config_path: str,
                 start_date: str | None = None) -> None:
    """Check data coverage and fetch/extend parquet data as needed.

    Uses the pipeline's ParquetStore.get() for incremental fetching —
    only downloads missing date ranges.
    """
    from pipeline.config import load_download_config
    from pipeline.provider import TradeStationProvider
    from pipeline.store import ParquetStore

    # Determine required date range
    # Use yesterday — today's daily bar isn't available until after market close
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start = datetime(today.year - 11, 1, 1)
    end = yesterday

    # Check cached coverage via meta.json
    meta_path = os.path.join(data_dir, f"{symbol.upper()}_1day.meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cached_start = meta.get("start", "")
        cached_end = meta.get("end", "")
        if cached_start <= start.isoformat() and cached_end >= end.isoformat():
            print(f"  {symbol}: data OK "
                  f"(cached {cached_start[:10]} to {cached_end[:10]})")
            return

    # Data missing or stale — fetch via pipeline
    print(f"  {symbol}: fetching 1day bars "
          f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})...")
    cfg = load_download_config(config_path)
    provider = TradeStationProvider(cfg["tradestation"])
    store = ParquetStore(cfg["storage"])
    try:
        store.get(symbol.upper(), "1day", start, end, provider)
        print(f"  {symbol}: data updated")
    except Exception as exc:
        # API errors (404 on holidays/weekends, network issues) shouldn't
        # kill the WFO if we already have cached data
        parquet_path = os.path.join(data_dir, f"{symbol.upper()}_1day.parquet")
        if os.path.exists(parquet_path):
            print(f"  {symbol}: fetch failed ({exc}), using cached data")
        else:
            raise RuntimeError(
                f"No cached data for {symbol} and fetch failed: {exc}"
            ) from exc


def main():
    parser = argparse.ArgumentParser(description="Numba WFO grid search")
    parser.add_argument("--symbol", default="SMH", help="Symbol to backtest")
    parser.add_argument("--strategy", default="atr_d_l",
                        help="Strategy kernel name")
    parser.add_argument("--grid", default="phase1_slim",
                        help="Grid name from grids/ module")
    parser.add_argument("--start", default=None, help="Data start date")
    parser.add_argument("--cost", type=float, default=0.0,
                        help="Round-trip cost per share")
    parser.add_argument("--min-phases", type=int, default=3,
                        help="Minimum valid phases for stability")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--symbol2", default=None,
                        help="Secondary symbol for DualS strategies")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")

    # Rescore mode: re-score cached .npy results without re-running
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score cached results (skip WFO run)")
    parser.add_argument("--wfo-dir", default=None,
                        help="WFO results directory (for --rescore)")

    # Plateau detection after WFO (on by default)
    parser.add_argument("--no-plateaus", action="store_true",
                        help="Skip plateau detection + TS range generation")

    # Strategy expansion (on by default)
    parser.add_argument("--no-expand", action="store_true",
                        help="Skip strategy expansion (run only specified strategy)")

    # Auto-fetch data (on by default)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip auto-fetch (use cached data only)")

    args = parser.parse_args()

    # Load config defaults
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    symbol = args.symbol or cfg.get("symbol", "SMH")
    strategy = args.strategy or cfg.get("strategy", "atr_d_l")
    grid_name = args.grid or cfg.get("grid", "phase1_slim")
    wfo_cfg = cfg.get("wfo", {})
    cost = args.cost if args.cost != 0.0 else wfo_cfg.get("cost_per_trade", 0.0)
    min_phases = args.min_phases or wfo_cfg.get("min_phases", 3)
    data_dir = cfg.get("data_dir", "data/parquet")
    results_dir = cfg.get("results_dir", "results")

    # Resolve symbol2 early so folder name matches TS convention (TQQQ_QQQ)
    symbol2 = None
    if _is_dual_symbol(strategy):
        symbol2 = args.symbol2 or cfg.get("symbol2")
        if not symbol2:
            print("ERROR: DualS strategies require --symbol2 "
                  "or symbol2 in config.yaml")
            sys.exit(1)
    symbol_key = f"{symbol}_{symbol2}" if symbol2 else symbol

    # Expand strategy into variants (or just use the one specified)
    if args.no_expand:
        variants = [strategy]
    else:
        variants = expand_strategy(strategy)

    if len(variants) > 1:
        print(f"\nStrategy expansion: {strategy} → {variants}")

    # Guard: --wfo-dir with expansion is ambiguous (each variant needs its own dir)
    if args.wfo_dir and len(variants) > 1:
        print("ERROR: --wfo-dir with strategy expansion is ambiguous. "
              "Use --no-expand or omit --wfo-dir to use default per-variant dirs.")
        sys.exit(1)

    # Auto-fetch data if needed
    if not args.rescore and not args.no_fetch:
        symbols_needed = [symbol]
        if symbol2:
            symbols_needed.append(symbol2)
        print("\nChecking data coverage...")
        for sym in symbols_needed:
            ensure_data(sym, data_dir, args.config, args.start)

    # Load primary data once (shared across all variants)
    data1 = None
    data2 = None
    if not args.rescore:
        data1 = load_data(symbol, data_dir, args.start)
        print(f"Transaction cost: ${cost:.4f}/share round-trip")

        # Load Data2 for DualS variants
        if symbol2:
            h2, l2, c2, _ = load_data(symbol2, data_dir, args.start)
            data2 = (h2, l2, c2)

    # Run WFO for each variant
    all_range_files = []
    output_dir_base = os.path.join(results_dir, symbol_key)

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant}")
        print(f"{'='*60}")

        # Import grid module for this variant
        grid_mod = importlib.import_module(f"grids.{variant}")
        grid = grid_mod.get_grid(grid_name)
        total = grid_mod.count_combinations(grid)

        print(f"Strategy: {variant}, Grid: {grid_name}")
        print(f"Parameters:")
        for name, vals in grid.items():
            print(f"  {name:>22}: {len(vals):>3} values  "
                  f"[{vals[0]:.2f} ... {vals[-1]:.2f}]")
        print(f"  {'TOTAL':>22}: {total:>12,} combinations")

        # Isolate WFO output per variant
        wfo_dir = args.wfo_dir or os.path.join(
            output_dir_base, "wfo", variant)

        if args.rescore:
            from wfo.runner import rescore_from_cache
            print(f"\nRe-scoring cached results from {wfo_dir}")
            results = rescore_from_cache(grid, wfo_dir, min_phases)
        else:
            highs, lows, closes, dates = data1

            # Non-dual variants skip Data2
            variant_is_dual = _is_dual_symbol(variant)
            highs2 = lows2 = closes2 = None
            if variant_is_dual and data2:
                highs2, lows2, closes2 = data2

            from wfo.runner import run_wfo
            results = run_wfo(
                highs, lows, closes, dates, grid,
                strategy=variant,
                output_dir=wfo_dir,
                cost_per_trade=cost,
                min_phases=min_phases,
                highs2=highs2, lows2=lows2, closes2=closes2,
            )

        # Print top 30
        print(f"\nTop 30 by Stability Score ({variant}):")
        display_cols = [
            "atr_period", "atr_multiplier", "bb_length",
            "atr_buy_period", "atr_buy_multiplier",
            "Stability_Score", "Phases_In_Top", "Total_OOS_Trades",
            "P1_OOS_PnL", "P2_OOS_PnL", "P3_OOS_PnL", "P4_OOS_PnL",
        ]
        avail = [c for c in display_cols if c in results.columns]
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:.2f}")
        print(results.head(30)[avail].to_string(index=False))

        # Save candidates CSV
        output = args.output if (args.output and len(variants) == 1) else \
            os.path.join(output_dir_base,
                         f"candidates_{variant}_{grid_name}.csv")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        results.to_csv(output, index=False)
        print(f"\nSaved {len(results):,} stable combos to {output}")

        # Chain into plateau detection (collect range_files, defer config)
        if not args.no_plateaus:
            range_files = run_plateaus(
                results, variant, symbol_key, cfg, results_dir, min_phases)
            all_range_files.extend(range_files)

    # Generate combined config ONCE after all variants complete
    if not args.no_plateaus and all_range_files:
        from plateaus.range_generator import generate_config
        print(f"\n{'='*60}")
        print("COMBINED CONFIG")
        print(f"{'='*60}")
        generate_config(all_range_files, output_dir_base)

    # Summary
    if len(variants) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(variants)} variants completed")
        print(f"{'='*60}")
        for v in variants:
            ts_name = STRATEGY_TS_MAP.get(v, v)
            print(f"  {v:25} ({ts_name})")
        if all_range_files:
            print(f"\n  Combined config: {len(all_range_files)} refinement jobs")

    if all_range_files or args.no_plateaus:
        print(f"\nAll files written to: {output_dir_base}/")
    if all_range_files:
        print("Next: Import NumbaRefinement TradingApp in TS and run refinement.")


def run_plateaus(candidates, strategy, symbol, cfg, results_dir, min_phases):
    """Run plateau detection + TS range generation on WFO candidates.

    Returns list of range_files (filenames). Does NOT call generate_config()
    — the caller collects all range_files and generates one combined config.
    """
    from plateaus.detector import detect_plateaus, get_plateau_top_combos
    from plateaus.range_generator import generate_ranges_for_plateau

    range_dir = cfg.get("ranges", {}).get("default_range_dir", "strategies/ranges")
    plateaus_cfg = cfg.get("plateaus", {})
    min_cell_count = plateaus_cfg.get("min_cell_count", 50)

    ts_strategy_name = STRATEGY_TS_MAP.get(strategy, strategy)

    df = candidates
    if min_phases > 0 and "Phases_In_Top" in df.columns:
        df = df[df["Phases_In_Top"] >= min_phases]
        print(f"  After filtering >= {min_phases} phases: {len(df):,}")

    print(f"\n{'='*60}")
    print(f"PLATEAU DETECTION ({strategy})")
    print(f"{'='*60}")

    plateaus = detect_plateaus(df, min_cell_count=min_cell_count)

    if not plateaus:
        print("No plateaus detected. Try adjusting min_cell_count in config.")
        return []

    output_dir = os.path.join(results_dir, symbol)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GENERATING TS RANGE FILES ({strategy})")
    print(f"{'='*60}")

    top_plateaus = 10
    range_files = []
    for i, plateau in enumerate(plateaus[:top_plateaus]):
        rank = i + 1
        best = get_plateau_top_combos(df, plateau)
        filename, n_combos = generate_ranges_for_plateau(
            plateau, best, ts_strategy_name, rank,
            output_dir, range_dir,
        )
        range_files.append(filename)

    return range_files


if __name__ == "__main__":
    main()
