#!/usr/bin/env python3
"""Run Numba WFO grid search.

Usage:
    python run_wfo.py --symbol SMH --strategy atr_d_l --grid phase1_slim
    python run_wfo.py --symbol SMH --rescore --wfo-dir results/SMH/wfo
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml


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

    # Import grid module dynamically based on strategy
    import importlib
    grid_mod = importlib.import_module(f"grids.{strategy}")
    get_grid = grid_mod.get_grid
    count_combinations = grid_mod.count_combinations
    grid = get_grid(grid_name)
    total = count_combinations(grid)

    print(f"Strategy: {strategy}, Grid: {grid_name}")
    print(f"Parameters:")
    for name, vals in grid.items():
        print(f"  {name:>22}: {len(vals):>3} values  [{vals[0]:.2f} ... {vals[-1]:.2f}]")
    print(f"  {'TOTAL':>22}: {total:>12,} combinations")

    output_dir = args.wfo_dir or os.path.join(results_dir, symbol, "wfo")

    if args.rescore:
        # Re-score cached results
        from wfo.runner import rescore_from_cache
        print(f"\nRe-scoring cached results from {output_dir}")
        results = rescore_from_cache(grid, output_dir, min_phases)
    else:
        # Full WFO run
        highs, lows, closes, dates = load_data(symbol, data_dir, args.start)
        print(f"Transaction cost: ${cost:.4f}/share round-trip")

        # Load Data2 for DualS strategies
        from wfo.runner import _is_dual_symbol
        highs2 = lows2 = closes2 = None
        if _is_dual_symbol(strategy):
            symbol2 = args.symbol2 or cfg.get("symbol2")
            if not symbol2:
                print("ERROR: DualS strategies require --symbol2 "
                      "or symbol2 in config.yaml")
                sys.exit(1)
            h2, l2, c2, _ = load_data(symbol2, data_dir, args.start)
            highs2, lows2, closes2 = h2, l2, c2

        from wfo.runner import run_wfo
        results = run_wfo(
            highs, lows, closes, dates, grid,
            strategy=strategy,
            output_dir=output_dir,
            cost_per_trade=cost,
            min_phases=min_phases,
            highs2=highs2, lows2=lows2, closes2=closes2,
        )

    # Print top 30
    print(f"\nTop 30 by Stability Score:")
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

    # Save
    output = args.output or os.path.join(
        results_dir, symbol, f"candidates_{strategy}_{grid_name}.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    results.to_csv(output, index=False)
    print(f"\nSaved {len(results):,} stable combos to {output}")

    # Chain into plateau detection (default)
    if not args.no_plateaus:
        run_plateaus(results, strategy, symbol, cfg, results_dir, min_phases)


def run_plateaus(candidates, strategy, symbol, cfg, results_dir, min_phases):
    """Run plateau detection + TS range generation on WFO candidates."""
    from plateaus.detector import detect_plateaus, get_plateau_top_combos
    from plateaus.range_generator import generate_ranges_for_plateau, generate_config

    range_dir = cfg.get("ranges", {}).get("default_range_dir", "strategies/ranges")
    plateaus_cfg = cfg.get("plateaus", {})
    min_cell_count = plateaus_cfg.get("min_cell_count", 50)

    strategy_ts_map = {
        "atr_d_l": "ATR-D-v1.5-L",
        "atr_d_ls": "ATR-D-v1.5-LS",
        "atr_d_hma_l": "ATR-D-HMA-v1.5-L",
        "atr_d_hma_ls": "ATR-D-HMA-v1.5-LS",
        "atr_d_duals_l": "ATR-D-DualS-v1.5-L",
        "atr_d_duals_ls": "ATR-D-DualS-v1.5-LS",
    }
    ts_strategy_name = strategy_ts_map.get(strategy, strategy)

    df = candidates
    if min_phases > 0 and "Phases_In_Top" in df.columns:
        df = df[df["Phases_In_Top"] >= min_phases]
        print(f"  After filtering >= {min_phases} phases: {len(df):,}")

    print(f"\n{'='*60}")
    print("PLATEAU DETECTION")
    print(f"{'='*60}")

    plateaus = detect_plateaus(df, min_cell_count=min_cell_count)

    if not plateaus:
        print("No plateaus detected. Try adjusting min_cell_count in config.")
        return

    output_dir = os.path.join(results_dir, symbol)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING TS RANGE FILES")
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

    print()
    generate_config(range_files, output_dir)

    print(f"\nAll files written to: {output_dir}/")
    print("Next: Import NumbaRefinement TradingApp in TS and run refinement.")


if __name__ == "__main__":
    main()
