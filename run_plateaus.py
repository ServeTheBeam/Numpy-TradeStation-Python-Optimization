#!/usr/bin/env python3
"""Detect plateaus and generate TS refinement range files.

Usage:
    python run_plateaus.py --symbol SMH --strategy atr_d_l --min-phases 3
    python run_plateaus.py --symbol SMH --candidates results/SMH/candidates_atr_d_l_phase1_slim.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

from plateaus.detector import detect_plateaus, get_plateau_top_combos
from plateaus.range_generator import generate_ranges_for_plateau, generate_config


def main():
    parser = argparse.ArgumentParser(
        description="Detect plateaus and generate TS refinement range files")
    parser.add_argument("--symbol", default="SMH", help="Symbol name")
    parser.add_argument("--strategy", default="atr_d_l",
                        help="Strategy name (for TS filename mapping)")
    parser.add_argument("--candidates", default=None,
                        help="Path to WFO candidates CSV")
    parser.add_argument("--min-phases", type=int, default=3,
                        help="Filter: minimum valid phases")
    parser.add_argument("--min-cell-count", type=int, default=50,
                        help="Minimum combos per (mult, buy_mult) cell")
    parser.add_argument("--top-plateaus", type=int, default=10,
                        help="Max number of plateaus to generate ranges for")
    parser.add_argument("--symbol2", default=None,
                        help="Secondary symbol for DualS strategies")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    symbol = args.symbol
    results_dir = cfg.get("results_dir", "results")

    # Resolve symbol2 for DualS folder naming (matches TS convention)
    from wfo.runner import _is_dual_symbol
    symbol2 = None
    if _is_dual_symbol(args.strategy):
        symbol2 = args.symbol2 or cfg.get("symbol2")
    symbol_key = f"{symbol}_{symbol2}" if symbol2 else symbol
    range_dir = cfg.get("ranges", {}).get("default_range_dir", "strategies/ranges")

    # Map strategy kernel names to TS strategy names
    strategy_ts_map = {
        "atr_d_l": "ATR-D-v1.5-L",
        "atr_d_ls": "ATR-D-v1.5-LS",
        "atr_d_hma_l": "ATR-D-HMA-v1.5-L",
        "atr_d_hma_ls": "ATR-D-HMA-v1.5-LS",
        "atr_d_duals_l": "ATR-D-DualS-v1.5-L",
        "atr_d_duals_ls": "ATR-D-DualS-v1.5-LS",
    }
    ts_strategy_name = strategy_ts_map.get(args.strategy, args.strategy)

    # Load candidates
    candidates_path = args.candidates or os.path.join(
        results_dir, symbol_key, f"candidates_{args.strategy}_phase1_slim.csv")
    if not os.path.exists(candidates_path):
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)

    df = pd.read_csv(candidates_path)
    print(f"Loaded {len(df):,} stable combos from {candidates_path}")

    if args.min_phases > 0 and "Phases_In_Top" in df.columns:
        df = df[df["Phases_In_Top"] >= args.min_phases]
        print(f"  After filtering >= {args.min_phases} phases: {len(df):,}")

    # Detect plateaus
    print(f"\n{'='*60}")
    print("PLATEAU DETECTION")
    print(f"{'='*60}")

    plateaus = detect_plateaus(
        df,
        min_cell_count=args.min_cell_count,
    )

    if not plateaus:
        print("No plateaus detected. Try adjusting --min-cell-count.")
        sys.exit(1)

    # Generate range files
    output_dir = os.path.join(results_dir, symbol_key)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING TS RANGE FILES")
    print(f"{'='*60}")

    range_files = []
    for i, plateau in enumerate(plateaus[:args.top_plateaus]):
        rank = i + 1
        best = get_plateau_top_combos(df, plateau)
        filename, n_combos = generate_ranges_for_plateau(
            plateau, best, ts_strategy_name, rank,
            output_dir, range_dir,
        )
        range_files.append(filename)

    # Generate config
    print()
    generate_config(range_files, output_dir)

    print(f"\nAll files written to: {output_dir}/")
    print("Next: Import NumbaRefinement TradingApp in TS and run refinement.")


if __name__ == "__main__":
    main()
