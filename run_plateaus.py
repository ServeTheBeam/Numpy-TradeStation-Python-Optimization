#!/usr/bin/env python3
"""Detect plateaus and generate TS refinement range files.

Usage:
    python run_plateaus.py --symbol SMH --strategy atr_d_l --min-phases 3
    python run_plateaus.py --symbol SMH --candidates results/SMH/candidates_atr_d_l_phase1_slim.csv
    python run_plateaus.py --symbol TQQQ --symbol2 QQQ --strategy atr_d_duals_ls

Multi-strategy expansion (default):
    atr_d_duals_ls → runs plateaus for 4 variants: duals_ls, duals_l, ls, l
    Use --no-expand to run only the specified strategy.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

from plateaus.detector import detect_plateaus, get_plateau_top_combos
from plateaus.range_generator import generate_ranges_for_plateau, generate_config
from strategies.expansion import expand_strategy, STRATEGY_TS_MAP


def main():
    parser = argparse.ArgumentParser(
        description="Detect plateaus and generate TS refinement range files")
    parser.add_argument("--symbol", default="SMH", help="Symbol name")
    parser.add_argument("--strategy", default="atr_d_l",
                        help="Strategy name (for TS filename mapping)")
    parser.add_argument("--candidates", default=None,
                        help="Path to WFO candidates CSV (single-strategy mode)")
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
    parser.add_argument("--grid", default="phase1_slim",
                        help="Grid name (for locating candidates CSVs)")
    parser.add_argument("--no-expand", action="store_true",
                        help="Skip strategy expansion (run only specified strategy)")
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

    # Expand strategy into variants
    if args.no_expand or args.candidates:
        # Explicit candidates path → single strategy, no expansion
        variants = [args.strategy]
    else:
        variants = expand_strategy(args.strategy)

    if len(variants) > 1:
        print(f"\nStrategy expansion: {args.strategy} → {variants}")

    output_dir = os.path.join(results_dir, symbol_key)
    os.makedirs(output_dir, exist_ok=True)

    all_range_files = []

    for variant in variants:
        ts_strategy_name = STRATEGY_TS_MAP.get(variant, variant)

        # Load candidates
        if args.candidates and len(variants) == 1:
            candidates_path = args.candidates
        else:
            candidates_path = os.path.join(
                results_dir, symbol_key,
                f"candidates_{variant}_{args.grid}.csv")

        if not os.path.exists(candidates_path):
            print(f"WARNING: Candidates file not found: {candidates_path}")
            print(f"  Skipping variant: {variant}")
            continue

        df = pd.read_csv(candidates_path)
        print(f"\nLoaded {len(df):,} stable combos from {candidates_path}")

        if args.min_phases > 0 and "Phases_In_Top" in df.columns:
            df = df[df["Phases_In_Top"] >= args.min_phases]
            print(f"  After filtering >= {args.min_phases} phases: {len(df):,}")

        # Detect plateaus
        print(f"\n{'='*60}")
        print(f"PLATEAU DETECTION ({variant})")
        print(f"{'='*60}")

        plateaus = detect_plateaus(
            df,
            min_cell_count=args.min_cell_count,
        )

        if not plateaus:
            print(f"No plateaus detected for {variant}. "
                  "Try adjusting --min-cell-count.")
            continue

        # Generate range files
        print(f"\n{'='*60}")
        print(f"GENERATING TS RANGE FILES ({variant})")
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

        all_range_files.extend(range_files)

    # Generate combined config
    if not all_range_files:
        print("\nNo range files generated across any variant.")
        sys.exit(1)

    print()
    generate_config(all_range_files, output_dir)

    # Summary
    if len(variants) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(variants)} variants processed")
        print(f"{'='*60}")
        for v in variants:
            ts_name = STRATEGY_TS_MAP.get(v, v)
            print(f"  {v:25} ({ts_name})")
        print(f"\n  Combined config: {len(all_range_files)} refinement jobs")

    print(f"\nAll files written to: {output_dir}/")
    print("Next: Import NumbaRefinement TradingApp in TS and run refinement.")


if __name__ == "__main__":
    main()
