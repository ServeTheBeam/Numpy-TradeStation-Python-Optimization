#!/usr/bin/env python3
"""Analyze TS refinement results.

Usage:
    python run_analysis.py --symbol SMH
    python run_analysis.py --symbol SMH --top 10
    python run_analysis.py --symbol TQQQ --symbol2 QQQ
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

from analysis.refinement_analyzer import (
    load_refinement_results, generate_report, save_combined_excel,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze TS refinement results")
    parser.add_argument("--symbol", default="SMH", help="Symbol name")
    parser.add_argument("--symbol2", default=None,
                        help="Secondary symbol for DualS strategies")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top results per strategy")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    results_dir = cfg.get("results_dir", "results")
    # Use SYMBOL_SYMBOL2 folder for DualS (matches TS convention)
    symbol_key = f"{args.symbol}_{args.symbol2}" if args.symbol2 else args.symbol
    symbol_folder = os.path.join(results_dir, symbol_key)

    if not os.path.exists(symbol_folder):
        print(f"ERROR: Symbol folder not found: {symbol_folder}")
        sys.exit(1)

    print(f"Analyzing refinement results in: {symbol_folder}")

    results = load_refinement_results(symbol_folder)
    if not results:
        print("No refinement results found!")
        sys.exit(1)

    by_strategy = generate_report(results, symbol_folder, args.top)
    save_combined_excel(by_strategy, symbol_folder, symbol_key)

    total_combos = sum(len(combos) for combos in by_strategy.values())
    print(f"\nAnalysis complete. {total_combos} combos ranked "
          f"across {len(by_strategy)} strategies.")
    print("Scoring: MAR>0.5 good, Sharpe>1.0 good, PF>1.5 good")
    print("Rank Sum = sum of ranks across 6 metrics (LOWER = BETTER)")
    print("NOTE: Rank sums are computed WITHIN each strategy (not cross-strategy)")


if __name__ == "__main__":
    main()
