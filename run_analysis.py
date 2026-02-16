#!/usr/bin/env python3
"""Analyze TS refinement results.

Usage:
    python run_analysis.py --symbol SMH
    python run_analysis.py --symbol SMH --top 10
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
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top results to show")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    results_dir = cfg.get("results_dir", "results")
    symbol_folder = os.path.join(results_dir, args.symbol)

    if not os.path.exists(symbol_folder):
        print(f"ERROR: Symbol folder not found: {symbol_folder}")
        sys.exit(1)

    print(f"Analyzing refinement results in: {symbol_folder}")

    results = load_refinement_results(symbol_folder)
    if not results:
        print("No refinement results found!")
        sys.exit(1)

    analyses = generate_report(results, symbol_folder, args.top)
    save_combined_excel(analyses, symbol_folder, args.symbol)

    print(f"\nAnalysis complete. {len(analyses)} plateaus analyzed.")
    print("Scoring: MAR>0.5 good, Sharpe>1.0 good, PF>1.5 good")
    print("Rank Sum = sum of ranks across 6 metrics (LOWER = BETTER)")


if __name__ == "__main__":
    main()
