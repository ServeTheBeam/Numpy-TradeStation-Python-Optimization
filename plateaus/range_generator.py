"""Generate TS-format range CSVs for Phase 5 refinement.

For each plateau, creates:
  1. A range CSV (ParameterName, Min, Max, Step) for TS optimization
  2. A summary row in phase5_refine_config.csv

Tier 1 params: use actual min/max from top combos (no expansion beyond
validated bounds — the Numba grid already confirms which regions work).
Tier 2/3 params: fixed at centroid value (Min=Max)
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np


def _load_range_defs(strategy_name: str,
                     range_dir: str = "strategies/ranges") -> tuple[dict, dict]:
    """Load coarse and fine range definitions for a strategy.

    Returns (coarse_dict, fine_dict) where each is
    {param_name: {min, max, step}}.
    """
    if "HMA" in strategy_name:
        prefix = "_ATR-D-HMA-v1.5-LS" if "-LS" in strategy_name else "_ATR-D-HMA-v1.5-L"
    elif "DualS" in strategy_name:
        prefix = "_ATR-D-DualS-v1.5-LS" if "-LS" in strategy_name else "_ATR-D-DualS-v1.5-L"
    else:
        prefix = "_ATR-D-v1.5-LS" if "-LS" in strategy_name else "_ATR-D-v1.5-L"

    coarse_df = pd.read_csv(os.path.join(range_dir, f"{prefix}_coarse.csv"))
    fine_df = pd.read_csv(os.path.join(range_dir, f"{prefix}_ranges.csv"))

    coarse = {row["ParameterName"]: {"min": row["Min"], "max": row["Max"], "step": row["Step"]}
              for _, row in coarse_df.iterrows()}
    fine = {row["ParameterName"]: {"min": row["Min"], "max": row["Max"], "step": row["Step"]}
            for _, row in fine_df.iterrows()}

    return coarse, fine


# Parameters that use fine step
FINE_PARAMS = {"ATR_Multiplier", "Short_Multiplier", "Trend_Veto_Threshold",
               "HMA_Fast_Length", "HMA_Slow_Length"}
# Parameters that use coarse/3 step
PERIOD_PARAMS = {"ATR_Period", "ATRBuy_Period", "Trend_Filter_Period"}
# Integer parameters
INTEGER_PARAMS = {"ATR_Period", "ATRBuy_Period", "BB_Length", "ATR_Smooth_Period",
                  "HMA_Fast_Length", "HMA_Slow_Length", "Scaling_Method",
                  "EquityPrice_Method", "Entry_Mode", "Both_Triggered_Mode",
                  "Trend_Filter_Period", "Entry_Ratchet"}

# Numba param name -> TS param name mapping
NUMBA_TO_TS = {
    # Base 11
    "atr_period": "ATR_Period",
    "atr_multiplier": "ATR_Multiplier",
    "bb_length": "BB_Length",
    "bb_deviation": "BB_Deviation",
    "vol_threshold": "Vol_Threshold",
    "min_tightening": "Min_Tightening",
    "scaling_method": "Scaling_Method",
    "atr_buy_period": "ATRBuy_Period",
    "atr_buy_multiplier": "ATRBuyMultiplier",
    "atr_smooth_period": "ATR_Smooth_Period",
    "equity_price_method": "EquityPrice_Method",
    # LS additions
    "short_multiplier": "Short_Multiplier",
    "entry_mode": "Entry_Mode",
    "trend_filter_period": "Trend_Filter_Period",
    "trend_veto_threshold": "Trend_Veto_Threshold",
    "both_triggered_mode": "Both_Triggered_Mode",
    # HMA additions
    "hma_fast_length": "HMA_Fast_Length",
    "hma_slow_length": "HMA_Slow_Length",
}


def _calculate_combinations(rows: list[dict]) -> int:
    """Calculate total parameter combinations from output rows."""
    total = 1
    for row in rows:
        if row["Step"] > 0 and row["Max"] > row["Min"]:
            values = int((row["Max"] - row["Min"]) / row["Step"]) + 1
        else:
            values = 1
        total *= values
    return total


def generate_ranges_for_plateau(
    plateau: dict,
    best_combos: pd.DataFrame,
    strategy_name: str,
    rank: int,
    output_dir: str,
    range_dir: str = "strategies/ranges",
) -> tuple[str, int]:
    """Generate a TS range CSV for a single plateau.

    Args:
        plateau: Plateau dict from detector.detect_plateaus()
        best_combos: DataFrame of top combos in this plateau
        strategy_name: e.g. "ATR-D-v1.5-L"
        rank: Plateau rank (1-based)
        output_dir: Where to write the range CSV
        range_dir: Where to find coarse/fine range definitions

    Returns (filename, n_combinations).
    """
    coarse, fine = _load_range_defs(strategy_name, range_dir)
    is_hma = "HMA" in strategy_name

    all_params = list(coarse.keys())

    # Tier 1: parameters expanded in refinement
    if is_hma:
        tier1 = {"ATR_Period", "ATR_Multiplier", "Short_Multiplier",
                 "HMA_Fast_Length", "HMA_Slow_Length",
                 "ATRBuy_Period", "ATRBuyMultiplier", "BB_Deviation",
                 "Min_Tightening", "ATR_Smooth_Period"}
    else:
        tier1 = {"ATR_Period", "ATR_Multiplier", "Short_Multiplier",
                 "Entry_Mode", "Trend_Veto_Threshold",
                 "ATRBuy_Period", "ATRBuyMultiplier", "BB_Deviation",
                 "Min_Tightening", "ATR_Smooth_Period"}

    # Build plateau range (min, max, median) from best combos
    param_range = {}
    for numba_name, ts_name in NUMBA_TO_TS.items():
        if numba_name in best_combos.columns:
            param_range[ts_name] = {
                "min": float(best_combos[numba_name].min()),
                "max": float(best_combos[numba_name].max()),
                "median": float(best_combos[numba_name].median()),
            }

    output_rows = []
    for param in all_params:
        pr = param_range.get(param)
        cent_val = pr["median"] if pr else coarse[param]["min"]

        if param in tier1 and param in fine:
            # Tier 1: use actual min/max from top combos, no expansion
            search_min = pr["min"] if pr else cent_val
            search_max = pr["max"] if pr else cent_val

            # Clamp to fine limits
            search_min = max(search_min, fine[param]["min"])
            search_max = min(search_max, fine[param]["max"])

            # Ensure valid range (at least 2 steps)
            if search_min >= search_max:
                step = fine[param]["step"]
                search_min = max(cent_val - step, fine[param]["min"])
                search_max = min(cent_val + step, fine[param]["max"])

            # Determine step
            if param in FINE_PARAMS:
                step = fine[param]["step"]
            elif param in PERIOD_PARAMS:
                step = max(1, round(coarse[param]["step"] / 3))
            elif param == "ATRBuyMultiplier":
                step = coarse[param]["step"] if is_hma else fine[param]["step"]
            else:
                step = coarse[param]["step"]
        else:
            # Tier 2/3: fixed at median
            search_min = cent_val
            search_max = cent_val
            step = 1

        if param in INTEGER_PARAMS:
            output_rows.append({
                "ParameterName": param,
                "Min": int(round(search_min)),
                "Max": int(round(search_max)),
                "Step": int(round(step)),
            })
        else:
            output_rows.append({
                "ParameterName": param,
                "Min": round(search_min, 4),
                "Max": round(search_max, 4),
                "Step": round(step, 4),
            })

    n_combos = _calculate_combinations(output_rows)
    filename = f"phase5_refine_{strategy_name}_rank{rank}.csv"
    filepath = os.path.join(output_dir, filename)

    pd.DataFrame(output_rows).to_csv(filepath, index=False)
    print(f"  Created: {filename} ({n_combos:,} combos)")

    return filename, n_combos


def generate_config(
    range_files: list[str],
    output_dir: str,
) -> str:
    """Generate phase5_refine_config.csv listing all refinement jobs.

    Args:
        range_files: List of range CSV filenames
        output_dir: Where to write the config CSV

    Returns path to config file.
    """
    config_rows = []
    for i, range_file in enumerate(range_files):
        base = range_file.replace("phase5_refine_", "").replace(".csv", "")
        parts = base.rsplit("_rank", 1)
        strategy_name = parts[0]
        rank = int(parts[1]) if len(parts) > 1 else 1

        strategy_file = "_" + strategy_name
        short_enabled = 1 if "-LS" in strategy_name else 0
        is_dual = 1 if "DualS" in strategy_name else 0

        config_rows.append({
            "JobNum": i + 1,
            "StrategyName": strategy_name,
            "StrategyFile": strategy_file,
            "RangeFile": range_file,
            "Rank": rank,
            "ShortEnabled": short_enabled,
            "IsDualSymbol": is_dual,
        })

    config_df = pd.DataFrame(config_rows)
    config_path = os.path.join(output_dir, "phase5_refine_config.csv")
    config_df.to_csv(config_path, index=False)
    print(f"Created: phase5_refine_config.csv ({len(config_rows)} jobs)")

    return config_path
