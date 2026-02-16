"""Grid definitions for ATR-D DualSymbol v1.5 Long-only.

Same parameter set as base L (11 params, p0..p10) but with different
default ranges. The DualS kernel takes 6 data arrays but the SAME
parameter grid structure.

Parameter order MUST match the kernel's p0..p10 signature:
  p0=atr_period, p1=atr_multiplier, p2=bb_length, p3=bb_deviation,
  p4=vol_threshold, p5=min_tightening, p6=scaling_method,
  p7=atr_buy_period, p8=atr_buy_multiplier, p9=atr_smooth_period,
  p10=equity_price_method
"""
from __future__ import annotations

import numpy as np

PARAM_ORDER = [
    "atr_period", "atr_multiplier", "bb_length", "bb_deviation",
    "vol_threshold", "min_tightening", "scaling_method",
    "atr_buy_period", "atr_buy_multiplier", "atr_smooth_period",
    "equity_price_method",
]

INT_PARAMS = {"atr_period", "bb_length", "scaling_method",
              "atr_buy_period", "atr_smooth_period", "equity_price_method"}

GRIDS = {}


def _register(name, grid_dict):
    ordered = {}
    for k in PARAM_ORDER:
        if k not in grid_dict:
            raise KeyError(f"Grid '{name}' missing parameter '{k}'")
        ordered[k] = grid_dict[k]
    GRIDS[name] = ordered


# ---------------------------------------------------------------------------
# ts_coarse — Match TS coarse grid (from _ATR-D-DualS-v1.5-L_coarse.csv)
# ---------------------------------------------------------------------------
_register("ts_coarse", {
    "atr_period":          np.array([8, 32, 56], dtype=np.int64),
    "atr_multiplier":      np.arange(2, 21, 2, dtype=np.float64),   # 2,4,...,20
    "bb_length":           np.array([15, 35, 55], dtype=np.int64),
    "bb_deviation":        np.array([1.5, 3.5], dtype=np.float64),
    "vol_threshold":       np.array([0.06, 0.18, 0.30], dtype=np.float64),
    "min_tightening":      np.array([0.2, 0.8], dtype=np.float64),
    "scaling_method":      np.array([0, 2], dtype=np.int64),
    "atr_buy_period":      np.array([10, 30, 50], dtype=np.int64),
    "atr_buy_multiplier":  np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
    "atr_smooth_period":   np.array([3, 9, 15], dtype=np.int64),
    "equity_price_method": np.array([0, 1], dtype=np.int64),
})

# ---------------------------------------------------------------------------
# phase1_slim — Expanded grid for WFO search
# ---------------------------------------------------------------------------
_register("phase1_slim", {
    "atr_period":          np.array([4, 16, 28, 40, 52, 64], dtype=np.int64),
    "atr_multiplier":      np.arange(2, 25, 2, dtype=np.float64),
    "bb_length":           np.array([10, 25, 40, 55], dtype=np.int64),
    "bb_deviation":        np.arange(0.5, 4.1, 0.5, dtype=np.float64),
    "vol_threshold":       np.arange(0.04, 0.32, 0.04, dtype=np.float64),
    "min_tightening":      np.arange(0.10, 0.96, 0.15, dtype=np.float64),
    "scaling_method":      np.array([0, 1, 2], dtype=np.int64),
    "atr_buy_period":      np.array([8, 20, 32, 44, 56], dtype=np.int64),
    "atr_buy_multiplier":  np.arange(0.5, 4.0, 0.5, dtype=np.float64),
    "atr_smooth_period":   np.array([2, 6, 10, 14, 18], dtype=np.int64),
    "equity_price_method": np.array([0], dtype=np.int64),
})


def get_grid(name: str) -> dict[str, np.ndarray]:
    if name not in GRIDS:
        available = ", ".join(sorted(GRIDS.keys()))
        raise ValueError(f"Unknown grid '{name}'. Available: {available}")
    return GRIDS[name]


def count_combinations(grid: dict[str, np.ndarray]) -> int:
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def decode_flat_indices(flat_indices, param_names, param_arrays):
    sizes = [len(a) for a in param_arrays]
    result = {}
    remainder = flat_indices.copy()
    for dim in reversed(range(len(sizes))):
        dim_indices = remainder % sizes[dim]
        remainder //= sizes[dim]
        result[param_names[dim]] = param_arrays[dim][dim_indices]
    return result
