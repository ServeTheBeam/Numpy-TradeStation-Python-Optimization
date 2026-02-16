"""Grid definitions for ATR-D HMA v1.5 Long/Short.

Parameter order MUST match the kernel's p0..p13 signature:
  p0=atr_period, p1=atr_multiplier, p2=bb_length, p3=bb_deviation,
  p4=vol_threshold, p5=min_tightening, p6=scaling_method,
  p7=atr_buy_period, p8=atr_buy_multiplier, p9=atr_smooth_period,
  p10=equity_price_method,
  p11=short_multiplier, p12=hma_fast_length, p13=hma_slow_length
"""
from __future__ import annotations

import numpy as np

PARAM_ORDER = [
    "atr_period", "atr_multiplier", "bb_length", "bb_deviation",
    "vol_threshold", "min_tightening", "scaling_method",
    "atr_buy_period", "atr_buy_multiplier", "atr_smooth_period",
    "equity_price_method",
    "short_multiplier", "hma_fast_length", "hma_slow_length",
]

INT_PARAMS = {"atr_period", "bb_length", "scaling_method",
              "atr_buy_period", "atr_smooth_period", "equity_price_method",
              "hma_fast_length", "hma_slow_length"}

GRIDS = {}


def _register(name, grid_dict):
    ordered = {}
    for k in PARAM_ORDER:
        if k not in grid_dict:
            raise KeyError(f"Grid '{name}' missing parameter '{k}'")
        ordered[k] = grid_dict[k]
    GRIDS[name] = ordered


# ---------------------------------------------------------------------------
# ts_coarse — Match TS coarse grid (from _ATR-D-HMA-v1.5-LS_coarse.csv)
# ---------------------------------------------------------------------------
_register("ts_coarse", {
    "atr_period":          np.array([8, 32, 56], dtype=np.int64),
    "atr_multiplier":      np.array([2, 9, 16], dtype=np.float64),
    "bb_length":           np.array([15, 55], dtype=np.int64),
    "bb_deviation":        np.array([1.5, 3.5], dtype=np.float64),
    "vol_threshold":       np.array([0.06, 0.30], dtype=np.float64),
    "min_tightening":      np.array([0.2, 0.8], dtype=np.float64),
    "scaling_method":      np.array([0, 2], dtype=np.int64),
    "atr_buy_period":      np.array([8, 32, 56], dtype=np.int64),
    "atr_buy_multiplier":  np.array([2.0, 8.0, 14.0], dtype=np.float64),
    "atr_smooth_period":   np.array([3, 9, 15], dtype=np.int64),
    "equity_price_method": np.array([0, 1], dtype=np.int64),
    "short_multiplier":    np.array([0.2, 0.6, 1.0], dtype=np.float64),
    "hma_fast_length":     np.arange(15, 56, 8, dtype=np.int64),
    "hma_slow_length":     np.arange(45, 86, 10, dtype=np.int64),
})

# ---------------------------------------------------------------------------
# phase1_slim — Expanded grid for WFO search
# ---------------------------------------------------------------------------
_register("phase1_slim", {
    "atr_period":          np.array([4, 16, 28, 40, 52, 64], dtype=np.int64),
    "atr_multiplier":      np.arange(2, 21, 2, dtype=np.float64),
    "bb_length":           np.array([10, 25, 40, 55], dtype=np.int64),
    "bb_deviation":        np.arange(1.0, 4.1, 1.0, dtype=np.float64),
    "vol_threshold":       np.arange(0.04, 0.32, 0.06, dtype=np.float64),
    "min_tightening":      np.arange(0.10, 0.91, 0.20, dtype=np.float64),
    "scaling_method":      np.array([0, 1, 2], dtype=np.int64),
    "atr_buy_period":      np.array([8, 20, 32, 44, 56], dtype=np.int64),
    "atr_buy_multiplier":  np.arange(1.0, 15.1, 2.0, dtype=np.float64),
    "atr_smooth_period":   np.array([3, 7, 12], dtype=np.int64),
    "equity_price_method": np.array([0], dtype=np.int64),
    "short_multiplier":    np.array([0.2, 0.5, 0.8, 1.0], dtype=np.float64),
    "hma_fast_length":     np.arange(10, 61, 5, dtype=np.int64),
    "hma_slow_length":     np.arange(40, 101, 10, dtype=np.int64),
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
