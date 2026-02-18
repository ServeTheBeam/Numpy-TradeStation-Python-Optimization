"""Parallel WFO runner — grid search across phases with Numba kernels.

Orchestrates: load data -> JIT warmup -> run all phases -> score -> save.
"""
from __future__ import annotations

import json
import os
import time

import numba
import numpy as np
import pandas as pd

from grids.atr_d_l import count_combinations, decode_flat_indices  # noqa: used as default
from wfo.phases import compute_wfo_windows, generate_phase_defs
from wfo.scoring import score_phases

CHUNK_SIZE = 10_000_000


def _is_dual_symbol(strategy: str) -> bool:
    """True if strategy uses dual data arrays (Data1 + Data2)."""
    return "duals" in strategy


def _get_kernel(strategy: str):
    """Import and return (run_strategy_wfo, run_chunk_wfo) for a strategy."""
    if strategy == "atr_d_l":
        from kernels.atr_d_l import run_strategy_wfo, run_chunk_wfo
        return run_strategy_wfo, run_chunk_wfo
    elif strategy == "atr_d_ls":
        from kernels.atr_d_ls import run_strategy_wfo, run_chunk_wfo
        return run_strategy_wfo, run_chunk_wfo
    elif strategy == "atr_d_hma_l":
        from kernels.atr_d_hma_l import run_strategy_wfo, run_chunk_wfo
        return run_strategy_wfo, run_chunk_wfo
    elif strategy == "atr_d_hma_ls":
        from kernels.atr_d_hma_ls import run_strategy_wfo, run_chunk_wfo
        return run_strategy_wfo, run_chunk_wfo
    elif strategy == "atr_d_duals_l":
        from kernels.atr_d_duals_l import run_strategy_wfo, run_chunk_wfo
        return run_strategy_wfo, run_chunk_wfo
    elif strategy == "atr_d_duals_ls":
        from kernels.atr_d_duals_ls import run_strategy_wfo, run_chunk_wfo
        return run_strategy_wfo, run_chunk_wfo
    else:
        raise ValueError(f"Unknown strategy kernel: {strategy}")


def _warmup_jit(strategy: str, param_arrays,
                highs=None, lows=None, closes=None,
                highs2=None, lows2=None, closes2=None,
                cost_per_trade=0.0, initial_capital=10_000.0):
    """Compile the strategy kernel and parallel chunk runner.

    For single-data strategies, pass highs/lows/closes.
    For dual-data (DualS) strategies, also pass highs2/lows2/closes2.
    """
    run_strategy_wfo, run_chunk_wfo = _get_kernel(strategy)
    dual = _is_dual_symbol(strategy)

    # Synthetic data for warmup
    if highs is None:
        highs = np.linspace(100, 110, 200, dtype=np.float64)
        lows = np.linspace(95, 105, 200, dtype=np.float64)
        closes = np.linspace(98, 108, 200, dtype=np.float64)
    h_w = highs[:200]; l_w = lows[:200]; c_w = closes[:200]

    if dual:
        if highs2 is None:
            highs2 = np.linspace(50, 55, 200, dtype=np.float64)
            lows2 = np.linspace(48, 53, 200, dtype=np.float64)
            closes2 = np.linspace(49, 54, 200, dtype=np.float64)
        h2_w = highs2[:200]; l2_w = lows2[:200]; c2_w = closes2[:200]

    print("Warming up JIT (WFO kernel)...", end=" ", flush=True)
    # Build warmup args for the single-run kernel
    base_args = (14, 2.0, 20, 2.0, 0.10, 0.10, 0, 14, 2.0, 5, 0)

    if strategy == "atr_d_l":
        run_strategy_wfo(h_w, l_w, c_w, *base_args,
                         10, cost_per_trade, initial_capital, 50)
    elif strategy == "atr_d_ls":
        run_strategy_wfo(h_w, l_w, c_w, *base_args,
                         0.5, 1, 5, 0.05, 0,
                         10, cost_per_trade, initial_capital, 50)
    elif strategy == "atr_d_hma_l":
        run_strategy_wfo(h_w, l_w, c_w, *base_args,
                         20, 50,
                         10, cost_per_trade, initial_capital, 50)
    elif strategy == "atr_d_hma_ls":
        run_strategy_wfo(h_w, l_w, c_w, *base_args,
                         0.5, 20, 50,
                         10, cost_per_trade, initial_capital, 50)
    elif strategy == "atr_d_duals_l":
        run_strategy_wfo(h_w, l_w, c_w, h2_w, l2_w, c2_w,
                         *base_args,
                         10, cost_per_trade, initial_capital, 50)
    elif strategy == "atr_d_duals_ls":
        run_strategy_wfo(h_w, l_w, c_w, h2_w, l2_w, c2_w,
                         *base_args,
                         0.5, 1, 5, 0.05, 0,
                         10, cost_per_trade, initial_capital, 50)
    print("done.")

    print("Warming up JIT (WFO parallel runner)...", end=" ", flush=True)
    tiny = 4
    out_args = (
        np.empty(tiny, dtype=np.float64),
        np.empty(tiny, dtype=np.float64),
        np.empty(tiny, dtype=np.int64),
        np.empty(tiny, dtype=np.int64),
        np.empty(tiny, dtype=np.float64),
        np.empty(tiny, dtype=np.float64),
    )
    if dual:
        run_chunk_wfo(h_w, l_w, c_w, h2_w, l2_w, c2_w,
                      *param_arrays, 10, 0, *out_args,
                      cost_per_trade, initial_capital, 50)
    else:
        run_chunk_wfo(h_w, l_w, c_w,
                      *param_arrays, 10, 0, *out_args,
                      cost_per_trade, initial_capital, 50)
    print("done.")


def _run_phase_wfo(run_chunk_wfo_fn, highs, lows, closes, param_arrays,
                   data_start, data_end, warmup_bars, oos_split_bar,
                   total, chunk_size, cost_per_trade,
                   initial_capital=10_000.0,
                   highs2=None, lows2=None, closes2=None):
    """Run all combos on full phase window with continuous IS/OOS split.

    For DualS strategies, pass highs2/lows2/closes2 for Data2.
    Returns (is_pnl, oos_pnl, is_trades, oos_trades, is_maxdd, oos_maxdd).
    """
    h = highs[data_start:data_end].copy()
    l = lows[data_start:data_end].copy()
    c = closes[data_start:data_end].copy()

    dual = highs2 is not None
    if dual:
        h2 = highs2[data_start:data_end].copy()
        l2 = lows2[data_start:data_end].copy()
        c2 = closes2[data_start:data_end].copy()

    split_bar = oos_split_bar - data_start

    out_is_pnl = np.empty(total, dtype=np.float64)
    out_oos_pnl = np.empty(total, dtype=np.float64)
    out_is_trades = np.empty(total, dtype=np.int64)
    out_oos_trades = np.empty(total, dtype=np.int64)
    out_is_maxdd = np.empty(total, dtype=np.float64)
    out_oos_maxdd = np.empty(total, dtype=np.float64)

    t0 = time.perf_counter()
    for cs_start in range(0, total, chunk_size):
        cs = min(chunk_size, total - cs_start)
        out_slice = (
            out_is_pnl[cs_start:cs_start + cs],
            out_oos_pnl[cs_start:cs_start + cs],
            out_is_trades[cs_start:cs_start + cs],
            out_oos_trades[cs_start:cs_start + cs],
            out_is_maxdd[cs_start:cs_start + cs],
            out_oos_maxdd[cs_start:cs_start + cs],
        )
        if dual:
            run_chunk_wfo_fn(
                h, l, c, h2, l2, c2,
                *param_arrays, warmup_bars, cs_start,
                *out_slice,
                cost_per_trade, initial_capital, split_bar,
            )
        else:
            run_chunk_wfo_fn(
                h, l, c,
                *param_arrays, warmup_bars, cs_start,
                *out_slice,
                cost_per_trade, initial_capital, split_bar,
            )
        processed = cs_start + cs
        elapsed = time.perf_counter() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else 0
        print(f"    {processed:>12,} / {total:,}  ({processed/total*100:.1f}%)  "
              f"{rate:,.0f}/sec  ETA: {remaining:.0f}s")

    elapsed = time.perf_counter() - t0
    is_prof = int((out_is_pnl > 0).sum())
    oos_prof = int((out_oos_pnl > 0).sum())
    print(f"    {elapsed:.1f}s  |  IS profitable: {is_prof:,}  "
          f"OOS profitable: {oos_prof:,}  "
          f"|  OOS PnL: ${out_oos_pnl.min():.0f} to ${out_oos_pnl.max():.0f}")

    return out_is_pnl, out_oos_pnl, out_is_trades, out_oos_trades, out_is_maxdd, out_oos_maxdd


def run_wfo(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    dates: pd.DatetimeIndex,
    grid: dict[str, np.ndarray],
    strategy: str = "atr_d_l",
    output_dir: str = "results/SMH/wfo",
    chunk_size: int = CHUNK_SIZE,
    warmup_bars: int = 100,
    top_n: int = 2_000_000,
    cost_per_trade: float = 0.0,
    min_phases: int | None = None,
    initial_capital: float = 10_000.0,
    highs2: np.ndarray | None = None,
    lows2: np.ndarray | None = None,
    closes2: np.ndarray | None = None,
    phase_defs: list | None = None,
) -> pd.DataFrame:
    """Walk-Forward Optimization with IS/OOS split matching TS pipeline.

    Each phase runs as a SINGLE continuous backtest (matching TS behaviour).
    Trades are assigned to IS or OOS by entry bar. Capital compounds from
    IS into OOS.

    Returns DataFrame of stable combos sorted by stability score.
    """
    param_names = list(grid.keys())
    param_arrays = [grid[k] for k in param_names]
    total = count_combinations(grid)
    dual = _is_dual_symbol(strategy)

    if dual and (highs2 is None or lows2 is None or closes2 is None):
        raise ValueError(f"Strategy '{strategy}' requires Data2 arrays "
                         "(highs2, lows2, closes2)")

    _, run_chunk_wfo_fn = _get_kernel(strategy)

    # Auto-generate phase definitions if not provided
    if phase_defs is None:
        phase_defs, auto_min_phases = generate_phase_defs(dates)
        if min_phases is None:
            min_phases = auto_min_phases
    if min_phases is None:
        min_phases = 3

    windows = compute_wfo_windows(dates, warmup_bars, phase_defs)

    print(f"WFO IS/OOS: {len(windows)} phases, {total:,} combos each")
    if dual:
        print(f"Dual-symbol mode: Data1 ({len(highs)} bars) + Data2 ({len(highs2)} bars)")
    print(f"Numba threads: {numba.config.NUMBA_NUM_THREADS}")
    for w in windows:
        print(f"  {w['label']}")

    os.makedirs(output_dir, exist_ok=True)

    # JIT warmup
    _warmup_jit(strategy, param_arrays,
                highs=highs[:200], lows=lows[:200], closes=closes[:200],
                highs2=highs2[:200] if dual else None,
                lows2=lows2[:200] if dual else None,
                closes2=closes2[:200] if dual else None,
                cost_per_trade=cost_per_trade,
                initial_capital=initial_capital)

    # --- Run all phases ---
    phase_is_pnl = {}
    phase_oos_pnl = {}
    phase_is_trades = {}
    phase_oos_trades = {}
    phase_is_maxdd = {}
    phase_oos_maxdd = {}
    t_total_start = time.perf_counter()

    for w in windows:
        pn = w["phase_num"]
        is_bars = w["oos_trade_start"] - w["is_trade_start"]
        oos_bars = w["oos_trade_end"] - w["oos_trade_start"]
        print(f"\n{'='*60}")
        print(f"{w['label']}")
        print(f"{'='*60}")

        print(f"  Continuous run ({is_bars}+{oos_bars} bars, "
              f"{w['is_warmup']} warmup):")
        is_pnl, oos_pnl, is_tr, oos_tr, is_dd, oos_dd = _run_phase_wfo(
            run_chunk_wfo_fn, highs, lows, closes, param_arrays,
            w["is_data_start"], w["oos_trade_end"],
            w["is_warmup"], w["oos_trade_start"],
            total, chunk_size, cost_per_trade, initial_capital,
            highs2=highs2, lows2=lows2, closes2=closes2,
        )

        phase_is_pnl[pn] = is_pnl
        phase_oos_pnl[pn] = oos_pnl
        phase_is_trades[pn] = is_tr
        phase_oos_trades[pn] = oos_tr
        phase_is_maxdd[pn] = is_dd
        phase_oos_maxdd[pn] = oos_dd

        # Save per-phase binary results
        phase_dir = os.path.join(output_dir, f"phase{pn}")
        os.makedirs(phase_dir, exist_ok=True)
        np.save(os.path.join(phase_dir, "is_pnl.npy"), is_pnl)
        np.save(os.path.join(phase_dir, "oos_pnl.npy"), oos_pnl)
        np.save(os.path.join(phase_dir, "is_trades.npy"), is_tr)
        np.save(os.path.join(phase_dir, "oos_trades.npy"), oos_tr)
        np.save(os.path.join(phase_dir, "is_maxdd.npy"), is_dd)
        np.save(os.path.join(phase_dir, "oos_maxdd.npy"), oos_dd)

    total_elapsed = time.perf_counter() - t_total_start
    print(f"\n{'='*60}")
    print(f"All phases completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Save metadata
    meta = {
        "total": total,
        "param_names": param_names,
        "param_sizes": [len(a) for a in param_arrays],
        "n_phases": len(windows),
        "windows": [w["label"] for w in windows],
        "min_phases": min_phases,
        "initial_capital": initial_capital,
        "cost_per_trade": cost_per_trade,
        "strategy": strategy,
    }
    for name, arr in zip(param_names, param_arrays):
        np.save(os.path.join(output_dir, f"param_{name}.npy"), arr)
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # --- Score ---
    print("\nComputing WFE scores per phase...")
    n_phases = len(windows)
    scoring_result = score_phases(
        phase_is_pnl, phase_oos_pnl, phase_oos_trades,
        phase_is_maxdd, phase_oos_maxdd,
        total, min_phases, top_n, n_phases=n_phases,
    )

    # --- Build output DataFrame ---
    top_idx = scoring_result["stable_idx"]
    stability_scores = scoring_result["stability_scores"]
    phases_valid = scoring_result["phases_valid"]

    params = decode_flat_indices(top_idx, param_names, param_arrays)
    df = pd.DataFrame(params)
    df["Stability_Score"] = stability_scores[top_idx]
    df["Phases_In_Top"] = phases_valid[top_idx]
    for w in windows:
        pn = w["phase_num"]
        df[f"P{pn}_OOS_PnL"] = phase_oos_pnl[pn][top_idx]
        df[f"P{pn}_IS_PnL"] = phase_is_pnl[pn][top_idx]
        df[f"P{pn}_OOS_Rank"] = scoring_result["phase_oos_rank"][pn][top_idx]
        df[f"P{pn}_OOS_Trades"] = phase_oos_trades[pn][top_idx]
        df[f"P{pn}_WFE"] = scoring_result["phase_wfe_score"][pn][top_idx]

    total_oos_pnl = np.zeros(len(top_idx), dtype=np.float64)
    total_oos_trades = np.zeros(len(top_idx), dtype=np.int64)
    for w in windows:
        pn = w["phase_num"]
        total_oos_pnl += phase_oos_pnl[pn][top_idx]
        total_oos_trades += phase_oos_trades[pn][top_idx]
    df["Total_OOS_PnL"] = total_oos_pnl
    df["Total_OOS_Trades"] = total_oos_trades

    return df


def rescore_from_cache(
    grid: dict[str, np.ndarray],
    wfo_dir: str,
    min_phases: int = 3,
    top_n: int = 2_000_000,
    n_phases: int | None = None,
) -> pd.DataFrame:
    """Re-score cached .npy WFO results with current scoring parameters.

    Useful for iterating on scoring without re-running the WFO grid.
    Auto-detects n_phases from meta.json or phase directories if not specified.
    """
    param_names = list(grid.keys())
    param_arrays = [grid[k] for k in param_names]
    total = count_combinations(grid)
    print(f"Grid: {total:,} combos")

    # Auto-detect n_phases (and min_phases) from meta.json or directory listing
    if n_phases is None:
        meta_path = os.path.join(wfo_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            n_phases = meta.get("n_phases", 4)
            min_phases = meta.get("min_phases", min_phases)
            print(f"Auto-detected {n_phases} phases, "
                  f"min_phases={min_phases} from meta.json")
        else:
            # Fallback: count phase directories
            pn = 1
            while os.path.isdir(os.path.join(wfo_dir, f"phase{pn}")):
                pn += 1
            n_phases = pn - 1
            if n_phases == 0:
                raise FileNotFoundError(
                    f"No phase directories found in {wfo_dir}")
            print(f"Auto-detected {n_phases} phases from directories")

    phase_is_pnl = {}
    phase_oos_pnl = {}
    phase_oos_trades = {}
    phase_is_maxdd = {}
    phase_oos_maxdd = {}

    for pn in range(1, n_phases + 1):
        pdir = os.path.join(wfo_dir, f"phase{pn}")
        phase_is_pnl[pn] = np.load(os.path.join(pdir, "is_pnl.npy"))
        phase_oos_pnl[pn] = np.load(os.path.join(pdir, "oos_pnl.npy"))
        phase_oos_trades[pn] = np.load(os.path.join(pdir, "oos_trades.npy"))
        phase_is_maxdd[pn] = np.load(os.path.join(pdir, "is_maxdd.npy"))
        phase_oos_maxdd[pn] = np.load(os.path.join(pdir, "oos_maxdd.npy"))

    scoring_result = score_phases(
        phase_is_pnl, phase_oos_pnl, phase_oos_trades,
        phase_is_maxdd, phase_oos_maxdd,
        total, min_phases, top_n, n_phases=n_phases,
    )

    top_idx = scoring_result["stable_idx"]
    stability_scores = scoring_result["stability_scores"]
    phases_valid = scoring_result["phases_valid"]

    params = decode_flat_indices(top_idx, param_names, param_arrays)
    df = pd.DataFrame(params)
    df["Stability_Score"] = stability_scores[top_idx]
    df["Phases_In_Top"] = phases_valid[top_idx]
    for pn in range(1, n_phases + 1):
        df[f"P{pn}_OOS_PnL"] = phase_oos_pnl[pn][top_idx]
        df[f"P{pn}_IS_PnL"] = phase_is_pnl[pn][top_idx]
        df[f"P{pn}_OOS_Rank"] = scoring_result["phase_oos_rank"][pn][top_idx]
        df[f"P{pn}_OOS_Trades"] = phase_oos_trades[pn][top_idx]
        df[f"P{pn}_WFE"] = scoring_result["phase_wfe_score"][pn][top_idx]

    total_oos_pnl = np.zeros(len(top_idx), dtype=np.float64)
    total_oos_trades = np.zeros(len(top_idx), dtype=np.int64)
    for pn in range(1, n_phases + 1):
        total_oos_pnl += phase_oos_pnl[pn][top_idx]
        total_oos_trades += phase_oos_trades[pn][top_idx]
    df["Total_OOS_PnL"] = total_oos_pnl
    df["Total_OOS_Trades"] = total_oos_trades

    return df
