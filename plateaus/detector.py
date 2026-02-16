"""Plateau detection on WFO stable results.

2D DBSCAN clustering on (atr_multiplier, atr_buy_multiplier) landscape.
Top combos by stability score provide focused parameter ranges for
TS refinement — no expansion needed since the Numba grid already validates
which regions work at fine resolution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def _aggregate_cells(df: pd.DataFrame, min_cell_count: int = 50) -> pd.DataFrame:
    """Aggregate stable combos to (atr_multiplier, atr_buy_multiplier) cells."""
    cells = df.groupby(
        ["atr_multiplier", "atr_buy_multiplier"], as_index=False
    ).agg(
        count=("Stability_Score", "size"),
        avg_stability=("Stability_Score", "mean"),
        max_stability=("Stability_Score", "max"),
        avg_phases=("Phases_In_Top", "mean"),
        in4=("Phases_In_Top", lambda x: (x == 4).sum()),
        avg_oos=("Total_OOS_PnL", "mean"),
        max_oos=("Total_OOS_PnL", "max"),
        avg_trades=("Total_OOS_Trades", "mean"),
    )
    print(f"  {len(cells)} unique (mult, buy_mult) cells")
    sig_cells = cells[cells["count"] >= min_cell_count].copy()
    print(f"  {len(sig_cells)} cells with {min_cell_count}+ stable combos")
    return sig_cells


def _classify_regime(avg_mult: float) -> str:
    """Classify stop-width regime by weighted average multiplier."""
    if avg_mult >= 12:
        return "WIDE"
    elif avg_mult >= 5:
        return "MEDIUM"
    else:
        return "TIGHT"


def detect_plateaus(
    df: pd.DataFrame,
    min_cell_count: int = 50,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    max_noise_pct: float = 60.0,
) -> list[dict]:
    """Detect plateaus via DBSCAN on the 2D (mult, buy_mult) landscape.

    Returns list of plateau dicts sorted by weighted avg stability (best first).
    """
    if eps_values is None:
        eps_values = [0.5, 0.8, 1.0, 1.5, 2.0]
    if min_samples_values is None:
        min_samples_values = [2, 3, 5]

    sig_cells = _aggregate_cells(df, min_cell_count)
    if len(sig_cells) < 5:
        print("  Too few significant cells for DBSCAN.")
        return []

    X = sig_cells[["atr_multiplier", "atr_buy_multiplier"]].values.astype(np.float64)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Grid search for best DBSCAN config
    best_config = None
    best_n = -1
    best_noise = 100.0
    best_labels = None

    for eps in eps_values:
        for ms in min_samples_values:
            cl = DBSCAN(eps=eps, min_samples=ms).fit(X_norm)
            n = len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)
            noise = (cl.labels_ == -1).sum() / len(cl.labels_) * 100
            if noise < max_noise_pct and (n > best_n or (n == best_n and noise < best_noise)):
                best_config = (eps, ms)
                best_n = n
                best_noise = noise
                best_labels = cl.labels_.copy()

    if best_n <= 0:
        print("  DBSCAN found no clusters.")
        return []

    sig_cells = sig_cells.copy()
    sig_cells["cluster"] = best_labels
    print(f"  DBSCAN: eps={best_config[0]}, min_samples={best_config[1]}")
    print(f"  {best_n} clusters, {best_noise:.0f}% noise")

    plateaus = []
    for cid in sorted(sig_cells["cluster"].unique()):
        if cid == -1:
            continue
        members = sig_cells[sig_cells["cluster"] == cid]
        total_combos = int(members["count"].sum())
        total_in4 = int(members["in4"].sum())
        wavg_stab = float(np.average(members["avg_stability"], weights=members["count"]))
        avg_mult = float(np.average(members["atr_multiplier"], weights=members["count"]))
        avg_buy_mult = float(np.average(members["atr_buy_multiplier"], weights=members["count"]))

        regime = _classify_regime(avg_mult)
        mult_range = (float(members["atr_multiplier"].min()),
                      float(members["atr_multiplier"].max()))
        buy_mult_range = (float(members["atr_buy_multiplier"].min()),
                          float(members["atr_buy_multiplier"].max()))

        plateau = {
            "id": int(cid),
            "regime": regime,
            "cells": members,
            "centroid": (avg_mult, avg_buy_mult),
            "mult_range": mult_range,
            "buy_mult_range": buy_mult_range,
            "total_combos": total_combos,
            "in4_combos": total_in4,
            "avg_stability": wavg_stab,
        }
        plateaus.append(plateau)

        print(f"\n  Cluster {cid}: {regime:>6} STOP")
        print(f"    mult={mult_range[0]:.0f}-{mult_range[1]:.0f}, "
              f"buy_mult={buy_mult_range[0]:.1f}-{buy_mult_range[1]:.1f}")
        print(f"    {total_combos:>8,} stable combos ({total_in4:,} in 4 phases)")
        print(f"    Weighted avg stability: {wavg_stab:.1f}")
        print(f"    Centroid: mult={avg_mult:.1f}, buy_mult={avg_buy_mult:.2f}")

    return plateaus


def get_plateau_top_combos(df: pd.DataFrame, plateau: dict,
                           top_pct: float = 10.0) -> pd.DataFrame:
    """Extract top combos from a plateau for range computation.

    Uses the top N% by Stability_Score to focus refinement ranges
    on the best-performing region rather than the full grid span.
    """
    cells = plateau["cells"]
    mask = df["atr_multiplier"].isin(cells["atr_multiplier"]) & \
           df["atr_buy_multiplier"].isin(cells["atr_buy_multiplier"])
    all_combos = df[mask]
    n = max(50, int(len(all_combos) * top_pct / 100))
    return all_combos.nlargest(n, "Stability_Score")
