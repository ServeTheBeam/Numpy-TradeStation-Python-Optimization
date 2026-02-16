"""v4 WFO stability scoring — quality filters + cross-phase stability.

Scoring evolution:
  v1-v3: Top 20% gate per phase -> cross-phase. Missed TS-optimal region.
  v4 (current): Quality filters only (IS>0, OOS>0, WFE>0.35, DD filters),
    no top-20% cutoff.  Cross-phase requirement (valid 3+ phases) is the
    real stability filter.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


# Scoring thresholds
INITIAL_CAPITAL = 10_000.0
WFE_DOLLAR_MIN = 0.35
MIN_DD_THRESHOLD = 2000.0


def score_phases(phase_is_pnl: dict, phase_oos_pnl: dict,
                 phase_oos_trades: dict,
                 phase_is_maxdd: dict, phase_oos_maxdd: dict,
                 total: int, min_phases: int = 3,
                 top_n: int = 2_000_000) -> dict:
    """Apply v4 quality filters and cross-phase stability scoring.

    Args:
        phase_is_pnl: {phase_num: np.ndarray} IS PnL per combo
        phase_oos_pnl: {phase_num: np.ndarray} OOS PnL per combo
        phase_oos_trades: {phase_num: np.ndarray} OOS trade count
        phase_is_maxdd: {phase_num: np.ndarray} IS max drawdown
        phase_oos_maxdd: {phase_num: np.ndarray} OOS max drawdown
        total: Total number of parameter combos
        min_phases: Minimum phases a combo must pass
        top_n: Max results to return

    Returns dict with:
        stable_idx: indices of stable combos (sorted by stability desc)
        stability_scores: score per combo
        phases_valid: count of valid phases per combo
        phase_valid_mask: {pn: bool array}
        phase_oos_rank: {pn: percentile rank array}
        phase_wfe_score: {pn: WFE score array}
    """
    phase_wfe_score = {}
    phase_oos_rank = {}
    phase_valid_mask = {}

    for pn in range(1, 5):
        is_pnl = phase_is_pnl[pn]
        oos_pnl = phase_oos_pnl[pn]
        is_maxdd_arr = phase_is_maxdd[pn]
        oos_maxdd = phase_oos_maxdd[pn]

        # Quality filters: both IS and OOS must be profitable
        valid = (is_pnl > 0) & (oos_pnl > 0)

        # WFE calculation
        is_pct = is_pnl / INITIAL_CAPITAL
        oos_starting = INITIAL_CAPITAL + is_pnl
        oos_pct = np.where(oos_starting > 0, oos_pnl / oos_starting, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            wfe = np.where(is_pct > 0, oos_pct / is_pct, 0.0)
        wfe_factor = np.clip(wfe, 0.8, 1.2)
        score = oos_pct * wfe_factor

        # Dollar WFE filter (OOS$/IS$ > threshold)
        with np.errstate(divide="ignore", invalid="ignore"):
            dollar_wfe = np.where(is_pnl > 0, oos_pnl / is_pnl, 0.0)
        valid &= dollar_wfe > WFE_DOLLAR_MIN

        # Drawdown filters (only if OOS DD > threshold)
        dd_check = oos_maxdd > MIN_DD_THRESHOLD
        valid &= ~(dd_check & (oos_maxdd > oos_pnl))
        valid &= ~(dd_check & (is_maxdd_arr > 0) & (oos_maxdd > 3 * is_maxdd_arr))

        score = np.where(valid, score, 0.0)

        # Percentile rank among valid combos
        oos_rank = np.zeros(total, dtype=np.float64)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            ranks = rankdata(oos_pnl[valid_idx], method="average")
            oos_rank[valid_idx] = ranks / len(ranks) * 100

        n_valid = int(valid.sum())
        print(f"  Phase {pn}: {n_valid:,} valid ({n_valid/total*100:.1f}%)")

        phase_wfe_score[pn] = score
        phase_oos_rank[pn] = oos_rank
        phase_valid_mask[pn] = valid

    # --- Cross-phase stability ---
    print(f"\nCross-phase stability (valid in {min_phases}+ phases)...")

    phases_valid = np.zeros(total, dtype=np.int64)
    for pn in range(1, 5):
        phases_valid += phase_valid_mask[pn].astype(np.int64)

    stable_mask = phases_valid >= min_phases
    n_stable = int(stable_mask.sum())
    print(f"  Stable combos: {n_stable:,} ({n_stable/total*100:.2f}%)")

    for count in range(4, 0, -1):
        n = int((phases_valid == count).sum())
        print(f"    In {count} phases: {n:,}")

    if n_stable == 0:
        print("WARNING: No stable combos found. Relaxing to min_phases=2.")
        min_phases = 2
        stable_mask = phases_valid >= min_phases
        n_stable = int(stable_mask.sum())

    # Stability score: avg_rank * phase_bonus * consistency_factor
    stable_idx = np.where(stable_mask)[0]
    stability_scores = np.zeros(total, dtype=np.float64)

    for idx in stable_idx:
        ranks = []
        for pn in range(1, 5):
            if phase_valid_mask[pn][idx]:
                ranks.append(phase_oos_rank[pn][idx])

        avg_rank = np.mean(ranks)
        phase_count = len(ranks)
        phase_bonus = 1.0 + (phase_count - 1) * 0.1
        rank_std = np.std(ranks) if len(ranks) > 1 else 0.0
        consistency_factor = 1.0 / (1.0 + rank_std / 20.0)

        stability_scores[idx] = avg_rank * phase_bonus * consistency_factor

    # Sort stable combos by stability score (descending)
    top_n = min(top_n, n_stable)
    if top_n > 0:
        sorted_idx = stable_idx[np.argsort(stability_scores[stable_idx])[::-1][:top_n]]
    else:
        sorted_idx = np.array([], dtype=np.int64)

    return {
        "stable_idx": sorted_idx,
        "stability_scores": stability_scores,
        "phases_valid": phases_valid,
        "phase_valid_mask": phase_valid_mask,
        "phase_oos_rank": phase_oos_rank,
        "phase_wfe_score": phase_wfe_score,
    }
