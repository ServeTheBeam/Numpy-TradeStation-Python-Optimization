"""WFO phase date configuration and IS/OOS window computation.

Phase definitions match the TS WFO pipeline:
  Phase 1: 10y-5y ago, 60% IS / 40% OOS
  Phase 2:  8y-3y ago, 60% IS / 40% OOS
  Phase 3:  6y-1y ago, 60% IS / 40% OOS
  Phase 4:  4y-present, 62% IS / 38% OOS

For shorter histories, generate_phase_defs() auto-creates 2-4 overlapping
windows scaled to the available data.
"""
from __future__ import annotations

import pandas as pd


# Default phase definitions: (years_back_start, years_back_end, oos_fraction)
DEFAULT_PHASES = [
    (10, 5, 0.40),
    (8,  3, 0.40),
    (6,  1, 0.40),
    (4,  0, 0.38),
]

# Recency-weighted phase weights (for future weighted scoring)
PHASE_WEIGHTS = {1: 0.22, 2: 0.24, 3: 0.26, 4: 0.28}


def generate_phase_defs(
    dates: pd.DatetimeIndex,
) -> tuple[list[tuple[float, float, float]], int]:
    """Auto-generate phase definitions based on actual data length.

    Returns (phase_defs, min_phases) where phase_defs is a list of
    (years_back_start, years_back_end, oos_fraction) tuples.

    Rules:
      >= 10y  → DEFAULT_PHASES, min_phases=3  (exact backward compat)
      >= 6y   → 4 auto-generated phases, min_phases=3
      >= 3y   → 3 phases, min_phases=2
      >= 1.5y → 2 phases, min_phases=2
      < 1.5y  → raise ValueError
    """
    data_days = (dates[-1] - dates[0]).days
    data_years = data_days / 365.25

    if data_years >= 10:
        return DEFAULT_PHASES, 3

    if data_years < 1.5:
        raise ValueError(
            f"Insufficient data for WFO: {data_years:.1f} years "
            f"({data_days} days). Need at least 1.5 years."
        )

    # Determine number of phases
    if data_years >= 6:
        n_phases = 4
        min_phases = 3
    elif data_years >= 3:
        n_phases = 3
        min_phases = 2
    else:
        n_phases = 2
        min_phases = 2

    # Phase geometry: each phase spans ~half the data, evenly spaced
    span = max(data_years * 0.5, 1.5)
    shift = (data_years - span) / (n_phases - 1) if n_phases > 1 else 0.0

    # Safety: if shift < 0.5y, reduce n_phases to avoid near-duplicate windows
    while n_phases > 2 and shift < 0.5:
        n_phases -= 1
        shift = (data_years - span) / (n_phases - 1)
        min_phases = min(min_phases, n_phases)

    phases = []
    for i in range(n_phases):
        yrs_back_start = data_years - i * shift
        yrs_back_end = yrs_back_start - span
        yrs_back_end = max(yrs_back_end, 0.0)
        # Last phase gets slightly lower OOS fraction (matches DEFAULT_PHASES pattern)
        oos_frac = 0.38 if i == n_phases - 1 else 0.40
        phases.append((round(yrs_back_start, 2), round(yrs_back_end, 2), oos_frac))

    print(f"Auto-generated {n_phases} phases for {data_years:.1f}y of data "
          f"(span={span:.2f}y, shift={shift:.2f}y, min_phases={min_phases}):")
    for i, (s, e, f) in enumerate(phases, 1):
        print(f"  Phase {i}: {s:.2f}y -> {e:.2f}y  "
              f"(span={s - e:.2f}y, OOS={f:.0%})")

    return phases, min_phases


def compute_wfo_windows(dates: pd.DatetimeIndex,
                        warmup_bars: int = 100,
                        phase_defs: list | None = None) -> list[dict]:
    """Create overlapping IS/OOS windows matching TS WFO phases.

    Each phase gets warmup bars before the IS period for indicator init.

    Returns list of dicts with:
      phase_num, is_data_start, is_trade_start, is_trade_end,
      is_warmup, oos_data_start, oos_trade_start, oos_trade_end,
      oos_warmup, label
    """
    if phase_defs is None:
        phase_defs = DEFAULT_PHASES

    end_date = dates[-1]
    windows = []

    for phase_num, (yrs_start, yrs_end, oos_frac) in enumerate(phase_defs, 1):
        phase_start = end_date - pd.Timedelta(days=int(yrs_start * 365.25))
        phase_end = end_date - pd.Timedelta(days=int(yrs_end * 365.25))

        # Find nearest bar indices
        start_idx = int(dates.searchsorted(phase_start))
        end_idx = int(dates.searchsorted(phase_end, side="right"))
        start_idx = max(0, start_idx)
        end_idx = min(len(dates), end_idx)

        phase_bars = end_idx - start_idx
        is_bars = int(phase_bars * (1 - oos_frac))
        oos_start_idx = start_idx + is_bars

        # Warmup: use bars preceding the IS start
        is_data_start = max(0, start_idx - warmup_bars)
        is_warmup = start_idx - is_data_start

        is_start_date = dates[start_idx].date()
        is_end_date = dates[min(oos_start_idx - 1, len(dates) - 1)].date()
        oos_start_date = dates[oos_start_idx].date()
        oos_end_date = dates[min(end_idx - 1, len(dates) - 1)].date()

        label = (f"Phase {phase_num}: IS {is_start_date} to {is_end_date} "
                 f"({is_bars}b), OOS {oos_start_date} to {oos_end_date} "
                 f"({end_idx - oos_start_idx}b)")

        windows.append({
            "phase_num": phase_num,
            "is_data_start": is_data_start,
            "is_trade_start": start_idx,
            "oos_trade_start": oos_start_idx,
            "oos_trade_end": end_idx,
            "is_warmup": is_warmup,
            "label": label,
        })

    return windows
