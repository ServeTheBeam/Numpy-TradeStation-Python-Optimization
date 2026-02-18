"""Strategy expansion and TS name mapping.

Expansion rules:
  - LS strategies add L variant (does long-short help vs long-only?)
  - DualS strategies add non-dual variant (does dual-symbol help vs single?)
  - These compound: DualS-LS → {DualS-LS, DualS-L, LS, L}
  - HMA stays within family (HMA-LS → HMA-L, not base LS/L)
"""
from __future__ import annotations

# Canonical mapping: Numba kernel name → TradeStation strategy name
STRATEGY_TS_MAP = {
    "atr_d_l": "ATR-D-v1.5-L",
    "atr_d_ls": "ATR-D-v1.5-LS",
    "atr_d_hma_l": "ATR-D-HMA-v1.5-L",
    "atr_d_hma_ls": "ATR-D-HMA-v1.5-LS",
    "atr_d_duals_l": "ATR-D-DualS-v1.5-L",
    "atr_d_duals_ls": "ATR-D-DualS-v1.5-LS",
}


def expand_strategy(strategy: str) -> list[str]:
    """Expand a strategy into its comparison variants.

    Rules:
      - LS adds L variant
      - DualS adds non-dual variant
      - These compound (DualS-LS → 4 variants)
      - HMA stays within HMA family

    Examples:
        atr_d_l            → [atr_d_l]
        atr_d_ls           → [atr_d_ls, atr_d_l]
        atr_d_hma_l        → [atr_d_hma_l]
        atr_d_hma_ls       → [atr_d_hma_ls, atr_d_hma_l]
        atr_d_duals_l      → [atr_d_duals_l, atr_d_l]
        atr_d_duals_ls     → [atr_d_duals_ls, atr_d_duals_l, atr_d_ls, atr_d_l]
    """
    variants = [strategy]

    # LS → add L variant (strip trailing _ls → _l)
    if strategy.endswith("_ls"):
        l_variant = strategy[:-3] + "_l"
        if l_variant not in variants:
            variants.append(l_variant)

    # DualS → add non-dual variants (remove _duals from name)
    if "_duals_" in strategy:
        non_dual = strategy.replace("_duals_", "_")
        if non_dual not in variants:
            variants.append(non_dual)
        # If the non-dual is also LS, add its L variant too
        if non_dual.endswith("_ls"):
            non_dual_l = non_dual[:-3] + "_l"
            if non_dual_l not in variants:
                variants.append(non_dual_l)

    return variants
