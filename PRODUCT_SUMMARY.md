# Numba WFO

**High-performance walk-forward optimization for systematic trading strategy discovery**

A research-grade pipeline that stress-tests 58M+ parameter combinations across overlapping market regimes, surfaces the robust ones, and exports refinement-ready configurations for TradeStation — all in about 15 minutes on commodity hardware.

---

## The Problem

Manual strategy optimization is slow, brittle, and biased. Most backtests find parameters that worked *in-sample* but collapse *out-of-sample*. Traders waste weeks tweaking knobs on a single symbol, with no systematic way to separate signal from noise.

Walk-forward optimization solves this — but brute-force WFO over millions of combinations is computationally prohibitive in Python, and TradeStation's built-in optimizer can't search at this scale.

## The Solution

Numba WFO replaces guesswork with exhaustive, parallelized parameter search. Every combination is tested across 4 overlapping time windows. Only parameters that prove stable across **multiple independent market regimes** survive. The output isn't a single "best" backtest — it's a *map of robust parameter regions* ready for live deployment.

---

## End-to-End Pipeline

```
  Data Ingestion        WFO Grid Search         Scoring & Filtering
 ┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
 │ TradeStation  │     │  58M+ combos     │     │ Quality gates:    │
 │ API ──────────┼────>│  4 phases        │────>│  - IS & OOS > 0   │
 │ Parquet cache │     │  Numba JIT       │     │  - WFE > 35%      │
 │ Auto-fetch    │     │  10M chunk ||    │     │  - DD sanity      │
 └──────────────┘     └──────────────────┘     │  - 3/4 phases     │
                                                └────────┬──────────┘
                                                         │
                   TS Refinement Export      Plateau Detection
                  ┌──────────────────┐     ┌───────────────────┐
                  │ Range CSVs       │     │ DBSCAN clustering │
                  │ Config manifest  │<────│ Regime labels     │
                  │ Tiered params    │     │ Top-combo extract │
                  └──────────────────┘     └───────────────────┘
```

---

## Strategy Library

Six JIT-compiled strategy kernels built on the **ATR-D v1.5** architecture:

| Variant | Direction | Signals | Data Inputs | Parameters |
|---|---|---|---|---|
| **atr_d_l** | Long-only | ATR trailing stop + buy-touch re-entry | 1 symbol | 11 |
| **atr_d_ls** | Long/Short | + short multiplier, trend veto, entry modes | 1 symbol | 16 |
| **atr_d_hma_l** | Long-only | + Hull MA crossover for trend confirmation | 1 symbol | 13 |
| **atr_d_hma_ls** | Long/Short | + HMA crossover + shorts | 1 symbol | 14 |
| **atr_d_duals_l** | Long-only | Price from Data1, BB Width from Data2 | 2 symbols | 11 |
| **atr_d_duals_ls** | Long/Short | Dual-symbol + full short logic | 2 symbols | 16 |

**Automatic strategy expansion** — requesting `atr_d_duals_ls` also runs `duals_l`, `ls`, and `l` variants so you can see whether dual-symbol and short-side logic actually improve results.

---

## Core Mechanics

### Adaptive Trailing Stop

The stop adjusts dynamically to volatility:

```
stop = highest_since_entry - ATR(period) x multiplier x bb_tightening
```

- **Bollinger Band tightening**: When BB Width drops below a volatility threshold, the stop narrows — protecting profits during low-vol squeezes
- **Three scaling modes**: Linear, square-root, and quadratic tightening curves
- **Max stop distance gate**: A 70-bar ATR baseline caps how wide the stop can drift, preventing catastrophic single-trade losses

### Buy-Touch Re-Entry

After a stop-out, the strategy re-enters when price pulls back to a configurable ATR-based threshold — capturing mean-reversion bounces without chasing momentum.

### Short-Side Logic (LS Variants)

- **Short multiplier** scales the ATR period and multiplier for shorts (e.g. 0.6x = tighter short stops)
- **Trend veto** blocks short entries when the EMA slope is below a threshold
- **Entry modes**: Flat-only transitions, long-priority, or full bidirectional reversals

### Hull Moving Average (HMA Variants)

Fast/slow HMA crossover gates entries. Re-entry via buy-touch requires the HMA trend to still be favorable — no re-entering against the trend.

### Dual-Symbol (DualS Variants)

Trades a leveraged ETF (e.g. TQQQ) using its own price for ATR stops, but computes BB Width from the underlying index (e.g. QQQ) — producing more stable volatility signals on inherently noisy leveraged instruments.

---

## Walk-Forward Optimization

### 4-Phase Overlapping Windows

Each phase is an independent continuous backtest with an IS/OOS split:

```
Phase 1:  ├── 6y IS ──┤── 4y OOS ──┤                    10y → 5y ago
Phase 2:       ├── 5y IS ──┤── 5y OOS ──┤               8y → 3y ago
Phase 3:            ├── 5y IS ──┤── 5y OOS ──┤           6y → 1y ago
Phase 4:                 ├── 4y IS ──┤── 4y OOS ──┤      4y → present
          ─────────────────────────────────────────────>
          past                                    today
```

- **Continuous capital flow**: IS profits compound into OOS, matching real-world equity curves
- **100-bar warmup**: Every phase initializes indicators before the first trade
- **Open position handoff**: Trades spanning the IS/OOS boundary are marked-to-market and carried forward

### Chunk-Parallel Execution

The 58M+ combination space is split into 10M-element chunks, each dispatched to a Numba `@njit(parallel=True)` kernel. Typical runtime: **8-12 minutes on a 32-thread machine**.

---

## Scoring System (v4)

### Quality Gates (Per Phase)

Every combination must clear three filters in each phase it participates in:

| Gate | Rule | Purpose |
|---|---|---|
| **Profitability** | IS PnL > 0 AND OOS PnL > 0 | Eliminates curve-fit losers |
| **Walk-Forward Efficiency** | OOS$ / IS$ > 0.35 | OOS must retain 35%+ of IS edge |
| **Drawdown Sanity** | OOS DD < OOS PnL, OOS DD < 3x IS DD | Catches regime-blown strategies |

### Cross-Phase Stability

A combination must pass quality gates in **3 of 4 phases** to be considered stable. This is the primary robustness filter — one bad phase is tolerated, but consistent failure is not.

If no combinations qualify at 3/4, the threshold auto-relaxes to 2/4 with a warning.

### Stability Score

```
Stability = avg_rank x phase_bonus x consistency_factor
```

| Component | Formula | Effect |
|---|---|---|
| **avg_rank** | Mean percentile rank across valid phases | Higher OOS PnL = higher rank |
| **phase_bonus** | 1 + (phases_valid - 1) x 0.1 | 4-phase combos get 1.3x; 3-phase get 1.2x |
| **consistency_factor** | 1 / (1 + stdev(ranks) / 20) | Penalizes erratic phase-to-phase swings |

The top 2M combinations by stability score are returned as candidates.

---

## Plateau Detection

Rather than picking a single "best" parameter set, the system identifies **regions of parameter space** where many nearby combinations all perform well — plateaus are inherently more robust than isolated peaks.

### How It Works

1. **Aggregate** all candidates into `(atr_multiplier, atr_buy_multiplier)` cells
2. **Filter** cells with fewer than 50 stable combinations
3. **Cluster** the surviving cells using DBSCAN with automated hyperparameter search
4. **Classify** each cluster's stop regime:

| Regime | ATR Multiplier | Character |
|---|---|---|
| **TIGHT** | < 5 | Aggressive stops, more trades, tighter risk |
| **MEDIUM** | 5 - 12 | Balanced risk/reward |
| **WIDE** | > 12 | Loose stops, fewer trades, trend-following |

5. **Extract** the top 10% of combinations from each plateau for refinement

---

## TradeStation Integration

### Range Export

For each plateau, the system generates a `phase5_refine_*.csv` with parameter ranges ready for TradeStation's optimizer:

| Tier | Parameters | Treatment |
|---|---|---|
| **Tier 1** (search) | ATR periods, multipliers, BB deviation, HMA lengths | Min/max/step from plateau bounds |
| **Tier 2-3** (fixed) | Scaling method, entry mode, vol threshold | Locked at median |

### Refinement Analysis

After TradeStation runs Phase 5 refinement, `run_analysis.py` ranks the final results across six metrics with **per-strategy rank-sum composite scoring**:

- **MAR Ratio** — CAGR / Max Drawdown
- **Sharpe Ratio** — risk-adjusted return
- **Sortino Ratio** — downside-risk-adjusted return
- **Profit Factor** — gross profit / gross loss
- **Profit / Drawdown** — net profit / max drawdown
- **Robustness** — % of profitable combinations in the source plateau

Output: a formatted Excel workbook with conditional formatting, ready for final selection.

---

## Data Management

### Auto-Fetch

`run_wfo.py` automatically checks data coverage before every run:

- Reads `{SYMBOL}_1day.meta.json` to determine cached date range
- If data is missing or stale, calls `ParquetStore.get()` to incrementally download only the gap
- Gracefully falls back to cached data if the API is unavailable (weekends, holidays, network issues)
- `--no-fetch` disables auto-fetch for offline/air-gapped environments

### Incremental Parquet Cache

The `ParquetStore` never re-downloads data you already have:

- **Cache hit**: meta.json covers the requested range — instant return
- **Partial miss**: Only fetches the pre-gap or post-gap, merges with existing data
- **Full miss**: Downloads the entire range, writes parquet + meta.json

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Continuous IS→OOS backtest** | Matches TradeStation's single-simulation model; capital compounds naturally |
| **Ring-buffer indicators** | O(1) per-bar updates, cache-friendly memory access for Numba |
| **2D DBSCAN on multipliers** | Interpretable axes (stop width vs. entry sensitivity), directly actionable for refinement |
| **v4 quality-gate scoring** | Earlier versions used top-20% cutoffs that missed high-quality regions; v4 uses hard filters + cross-phase stability |
| **Strategy expansion** | Automatically answers "does adding shorts/dual-symbol/HMA actually help?" without separate manual runs |
| **Yesterday as fetch end-date** | Today's daily bar isn't available until after market close; avoids API 404s on holidays and weekends |

---

## Quick Start

```bash
# Single symbol, long-only — auto-downloads data, runs WFO, detects plateaus, exports ranges
python run_wfo.py --symbol SMH --strategy atr_d_l --grid phase1_slim

# Dual-symbol with full expansion (runs 4 variants automatically)
python run_wfo.py --symbol TQQQ --symbol2 QQQ --strategy atr_d_duals_ls

# Re-score existing results with updated scoring logic
python run_wfo.py --symbol SMH --rescore --wfo-dir results/SMH/wfo/atr_d_l

# Analyze TradeStation refinement output
python run_analysis.py --symbol SMH --top 50

# Manual data download (minute bars, specific date range)
python download_data.py --symbols SMH QQQ --timeframe 1min --start 2016-01-01 --end 2026-02-14
```

---

## Output Structure

```
results/SMH/
├── wfo/
│   ├── atr_d_l/phase1/          # Binary cache: is_pnl.npy, oos_pnl.npy, ...
│   ├── atr_d_l/phase2/
│   ├── atr_d_l/phase3/
│   ├── atr_d_l/phase4/
│   ├── atr_d_ls/phase1/ ... phase4/
│   └── ...
├── candidates_atr_d_l_phase1_slim.csv      # Stable combos with scores
├── candidates_atr_d_ls_phase1_slim.csv
├── phase5_refine_ATR-D-v1.5-L_rank1.csv    # TS refinement ranges (per plateau)
├── phase5_refine_ATR-D-v1.5-L_rank2.csv
├── phase5_refine_ATR-D-v1.5-LS_rank1.csv
├── phase5_refine_config.csv                 # Combined config for all refinement jobs
└── Refinement_Analysis_SMH.xlsx             # Final ranked results
```

---

## Requirements

- Python 3.10+
- Numba, NumPy, Pandas, SciPy, scikit-learn, PyArrow
- TradeStation API credentials (for data download)
- TradeStation desktop (for Phase 5 refinement)
