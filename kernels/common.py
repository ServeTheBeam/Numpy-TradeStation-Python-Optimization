"""Shared Numba-jitted helpers used by all strategy kernels.

All functions are @njit(cache=True) so they compile once and are reusable
across kernels without re-compilation overhead.
"""
from __future__ import annotations

import math
import numba
import numpy as np


# ---------------------------------------------------------------------------
# ATR (SMA of TrueRange) via ring buffer
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def atr_sma_init(period: int):
    """Return initial state for an ATR SMA ring buffer.

    Returns (buf, idx, count, running_sum, atr_val).
    """
    buf = np.zeros(period, dtype=np.float64)
    return buf, 0, 0, 0.0, 0.0


@numba.njit(cache=True)
def atr_sma_update(buf, idx, count, running_sum, tr, period):
    """Push a new TrueRange value and return updated state + current ATR.

    Returns (buf, idx, count, running_sum, atr_val).
    """
    if count < period:
        buf[idx] = tr
        running_sum += tr
        count += 1
        atr_val = running_sum / count
    else:
        old = buf[idx]
        buf[idx] = tr
        running_sum += tr - old
        atr_val = running_sum / period
    idx = (idx + 1) % period
    return buf, idx, count, running_sum, atr_val


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def ema_update(prev_ema: float, new_val: float, alpha: float,
               started: bool) -> tuple:
    """Single EMA step.  Returns (new_ema, started=True)."""
    if not started:
        return new_val, True
    return prev_ema + alpha * (new_val - prev_ema), True


# ---------------------------------------------------------------------------
# Bollinger Band Width on a Close buffer
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def calc_bb_width(close_buf, bb_length: int, bb_deviation: float,
                  buf_count: int) -> float:
    """Compute BB Width = (upper - lower) / mean using population stdev."""
    if buf_count < bb_length:
        return 0.0
    total = 0.0
    for j in range(bb_length):
        total += close_buf[j]
    mean = total / bb_length
    if mean <= 0.0:
        return 0.0
    var_sum = 0.0
    for j in range(bb_length):
        diff = close_buf[j] - mean
        var_sum += diff * diff
    std = math.sqrt(var_sum / bb_length)
    upper = mean + bb_deviation * std
    lower = mean - bb_deviation * std
    return (upper - lower) / mean


@numba.njit(cache=True)
def calc_bb_multiplier(bb_width: float, vol_threshold: float,
                       min_tightening: float, scaling_method: int) -> float:
    """Compute the BB tightening multiplier (0..1)."""
    if bb_width >= vol_threshold:
        return 1.0
    if vol_threshold > 0:
        ratio = bb_width / vol_threshold
    else:
        ratio = 0.0
    if scaling_method == 0:      # Linear
        mult = ratio
    elif scaling_method == 1:    # Sqrt
        mult = math.sqrt(ratio) if ratio > 0 else 0.0
    else:                        # Exponential (quadratic)
        mult = ratio * ratio
    if mult < min_tightening:
        mult = min_tightening
    return mult


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def calc_position_size(equity: float, price: float) -> int:
    """Floor(equity / price) shares — returns 0 if can't afford 1 share."""
    if price <= 0.0:
        return 0
    return int(equity / price)


# ---------------------------------------------------------------------------
# TrueRange
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def true_range(high: float, low: float, prev_close: float,
               is_first_bar: bool) -> float:
    if is_first_bar:
        return high - low
    return max(prev_close, high) - min(low, prev_close)


# ---------------------------------------------------------------------------
# WMA (Weighted Moving Average) — needed for HMA
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def calc_wma(buf, buf_idx, buf_count, period):
    """Compute WMA from a ring buffer. Returns 0.0 if not enough data."""
    if buf_count < period:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for j in range(period):
        # Most recent value gets weight=period, oldest gets weight=1
        # Ring buffer: most recent is at (buf_idx - 1) % period
        actual_idx = (buf_idx - period + j) % buf_count
        weight = float(j + 1)
        numerator += buf[actual_idx] * weight
        denominator += weight
    return numerator / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# HMA (Hull Moving Average) — WMA(2*WMA(n/2) - WMA(n), sqrt(n))
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def calc_hma_from_series(price_series, n_bars, length):
    """Compute HMA for the last bar from a price series.

    HMA(n) = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )

    Args:
        price_series: full price array up to current bar
        n_bars: number of valid bars available
        length: HMA period

    Returns current HMA value (0.0 if insufficient data).
    """
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(math.sqrt(length) + 0.5))
    # Need at least length + sqrt_len bars
    min_bars = length + sqrt_len
    if n_bars < min_bars:
        return 0.0

    # Compute difference series: 2*WMA(half) - WMA(full)
    # We need sqrt_len values of this difference
    diff_buf = np.empty(sqrt_len, dtype=np.float64)
    for k in range(sqrt_len):
        bar_end = n_bars - sqrt_len + k + 1
        # WMA(half_len) ending at bar_end
        wma_half = 0.0
        denom_half = 0.0
        for j in range(half_len):
            idx = bar_end - half_len + j
            w = float(j + 1)
            wma_half += price_series[idx] * w
            denom_half += w
        wma_half /= denom_half if denom_half > 0 else 1.0

        # WMA(length) ending at bar_end
        wma_full = 0.0
        denom_full = 0.0
        for j in range(length):
            idx = bar_end - length + j
            w = float(j + 1)
            wma_full += price_series[idx] * w
            denom_full += w
        wma_full /= denom_full if denom_full > 0 else 1.0

        diff_buf[k] = 2.0 * wma_half - wma_full

    # WMA(sqrt_len) of the difference series
    numerator = 0.0
    denominator = 0.0
    for j in range(sqrt_len):
        w = float(j + 1)
        numerator += diff_buf[j] * w
        denominator += w
    return numerator / denominator if denominator > 0 else 0.0
