"""ATR-D HMA v1.5 Long/Short Numba kernel.

Bidirectional HMA crossover with:
  - Long on bull cross, short on bear cross (immediate flips)
  - Re-entry via buy/sell-touch after stop-out if HMA trend persists
  - Short_Multiplier for short ATR scaling
  - StoppedOutLong / StoppedOutShort flags
"""
from __future__ import annotations

import math
import numba
import numpy as np
from kernels.common import calc_hma_from_series


@numba.njit(cache=True)
def run_strategy_wfo(
    highs, lows, closes,
    atr_period, atr_multiplier, bb_length, bb_deviation,
    vol_threshold, min_tightening, scaling_method,
    atr_buy_period, atr_buy_multiplier, atr_smooth_period,
    equity_price_method,
    short_multiplier,
    hma_fast_length, hma_slow_length,
    max_bars_back,
    cost_per_trade,
    initial_capital,
    oos_start_bar,
):
    """Run ATR-D HMA v1.5-LS with continuous IS/OOS split."""
    n = len(closes)
    min_hma_bars = max(hma_fast_length, hma_slow_length) + int(math.sqrt(hma_slow_length) + 1)
    if n < max_bars_back + max(atr_period, atr_buy_period, min_hma_bars) + 2:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)

    atr_period_short = max(1, int(atr_period * short_multiplier + 0.5))

    # Price series for HMA
    price_series = np.empty(n, dtype=np.float64)
    for ii in range(n):
        if equity_price_method == 0:
            price_series[ii] = (highs[ii] + lows[ii] + closes[ii]) / 3.0
        else:
            price_series[ii] = closes[ii]

    # ATR buffers
    tr_stop_buf = np.zeros(atr_period)
    tr_stop_idx = 0; tr_stop_count = 0; tr_stop_sum = 0.0; atr_stop_val = 0.0

    tr_short_buf = np.zeros(atr_period_short)
    tr_short_idx = 0; tr_short_count = 0; tr_short_sum = 0.0; atr_short_val = 0.0

    tr_buy_buf = np.zeros(atr_buy_period)
    tr_buy_idx = 0; tr_buy_count = 0; tr_buy_sum = 0.0; atr_buy_val = 0.0

    tr_base_buf = np.zeros(70)
    tr_base_idx = 0; tr_base_count = 0; tr_base_sum = 0.0; atr_baseline_val = 0.0

    close_buf = np.zeros(bb_length)
    close_buf_idx = 0; close_buf_count = 0

    atr_ema_value = 0.0; atr_ema_started = False
    ema_alpha = 2.0 / (atr_smooth_period + 1)

    hma_fast_prev = 0.0; hma_slow_prev = 0.0

    eval_bar_count = 0
    position = 0  # 0=flat, 1=long, -1=short
    stopped_out_long = False
    stopped_out_short = False
    entry_price = 0.0
    final_stop_long = 0.0; final_stop_short = 0.0
    highest_since_entry = 0.0; lowest_since_entry = 999999.0
    trail_amt = 0.0
    bb_multiplier = 1.0

    equity = initial_capital; shares = 0; entry_bar = -1
    is_pnl = 0.0; oos_pnl = 0.0; is_trades = 0; oos_trades = 0
    is_peak = initial_capital; is_maxdd = 0.0
    oos_peak = 0.0; oos_maxdd = 0.0; oos_peak_set = False

    prev_close = closes[0]

    for i in range(n):
        h = highs[i]; l = lows[i]; c = closes[i]
        hl2 = (h + l) / 2.0

        if i == 0:
            tr = h - l
        else:
            tr = max(prev_close, h) - min(l, prev_close)

        # ATR updates (long, short, buy, baseline)
        if tr_stop_count < atr_period:
            tr_stop_buf[tr_stop_idx] = tr; tr_stop_sum += tr; tr_stop_count += 1
            atr_stop_val = tr_stop_sum / tr_stop_count
        else:
            old = tr_stop_buf[tr_stop_idx]; tr_stop_buf[tr_stop_idx] = tr
            tr_stop_sum += tr - old; atr_stop_val = tr_stop_sum / atr_period
        tr_stop_idx = (tr_stop_idx + 1) % atr_period

        if tr_short_count < atr_period_short:
            tr_short_buf[tr_short_idx] = tr; tr_short_sum += tr; tr_short_count += 1
            atr_short_val = tr_short_sum / tr_short_count
        else:
            old = tr_short_buf[tr_short_idx]; tr_short_buf[tr_short_idx] = tr
            tr_short_sum += tr - old; atr_short_val = tr_short_sum / atr_period_short
        tr_short_idx = (tr_short_idx + 1) % atr_period_short

        if tr_buy_count < atr_buy_period:
            tr_buy_buf[tr_buy_idx] = tr; tr_buy_sum += tr; tr_buy_count += 1
            atr_buy_val = tr_buy_sum / tr_buy_count
        else:
            old = tr_buy_buf[tr_buy_idx]; tr_buy_buf[tr_buy_idx] = tr
            tr_buy_sum += tr - old; atr_buy_val = tr_buy_sum / atr_buy_period
        tr_buy_idx = (tr_buy_idx + 1) % atr_buy_period

        if tr_base_count < 70:
            tr_base_buf[tr_base_idx] = tr; tr_base_sum += tr; tr_base_count += 1
            atr_baseline_val = tr_base_sum / tr_base_count
        else:
            old = tr_base_buf[tr_base_idx]; tr_base_buf[tr_base_idx] = tr
            tr_base_sum += tr - old; atr_baseline_val = tr_base_sum / 70
        tr_base_idx = (tr_base_idx + 1) % 70

        close_buf[close_buf_idx] = c
        close_buf_idx = (close_buf_idx + 1) % bb_length
        if close_buf_count < bb_length:
            close_buf_count += 1

        prev_close = c

        if i < max_bars_back:
            continue
        eval_bar_count += 1

        if i >= oos_start_bar and not oos_peak_set:
            oos_peak = equity; oos_peak_set = True

        atr_stop_ready = eval_bar_count > atr_period and tr_stop_count >= atr_period
        atr_buy_ready = eval_bar_count > atr_buy_period and tr_buy_count >= atr_buy_period

        xavg_input = atr_buy_val if atr_buy_ready else 0.0
        if not atr_ema_started:
            atr_ema_value = xavg_input; atr_ema_started = True
        else:
            atr_ema_value += ema_alpha * (xavg_input - atr_ema_value)

        if not (atr_stop_ready and atr_buy_ready):
            continue

        atr_val = atr_stop_val
        atr_buy_ema = atr_ema_value

        # HMA
        bars_avail = i + 1
        hma_fast = calc_hma_from_series(price_series, bars_avail, hma_fast_length)
        hma_slow = calc_hma_from_series(price_series, bars_avail, hma_slow_length)

        hma_bull_cross = (hma_fast > hma_slow) and (hma_fast_prev <= hma_slow_prev)
        hma_bear_cross = (hma_fast < hma_slow) and (hma_fast_prev >= hma_slow_prev)
        hma_bull_trend = hma_fast > hma_slow
        hma_bear_trend = hma_fast < hma_slow

        # BB Width + Multiplier
        bb_width = 0.0
        if eval_bar_count > bb_length and close_buf_count >= bb_length:
            total = 0.0
            for j in range(bb_length):
                total += close_buf[j]
            mean = total / bb_length
            if mean > 0:
                var_sum = 0.0
                for j in range(bb_length):
                    diff = close_buf[j] - mean
                    var_sum += diff * diff
                std = math.sqrt(var_sum / bb_length)
                bb_width = (mean + bb_deviation * std - (mean - bb_deviation * std)) / mean

        if bb_width >= vol_threshold:
            bb_multiplier = 1.0
        else:
            ratio = bb_width / vol_threshold if vol_threshold > 0 else 0.0
            if scaling_method == 0:
                bb_multiplier = ratio
            elif scaling_method == 1:
                bb_multiplier = math.sqrt(ratio) if ratio > 0 else 0.0
            else:
                bb_multiplier = ratio * ratio
            if bb_multiplier < min_tightening:
                bb_multiplier = min_tightening

        max_stop_distance = atr_baseline_val * 4.0 if (eval_bar_count > 70 and tr_base_count >= 70) else 0.0

        # PnL helper
        def _rec(pnl_val, eb, exit_bar):
            nonlocal equity, is_pnl, oos_pnl, is_trades, oos_trades
            nonlocal is_peak, is_maxdd, oos_peak, oos_maxdd, oos_peak_set
            equity += pnl_val
            if eb < oos_start_bar:
                is_pnl += pnl_val; is_trades += 1
            else:
                oos_pnl += pnl_val; oos_trades += 1
            if exit_bar < oos_start_bar:
                if equity > is_peak: is_peak = equity
                dd = is_peak - equity
                if dd > is_maxdd: is_maxdd = dd
            else:
                if not oos_peak_set: oos_peak = equity; oos_peak_set = True
                if equity > oos_peak: oos_peak = equity
                dd = oos_peak - equity
                if dd > oos_maxdd: oos_maxdd = dd

        # ================================================================
        # CROSSOVER FLIPS (handle regardless of position)
        # ================================================================
        if hma_bull_cross:
            # Close short if in short
            if position == -1:
                pnl = (entry_price - c) * shares - cost_per_trade * shares
                _rec(pnl, entry_bar, i)
                position = 0; shares = 0
            # Enter long
            if position == 0:
                stopped_out_long = False; stopped_out_short = False
                entry_price = c
                shares = int(equity / entry_price) if entry_price > 0 else 0
                if shares >= 1:
                    position = 1; entry_bar = i
                    atr_stop_dist = atr_val * atr_multiplier
                    if max_stop_distance > 0 and atr_stop_dist > max_stop_distance:
                        atr_stop_dist = max_stop_distance
                    final_stop_long = hl2 - atr_stop_dist
                    trail_amt = c - final_stop_long
                    highest_since_entry = c

        elif hma_bear_cross:
            # Close long if in long
            if position == 1:
                pnl = (c - entry_price) * shares - cost_per_trade * shares
                _rec(pnl, entry_bar, i)
                position = 0; shares = 0
            # Enter short
            if position == 0:
                stopped_out_long = False; stopped_out_short = False
                entry_price = c
                shares = int(equity / entry_price) if entry_price > 0 else 0
                if shares >= 1:
                    position = -1; entry_bar = i
                    short_stop_dist = atr_short_val * atr_multiplier * short_multiplier
                    if max_stop_distance > 0 and short_stop_dist > max_stop_distance:
                        short_stop_dist = max_stop_distance
                    final_stop_short = hl2 + short_stop_dist
                    trail_amt = final_stop_short - c
                    lowest_since_entry = c

        # ================================================================
        # FLAT: Re-entry after stop-out via buy/sell-touch
        # ================================================================
        elif position == 0:
            if stopped_out_long and hma_bull_trend and atr_buy_ema > 0:
                buy_touch_level = c + (atr_buy_ema * atr_buy_multiplier)
                if i + 1 < n and buy_touch_level > 0:
                    if highs[i + 1] >= buy_touch_level:
                        entry_price = buy_touch_level
                        shares = int(equity / entry_price) if entry_price > 0 else 0
                        if shares >= 1:
                            position = 1; entry_bar = i
                            atr_stop_dist = atr_val * atr_multiplier
                            if max_stop_distance > 0 and atr_stop_dist > max_stop_distance:
                                atr_stop_dist = max_stop_distance
                            final_stop_long = hl2 - atr_stop_dist
                            trail_amt = c - final_stop_long
                            highest_since_entry = c

            elif stopped_out_short and hma_bear_trend and atr_buy_ema > 0:
                sell_touch_level = c - (atr_buy_ema * atr_buy_multiplier)
                if i + 1 < n and sell_touch_level > 0:
                    if lows[i + 1] <= sell_touch_level:
                        entry_price = sell_touch_level
                        shares = int(equity / entry_price) if entry_price > 0 else 0
                        if shares >= 1:
                            position = -1; entry_bar = i
                            short_stop_dist = atr_short_val * atr_multiplier * short_multiplier
                            if max_stop_distance > 0 and short_stop_dist > max_stop_distance:
                                short_stop_dist = max_stop_distance
                            final_stop_short = hl2 + short_stop_dist
                            trail_amt = final_stop_short - c
                            lowest_since_entry = c

        # ================================================================
        # LONG: ATR trailing stop
        # ================================================================
        elif position == 1:
            if h > highest_since_entry:
                highest_since_entry = h
            basic_stop = hl2 - (atr_val * atr_multiplier)
            if final_stop_long == 0.0:
                final_stop_long = basic_stop
            elif basic_stop > final_stop_long:
                final_stop_long = basic_stop
            stop_distance = c - final_stop_long
            trail_amt = stop_distance * bb_multiplier

            if trail_amt < 0:
                pnl = (c - entry_price) * shares - cost_per_trade * shares
                _rec(pnl, entry_bar, i)
                position = 0; shares = 0; stopped_out_long = True
            else:
                effective_stop = highest_since_entry - trail_amt
                if effective_stop > 0 and l <= effective_stop:
                    pnl = (effective_stop - entry_price) * shares - cost_per_trade * shares
                    _rec(pnl, entry_bar, i)
                    position = 0; shares = 0; stopped_out_long = True

        # ================================================================
        # SHORT: ATR trailing stop
        # ================================================================
        elif position == -1:
            if l < lowest_since_entry:
                lowest_since_entry = l
            basic_stop = hl2 + (atr_short_val * atr_multiplier * short_multiplier)
            if final_stop_short == 0.0:
                final_stop_short = basic_stop
            elif basic_stop < final_stop_short:
                final_stop_short = basic_stop
            stop_distance = final_stop_short - c
            trail_amt = stop_distance * bb_multiplier

            if trail_amt < 0:
                pnl = (entry_price - c) * shares - cost_per_trade * shares
                _rec(pnl, entry_bar, i)
                position = 0; shares = 0; stopped_out_short = True
            else:
                effective_stop = lowest_since_entry + trail_amt
                if effective_stop > 0 and h >= effective_stop:
                    pnl = (entry_price - effective_stop) * shares - cost_per_trade * shares
                    _rec(pnl, entry_bar, i)
                    position = 0; shares = 0; stopped_out_short = True

        hma_fast_prev = hma_fast
        hma_slow_prev = hma_slow

    # Close open position
    if position == 1:
        pnl = (closes[-1] - entry_price) * shares - cost_per_trade * shares
        _rec(pnl, entry_bar, n - 1)
    elif position == -1:
        pnl = (entry_price - closes[-1]) * shares - cost_per_trade * shares
        _rec(pnl, entry_bar, n - 1)

    return (is_pnl, oos_pnl, is_trades, oos_trades, is_maxdd, oos_maxdd)


@numba.njit(parallel=True)
def run_chunk_wfo(
    highs, lows, closes,
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
    max_bars_back, chunk_start,
    out_is_pnl, out_oos_pnl, out_is_trades, out_oos_trades,
    out_is_maxdd, out_oos_maxdd,
    cost_per_trade, initial_capital, oos_start_bar,
):
    """p0..p10=base, p11=short_mult, p12=hma_fast, p13=hma_slow."""
    chunk_size = len(out_is_pnl)
    s0 = len(p0); s1 = len(p1); s2 = len(p2); s3 = len(p3)
    s4 = len(p4); s5 = len(p5); s6 = len(p6); s7 = len(p7)
    s8 = len(p8); s9 = len(p9); s10 = len(p10)
    s11 = len(p11); s12 = len(p12); s13 = len(p13)

    for i in numba.prange(chunk_size):
        flat = chunk_start + i
        rem = flat
        i13 = rem % s13; rem //= s13
        i12 = rem % s12; rem //= s12
        i11 = rem % s11; rem //= s11
        i10 = rem % s10; rem //= s10
        i9  = rem % s9;  rem //= s9
        i8  = rem % s8;  rem //= s8
        i7  = rem % s7;  rem //= s7
        i6  = rem % s6;  rem //= s6
        i5  = rem % s5;  rem //= s5
        i4  = rem % s4;  rem //= s4
        i3  = rem % s3;  rem //= s3
        i2  = rem % s2;  rem //= s2
        i1  = rem % s1;  rem //= s1
        i0  = rem

        is_p, oos_p, is_t, oos_t, is_dd, oos_dd = run_strategy_wfo(
            highs, lows, closes,
            int(p0[i0]),   float(p1[i1]),
            int(p2[i2]),   float(p3[i3]),
            float(p4[i4]), float(p5[i5]),
            int(p6[i6]),
            int(p7[i7]),   float(p8[i8]),
            int(p9[i9]),   int(p10[i10]),
            float(p11[i11]),
            int(p12[i12]), int(p13[i13]),
            max_bars_back,
            cost_per_trade, initial_capital, oos_start_bar,
        )
        out_is_pnl[i] = is_p; out_oos_pnl[i] = oos_p
        out_is_trades[i] = is_t; out_oos_trades[i] = oos_t
        out_is_maxdd[i] = is_dd; out_oos_maxdd[i] = oos_dd
