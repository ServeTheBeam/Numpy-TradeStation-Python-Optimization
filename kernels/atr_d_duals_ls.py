"""ATR-D DualSymbol v1.5 Long/Short Numba kernel.

Hybrid model combining:
  - DualS: Data1 (leveraged ETF) for ATR/stops, Data2 (index ETF) for BB tightening
  - LS: Bidirectional with Entry_Mode, Trend_Veto, Short_Multiplier
"""
from __future__ import annotations

import math
import numba
import numpy as np


@numba.njit(cache=True)
def run_strategy_wfo(
    # Data1 (leveraged ETF) — entry, stops, position sizing
    highs1, lows1, closes1,
    # Data2 (index ETF) — BB Width tightening only
    highs2, lows2, closes2,
    # Base params
    atr_period, atr_multiplier, bb_length, bb_deviation,
    vol_threshold, min_tightening, scaling_method,
    atr_buy_period, atr_buy_multiplier, atr_smooth_period,
    equity_price_method,
    # LS-specific params
    short_multiplier,
    entry_mode,
    trend_filter_period,
    trend_veto_threshold,
    both_triggered_mode,
    # WFO params
    max_bars_back,
    cost_per_trade,
    initial_capital,
    oos_start_bar,
):
    """Run ATR-D DualS v1.5-LS with continuous IS/OOS split.

    Data1 = leveraged ETF for price/ATR/stops/trend.
    Data2 = index ETF for BB Width tightening.

    Returns (is_pnl, oos_pnl, is_trades, oos_trades, is_maxdd, oos_maxdd).
    """
    n = len(closes1)
    n2 = len(closes2)
    if n < max_bars_back + max(atr_period, atr_buy_period) + 2:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)
    if n2 < bb_length + 1:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)

    n_bars = min(n, n2)

    atr_period_short = max(1, int(atr_period * short_multiplier + 0.5))

    # ATR buffers (Data1)
    tr_stop_buf = np.zeros(atr_period)
    tr_stop_idx = 0; tr_stop_count = 0; tr_stop_sum = 0.0; atr_stop_val = 0.0

    tr_short_buf = np.zeros(atr_period_short)
    tr_short_idx = 0; tr_short_count = 0; tr_short_sum = 0.0; atr_short_val = 0.0

    tr_buy_buf = np.zeros(atr_buy_period)
    tr_buy_idx = 0; tr_buy_count = 0; tr_buy_sum = 0.0; atr_buy_val = 0.0

    tr_base_buf = np.zeros(70)
    tr_base_idx = 0; tr_base_count = 0; tr_base_sum = 0.0; atr_baseline_val = 0.0

    # BB Width buffer — uses Data2 prices
    close_buf2 = np.zeros(bb_length)
    close_buf2_idx = 0; close_buf2_count = 0

    # EMA for ATR Buy smoothing
    atr_ema_value = 0.0; atr_ema_started = False
    ema_alpha = 2.0 / (atr_smooth_period + 1)

    # Trend filter EMA (Data1)
    trend_ema_value = 0.0; trend_ema_prev = 0.0; trend_ema_started = False
    trend_alpha = 2.0 / (trend_filter_period + 1)

    eval_bar_count = 0
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    final_stop_long = 0.0; final_stop_short = 0.0
    highest_since_entry = 0.0; lowest_since_entry = 999999.0
    trail_amt = 0.0
    buy_touch_level = 0.0; sell_touch_level = 0.0
    bb_multiplier = 1.0

    equity = initial_capital; shares = 0; entry_bar = -1
    is_pnl = 0.0; oos_pnl = 0.0; is_trades = 0; oos_trades = 0
    is_peak = initial_capital; is_maxdd = 0.0
    oos_peak = 0.0; oos_maxdd = 0.0; oos_peak_set = False

    prev_close1 = closes1[0]

    for i in range(n_bars):
        h = highs1[i]; l = lows1[i]; c = closes1[i]
        hl2 = (h + l) / 2.0
        c2 = closes2[i]

        # TrueRange on Data1
        if i == 0:
            tr = h - l
        else:
            tr = max(prev_close1, h) - min(l, prev_close1)

        # ATR Stop (long, Data1)
        if tr_stop_count < atr_period:
            tr_stop_buf[tr_stop_idx] = tr; tr_stop_sum += tr; tr_stop_count += 1
            atr_stop_val = tr_stop_sum / tr_stop_count
        else:
            old = tr_stop_buf[tr_stop_idx]; tr_stop_buf[tr_stop_idx] = tr
            tr_stop_sum += tr - old; atr_stop_val = tr_stop_sum / atr_period
        tr_stop_idx = (tr_stop_idx + 1) % atr_period

        # ATR Stop (short, Data1)
        if tr_short_count < atr_period_short:
            tr_short_buf[tr_short_idx] = tr; tr_short_sum += tr; tr_short_count += 1
            atr_short_val = tr_short_sum / tr_short_count
        else:
            old = tr_short_buf[tr_short_idx]; tr_short_buf[tr_short_idx] = tr
            tr_short_sum += tr - old; atr_short_val = tr_short_sum / atr_period_short
        tr_short_idx = (tr_short_idx + 1) % atr_period_short

        # ATR Buy (Data1)
        if tr_buy_count < atr_buy_period:
            tr_buy_buf[tr_buy_idx] = tr; tr_buy_sum += tr; tr_buy_count += 1
            atr_buy_val = tr_buy_sum / tr_buy_count
        else:
            old = tr_buy_buf[tr_buy_idx]; tr_buy_buf[tr_buy_idx] = tr
            tr_buy_sum += tr - old; atr_buy_val = tr_buy_sum / atr_buy_period
        tr_buy_idx = (tr_buy_idx + 1) % atr_buy_period

        # ATR Baseline 70 (Data1)
        if tr_base_count < 70:
            tr_base_buf[tr_base_idx] = tr; tr_base_sum += tr; tr_base_count += 1
            atr_baseline_val = tr_base_sum / tr_base_count
        else:
            old = tr_base_buf[tr_base_idx]; tr_base_buf[tr_base_idx] = tr
            tr_base_sum += tr - old; atr_baseline_val = tr_base_sum / 70
        tr_base_idx = (tr_base_idx + 1) % 70

        # Close buffer for BB Width — uses Data2
        if equity_price_method == 0:
            bb_price = (highs2[i] + lows2[i] + closes2[i]) / 3.0
        else:
            bb_price = c2
        close_buf2[close_buf2_idx] = bb_price
        close_buf2_idx = (close_buf2_idx + 1) % bb_length
        if close_buf2_count < bb_length:
            close_buf2_count += 1

        prev_close1 = c

        if i < max_bars_back:
            continue
        eval_bar_count += 1

        # -- Mark-to-market split at IS/OOS boundary --
        if i == oos_start_bar and position != 0 and entry_bar < oos_start_bar:
            split_price = closes1[i - 1]
            if position == 1:
                mtm_pnl = (split_price - entry_price) * shares
            else:
                mtm_pnl = (entry_price - split_price) * shares
            is_pnl += mtm_pnl
            is_trades += 1
            equity += mtm_pnl
            if equity > is_peak:
                is_peak = equity
            dd = is_peak - equity
            if dd > is_maxdd:
                is_maxdd = dd
            entry_price = split_price
            entry_bar = oos_start_bar

        if i >= oos_start_bar and not oos_peak_set:
            oos_peak = equity; oos_peak_set = True

        atr_stop_ready = eval_bar_count > atr_period and tr_stop_count >= atr_period
        atr_buy_ready = eval_bar_count > atr_buy_period and tr_buy_count >= atr_buy_period

        # EMA of ATR Buy
        xavg_input = atr_buy_val if atr_buy_ready else 0.0
        if not atr_ema_started:
            atr_ema_value = xavg_input; atr_ema_started = True
        else:
            atr_ema_value += ema_alpha * (xavg_input - atr_ema_value)

        # Trend filter EMA (Data1)
        trend_ema_prev = trend_ema_value
        if not trend_ema_started:
            trend_ema_value = c; trend_ema_started = True
        else:
            trend_ema_value += trend_alpha * (c - trend_ema_value)
        trend_slope = trend_ema_value - trend_ema_prev

        if not (atr_stop_ready and atr_buy_ready):
            continue
        # Data2 gate
        if close_buf2_count < bb_length:
            continue

        atr_val = atr_stop_val
        atr_buy_ema = atr_ema_value

        # BB Width on Data2
        if eval_bar_count > bb_length and close_buf2_count >= bb_length:
            total_bb = 0.0
            for j in range(bb_length):
                total_bb += close_buf2[j]
            mean = total_bb / bb_length
            if mean > 0:
                var_sum = 0.0
                for j in range(bb_length):
                    diff = close_buf2[j] - mean
                    var_sum += diff * diff
                std = math.sqrt(var_sum / bb_length)
                upper = mean + bb_deviation * std
                lower = mean - bb_deviation * std
                bb_width = (upper - lower) / mean
            else:
                bb_width = 0.0
        else:
            bb_width = 0.0

        # BB Multiplier
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
        def _record_pnl(pnl_val, entry_bar_val, exit_bar):
            nonlocal equity, is_pnl, oos_pnl, is_trades, oos_trades
            nonlocal is_peak, is_maxdd, oos_peak, oos_maxdd, oos_peak_set
            equity += pnl_val
            if entry_bar_val < oos_start_bar:
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
        # FLAT: Entry Logic (same as LS but uses Data1 prices)
        # ================================================================
        if position == 0:
            final_stop_long = 0.0; final_stop_short = 0.0
            trail_amt = 0.0
            highest_since_entry = 0.0; lowest_since_entry = 999999.0

            if atr_buy_ema > 0:
                buy_touch_level = c + (atr_buy_ema * atr_buy_multiplier)
                sell_touch_level = c - (atr_buy_ema * atr_buy_multiplier)

            if i + 1 < n_bars and buy_touch_level > 0:
                next_h = highs1[i + 1]
                next_l = lows1[i + 1]

                long_triggered = next_h >= buy_touch_level
                short_triggered = next_l <= sell_touch_level

                enter_long = False; enter_short = False

                if long_triggered and short_triggered:
                    if both_triggered_mode == 0:
                        enter_long = True
                elif long_triggered:
                    enter_long = True
                elif short_triggered:
                    if trend_slope < -trend_veto_threshold:
                        enter_short = True

                if enter_long:
                    entry_price = buy_touch_level
                    shares = int(equity / entry_price) if entry_price > 0 else 0
                    if shares >= 1:
                        position = 1; entry_bar = i
                        atr_stop_dist = atr_val * atr_multiplier
                        if max_stop_distance > 0 and atr_stop_dist > max_stop_distance:
                            atr_stop_dist = max_stop_distance
                        capped_fsl = hl2 - atr_stop_dist
                        trail_amt = c - capped_fsl
                        highest_since_entry = c

                elif enter_short:
                    entry_price = sell_touch_level
                    shares = int(equity / entry_price) if entry_price > 0 else 0
                    if shares >= 1:
                        position = -1; entry_bar = i
                        short_stop_dist = atr_short_val * atr_multiplier * short_multiplier
                        if max_stop_distance > 0 and short_stop_dist > max_stop_distance:
                            short_stop_dist = max_stop_distance
                        capped_fss = hl2 + short_stop_dist
                        trail_amt = capped_fss - c
                        lowest_since_entry = c

        # ================================================================
        # LONG: Stop Logic
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

            exited = False
            if trail_amt < 0:
                pnl = (c - entry_price) * shares - cost_per_trade * shares
                _record_pnl(pnl, entry_bar, i)
                position = 0; shares = 0; exited = True
            else:
                effective_stop = highest_since_entry - trail_amt
                if effective_stop > 0 and l <= effective_stop:
                    exit_price = effective_stop
                    pnl = (exit_price - entry_price) * shares - cost_per_trade * shares
                    _record_pnl(pnl, entry_bar, i)
                    position = 0; shares = 0; exited = True

            # Entry_Mode 2: flip to short after long exit
            if exited and entry_mode == 2 and i + 1 < n_bars:
                if sell_touch_level > 0:
                    next_l = lows1[i + 1]
                    if next_l <= sell_touch_level:
                        if trend_slope < -trend_veto_threshold:
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
        # SHORT: Stop Logic
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

            exited = False
            if trail_amt < 0:
                pnl = (entry_price - c) * shares - cost_per_trade * shares
                _record_pnl(pnl, entry_bar, i)
                position = 0; shares = 0; exited = True
            else:
                effective_stop = lowest_since_entry + trail_amt
                if effective_stop > 0 and h >= effective_stop:
                    exit_price = effective_stop
                    pnl = (entry_price - exit_price) * shares - cost_per_trade * shares
                    _record_pnl(pnl, entry_bar, i)
                    position = 0; shares = 0; exited = True

            # Entry_Mode 1 or 2: flip to long after short exit
            if exited and entry_mode >= 1 and i + 1 < n_bars:
                if buy_touch_level > 0:
                    next_h = highs1[i + 1]
                    if next_h >= buy_touch_level:
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

    # Close open position at last bar
    if position == 1:
        pnl = (closes1[n_bars - 1] - entry_price) * shares - cost_per_trade * shares
        equity += pnl
        if entry_bar < oos_start_bar: is_pnl += pnl; is_trades += 1
        else: oos_pnl += pnl; oos_trades += 1
    elif position == -1:
        pnl = (entry_price - closes1[n_bars - 1]) * shares - cost_per_trade * shares
        equity += pnl
        if entry_bar < oos_start_bar: is_pnl += pnl; is_trades += 1
        else: oos_pnl += pnl; oos_trades += 1

    return (is_pnl, oos_pnl, is_trades, oos_trades, is_maxdd, oos_maxdd)


# ---------------------------------------------------------------------------
# Parallel chunk runner
# ---------------------------------------------------------------------------

@numba.njit(parallel=True)
def run_chunk_wfo(
    highs1, lows1, closes1,
    highs2, lows2, closes2,
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
    p11, p12, p13, p14, p15,
    max_bars_back, chunk_start,
    out_is_pnl, out_oos_pnl, out_is_trades, out_oos_trades,
    out_is_maxdd, out_oos_maxdd,
    cost_per_trade, initial_capital, oos_start_bar,
):
    """Parallel chunk runner for DualS-LS: 6 data arrays + 16 params.

    p0=atr_period, p1=atr_multiplier, p2=bb_length, p3=bb_deviation,
    p4=vol_threshold, p5=min_tightening, p6=scaling_method,
    p7=atr_buy_period, p8=atr_buy_multiplier, p9=atr_smooth_period,
    p10=equity_price_method,
    p11=short_multiplier, p12=entry_mode, p13=trend_filter_period,
    p14=trend_veto_threshold, p15=both_triggered_mode
    """
    chunk_size = len(out_is_pnl)
    s0 = len(p0);   s1 = len(p1);   s2 = len(p2);   s3 = len(p3)
    s4 = len(p4);   s5 = len(p5);   s6 = len(p6);   s7 = len(p7)
    s8 = len(p8);   s9 = len(p9);   s10 = len(p10);  s11 = len(p11)
    s12 = len(p12);  s13 = len(p13);  s14 = len(p14);  s15 = len(p15)

    for i in numba.prange(chunk_size):
        flat = chunk_start + i
        rem = flat
        i15 = rem % s15; rem //= s15
        i14 = rem % s14; rem //= s14
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
            highs1, lows1, closes1,
            highs2, lows2, closes2,
            int(p0[i0]),   float(p1[i1]),
            int(p2[i2]),   float(p3[i3]),
            float(p4[i4]), float(p5[i5]),
            int(p6[i6]),
            int(p7[i7]),   float(p8[i8]),
            int(p9[i9]),   int(p10[i10]),
            float(p11[i11]), int(p12[i12]),
            int(p13[i13]),  float(p14[i14]),
            int(p15[i15]),
            max_bars_back,
            cost_per_trade, initial_capital, oos_start_bar,
        )

        out_is_pnl[i]     = is_p
        out_oos_pnl[i]    = oos_p
        out_is_trades[i]  = is_t
        out_oos_trades[i] = oos_t
        out_is_maxdd[i]   = is_dd
        out_oos_maxdd[i]  = oos_dd
