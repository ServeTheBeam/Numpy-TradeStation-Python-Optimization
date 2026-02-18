"""ATR-D v1.5 Long-only Numba kernel.

Ported from NautilusTrader/numba_prefilter.py — proven to match TS output.

Two entry points:
  - run_strategy_wfo(): Single continuous IS/OOS backtest (for WFO phases)
  - run_chunk_wfo():    Parallel chunk runner across parameter combos
"""
from __future__ import annotations

import math
import numba
import numpy as np


# ---------------------------------------------------------------------------
# WFO Strategy Kernel — single continuous IS/OOS backtest
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def run_strategy_wfo(
    highs, lows, closes,
    atr_period, atr_multiplier, bb_length, bb_deviation,
    vol_threshold, min_tightening, scaling_method,
    atr_buy_period, atr_buy_multiplier, atr_smooth_period,
    equity_price_method,
    max_bars_back,
    cost_per_trade,
    initial_capital,
    oos_start_bar,
):
    """Run ATR-D v1.5-L as single continuous backtest with IS/OOS PnL split.

    Runs the FULL phase window (warmup + IS + OOS) as one continuous backtest.
    Trades are assigned to IS or OOS by entry bar index:
    if entry bar < oos_start_bar the trade's P/L goes to IS, else OOS.
    Capital compounds continuously from IS into OOS (matching TS behaviour).
    Drawdown is tracked by current-bar period.

    Returns (is_pnl, oos_pnl, is_trades, oos_trades, is_maxdd, oos_maxdd).
    """
    n = len(closes)
    if n < max_bars_back + max(atr_period, atr_buy_period) + 2:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)

    # -- ATR (SMA of TrueRange) -- ring buffers --
    tr_stop_buf = np.zeros(atr_period)
    tr_stop_idx = 0
    tr_stop_count = 0
    tr_stop_sum = 0.0
    atr_stop_val = 0.0

    tr_buy_buf = np.zeros(atr_buy_period)
    tr_buy_idx = 0
    tr_buy_count = 0
    tr_buy_sum = 0.0
    atr_buy_val = 0.0

    tr_base_buf = np.zeros(70)
    tr_base_idx = 0
    tr_base_count = 0
    tr_base_sum = 0.0
    atr_baseline_val = 0.0

    # -- BB Width on Close -- ring buffer --
    close_buf = np.zeros(bb_length)
    close_buf_idx = 0
    close_buf_count = 0

    # -- EMA for ATR Buy smoothing --
    atr_ema_value = 0.0
    atr_ema_started = False
    ema_alpha = 2.0 / (atr_smooth_period + 1)

    # -- State --
    eval_bar_count = 0
    in_position = False
    entry_price = 0.0
    final_stop_long = 0.0
    highest_since_entry = 0.0
    trail_amt = 0.0
    buy_touch_level = 0.0
    bb_multiplier = 1.0

    # -- Position sizing & IS/OOS tracking --
    equity = initial_capital
    shares = 0
    entry_bar = -1

    is_pnl = 0.0
    oos_pnl = 0.0
    is_trades = 0
    oos_trades = 0

    # Drawdown tracked by current bar period
    is_peak = initial_capital
    is_maxdd = 0.0
    oos_peak = 0.0
    oos_maxdd = 0.0
    oos_peak_set = False

    prev_close = closes[0]

    for i in range(n):
        h = highs[i]
        l = lows[i]
        c = closes[i]
        hl2 = (h + l) / 2.0

        # -- TrueRange --
        if i == 0:
            tr = h - l
        else:
            tr = max(prev_close, h) - min(l, prev_close)

        # -- ATR Stop (SMA ring buffer) --
        if tr_stop_count < atr_period:
            tr_stop_buf[tr_stop_idx] = tr
            tr_stop_sum += tr
            tr_stop_count += 1
            atr_stop_val = tr_stop_sum / tr_stop_count
        else:
            old = tr_stop_buf[tr_stop_idx]
            tr_stop_buf[tr_stop_idx] = tr
            tr_stop_sum += tr - old
            atr_stop_val = tr_stop_sum / atr_period
        tr_stop_idx = (tr_stop_idx + 1) % atr_period

        # -- ATR Buy (SMA ring buffer) --
        if tr_buy_count < atr_buy_period:
            tr_buy_buf[tr_buy_idx] = tr
            tr_buy_sum += tr
            tr_buy_count += 1
            atr_buy_val = tr_buy_sum / tr_buy_count
        else:
            old = tr_buy_buf[tr_buy_idx]
            tr_buy_buf[tr_buy_idx] = tr
            tr_buy_sum += tr - old
            atr_buy_val = tr_buy_sum / atr_buy_period
        tr_buy_idx = (tr_buy_idx + 1) % atr_buy_period

        # -- ATR Baseline 70 (SMA ring buffer) --
        if tr_base_count < 70:
            tr_base_buf[tr_base_idx] = tr
            tr_base_sum += tr
            tr_base_count += 1
            atr_baseline_val = tr_base_sum / tr_base_count
        else:
            old = tr_base_buf[tr_base_idx]
            tr_base_buf[tr_base_idx] = tr
            tr_base_sum += tr - old
            atr_baseline_val = tr_base_sum / 70
        tr_base_idx = (tr_base_idx + 1) % 70

        # -- Close buffer for BB Width --
        close_buf[close_buf_idx] = c
        close_buf_idx = (close_buf_idx + 1) % bb_length
        if close_buf_count < bb_length:
            close_buf_count += 1

        prev_close = c

        # -- Skip warmup bars --
        if i < max_bars_back:
            continue

        eval_bar_count += 1

        # -- Mark-to-market split at IS/OOS boundary --
        if i == oos_start_bar and in_position and entry_bar < oos_start_bar:
            split_price = closes[i - 1]
            mtm_pnl = (split_price - entry_price) * shares
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

        # -- Initialize OOS peak when entering OOS period --
        if i >= oos_start_bar and not oos_peak_set:
            oos_peak = equity
            oos_peak_set = True

        # -- Gate ATR values by CurrentBar > Period --
        atr_stop_ready = eval_bar_count > atr_period and tr_stop_count >= atr_period
        atr_buy_ready = eval_bar_count > atr_buy_period and tr_buy_count >= atr_buy_period

        # -- EMA of ATR Buy (XAverage) --
        xavg_input = atr_buy_val if atr_buy_ready else 0.0
        if not atr_ema_started:
            atr_ema_value = xavg_input
            atr_ema_started = True
        else:
            atr_ema_value += ema_alpha * (xavg_input - atr_ema_value)

        # -- Entry gate --
        if not (atr_stop_ready and atr_buy_ready):
            continue

        atr_val = atr_stop_val
        atr_buy_ema = atr_ema_value

        # -- BB Width on Close --
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
                upper = mean + bb_deviation * std
                lower = mean - bb_deviation * std
                bb_width = (upper - lower) / mean
            else:
                bb_width = 0.0
        else:
            bb_width = 0.0

        # -- BB Multiplier --
        if bb_width >= vol_threshold:
            bb_multiplier = 1.0
        else:
            if vol_threshold > 0:
                ratio = bb_width / vol_threshold
            else:
                ratio = 0.0
            if scaling_method == 0:
                bb_multiplier = ratio
            elif scaling_method == 1:
                bb_multiplier = math.sqrt(ratio) if ratio > 0 else 0.0
            else:
                bb_multiplier = ratio * ratio
            if bb_multiplier < min_tightening:
                bb_multiplier = min_tightening

        # -- Opening stop protection --
        if eval_bar_count > 70 and tr_base_count >= 70:
            max_stop_distance = atr_baseline_val * 4.0
        else:
            max_stop_distance = 0.0

        # -- FLAT: Entry Logic --
        if not in_position:
            final_stop_long = 0.0
            trail_amt = 0.0
            highest_since_entry = 0.0

            if atr_buy_ema > 0:
                buy_touch_level = c + (atr_buy_ema * atr_buy_multiplier)

            if i + 1 < n and buy_touch_level > 0:
                next_h = highs[i + 1]
                if next_h >= buy_touch_level:
                    entry_price = buy_touch_level
                    shares = int(equity / entry_price) if entry_price > 0 else 0
                    if shares < 1:
                        continue
                    in_position = True
                    entry_bar = i

                    atr_stop_dist = atr_val * atr_multiplier
                    if max_stop_distance > 0 and atr_stop_dist > max_stop_distance:
                        atr_stop_dist = max_stop_distance
                    capped_fsl = hl2 - atr_stop_dist
                    trail_amt = c - capped_fsl
                    highest_since_entry = c

        # -- IN POSITION: Stop Logic --
        else:
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
                equity += pnl
                if entry_bar < oos_start_bar:
                    is_pnl += pnl
                    is_trades += 1
                else:
                    oos_pnl += pnl
                    oos_trades += 1
                in_position = False
                shares = 0

                if i < oos_start_bar:
                    if equity > is_peak:
                        is_peak = equity
                    dd = is_peak - equity
                    if dd > is_maxdd:
                        is_maxdd = dd
                else:
                    if not oos_peak_set:
                        oos_peak = equity
                        oos_peak_set = True
                    if equity > oos_peak:
                        oos_peak = equity
                    dd = oos_peak - equity
                    if dd > oos_maxdd:
                        oos_maxdd = dd
            else:
                effective_stop = highest_since_entry - trail_amt
                if effective_stop > 0 and l <= effective_stop:
                    exit_price = effective_stop
                    pnl = (exit_price - entry_price) * shares - cost_per_trade * shares
                    equity += pnl
                    if entry_bar < oos_start_bar:
                        is_pnl += pnl
                        is_trades += 1
                    else:
                        oos_pnl += pnl
                        oos_trades += 1
                    in_position = False
                    shares = 0

                    if i < oos_start_bar:
                        if equity > is_peak:
                            is_peak = equity
                        dd = is_peak - equity
                        if dd > is_maxdd:
                            is_maxdd = dd
                    else:
                        if not oos_peak_set:
                            oos_peak = equity
                            oos_peak_set = True
                        if equity > oos_peak:
                            oos_peak = equity
                        dd = oos_peak - equity
                        if dd > oos_maxdd:
                            oos_maxdd = dd

    # Close open position at last bar's close
    if in_position:
        pnl = (closes[-1] - entry_price) * shares - cost_per_trade * shares
        equity += pnl
        if entry_bar < oos_start_bar:
            is_pnl += pnl
            is_trades += 1
        else:
            oos_pnl += pnl
            oos_trades += 1

        if n - 1 < oos_start_bar:
            if equity > is_peak:
                is_peak = equity
            dd = is_peak - equity
            if dd > is_maxdd:
                is_maxdd = dd
        else:
            if not oos_peak_set:
                oos_peak = equity
                oos_peak_set = True
            if equity > oos_peak:
                oos_peak = equity
            dd = oos_peak - equity
            if dd > oos_maxdd:
                oos_maxdd = dd

    return (is_pnl, oos_pnl, is_trades, oos_trades, is_maxdd, oos_maxdd)


# ---------------------------------------------------------------------------
# Parallel chunk runner (uses all CPU cores via prange)
# ---------------------------------------------------------------------------

@numba.njit(parallel=True)
def run_chunk_wfo(
    highs, lows, closes,
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
    max_bars_back, chunk_start,
    out_is_pnl, out_oos_pnl, out_is_trades, out_oos_trades,
    out_is_maxdd, out_oos_maxdd,
    cost_per_trade, initial_capital, oos_start_bar,
):
    """Run WFO kernel for a range of flat parameter indices in parallel.

    Parameter arrays p0..p10 correspond to:
      p0=atr_period, p1=atr_multiplier, p2=bb_length, p3=bb_deviation,
      p4=vol_threshold, p5=min_tightening, p6=scaling_method,
      p7=atr_buy_period, p8=atr_buy_multiplier, p9=atr_smooth_period,
      p10=equity_price_method

    Last dimension (p10) varies fastest in the flat index.
    """
    chunk_size = len(out_is_pnl)
    s0 = len(p0);   s1 = len(p1);   s2 = len(p2);   s3 = len(p3)
    s4 = len(p4);   s5 = len(p5);   s6 = len(p6);   s7 = len(p7)
    s8 = len(p8);   s9 = len(p9);   s10 = len(p10)

    for i in numba.prange(chunk_size):
        flat = chunk_start + i
        rem = flat
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
            max_bars_back,
            cost_per_trade,
            initial_capital,
            oos_start_bar,
        )

        out_is_pnl[i]     = is_p
        out_oos_pnl[i]    = oos_p
        out_is_trades[i]  = is_t
        out_oos_trades[i] = oos_t
        out_is_maxdd[i]   = is_dd
        out_oos_maxdd[i]  = oos_dd
