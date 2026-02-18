#!/usr/bin/env python3
"""Generate an attractive PDF product summary for Numba WFO."""

from __future__ import annotations

import re
from fpdf import FPDF


# ── Color palette ──────────────────────────────────────────────────
NAVY       = (15, 23, 42)
SLATE      = (51, 65, 85)
GRAY_600   = (75, 85, 99)
GRAY_400   = (156, 163, 175)
GRAY_200   = (229, 231, 235)
GRAY_50    = (249, 250, 251)
WHITE      = (255, 255, 255)
BLUE_700   = (29, 78, 216)
BLUE_100   = (219, 234, 254)
BLUE_50    = (239, 246, 255)
AMBER_600  = (217, 119, 6)
AMBER_50   = (255, 251, 235)
GREEN_700  = (21, 128, 61)
GREEN_50   = (240, 253, 244)
TEAL_700   = (15, 118, 110)

FONT_DIR = "C:/Windows/Fonts"

# Font family aliases used throughout
SANS = "segoe"
MONO = "consolas"


class SummaryPDF(FPDF):
    """Custom PDF with header/footer and helper methods."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(18, 18, 18)
        self.page_w = 210 - 36  # usable width

        # Register Unicode-capable TTF fonts
        self.add_font(SANS, "",  f"{FONT_DIR}/segoeui.ttf")
        self.add_font(SANS, "B", f"{FONT_DIR}/segoeuib.ttf")
        self.add_font(SANS, "I", f"{FONT_DIR}/segoeuii.ttf")
        self.add_font(SANS, "BI", f"{FONT_DIR}/segoeuiz.ttf")
        self.add_font(MONO, "",  f"{FONT_DIR}/consola.ttf")
        self.add_font(MONO, "B", f"{FONT_DIR}/consolab.ttf")

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(SANS, "I", 8)
        self.set_text_color(*GRAY_400)
        self.cell(0, 6, "Numba WFO  |  Product Summary", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(SANS, "I", 8)
        self.set_text_color(*GRAY_400)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── Drawing helpers ────────────────────────────────────────────

    def _hr(self, y=None, color=GRAY_200):
        if y is None:
            y = self.get_y()
        self.set_draw_color(*color)
        self.set_line_width(0.3)
        self.line(18, y, 192, y)
        self.set_y(y + 3)

    def _section_title(self, text: str):
        self.ln(4)
        self.set_font(SANS, "B", 15)
        self.set_text_color(*NAVY)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self._hr(color=BLUE_700)
        self.ln(2)

    def _subsection_title(self, text: str):
        self.ln(3)
        self.set_font(SANS, "B", 11)
        self.set_text_color(*SLATE)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def _body(self, text: str):
        self.set_font(SANS, "", 9.5)
        self.set_text_color(*SLATE)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def _body_md(self, text: str):
        """Render body text with **bold** and *italic* spans."""
        self.set_font(SANS, "", 9.5)
        self.set_text_color(*SLATE)
        # Split on bold/italic markers
        parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`[^`]+`)', text)
        x_start = self.get_x()
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font(SANS, "B", 9.5)
                self._write_inline(part[2:-2])
                self.set_font(SANS, "", 9.5)
            elif part.startswith("*") and part.endswith("*"):
                self.set_font(SANS, "I", 9.5)
                self._write_inline(part[1:-1])
                self.set_font(SANS, "", 9.5)
            elif part.startswith("`") and part.endswith("`"):
                self.set_font(MONO, "", 8.5)
                self.set_text_color(*BLUE_700)
                self._write_inline(part[1:-1])
                self.set_font(SANS, "", 9.5)
                self.set_text_color(*SLATE)
            else:
                self._write_inline(part)
        self.ln(5)

    def _write_inline(self, text: str):
        self.write(5, text)

    def _bullet(self, text: str, indent: float = 4):
        x = self.get_x()
        self.set_x(x + indent)
        self.set_font(SANS, "", 9.5)
        self.set_text_color(*SLATE)
        # Parse bold
        parts = re.split(r'(\*\*.*?\*\*)', text)
        self.write(5, "\u2022  ")
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font(SANS, "B", 9.5)
                self.write(5, part[2:-2])
                self.set_font(SANS, "", 9.5)
            else:
                # Handle inline code
                code_parts = re.split(r'(`[^`]+`)', part)
                for cp in code_parts:
                    if cp.startswith("`") and cp.endswith("`"):
                        self.set_font(MONO, "", 8.5)
                        self.set_text_color(*BLUE_700)
                        self.write(5, cp[1:-1])
                        self.set_font(SANS, "", 9.5)
                        self.set_text_color(*SLATE)
                    else:
                        self.write(5, cp)
        self.ln(5)

    def _code_block(self, text: str, font_size: float = 7.5):
        self.ln(1)
        lines = text.rstrip("\n").split("\n")
        line_h = 4.2
        block_h = len(lines) * line_h + 6
        # Background
        self.set_fill_color(*GRAY_50)
        self.set_draw_color(*GRAY_200)
        y0 = self.get_y()
        if y0 + block_h > self.h - 20:
            self.add_page()
            y0 = self.get_y()
        self.rect(18, y0, self.page_w, block_h, style="DF")
        self.set_y(y0 + 3)
        self.set_font(MONO, "", font_size)
        self.set_text_color(*SLATE)
        for line in lines:
            self.set_x(22)
            self.cell(0, line_h, line)
            self.ln(line_h)
        self.set_y(y0 + block_h + 2)

    def _table(self, headers: list[str], rows: list[list[str]],
               col_widths: list[float] | None = None):
        if col_widths is None:
            col_widths = [self.page_w / len(headers)] * len(headers)
        row_h = 6.5

        # Check if table fits, if not add page
        total_h = (len(rows) + 1) * row_h + 4
        if self.get_y() + total_h > self.h - 20:
            self.add_page()

        # Header row
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font(SANS, "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h, f"  {h}", border=0, fill=True)
        self.ln(row_h)

        # Data rows
        self.set_font(SANS, "", 8)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 0:
                self.set_fill_color(*WHITE)
            else:
                self.set_fill_color(*BLUE_50)
            self.set_text_color(*SLATE)
            for i, cell in enumerate(row):
                style = ""
                text = cell
                if text.startswith("**") and text.endswith("**"):
                    text = text[2:-2]
                    style = "B"
                self.set_font(SANS, style, 8)
                self.cell(col_widths[i], row_h, f"  {text}", border=0, fill=True)
            self.ln(row_h)
        self.ln(2)

    def _stat_card(self, label: str, value: str, x: float, y: float,
                   w: float, accent_color=BLUE_700):
        self.set_fill_color(*BLUE_50)
        self.rect(x, y, w, 18, style="F")
        # Accent bar
        self.set_fill_color(*accent_color)
        self.rect(x, y, 2, 18, style="F")
        # Value
        self.set_xy(x + 5, y + 2)
        self.set_font(SANS, "B", 14)
        self.set_text_color(*accent_color)
        self.cell(w - 8, 8, value)
        # Label
        self.set_xy(x + 5, y + 10)
        self.set_font(SANS, "", 7.5)
        self.set_text_color(*GRAY_600)
        self.cell(w - 8, 6, label)

    def _callout_box(self, text: str, bg_color=AMBER_50, border_color=AMBER_600):
        self.ln(1)
        self.set_fill_color(*bg_color)
        self.set_draw_color(*border_color)
        y0 = self.get_y()
        # Measure height
        self.set_font(SANS, "", 9)
        lines = self.multi_cell(self.page_w - 12, 5, text, dry_run=True, output="LINES")
        block_h = max(len(lines) * 5 + 6, 14)
        if y0 + block_h > self.h - 20:
            self.add_page()
            y0 = self.get_y()
        self.rect(18, y0, self.page_w, block_h, style="DF")
        # Accent bar
        self.set_fill_color(*border_color)
        self.rect(18, y0, 2.5, block_h, style="F")
        self.set_xy(24, y0 + 3)
        self.set_font(SANS, "", 9)
        self.set_text_color(*SLATE)
        self.multi_cell(self.page_w - 12, 5, text)
        self.set_y(y0 + block_h + 3)


def build_pdf(output_path: str = "Numba_WFO_Product_Summary.pdf"):
    pdf = SummaryPDF()
    pdf.alias_nb_pages()

    # ══════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()

    # Background band
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 120, style="F")

    # Title
    pdf.set_y(35)
    pdf.set_font(SANS, "B", 36)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 15, "Numba WFO", align="C", new_x="LMARGIN", new_y="NEXT")

    # Subtitle
    pdf.set_font(SANS, "", 13)
    pdf.set_text_color(180, 200, 255)
    pdf.cell(0, 8, "Walk-Forward Optimization for Systematic Trading", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Strategy Discovery", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Accent line
    pdf.set_draw_color(100, 150, 255)
    pdf.set_line_width(0.8)
    pdf.line(70, 82, 140, 82)

    # Tagline
    pdf.set_y(90)
    pdf.set_font(SANS, "I", 10)
    pdf.set_text_color(200, 215, 255)
    pdf.cell(0, 6, "58M+ parameter combinations. 4 overlapping market regimes.",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Only the robust survive.", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Stats row
    y_cards = 132
    card_w = 40
    gap = 5.3
    x_start = 18
    cards = [
        ("Combinations Tested", "58M+", BLUE_700),
        ("Strategy Variants", "6", TEAL_700),
        ("WFO Phases", "4", GREEN_700),
        ("Runtime (32 threads)", "~15 min", AMBER_600),
    ]
    for i, (label, value, color) in enumerate(cards):
        pdf._stat_card(label, value, x_start + i * (card_w + gap), y_cards,
                       card_w, accent_color=color)

    # Body intro
    pdf.set_y(162)
    pdf.set_text_color(*SLATE)
    pdf._body(
        "A research-grade pipeline that stress-tests millions of parameter "
        "combinations across overlapping market regimes, surfaces the robust "
        "ones, and exports refinement-ready configurations for TradeStation."
    )

    pdf.ln(2)

    pdf._body(
        "The output is not a single \"best\" backtest. It is a map of robust "
        "parameter regions \u2014 plateaus where many nearby combinations all "
        "perform well \u2014 ready for live deployment."
    )

    # ══════════════════════════════════════════════════════════════
    # THE PROBLEM / THE SOLUTION
    # ══════════════════════════════════════════════════════════════
    pdf.ln(4)
    pdf._section_title("The Problem")
    pdf._body(
        "Manual strategy optimization is slow, brittle, and biased. Most "
        "backtests find parameters that worked in-sample but collapse "
        "out-of-sample. Traders waste weeks tweaking knobs on a single "
        "symbol, with no systematic way to separate signal from noise."
    )
    pdf._body(
        "Walk-forward optimization solves this \u2014 but brute-force WFO "
        "over millions of combinations is computationally prohibitive in "
        "Python, and TradeStation's built-in optimizer can't search at this scale."
    )

    pdf._section_title("The Solution")
    pdf._body(
        "Numba WFO replaces guesswork with exhaustive, parallelized parameter "
        "search. Every combination is tested across 4 overlapping time windows. "
        "Only parameters that prove stable across multiple independent market "
        "regimes survive."
    )

    # ══════════════════════════════════════════════════════════════
    # PIPELINE OVERVIEW
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._section_title("End-to-End Pipeline")

    pdf._code_block(
        "  Data Ingestion        WFO Grid Search         Scoring & Filtering\n"
        " +----------------+   +--------------------+   +---------------------+\n"
        " | TradeStation   |   |  58M+ combos       |   | Quality gates:      |\n"
        " | API ---------->+-->|  4 phases           +-->|  - IS & OOS > 0     |\n"
        " | Parquet cache  |   |  Numba JIT          |   |  - WFE > 35%        |\n"
        " | Auto-fetch     |   |  10M chunk //       |   |  - DD sanity        |\n"
        " +----------------+   +--------------------+   |  - 3/4 phases       |\n"
        "                                                +----------+----------+\n"
        "                                                           |\n"
        "                   TS Refinement Export      Plateau Detection\n"
        "                  +--------------------+   +---------------------+\n"
        "                  | Range CSVs         |   | DBSCAN clustering   |\n"
        "                  | Config manifest    |<--| Regime labels       |\n"
        "                  | Tiered params      |   | Top-combo extract   |\n"
        "                  +--------------------+   +---------------------+",
        font_size=7,
    )

    # Phase descriptions
    stages = [
        ("1  Data Ingestion",
         "TradeStation API with OAuth 2.0, incremental Parquet cache, "
         "auto-fetch with graceful fallback on holidays/weekends."),
        ("2  WFO Grid Search",
         "58M+ parameter combinations tested across 4 overlapping IS/OOS "
         "phases. Numba JIT kernels split the work into 10M-element parallel chunks."),
        ("3  Scoring & Filtering",
         "Three quality gates (profitability, WFE > 35%, drawdown sanity) per phase. "
         "Cross-phase stability requires passing 3 of 4 windows."),
        ("4  Plateau Detection",
         "DBSCAN clustering on the (atr_multiplier, atr_buy_multiplier) landscape. "
         "Identifies regions of robust performance, not isolated peaks."),
        ("5  TS Refinement Export",
         "Generates tiered parameter range CSVs and a config manifest for "
         "TradeStation's Phase 5 fine-grained optimizer."),
    ]
    for label, desc in stages:
        pdf._subsection_title(f"Stage {label}")
        pdf._body(desc)

    # ══════════════════════════════════════════════════════════════
    # STRATEGY LIBRARY
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._section_title("Strategy Library")

    pdf._body_md(
        "Six JIT-compiled strategy kernels built on the **ATR-D v1.5** "
        "architecture. Each is a self-contained Numba function that processes "
        "raw OHLC arrays and returns trade-level PnL."
    )

    pdf._table(
        ["Variant", "Direction", "Key Signals", "Inputs", "Params"],
        [
            ["**atr_d_l**", "Long-only", "ATR trailing stop + buy-touch", "1 sym", "11"],
            ["**atr_d_ls**", "Long/Short", "+ short mult, trend veto", "1 sym", "16"],
            ["**atr_d_hma_l**", "Long-only", "+ Hull MA crossover", "1 sym", "13"],
            ["**atr_d_hma_ls**", "Long/Short", "+ HMA + shorts", "1 sym", "14"],
            ["**atr_d_duals_l**", "Long-only", "Data1 price, Data2 BB Width", "2 sym", "11"],
            ["**atr_d_duals_ls**", "Long/Short", "Dual-symbol + short logic", "2 sym", "16"],
        ],
        col_widths=[32, 22, 48, 16, 14],
    )

    pdf._callout_box(
        "Automatic strategy expansion \u2014 requesting atr_d_duals_ls also "
        "runs duals_l, ls, and l variants, so you can see whether dual-symbol "
        "and short-side logic actually improve results."
    )

    # ── Core Mechanics ─────────────────────────────────────────────
    pdf.ln(2)
    pdf._section_title("Core Mechanics")

    pdf._subsection_title("Adaptive Trailing Stop")
    pdf._body("The stop adjusts dynamically to volatility:")
    pdf._code_block("stop = highest_since_entry - ATR(period) x multiplier x bb_tightening")
    pdf._bullet("**Bollinger Band tightening**: When BB Width drops below a volatility "
                "threshold, the stop narrows \u2014 protecting profits during low-vol squeezes")
    pdf._bullet("**Three scaling modes**: Linear, square-root, and quadratic tightening curves")
    pdf._bullet("**Max stop distance gate**: A 70-bar ATR baseline caps how wide the stop "
                "can drift, preventing catastrophic single-trade losses")

    pdf._subsection_title("Buy-Touch Re-Entry")
    pdf._body(
        "After a stop-out, the strategy re-enters when price pulls back to a "
        "configurable ATR-based threshold \u2014 capturing mean-reversion bounces "
        "without chasing momentum."
    )

    pdf._subsection_title("Short-Side Logic (LS Variants)")
    pdf._bullet("**Short multiplier** scales the ATR period and multiplier for shorts "
                "(e.g. 0.6x = tighter short stops)")
    pdf._bullet("**Trend veto** blocks short entries when the EMA slope is below a threshold")
    pdf._bullet("**Entry modes**: Flat-only transitions, long-priority, or full "
                "bidirectional reversals")

    pdf._subsection_title("Hull Moving Average (HMA Variants)")
    pdf._body(
        "Fast/slow HMA crossover gates entries. Re-entry via buy-touch "
        "requires the HMA trend to still be favorable \u2014 no re-entering "
        "against the trend."
    )

    pdf._subsection_title("Dual-Symbol (DualS Variants)")
    pdf._body(
        "Trades a leveraged ETF (e.g. TQQQ) using its own price for ATR "
        "stops, but computes BB Width from the underlying index (e.g. QQQ) "
        "\u2014 producing more stable volatility signals on inherently noisy "
        "leveraged instruments."
    )

    # ══════════════════════════════════════════════════════════════
    # WALK-FORWARD OPTIMIZATION
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._section_title("Walk-Forward Optimization")

    pdf._subsection_title("4-Phase Overlapping Windows")
    pdf._body(
        "Each phase is an independent continuous backtest with an IS/OOS split. "
        "Overlapping windows ensure every year of data participates in "
        "multiple roles (in-sample and out-of-sample):"
    )

    pdf._code_block(
        "Phase 1:  |--- 6y IS ---|--- 4y OOS ---|                  10y -> 5y ago\n"
        "Phase 2:       |--- 5y IS ---|--- 5y OOS ---|             8y -> 3y ago\n"
        "Phase 3:            |--- 5y IS ---|--- 5y OOS ---|        6y -> 1y ago\n"
        "Phase 4:                 |--- 4y IS ---|--- 4y OOS ---|   4y -> present\n"
        "          ------------------------------------------------->\n"
        "          past                                      today"
    )

    pdf._bullet("**Continuous capital flow**: IS profits compound into OOS, "
                "matching real-world equity curves")
    pdf._bullet("**100-bar warmup**: Every phase initializes indicators before "
                "the first trade")
    pdf._bullet("**Open position handoff**: Trades spanning the IS/OOS boundary "
                "are marked-to-market and carried forward")

    pdf._subsection_title("Chunk-Parallel Execution")
    pdf._body(
        "The 58M+ combination space is split into 10M-element chunks, each "
        "dispatched to a Numba @njit(parallel=True) kernel. Typical runtime: "
        "8-12 minutes on a 32-thread machine."
    )

    # ══════════════════════════════════════════════════════════════
    # SCORING SYSTEM
    # ══════════════════════════════════════════════════════════════
    pdf._section_title("Scoring System (v4)")

    pdf._subsection_title("Quality Gates (Per Phase)")
    pdf._body(
        "Every combination must clear three filters in each phase it "
        "participates in:"
    )

    pdf._table(
        ["Gate", "Rule", "Purpose"],
        [
            ["**Profitability**", "IS PnL > 0 AND OOS PnL > 0",
             "Eliminates curve-fit losers"],
            ["**Walk-Forward Efficiency**", "OOS$ / IS$ > 0.35",
             "OOS retains 35%+ of IS edge"],
            ["**Drawdown Sanity**", "OOS DD < OOS PnL, OOS DD < 3x IS DD",
             "Catches regime-blown combos"],
        ],
        col_widths=[36, 55, 43],
    )

    pdf._subsection_title("Cross-Phase Stability")
    pdf._body(
        "A combination must pass quality gates in 3 of 4 phases to be "
        "considered stable. This is the primary robustness filter \u2014 one "
        "bad phase is tolerated, but consistent failure is not."
    )
    pdf._body(
        "If no combinations qualify at 3/4, the threshold auto-relaxes "
        "to 2/4 with a warning."
    )

    pdf._subsection_title("Stability Score")
    pdf._code_block("Stability = avg_rank x phase_bonus x consistency_factor")

    pdf._table(
        ["Component", "Formula", "Effect"],
        [
            ["**avg_rank**", "Mean percentile rank across valid phases",
             "Higher OOS PnL = higher rank"],
            ["**phase_bonus**", "1 + (phases_valid - 1) x 0.1",
             "4-phase: 1.3x; 3-phase: 1.2x"],
            ["**consistency_factor**", "1 / (1 + stdev(ranks) / 20)",
             "Penalizes erratic phase swings"],
        ],
        col_widths=[36, 52, 46],
    )

    pdf._body("The top 2M combinations by stability score are returned as candidates.")

    # ══════════════════════════════════════════════════════════════
    # PLATEAU DETECTION
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._section_title("Plateau Detection")

    pdf._callout_box(
        "Rather than picking a single \"best\" parameter set, the system "
        "identifies regions of parameter space where many nearby combinations "
        "all perform well. Plateaus are inherently more robust than isolated peaks.",
        bg_color=GREEN_50, border_color=GREEN_700,
    )

    pdf._subsection_title("How It Works")
    numbered = [
        "Aggregate all candidates into (atr_multiplier, atr_buy_multiplier) cells",
        "Filter cells with fewer than 50 stable combinations",
        "Cluster the surviving cells using DBSCAN with automated hyperparameter search",
        "Classify each cluster's stop regime (see table below)",
        "Extract the top 10% of combinations from each plateau for refinement",
    ]
    for i, item in enumerate(numbered, 1):
        pdf.set_font(SANS, "", 9.5)
        pdf.set_text_color(*SLATE)
        pdf.set_x(22)
        pdf.set_font(SANS, "B", 9.5)
        pdf.write(5, f"{i}.  ")
        pdf.set_font(SANS, "", 9.5)
        pdf.write(5, item)
        pdf.ln(6)

    pdf.ln(2)
    pdf._table(
        ["Regime", "ATR Multiplier", "Character"],
        [
            ["**TIGHT**", "< 5", "Aggressive stops, more trades, tighter risk"],
            ["**MEDIUM**", "5 \u2013 12", "Balanced risk/reward"],
            ["**WIDE**", "> 12", "Loose stops, fewer trades, trend-following"],
        ],
        col_widths=[30, 35, 69],
    )

    # ══════════════════════════════════════════════════════════════
    # TRADESTATION INTEGRATION
    # ══════════════════════════════════════════════════════════════
    pdf._section_title("TradeStation Integration")

    pdf._subsection_title("Range Export")
    pdf._body(
        "For each plateau, the system generates a phase5_refine_*.csv with "
        "parameter ranges ready for TradeStation's optimizer:"
    )

    pdf._table(
        ["Tier", "Parameters", "Treatment"],
        [
            ["**Tier 1** (search)", "ATR periods, multipliers, BB deviation, HMA lengths",
             "Min/max/step from plateau bounds"],
            ["**Tier 2\u20133** (fixed)", "Scaling method, entry mode, vol threshold",
             "Locked at median"],
        ],
        col_widths=[36, 60, 38],
    )

    pdf._subsection_title("Refinement Analysis")
    pdf._body(
        "After TradeStation runs Phase 5 refinement, run_analysis.py ranks "
        "the final results across six metrics with per-strategy rank-sum "
        "composite scoring:"
    )

    metrics = [
        ("MAR Ratio", "CAGR / Max Drawdown"),
        ("Sharpe Ratio", "risk-adjusted return"),
        ("Sortino Ratio", "downside-risk-adjusted return"),
        ("Profit Factor", "gross profit / gross loss"),
        ("Profit / Drawdown", "net profit / max drawdown"),
        ("Robustness", "% of profitable combinations in the source plateau"),
    ]
    for name, desc in metrics:
        pdf._bullet(f"**{name}** \u2014 {desc}")

    pdf._body(
        "Output: a formatted Excel workbook with conditional formatting, "
        "ready for final selection."
    )

    # ══════════════════════════════════════════════════════════════
    # DATA MANAGEMENT
    # ══════════════════════════════════════════════════════════════
    pdf._section_title("Data Management")

    pdf._subsection_title("Auto-Fetch")
    pdf._body(
        "run_wfo.py automatically checks data coverage before every run:"
    )
    pdf._bullet("Reads `{SYMBOL}_1day.meta.json` to determine cached date range")
    pdf._bullet("If data is missing or stale, calls `ParquetStore.get()` to "
                "incrementally download only the gap")
    pdf._bullet("Gracefully falls back to cached data if the API is unavailable "
                "(weekends, holidays, network issues)")
    pdf._bullet("`--no-fetch` disables auto-fetch for offline/air-gapped environments")

    pdf._subsection_title("Incremental Parquet Cache")
    pdf._bullet("**Cache hit**: meta.json covers the requested range \u2014 instant return")
    pdf._bullet("**Partial miss**: Only fetches the pre-gap or post-gap, merges "
                "with existing data")
    pdf._bullet("**Full miss**: Downloads the entire range, writes parquet + meta.json")

    # ══════════════════════════════════════════════════════════════
    # DESIGN DECISIONS
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._section_title("Key Design Decisions")

    decisions = [
        ("Continuous IS\u2192OOS backtest",
         "Matches TradeStation's single-simulation model; capital compounds naturally."),
        ("Ring-buffer indicators",
         "O(1) per-bar updates, cache-friendly memory access for Numba JIT kernels."),
        ("2D DBSCAN on multipliers",
         "Interpretable axes (stop width vs. entry sensitivity), directly actionable "
         "for refinement."),
        ("v4 quality-gate scoring",
         "Earlier versions used top-20% cutoffs that missed high-quality regions; v4 "
         "uses hard filters + cross-phase stability."),
        ("Strategy expansion",
         "Automatically answers \"does adding shorts/dual-symbol/HMA actually help?\" "
         "without separate manual runs."),
        ("Yesterday as fetch end-date",
         "Today's daily bar isn't available until after market close; avoids API 404s "
         "on holidays and weekends."),
    ]

    pdf._table(
        ["Decision", "Rationale"],
        [[f"**{d}**", r] for d, r in decisions],
        col_widths=[48, 86],
    )

    # ══════════════════════════════════════════════════════════════
    # QUICK START
    # ══════════════════════════════════════════════════════════════
    pdf._section_title("Quick Start")

    commands = [
        ("Single symbol, long-only \u2014 auto-downloads, runs WFO, plateaus, exports ranges",
         "python run_wfo.py --symbol SMH --strategy atr_d_l --grid phase1_slim"),
        ("Dual-symbol with full expansion (runs 4 variants automatically)",
         "python run_wfo.py --symbol TQQQ --symbol2 QQQ --strategy atr_d_duals_ls"),
        ("Re-score existing results with updated scoring logic",
         "python run_wfo.py --symbol SMH --rescore --wfo-dir results/SMH/wfo/atr_d_l"),
        ("Analyze TradeStation refinement output",
         "python run_analysis.py --symbol SMH --top 50"),
        ("Manual data download (minute bars, specific date range)",
         "python download_data.py --symbols SMH QQQ --timeframe 1min "
         "--start 2016-01-01 --end 2026-02-14"),
    ]
    for desc, cmd in commands:
        pdf.set_font(SANS, "I", 8)
        pdf.set_text_color(*GRAY_600)
        pdf.cell(0, 4, f"# {desc}", new_x="LMARGIN", new_y="NEXT")
        pdf._code_block(cmd)
        pdf.ln(1)

    # ══════════════════════════════════════════════════════════════
    # OUTPUT STRUCTURE
    # ══════════════════════════════════════════════════════════════
    pdf._section_title("Output Structure")

    pdf._code_block(
        "results/SMH/\n"
        "+-- wfo/\n"
        "|   +-- atr_d_l/phase1/         # Binary cache: is_pnl.npy, oos_pnl.npy, ...\n"
        "|   +-- atr_d_l/phase2/\n"
        "|   +-- atr_d_l/phase3/\n"
        "|   +-- atr_d_l/phase4/\n"
        "|   +-- atr_d_ls/phase1/ ... phase4/\n"
        "|   +-- ...\n"
        "+-- candidates_atr_d_l_phase1_slim.csv       # Stable combos with scores\n"
        "+-- candidates_atr_d_ls_phase1_slim.csv\n"
        "+-- phase5_refine_ATR-D-v1.5-L_rank1.csv     # TS refinement ranges\n"
        "+-- phase5_refine_ATR-D-v1.5-L_rank2.csv\n"
        "+-- phase5_refine_ATR-D-v1.5-LS_rank1.csv\n"
        "+-- phase5_refine_config.csv                  # Combined config\n"
        "+-- Refinement_Analysis_SMH.xlsx              # Final ranked results"
    )

    # ── Requirements ───────────────────────────────────────────────
    pdf.ln(4)
    pdf._section_title("Requirements")
    pdf._bullet("Python 3.10+")
    pdf._bullet("Numba, NumPy, Pandas, SciPy, scikit-learn, PyArrow")
    pdf._bullet("TradeStation API credentials (for data download)")
    pdf._bullet("TradeStation desktop (for Phase 5 refinement)")

    # ══════════════════════════════════════════════════════════════
    # WRITE
    # ══════════════════════════════════════════════════════════════
    pdf.output(output_path)
    print(f"PDF written to: {output_path}")


if __name__ == "__main__":
    build_pdf()
