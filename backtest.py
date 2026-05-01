#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  FVG Trading Bot — Backtester  (optimised)
#  Uses REAL candles fetched live from Binance
#  Run: python backtest.py
#
#  Performance:
#    FVG detection is fully vectorised (numpy / CuPy GPU).
#    --compare runs all 4 TP modes in parallel processes.
#
#  TP MODES:
#   fixed     — exit at N× reward-to-risk
#   structure — exit at nearest swing high/low
#   trailing  — ATR trail after 1R profit
#   partial   — 50% at 2R, trail rest to structure (default)
# ─────────────────────────────────────────────

import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, NamedTuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    RISK_PCT, MIN_RR,
    FVG_MIN_SIZE_PCT, FVG_EXPIRY_CANDLES, SL_BUFFER_PCT,
    FVG_SCORE_MIN, HTF_TF, HTF_EMA_PERIOD,
)
from exchange import get_exchange, fetch_historical_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Optional GPU (CuPy) ───────────────────────
try:
    import cupy as cp
    _GPU = True
    logger.info("CuPy detected — FVG scan will use GPU")
except ImportError:
    cp = None
    _GPU = False


# ── Pre-computed FVG record ───────────────────
class _RawFVG(NamedTuple):
    idx:       int    # C3 index in original df (formation candle)
    direction: str    # "bullish" | "bearish"
    gap_low:   float
    gap_high:  float
    sl:        float  # stop-loss price
    score:     float  # quality score 0-1 (matches live fvg_quality_score)


# ── Vectorised FVG scan ───────────────────────
def detect_all_fvgs(df: pd.DataFrame,
                    min_size_pct: float = FVG_MIN_SIZE_PCT,
                    sl_buffer_pct: float = SL_BUFFER_PCT) -> list[_RawFVG]:
    """
    Scan the entire OHLCV DataFrame in ONE pass using numpy array ops
    (or CuPy on GPU if available).

    Replaces the original O(n²) approach of calling detect_fvgs() inside
    the simulation loop for every candle step.  This function is O(n) and
    typically runs in < 10 ms even for 90 days of 1m data.

    Returns: list of _RawFVG sorted by formation index (ascending).
    """
    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n      = len(highs)
    if n < 3:
        return []

    t0 = time.perf_counter()

    # ── Sliced views for C1 / C2 / C3 ────────
    h1, l1 = highs[:-2], lows[:-2]   # C1
    h2, l2 = highs[1:-1], lows[1:-1] # C2 (SL source)
    h3, l3 = highs[2:],  lows[2:]    # C3
    px     = closes[2:]               # C3 close for size normalisation

    if _GPU:
        # Transfer slice arrays to GPU, run boolean masks there, pull indices back
        h1g = cp.asarray(h1); l1g = cp.asarray(l1)
        h3g = cp.asarray(h3); l3g = cp.asarray(l3)
        pxg = cp.asarray(px)

        gap_b   = l3g - h1g
        bull_ok = (gap_b > 0) & (gap_b / pxg >= min_size_pct)
        bull_k  = cp.asnumpy(cp.where(bull_ok)[0])

        gap_r   = l1g - h3g
        bear_ok = (gap_r > 0) & (gap_r / pxg >= min_size_pct)
        bear_k  = cp.asnumpy(cp.where(bear_ok)[0])
    else:
        gap_b   = l3 - h1
        bull_ok = (gap_b > 0) & (gap_b / px >= min_size_pct)
        bull_k  = np.where(bull_ok)[0]

        gap_r   = l1 - h3
        bear_ok = (gap_r > 0) & (gap_r / px >= min_size_pct)
        bear_k  = np.where(bear_ok)[0]

    # ── Pre-compute score inputs ATR + rolling avg volume ──
    # Mirrors fvg_quality_score() in fvg_detector.py:
    #   score = 0.4 * (gap/atr capped at 3) + 0.3 * (vol/avg capped at 3) + 0.3 * displacement
    atr_arr = calc_atr(df, period=14).values
    vol_arr = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.zeros(n)
    opens   = df["open"].values.astype(np.float64)
    # 100-candle rolling mean volume (matches live "avg_vol = df.volume.mean()" on FVG_LOOKBACK)
    if vol_arr.sum() > 0:
        vol_series = pd.Series(vol_arr).rolling(window=100, min_periods=1).mean().values
    else:
        vol_series = np.zeros(n)

    def _score(idx: int, gap_size: float) -> float:
        atr_i = atr_arr[idx] if idx < len(atr_arr) and not np.isnan(atr_arr[idx]) else 0.0
        if atr_i <= 0:
            return 0.5
        gap_atr = min(gap_size / atr_i, 3.0) / 3.0
        avg_vol = vol_series[idx] if idx < len(vol_series) else 0.0
        v       = vol_arr[idx]   if idx < len(vol_arr)   else 0.0
        vol_ratio = 0.5 if avg_vol <= 0 else min(v / avg_vol, 3.0) / 3.0
        # Displacement: middle candle (C2 = idx-1) body-to-range ratio.
        # Full-body candle = strong institutional displacement = 1.0.
        # Doji (tiny body, long wicks) = weak = near 0.
        # Replaces old proximity=1.0 which was always maxed at formation.
        displacement = 0.5
        mid = idx - 1
        if 0 <= mid < n:
            c2_body  = abs(closes[mid] - opens[mid])
            c2_range = highs[mid] - lows[mid]
            if c2_range > 0:
                displacement = min(c2_body / c2_range, 1.0)
        return round(min(gap_atr * 0.4 + vol_ratio * 0.3 + displacement * 0.3, 1.0), 3)

    # Build list (back on CPU for Python simulation loop)
    fvgs: list[_RawFVG] = []

    for k in bull_k:
        k = int(k)
        i = k + 2  # C3 index in original array
        gap_low  = float(h1[k])
        gap_high = float(l3[k])
        fvgs.append(_RawFVG(
            idx       = i,
            direction = "bullish",
            gap_low   = gap_low,
            gap_high  = gap_high,
            sl        = float(l2[k]) * (1 - sl_buffer_pct),
            score     = _score(i, gap_high - gap_low),
        ))

    for k in bear_k:
        k = int(k)
        i = k + 2
        gap_high = float(l1[k])
        gap_low  = float(h3[k])
        fvgs.append(_RawFVG(
            idx       = i,
            direction = "bearish",
            gap_high  = gap_high,
            gap_low   = gap_low,
            sl        = float(h2[k]) * (1 + sl_buffer_pct),
            score     = _score(i, gap_high - gap_low),
        ))

    fvgs.sort(key=lambda x: x.idx)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"detect_all_fvgs: {len(fvgs)} FVGs from {n} candles "
        f"in {elapsed:.1f} ms ({'GPU' if _GPU else 'CPU'})"
    )
    return fvgs


# ── ATR (Wilder's RMA) ────────────────────────
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (RMA / alpha=1/period) — matches TradingView and order_manager.py."""
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ── Swing structure target ────────────────────
def find_structure_target(highs: np.ndarray, lows: np.ndarray,
                           direction: str, entry: float,
                           min_rr: float, sl_dist: float,
                           lookback: int = 50) -> float:
    """Nearest swing high/low beyond min R:R. Falls back to 3R."""
    h = highs[-lookback:] if len(highs) >= lookback else highs
    l = lows[-lookback:]  if len(lows)  >= lookback else lows
    min_tp = entry + sl_dist * min_rr if direction == "bullish" else entry - sl_dist * min_rr

    pivots = []
    for j in range(2, len(h) - 2):
        if direction == "bullish":
            if h[j] == h[j-2:j+3].max() and h[j] > min_tp:
                pivots.append(float(h[j]))
        else:
            if l[j] == l[j-2:j+3].min() and l[j] < min_tp:
                pivots.append(float(l[j]))

    if pivots:
        return min(pivots) if direction == "bullish" else max(pivots)
    # No real swing ≥ min_rr away — return entry so the caller's rr gate
    # rejects the trade. Backtest proved blind fallback targets degrade
    # the strategy (PF 1.80 → 1.33). Skip is the right call.
    return entry  # rr = 0 → guaranteed gate rejection


# ── Trade record ──────────────────────────────
@dataclass
class BTTrade:
    symbol:        str
    direction:     str     # "long" | "short"
    entry_price:   float
    sl_price:      float
    tp_price:      float
    qty:           float
    risk_usdt:     float
    open_time:     pd.Timestamp
    close_time:    Optional[pd.Timestamp] = None
    exit_price:    Optional[float]        = None
    result:        str                    = "open"
    pnl_usdt:      float                  = 0.0   # net of fees
    gross_pnl:     float                  = 0.0   # before fees
    fees_paid:     float                  = 0.0   # total fees this trade (entry + partial + exit)
    pnl_r:         float                  = 0.0
    fvg_tf:        str                    = ""
    liq_price:     float                  = 0.0   # 0 = no leverage / no risk
    # Dynamic tracking
    current_sl:    float                  = 0.0
    be_moved:      bool                   = False
    partial_done:  bool                   = False
    qty_remaining: float                  = 0.0
    partial_pnl:   float                  = 0.0   # net partial P&L (after partial fee)
    gross_partial: float                  = 0.0   # gross partial P&L (price-only)


# ── Backtester ────────────────────────────────
class Backtester:
    def __init__(self,
                 symbol:          str   = "BTC/USDT",
                 timeframe:       str   = "5m",
                 days:            int   = 90,
                 initial_balance: float = 1000.0,
                 risk_pct:        float = RISK_PCT,
                 min_rr:          float = MIN_RR,
                 tp_mode:         str   = "partial",
                 tp_multiplier:   float = 2.0,
                 fvg_min_size:    float = FVG_MIN_SIZE_PCT,
                 fvg_expiry:      int   = FVG_EXPIRY_CANDLES,
                 sl_buffer:       float = SL_BUFFER_PCT,
                 atr_period:      int   = 14,
                 trail_atr_mult:  float = 1.5,
                 leverage:        float = 1.0,
                 fee_rate:        float = 0.0005,   # 0.05% taker
                 max_notional_pct: float = 0.20,    # max position size as % of balance
                 min_sl_pct:      float = 0.001,    # minimum SL distance = 0.1% of price
                 slippage_pct:    float = 0.0,      # one-way slippage at entry (e.g. 0.0005)
                 max_trades_day:  int   = 0,        # 0 = unlimited (live bot uses 10)
                 score_min:       float = 0.0,      # FVG quality gate (live default 0.40)
                 use_htf_filter:  bool  = False):   # require HTF bias alignment (live default True)
        self.symbol           = symbol
        self.timeframe        = timeframe
        self.days             = days
        self.initial_balance  = initial_balance
        self.balance          = initial_balance
        self.risk_pct         = risk_pct
        self.min_rr           = min_rr
        self.tp_mode          = tp_mode
        self.tp_multiplier    = tp_multiplier
        self.fvg_min_size     = fvg_min_size
        self.fvg_expiry       = fvg_expiry
        self.sl_buffer        = sl_buffer
        self.atr_period       = atr_period
        self.trail_atr_mult   = trail_atr_mult
        self.leverage         = max(1.0, leverage)
        self.fee_rate         = fee_rate
        self.max_notional_pct = max_notional_pct
        self.min_sl_pct       = min_sl_pct
        self.slippage_pct     = slippage_pct
        self.max_trades_day   = max_trades_day
        self.score_min        = score_min
        self.use_htf_filter   = use_htf_filter

        self.trades:       list[BTTrade] = []
        self.equity_curve: list[float]  = [initial_balance]

    # ── Helpers ───────────────────────────────
    def _fee(self, notional: float) -> float:
        """Deduct fee from balance and return the fee amount."""
        f = notional * self.fee_rate
        self.balance -= f
        return f

    def _calc_qty(self, entry: float, sl: float) -> tuple[float, float]:
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return 0.0, 0.0

        # Reject trades where SL is unrealistically tight (< min_sl_pct of price).
        # On 1m candles, SL distances of 0.01–0.05% create notional 10–50× the
        # account balance, which is impossible to fill in a real spot market.
        if sl_dist / entry < self.min_sl_pct:
            return 0.0, 0.0

        risk_usdt = self.balance * self.risk_pct
        qty       = risk_usdt / sl_dist

        # ALWAYS cap position notional, regardless of leverage.
        # For spot (leverage=1): you cannot spend more than your balance.
        # Without this cap a 0.05% SL on ETH@$2000 gives qty=50 ETH on a
        # $1000 account — $100,000 notional — which is physically impossible.
        cap_by_margin   = self.balance * self.leverage / entry      # max by account size
        cap_by_notional = self.balance * self.max_notional_pct / entry  # max 20% of balance

        max_qty = min(cap_by_margin, cap_by_notional)
        if qty > max_qty:
            qty       = max_qty
            risk_usdt = qty * sl_dist   # actual risk after cap (may be < risk_pct × balance)

        return qty, risk_usdt

    def _count_open(self) -> int:
        return sum(1 for t in self.trades if t.result == "open")

    def _liq_price(self, entry: float, is_long: bool) -> float:
        """Approximate liquidation price (ignores maintenance margin for simplicity)."""
        if self.leverage <= 1:
            return 0.0
        return entry * (1 - 1 / self.leverage) if is_long \
               else entry * (1 + 1 / self.leverage)

    # ── Main simulation ───────────────────────
    def run(self, df: pd.DataFrame) -> "BacktestResult":
        t_start = time.perf_counter()
        logger.info(
            f"Backtest [{self.tp_mode.upper()}] {self.symbol} {self.timeframe} "
            f"| {len(df)} candles | ${self.initial_balance:.0f} "
            f"| {self.leverage:.0f}x lev | fee={self.fee_rate*100:.3f}%"
        )

        # ── Pre-compute (vectorised) ───────────
        highs     = df["high"].values
        lows      = df["low"].values
        closes    = df["close"].values
        times     = df.index
        n         = len(df)
        atr_arr   = calc_atr(df, self.atr_period).values

        # All FVGs detected in ONE vectorised pass — O(n) instead of O(n²)
        all_fvgs = detect_all_fvgs(df, self.fvg_min_size, self.sl_buffer)
        fvg_ptr  = 0   # advances through all_fvgs as the sim progresses

        # ── HTF bias series (1h EMA-50, aligned to 5m index) ──
        # Mirrors get_htf_bias() in bot.py: bullish if price > ema * 1.001,
        # bearish if price < ema * 0.999, neutral in between.
        if self.use_htf_filter:
            htf_df = df.resample(HTF_TF).agg({
                "open":  "first", "high": "max", "low": "min",
                "close": "last",  "volume": "sum",
            }).dropna()
            htf_ema   = htf_df["close"].ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
            htf_close = htf_df["close"]
            # Forward-fill 1h values to every 5m candle so we can index by i
            htf_ema_5m   = htf_ema.reindex(df.index, method="ffill").values
            htf_close_5m = htf_close.reindex(df.index, method="ffill").values
        else:
            htf_ema_5m   = None
            htf_close_5m = None

        def _bias_at(i: int) -> str:
            if not self.use_htf_filter or htf_ema_5m is None:
                return "neutral"
            ema_v   = htf_ema_5m[i]
            price_v = htf_close_5m[i]
            if np.isnan(ema_v) or np.isnan(price_v):
                return "neutral"
            if price_v > ema_v * 1.001:
                return "bullish"
            if price_v < ema_v * 0.999:
                return "bearish"
            return "neutral"

        MAX_OPEN = 3

        # Active FVG state: list of dicts (lightweight vs full FVG objects)
        active: list[dict] = []

        # Daily trade counter — mirrors the live bot's MAX_TRADES_DAY limit.
        # Key is the date string (YYYY-MM-DD) so it rolls over at midnight.
        daily_trades: dict[str, int] = {}

        # Per-symbol cooldown after stop-loss — mirrors live bot's 5-min cooldown.
        # Maps symbol → candle index when cooldown expires.
        cooldown_until: dict[str, int] = {}

        # Convert timeframe string to seconds for cooldown calculation
        _tf_map = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}
        tf_seconds = _tf_map.get(self.timeframe, 300)
        cooldown_candles = max(1, int(300 / tf_seconds))  # 5 min cooldown

        # ── Simulation loop ───────────────────
        for i in range(3, n):
            c_high  = highs[i]
            c_low   = lows[i]
            c_close = closes[i]
            c_time  = times[i]
            c_atr   = float(atr_arr[i])

            # ── Add FVGs that formed strictly before this candle ──
            # (a FVG formed at candle i can only be retested from candle i+1)
            while fvg_ptr < len(all_fvgs) and all_fvgs[fvg_ptr].idx < i:
                rf = all_fvgs[fvg_ptr]
                active.append({
                    "rf":      rf,
                    "status":  "waiting",
                    "candles": 0,
                    "c_sl":    rf.sl,   # live (trailing) SL for this FVG slot
                })
                fvg_ptr += 1

            # ── Manage open trades ─────────────
            for trade in self.trades:
                if trade.result != "open":
                    continue

                sl_dist = abs(trade.entry_price - trade.sl_price)
                is_long = trade.direction == "long"
                sign    = 1 if is_long else -1

                if self.tp_mode in ("trailing", "partial"):
                    profit_r = (c_close - trade.entry_price) / sl_dist * sign

                    # Breakeven at 1R
                    if profit_r >= 1.0 and not trade.be_moved:
                        be = trade.entry_price
                        trade.current_sl = max(trade.current_sl, be) if is_long \
                                           else min(trade.current_sl, be)
                        trade.be_moved = True

                    # ATR trail at 1.5R
                    if profit_r >= 1.5:
                        if is_long:
                            new_trail = c_low - c_atr * self.trail_atr_mult
                            trade.current_sl = max(trade.current_sl, new_trail)
                        else:
                            new_trail = c_high + c_atr * self.trail_atr_mult
                            trade.current_sl = min(trade.current_sl, new_trail)

                    # Partial exit at TP1 (2R): close 50%, extend TP to structure
                    if self.tp_mode == "partial" and not trade.partial_done:
                        partial_hit = (is_long  and c_high >= trade.tp_price) or \
                                      (not is_long and c_low  <= trade.tp_price)
                        if partial_hit:
                            half_qty     = trade.qty / 2
                            gross_half   = (trade.tp_price - trade.entry_price) * half_qty * sign
                            partial_fee  = self._fee(half_qty * trade.tp_price)
                            net_half     = gross_half - partial_fee
                            self.balance += gross_half   # gross credited; fee already deducted
                            trade.partial_pnl   = net_half
                            trade.gross_partial = gross_half
                            trade.fees_paid    += partial_fee
                            trade.partial_done  = True
                            trade.qty_remaining = half_qty
                            # Extend TP to structure
                            struct_tp = find_structure_target(
                                highs[:i], lows[:i],
                                "bullish" if is_long else "bearish",
                                trade.entry_price, self.min_rr, sl_dist
                            )
                            trade.tp_price = struct_tp

                # ── SL / TP check ──────────────
                active_sl = trade.current_sl if self.tp_mode in ("trailing", "partial") \
                            else trade.sl_price

                sl_hit = (is_long  and c_low  <= active_sl) or \
                         (not is_long and c_high >= active_sl)

                # TRAILING mode has no fixed TP — exits ONLY via the trailing
                # SL (or liquidation). The 2R tp_price stored at entry is just
                # a placeholder so the rr-validation gate at entry passes.
                if self.tp_mode == "trailing":
                    tp_hit = False
                else:
                    tp_hit = (is_long  and c_high >= trade.tp_price) or \
                             (not is_long and c_low  <= trade.tp_price)

                # Check liquidation (only matters if leverage > 1)
                if trade.liq_price > 0:
                    liq_hit = (is_long  and c_low  <= trade.liq_price) or \
                              (not is_long and c_high >= trade.liq_price)
                    if liq_hit and not sl_hit:
                        active_sl = trade.liq_price
                        sl_hit = True

                if sl_hit and tp_hit:
                    sl_hit, tp_hit = True, False   # conservative

                remaining = trade.qty_remaining if trade.partial_done else trade.qty

                if sl_hit or tp_hit:
                    exit_p        = active_sl if sl_hit else trade.tp_price
                    gross_r       = (exit_p - trade.entry_price) * remaining * sign
                    exit_fee      = self._fee(remaining * exit_p)
                    self.balance += gross_r
                    trade.fees_paid += exit_fee

                    # Correct accounting:
                    #   gross_total = price-only PnL across both legs (no fees)
                    #   net_total   = gross_total minus ALL fees (entry + partial + exit)
                    # Previously net_total omitted the entry fee, inflating per-trade
                    # PnL stats (win rate, R:R, profit factor) by the entry-fee amount.
                    gross_total      = gross_r + trade.gross_partial
                    net_total        = gross_total - trade.fees_paid
                    trade.pnl_usdt   = net_total
                    trade.gross_pnl  = gross_total
                    trade.result     = "loss" if net_total < 0 else "win"
                    trade.exit_price = exit_p
                    trade.close_time = c_time
                    trade.pnl_r      = net_total / trade.risk_usdt if trade.risk_usdt else 0.0
                    self.equity_curve.append(self.balance)

                    # Set cooldown after stop-loss (matches live bot's 5-min cooldown)
                    if sl_hit:
                        cooldown_until[trade.symbol] = i + cooldown_candles

            # ── Check FVG retests (new entries) ──
            if self._count_open() < MAX_OPEN:

                for item in active:
                    if item["status"] != "waiting":
                        continue

                    # Per-symbol cooldown check (matches live bot)
                    if cooldown_until.get(self.symbol, 0) > i:
                        continue
                    if self._count_open() >= MAX_OPEN:
                        break

                    rf = item["rf"]

                    # Expiry and invalidation
                    item["candles"] += 1
                    if item["candles"] > self.fvg_expiry:
                        item["status"] = "expired"
                        continue
                    # Invalidation uses CLOSE, not wick — wicks that recover
                    # leave the FVG tradable for the next retest.
                    if rf.direction == "bullish" and c_close < rf.gap_low:
                        item["status"] = "invalidated"
                        continue
                    if rf.direction == "bearish" and c_close > rf.gap_high:
                        item["status"] = "invalidated"
                        continue

                    # Quality score gate (matches live FVG_SCORE_MIN)
                    if rf.score < self.score_min:
                        continue

                    # HTF bias gate (matches live get_htf_bias)
                    bias = _bias_at(i)
                    if bias == "bullish" and rf.direction == "bearish":
                        continue
                    if bias == "bearish" and rf.direction == "bullish":
                        continue

                    # Retest check — wick touches zone, entry at current price.
                    # The live bot places a MARKET order which fills at current
                    # price (c_close), NOT at gap_mid. Using gap_mid was
                    # optimistic — you can't fill inside a gap with a market order.
                    in_zone = False
                    if rf.direction == "bullish":
                        if c_low <= rf.gap_high and c_close >= rf.gap_low:
                            in_zone = True
                    else:
                        if c_high >= rf.gap_low and c_close <= rf.gap_high:
                            in_zone = True

                    if not in_zone:
                        continue

                    # Use current price as entry — matches live bot market fill
                    entry = c_close

                    # Drift check: skip if price drifted > 0.5% from gap_mid
                    gap_mid = (rf.gap_high + rf.gap_low) / 2
                    drift = abs(c_close - gap_mid) / gap_mid if gap_mid > 0 else 0
                    if drift > 0.005:
                        continue  # too far from FVG zone

                    # Daily trade cap (mirrors live bot's MAX_TRADES_DAY)
                    if self.max_trades_day > 0:
                        day_key = str(c_time.date())
                        if daily_trades.get(day_key, 0) >= self.max_trades_day:
                            continue

                    is_long = rf.direction == "bullish"
                    sign    = 1 if is_long else -1

                    # Slippage: entry price moves against the trader at fill
                    entry_slip = entry * (1 + self.slippage_pct) if is_long \
                                 else entry * (1 - self.slippage_pct)

                    sl      = rf.sl
                    sl_dist = abs(entry_slip - sl)
                    if sl_dist == 0:
                        continue

                    # TP selection by mode (anchored to slipped entry)
                    if self.tp_mode == "fixed":
                        tp = entry_slip + sl_dist * self.tp_multiplier * sign
                    elif self.tp_mode == "structure":
                        tp = find_structure_target(
                            highs[:i], lows[:i],
                            rf.direction, entry_slip, self.min_rr, sl_dist
                        )
                    else:  # trailing | partial — initial 2R target
                        tp = entry_slip + sl_dist * 2.0 * sign

                    # The rr gate only makes sense when TP is the actual exit
                    # target (fixed, structure). For partial it's just the 50%
                    # trigger; for trailing it's unused. Skip in those modes.
                    if self.tp_mode in ("fixed", "structure"):
                        rr_sign = 1 if is_long else -1
                        if (tp - entry_slip) * rr_sign / sl_dist < self.min_rr:
                            continue

                    qty, risk_usdt = self._calc_qty(entry_slip, sl)
                    if qty == 0 or self.balance <= 0:
                        continue

                    # Deduct entry fee immediately
                    entry_fee = self._fee(qty * entry_slip)
                    entry = entry_slip   # use slipped price for the trade record

                    if self.max_trades_day > 0:
                        day_key = str(c_time.date())
                        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
                    liq_p     = self._liq_price(entry, is_long)

                    trade = BTTrade(
                        symbol        = self.symbol,
                        direction     = "long" if is_long else "short",
                        entry_price   = entry,
                        sl_price      = sl,
                        tp_price      = tp,
                        qty           = qty,
                        risk_usdt     = risk_usdt,
                        open_time     = c_time,
                        fvg_tf        = self.timeframe,
                        current_sl    = sl,
                        qty_remaining = qty,
                        fees_paid     = entry_fee,
                        liq_price     = liq_p,
                    )
                    self.trades.append(trade)
                    item["status"] = "retested"

        # ── Close any trades still open at last bar ──
        last_price = float(df["close"].iloc[-1])
        for trade in self.trades:
            if trade.result == "open":
                trade.result     = "open (expired)"
                trade.exit_price = last_price
                trade.close_time = times[-1]

        elapsed = time.perf_counter() - t_start
        logger.info(f"Backtest finished in {elapsed:.2f} s | {len(self.trades)} trades")
        return self._build_result()

    def _build_result(self) -> "BacktestResult":
        return BacktestResult(
            self.symbol, self.timeframe, self.days,
            self.initial_balance, self.balance,
            self.trades, self.equity_curve,
            self.risk_pct, self.min_rr, self.tp_multiplier,
            self.tp_mode, self.leverage, self.fee_rate,
            self.max_notional_pct, self.min_sl_pct,
        )


# ── Result analytics ──────────────────────────
class BacktestResult:
    def __init__(self, symbol, timeframe, days, initial_balance, final_balance,
                 trades, equity_curve, risk_pct, min_rr, tp_mult, tp_mode,
                 leverage=1.0, fee_rate=0.0005,
                 max_notional_pct=0.20, min_sl_pct=0.001):
        self.symbol           = symbol
        self.timeframe        = timeframe
        self.days             = days
        self.initial_balance  = initial_balance
        self.final_balance    = final_balance
        self.trades           = trades
        self.equity_curve     = equity_curve
        self.risk_pct         = risk_pct
        self.min_rr           = min_rr
        self.tp_mult          = tp_mult
        self.tp_mode          = tp_mode
        self.leverage         = leverage
        self.fee_rate         = fee_rate
        self.max_notional_pct = max_notional_pct
        self.min_sl_pct       = min_sl_pct

    @property
    def closed_trades(self):
        return [t for t in self.trades if t.result in ("win", "loss")]

    @property
    def wins(self):
        return [t for t in self.closed_trades if t.result == "win"]

    @property
    def losses(self):
        return [t for t in self.closed_trades if t.result == "loss"]

    @property
    def win_rate(self):
        n = len(self.closed_trades)
        return len(self.wins) / n * 100 if n else 0

    @property
    def total_pnl(self):
        return self.final_balance - self.initial_balance   # net of all fees

    @property
    def total_pnl_pct(self):
        return self.total_pnl / self.initial_balance * 100

    @property
    def total_fees(self):
        return sum(t.fees_paid for t in self.closed_trades)

    @property
    def gross_pnl(self):
        return sum(t.gross_pnl for t in self.closed_trades)

    @property
    def avg_rr(self):
        rrs = [t.pnl_r for t in self.wins]
        return float(np.mean(rrs)) if rrs else 0.0

    @property
    def profit_factor(self):
        gross_win  = sum(t.pnl_usdt for t in self.wins)
        gross_loss = abs(sum(t.pnl_usdt for t in self.losses))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self):
        eq   = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / peak * 100
        return float(dd.min())

    @property
    def sharpe_ratio(self):
        if len(self.equity_curve) < 2:
            return 0.0
        eq    = np.array(self.equity_curve)
        rets  = np.diff(eq) / eq[:-1]
        std_r = np.std(rets)
        return float(np.mean(rets) / std_r * np.sqrt(252) if std_r > 0 else 0.0)

    @property
    def avg_win_usdt(self):
        return float(np.mean([t.pnl_usdt for t in self.wins])) if self.wins else 0.0

    @property
    def avg_loss_usdt(self):
        return float(np.mean([t.pnl_usdt for t in self.losses])) if self.losses else 0.0

    @property
    def expectancy(self):
        wr = self.win_rate / 100
        return wr * self.avg_win_usdt + (1 - wr) * self.avg_loss_usdt

    def print_report(self):
        sep      = "─" * 57
        tp_label = f"TP {self.tp_mode.upper()}" + \
                   (f" ({self.tp_mult}x)" if self.tp_mode == "fixed" else "")
        lev_str  = f"{self.leverage:.0f}x"   # market type is separate from leverage
        pnl_sign = "+" if self.total_pnl >= 0 else ""

        print(f"\n{'═'*57}")
        print(f"  BACKTEST REPORT — {self.symbol} {self.timeframe} ({self.days}d)")
        print(f"{'═'*57}")
        print(f"  Period        {self.days} days")
        print(f"  Risk/trade    {self.risk_pct*100:.2f}%  |  Min R:R {self.min_rr}  |  {tp_label}")
        print(f"  Leverage      {lev_str}  |  Fee {self.fee_rate*100:.3f}%  "
              f"|  Max notional {self.max_notional_pct*100:.0f}%  "
              f"|  Min SL {self.min_sl_pct*100:.2f}%")
        print(sep)
        print(f"  Initial bal   ${self.initial_balance:,.2f}")
        print(f"  Final bal     ${self.final_balance:,.2f}")
        print(f"  Gross P&L     {'+' if self.gross_pnl>=0 else ''}${self.gross_pnl:,.2f}")
        print(f"  Fees paid     -${self.total_fees:,.2f}")
        print(f"  Net P&L       {pnl_sign}${self.total_pnl:,.2f}  ({pnl_sign}{self.total_pnl_pct:.2f}%)")
        print(sep)
        print(f"  Total trades  {len(self.trades)}")
        print(f"  Closed        {len(self.closed_trades)}  "
              f"(wins: {len(self.wins)}  losses: {len(self.losses)})")
        print(f"  Win rate      {self.win_rate:.1f}%")
        print(f"  Avg R:R       {self.avg_rr:.2f}x")
        print(f"  Profit factor {self.profit_factor:.2f}")
        print(f"  Expectancy    ${self.expectancy:.2f}/trade")
        print(sep)
        print(f"  Max drawdown  {self.max_drawdown:.2f}%")
        print(f"  Sharpe ratio  {self.sharpe_ratio:.2f}")
        print(f"  Avg win       +${self.avg_win_usdt:.2f}")
        print(f"  Avg loss      -${abs(self.avg_loss_usdt):.2f}")
        print(f"{'═'*57}\n")

    def to_csv(self, path: str = "backtest_trades.csv"):
        rows = []
        for t in self.closed_trades:
            rows.append({
                "symbol":     t.symbol,
                "direction":  t.direction,
                "open_time":  t.open_time,
                "close_time": t.close_time,
                "entry":      round(t.entry_price, 6),
                "sl":         round(t.sl_price, 6),
                "tp":         round(t.tp_price, 6),
                "exit":       round(t.exit_price, 6) if t.exit_price else None,
                "qty":        round(t.qty, 6),
                "result":     t.result,
                "gross_pnl":  round(t.gross_pnl, 4),
                "fees_paid":  round(t.fees_paid, 4),
                "pnl_usdt":   round(t.pnl_usdt, 4),
                "pnl_r":      round(t.pnl_r, 3),
                "liq_price":  round(t.liq_price, 4) if t.liq_price else None,
            })
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info(f"Trades saved → {path}")

    def to_json(self, path: str = "backtest_summary.json"):
        summary = {
            "symbol":          self.symbol,
            "timeframe":       self.timeframe,
            "days":            self.days,
            "leverage":        self.leverage,
            "fee_rate":        self.fee_rate,
            "initial_balance": self.initial_balance,
            "final_balance":   round(self.final_balance, 2),
            "gross_pnl":       round(self.gross_pnl, 2),
            "total_fees":      round(self.total_fees, 2),
            "total_pnl":       round(self.total_pnl, 2),
            "total_pnl_pct":   round(self.total_pnl_pct, 2),
            "total_trades":    len(self.trades),
            "closed_trades":   len(self.closed_trades),
            "wins":            len(self.wins),
            "losses":          len(self.losses),
            "win_rate":        round(self.win_rate, 2),
            "avg_rr":          round(self.avg_rr, 3),
            "profit_factor":   round(self.profit_factor, 3),
            "expectancy":      round(self.expectancy, 4),
            "max_drawdown":    round(self.max_drawdown, 2),
            "sharpe_ratio":    round(self.sharpe_ratio, 3),
            "equity_curve":    [round(e, 2) for e in self.equity_curve],
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved → {path}")


# ── Multiprocessing worker (must be module-level for pickling) ────
def _run_single(packed: tuple) -> "BacktestResult":
    """Worker used by ProcessPoolExecutor for --compare parallelism."""
    mode, kwargs, df = packed
    bt = Backtester(**kwargs, tp_mode=mode)
    return bt.run(df)


# ── Entry point ───────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FVG Backtest — real Binance candles, GPU-accelerated FVG scan"
    )
    parser.add_argument("--symbol",   default="BTC/USDT", help="Trading pair")
    parser.add_argument("--tf",       default="5m",        help="Timeframe (1m, 5m, 15m)")
    parser.add_argument("--days",     type=int,   default=90,   help="Days of history")
    parser.add_argument("--balance",  type=float, default=1000.0, help="Starting balance USDT")
    parser.add_argument("--risk",     type=float, default=0.5,  help="Risk per trade (%%)")
    parser.add_argument("--rr",       type=float, default=MIN_RR, help=f"Min R:R ratio (default={MIN_RR}, matches live bot)")
    parser.add_argument("--tp",       type=float, default=2.0,  help="TP multiplier (fixed mode)")
    parser.add_argument("--leverage",    type=float, default=1.0,  help="Futures leverage (1=spot)")
    parser.add_argument("--fee",         type=float, default=0.05, help="Taker fee %% (default 0.05)")
    parser.add_argument("--max-notional",type=float, default=20.0, help="Max position as %% of balance (default 20)")
    parser.add_argument("--min-sl",      type=float, default=0.1,  help="Min SL distance as %% of price (default 0.1)")
    parser.add_argument("--slippage",    type=float, default=0.0,  help="Entry slippage %% (e.g. 0.05 for 0.05%%)")
    parser.add_argument("--max-trades-day", type=int, default=0,   help="Daily trade cap (0=unlimited; live bot uses 10)")
    parser.add_argument("--score-min",   type=float, default=0.0,  help=f"FVG quality score gate (live={FVG_SCORE_MIN})")
    parser.add_argument("--htf-filter",  action="store_true",      help="Enable 1h EMA-50 trend filter (matches live)")
    parser.add_argument("--match-live",  action="store_true",      help="Shortcut: enable score + HTF + slippage + trade cap with live defaults")
    parser.add_argument("--tp-mode",  default="partial",
                        choices=["fixed", "structure", "trailing", "partial"],
                        help="TP mode (default: partial)")
    parser.add_argument("--compare",  action="store_true",
                        help="Run all 4 TP modes in parallel and compare")
    parser.add_argument("--fvg-size", type=float, default=0.1,
                        help="Min FVG size %% of price (default 0.1)")
    parser.add_argument("--no-save",  action="store_true", help="Skip CSV/JSON output")
    args = parser.parse_args()

    print(f"\n  Connecting to Binance (public endpoint)...")
    exchange = get_exchange(testnet=True)

    print(f"  Fetching {args.days} days of {args.symbol} {args.tf} candles...\n")
    df = fetch_historical_ohlcv(exchange, args.symbol, args.tf, days=args.days)

    if df.empty:
        print("ERROR: No candle data received. Check your internet connection.")
        sys.exit(1)

    # --match-live shortcut: turn on every gate the live bot uses, with live defaults
    if args.match_live:
        if args.score_min == 0.0:      args.score_min = FVG_SCORE_MIN
        if not args.htf_filter:        args.htf_filter = True
        if args.slippage == 0.0:       args.slippage = 0.05
        if args.max_trades_day == 0:   args.max_trades_day = 10

    base_kwargs = dict(
        symbol           = args.symbol,
        timeframe        = args.tf,
        days             = args.days,
        initial_balance  = args.balance,
        risk_pct         = args.risk / 100,
        min_rr           = args.rr,
        tp_multiplier    = args.tp,
        fvg_min_size     = args.fvg_size / 100,
        fvg_expiry       = FVG_EXPIRY_CANDLES,
        sl_buffer        = SL_BUFFER_PCT,
        leverage         = args.leverage,
        fee_rate         = args.fee / 100,
        max_notional_pct = args.max_notional / 100,
        min_sl_pct       = args.min_sl / 100,
        slippage_pct     = args.slippage / 100,
        max_trades_day   = args.max_trades_day,
        score_min        = args.score_min,
        use_htf_filter   = args.htf_filter,
    )

    if args.compare:
        # ── Run all 4 modes in PARALLEL processes ──
        modes   = ["fixed", "structure", "trailing", "partial"]
        results = []
        jobs    = [(mode, base_kwargs, df) for mode in modes]

        print(f"  Running all 4 TP modes in parallel ({len(modes)} workers)...\n")
        t0 = time.perf_counter()

        with ProcessPoolExecutor(max_workers=len(modes)) as pool:
            futures = {pool.submit(_run_single, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                mode = futures[fut]
                try:
                    results.append(fut.result())
                    print(f"    ✓ {mode.upper()} done")
                except Exception as exc:
                    print(f"    ✗ {mode.upper()} failed: {exc}")

        elapsed = time.perf_counter() - t0
        # Sort to consistent order for display
        results.sort(key=lambda r: modes.index(r.tp_mode))

        print(f"\n  All modes finished in {elapsed:.1f} s")
        print(f"\n{'═'*80}")
        print(f"  TP MODE COMPARISON — {args.symbol} {args.tf} ({args.days}d)"
              f"  |  {args.leverage:.0f}x lev  |  fee {args.fee:.3f}%")
        print(f"{'═'*80}")
        print(f"  {'Mode':<12} {'Final $':>10} {'Net%':>8} {'WinRate':>8} "
              f"{'AvgR:R':>8} {'PF':>6} {'MaxDD':>8} {'Fees $':>8} {'Sharpe':>7}")
        print(f"  {'─'*76}")
        for r in results:
            sign = "+" if r.total_pnl_pct >= 0 else ""
            print(f"  {r.tp_mode.upper():<12} "
                  f"${r.final_balance:>9,.0f} "
                  f"{sign}{r.total_pnl_pct:>6.1f}% "
                  f"{r.win_rate:>7.1f}% "
                  f"{r.avg_rr:>8.2f}x "
                  f"{r.profit_factor:>6.2f} "
                  f"{r.max_drawdown:>7.1f}% "
                  f"${r.total_fees:>7,.2f} "
                  f"{r.sharpe_ratio:>7.2f}")
        print(f"{'═'*80}\n")

        best = max(results, key=lambda r: r.final_balance)
        print(f"  Best: {best.tp_mode.upper()} — ${best.final_balance:,.2f} "
              f"(net {'+' if best.total_pnl_pct>=0 else ''}{best.total_pnl_pct:.1f}%)\n")

        if not args.no_save:
            best.to_csv(f"backtest_trades_{best.tp_mode}.csv")
            best.to_json(f"backtest_summary_{best.tp_mode}.json")

    else:
        bt     = Backtester(**base_kwargs, tp_mode=args.tp_mode)
        result = bt.run(df)
        result.print_report()

        if not args.no_save:
            result.to_csv(f"backtest_trades_{args.tp_mode}.csv")
            result.to_json(f"backtest_summary_{args.tp_mode}.json")
