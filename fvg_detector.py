# ─────────────────────────────────────────────
#  FVG Trading Bot — FVG Detection Engine
# ─────────────────────────────────────────────
"""
Fair Value Gap (FVG) logic:

  Bullish FVG:  candle[i-2].high < candle[i].low
                → gap between top of C1 and bottom of C3

  Bearish FVG:  candle[i-2].low > candle[i].high
                → gap between bottom of C1 and top of C3

Entry on RETEST: price pulls back into the gap zone.
SL: just outside the low/high of the middle candle (candle[i-1]).

Invalidation: price closes through the full gap zone
  (below gap_low for bullish, above gap_high for bearish).
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class FVG:
    symbol: str
    timeframe: str
    direction: str          # "bullish" | "bearish"
    gap_high: float
    gap_low: float
    sl_price: float         # stop loss = middle-candle low/high ± buffer
    formed_at: pd.Timestamp
    formed_index: int
    status: str = "waiting" # waiting | retested | expired | invalidated
    entry_price: Optional[float] = None
    candles_since_formed: int = 0
    # Tracks the last PRIMARY_TF candle we processed — used to increment
    # candles_since_formed exactly once per candle close, not per loop tick.
    last_candle_time: Optional[pd.Timestamp] = None
    score: float = 0.0      # quality score 0-1 (set after detection)

    @property
    def gap_size(self) -> float:
        return self.gap_high - self.gap_low

    @property
    def gap_mid(self) -> float:
        return (self.gap_high + self.gap_low) / 2

    def sl_distance(self, entry: float) -> float:
        if self.direction == "bullish":
            return entry - self.sl_price
        return self.sl_price - entry

    def tp_price(self, entry: float, rr: float) -> float:
        dist = self.sl_distance(entry)
        if self.direction == "bullish":
            return entry + dist * rr
        return entry - dist * rr

    def __repr__(self):
        return (f"FVG({self.symbol} {self.timeframe} {self.direction.upper()} "
                f"| gap=[{self.gap_low:.4f}-{self.gap_high:.4f}] "
                f"| SL={self.sl_price:.4f} | score={self.score:.2f} | {self.status})")


def detect_fvgs(df: pd.DataFrame, symbol: str, timeframe: str,
                min_size_pct: float = 0.001,
                sl_buffer_pct: float = 0.0002) -> list[FVG]:
    """
    Scan an OHLCV DataFrame for Fair Value Gaps.

    df columns: open, high, low, close, volume  (DatetimeIndex)
    Returns list of FVG objects sorted newest first.
    """
    fvgs = []
    if len(df) < 3:
        return fvgs

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    times  = df.index

    for i in range(2, len(df)):
        c1_high = highs[i - 2]
        c1_low  = lows[i - 2]
        c2_high = highs[i - 1]
        c2_low  = lows[i - 1]
        c3_high = highs[i]
        c3_low  = lows[i]
        price   = closes[i]

        # ── Bullish FVG ──────────────────────────────
        if c3_low > c1_high:
            gap_low  = c1_high
            gap_high = c3_low
            if (gap_high - gap_low) / price >= min_size_pct:
                sl = c2_low * (1 - sl_buffer_pct)
                fvgs.append(FVG(
                    symbol=symbol, timeframe=timeframe,
                    direction="bullish",
                    gap_high=gap_high, gap_low=gap_low,
                    sl_price=sl,
                    formed_at=times[i],
                    formed_index=i,
                ))

        # ── Bearish FVG ──────────────────────────────
        elif c3_high < c1_low:
            gap_high = c1_low
            gap_low  = c3_high
            if (gap_high - gap_low) / price >= min_size_pct:
                sl = c2_high * (1 + sl_buffer_pct)
                fvgs.append(FVG(
                    symbol=symbol, timeframe=timeframe,
                    direction="bearish",
                    gap_high=gap_high, gap_low=gap_low,
                    sl_price=sl,
                    formed_at=times[i],
                    formed_index=i,
                ))

    logger.debug(f"Detected {len(fvgs)} FVGs on {symbol} {timeframe}")
    return sorted(fvgs, key=lambda x: x.formed_at, reverse=True)


def check_retest(fvg: FVG, current_high: float, current_low: float,
                 current_close: float) -> bool:
    """
    Returns True if the current candle retests (enters) the FVG zone.
    Bullish: price wicks into gap from above and closes inside or above gap_low.
    Bearish: price wicks into gap from below and closes inside or below gap_high.
    """
    if fvg.status != "waiting":
        return False

    if fvg.direction == "bullish":
        return current_low <= fvg.gap_high and current_close >= fvg.gap_low
    elif fvg.direction == "bearish":
        return current_high >= fvg.gap_low and current_close <= fvg.gap_high
    return False


def is_fvg_invalidated(fvg: FVG, current_close: float) -> bool:
    """
    FVG is invalidated when price *closes* through the entire gap zone.

    Bullish FVG: a close below gap_low means the gap was fully filled from above
                 — the price imbalance no longer exists as a support level.
    Bearish FVG: a close above gap_high means the gap was fully filled from below.

    Wicks alone don't invalidate — price often pokes into / through a gap and
    snaps back. Treating wicks as invalidation kills FVGs that would have
    given multiple retest opportunities (the user can revisit a gap many times
    before it's structurally broken).

    The trade SL (sl_price) is a separate, tighter level used only after entry.
    """
    if fvg.direction == "bullish":
        return current_close < fvg.gap_low
    return current_close > fvg.gap_high


def fvg_quality_score(fvg: FVG, df: pd.DataFrame, atr: float) -> float:
    """
    Score an FVG's quality from 0.0 to 1.0.

    Three equally-weighted factors:
      1. Gap size relative to ATR — larger imbalance = stronger institutional signal
      2. Volume at formation candle vs average — high volume = smart-money presence
      3. Displacement strength — middle candle body-to-range ratio measures how
         impulsive the move was (clean body = strong, doji = weak)

    Returns 0.5 (neutral) if data is insufficient.
    """
    if atr <= 0 or df.empty:
        return 0.5

    current_price = float(df["close"].iloc[-1])
    if current_price <= 0:
        return 0.5

    avg_vol = float(df["volume"].mean()) if "volume" in df.columns and df["volume"].sum() > 0 else 0.0

    # 1. Gap-to-ATR ratio (capped at 3× to normalise)
    gap_atr = min(fvg.gap_size / atr, 3.0) / 3.0

    # 2. Formation-candle volume vs rolling average
    vol_ratio = 0.5  # default when unknown
    if avg_vol > 0 and fvg.formed_at in df.index:
        try:
            formation_vol = float(df.loc[fvg.formed_at, "volume"])
            vol_ratio = min(formation_vol / avg_vol, 3.0) / 3.0
        except Exception:
            pass

    # 3. Displacement strength — how impulsive was the middle candle?
    #    A full-body candle (open≈low, close≈high or vice versa) scores 1.0.
    #    A doji (tiny body, long wicks) scores near 0.
    #    Previously this was "proximity to current price" which was always 1.0
    #    at formation time, compressing all scores into 0.45–0.70.
    displacement = 0.5  # default
    mid_idx = fvg.formed_index - 1
    if 0 <= mid_idx < len(df):
        try:
            c2 = df.iloc[mid_idx]
            c2_body  = abs(float(c2["close"]) - float(c2["open"]))
            c2_range = float(c2["high"]) - float(c2["low"])
            if c2_range > 0:
                displacement = min(c2_body / c2_range, 1.0)
        except Exception:
            pass

    score = gap_atr * 0.4 + vol_ratio * 0.3 + displacement * 0.3
    return round(min(score, 1.0), 3)
