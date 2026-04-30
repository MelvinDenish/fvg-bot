# ─────────────────────────────────────────────
#  FVG Trading Bot — Order Manager
# ─────────────────────────────────────────────

import math
import ccxt
from exchange import get_ticker_price
import logging
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from fvg_detector import FVG
from config import (
    RISK_PCT, MIN_RR, TP_MULTIPLIERS, ATR_PERIOD,
    MIN_NOTIONAL_USDT, MAX_LEVERAGE,
    DEFAULT_LEVERAGE, MAX_POSITION_PCT, MARGIN_MODE,
)

logger = logging.getLogger(__name__)


# ── ATR (Wilder's RMA — matches TradingView standard) ────────────
def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """
    Wilder's Average True Range using RMA smoothing (alpha = 1/period).
    Identical to TradingView's ta.atr() and MT4's ATR indicator.
    Previously used ewm(span=period) which gave different values.
    """
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


# ── Structure-based TP ────────────────────────
def find_structure_tp(df: pd.DataFrame, direction: str,
                      entry: float, sl_dist: float,
                      min_rr: float = MIN_RR, lookback: int = 50) -> float:
    """
    Find the nearest swing high (longs) / swing low (shorts) that lies
    beyond the minimum R:R target. Falls back to 3R if no pivot found.
    Uses a 5-candle pivot window to filter noise.
    """
    window  = df.iloc[-lookback:] if len(df) >= lookback else df
    highs   = window["high"].values
    lows    = window["low"].values
    min_tp  = entry + sl_dist * min_rr if direction == "bullish" else entry - sl_dist * min_rr
    pivots  = []

    for j in range(2, len(highs) - 2):
        if direction == "bullish":
            if highs[j] == max(highs[j - 2: j + 3]) and highs[j] > min_tp:
                pivots.append(highs[j])
        else:
            if lows[j] == min(lows[j - 2: j + 3]) and lows[j] < min_tp:
                pivots.append(lows[j])

    if pivots:
        return min(pivots) if direction == "bullish" else max(pivots)
    # No real swing exists ≥ min_rr away. Return a value BELOW min_rr so the
    # caller's rr gate rejects the trade. Backtest proved that taking blind
    # fallback trades at synthetic min_rr targets DESTROYS the strategy
    # (PF 1.80 → 1.33, Sharpe 3.65 → 1.67) — those targets have no structural
    # reason to be reached. Better to skip than to take blind shots.
    return entry  # rr = 0 → guaranteed gate rejection


# ── Trade dataclass ───────────────────────────
@dataclass
class Trade:
    symbol:        str
    direction:     str          # "long" | "short"
    entry_price:   float
    sl_price:      float
    tp_prices:     list[float]
    qty:           float
    risk_amount:   float        # USDT at risk for this trade
    fvg:           Optional[FVG]
    opened_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at:     Optional[datetime] = None
    exit_price:    Optional[float] = None
    pnl_pct:       Optional[float] = None
    pnl_usdt:      Optional[float] = None
    status:        str = "open"   # open | closed | cancelled
    close_reason:  str = ""
    order_ids:     list[str] = field(default_factory=list)
    # Dynamic SL / partial-exit tracking
    current_sl:    float = 0.0
    be_moved:      bool  = False
    partial_done:  bool  = False
    qty_remaining: float = 0.0
    sl_order_id:   str   = ""    # exchange order ID of the active SL order
    tp_order_id:   str   = ""    # exchange order ID of the active TP limit order (structure mode)
    partial_pnl:   float = 0.0   # USDT banked at the 50% partial close

    def __post_init__(self):
        if self.current_sl == 0.0:
            self.current_sl = self.sl_price
        if self.qty_remaining == 0.0:
            self.qty_remaining = self.qty

    @property
    def rr_achieved(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        sl_dist = abs(self.entry_price - self.sl_price)
        if sl_dist == 0:
            return None
        sign = 1 if self.direction == "long" else -1
        return (self.exit_price - self.entry_price) * sign / sl_dist

    def __repr__(self):
        return (f"Trade({self.symbol} {self.direction.upper()} "
                f"| entry={self.entry_price:.4f} SL={self.sl_price:.4f} "
                f"| qty={self.qty:.6f} | {self.status})")


# ── Position sizing ───────────────────────────
def calculate_position_size(balance: float,
                             entry_price: float,
                             sl_price: float,
                             risk_pct: float = RISK_PCT,
                             leverage: int = DEFAULT_LEVERAGE,
                             max_position_pct: float = MAX_POSITION_PCT,
                             ) -> tuple[float, float, int]:
    """
    Returns (quantity, risk_amount_usdt, leverage).

    Mirrors the backtest's sizing exactly:
      1. Risk-based qty:         qty = (balance × risk_pct) / |entry - sl|
      2. Cap NOTIONAL at max_position_pct × balance. This is what the
         backtest does — caps gross position size, not margin.
      3. After cap, recompute risk_usdt (will be < risk_pct × balance when
         the cap binds — typical for tight SLs).
      4. Pick leverage so margin requirement (= notional / leverage) fits
         comfortably. Default DEFAULT_LEVERAGE keeps margin small for
         capital efficiency; bumps higher only if needed.
      5. Floor at Binance MIN_NOTIONAL_USDT ($50).

    Concrete example ($5000 balance, 5x default lev, 20% notional cap):
      ETH @ $2850, SL $8.55. risk_pct=0.5% → risk_usdt=$25.
      qty (risk-based) = 25/8.55 = 2.92 ETH, notional = $8,326.
      → notional cap = 20% × $5000 = $1,000. qty shrunk to 0.351 ETH.
      → actual risk = 0.351 × $8.55 = $3 (cap binds, so risk < 0.5%).
      → margin at 5x = $1000/5 = $200. Three open positions = $600 (12%).

    Risk profile (matches backtest):
      Per-trade risk is bounded by min(risk_pct × balance, qty × sl_dist
      after notional cap). Worst case 3 SL hits = 3 × actual_risk ≈ $9
      on $5000 (~0.18%). Very conservative.
    """
    sl_dist = abs(entry_price - sl_price)
    if sl_dist == 0:
        return 0.0, 0.0, 1

    risk_usdt    = balance * risk_pct
    qty          = risk_usdt / sl_dist
    notional     = qty * entry_price
    max_notional = balance * max_position_pct

    # Cap notional at max_position_pct of balance (matches backtest exactly)
    if notional > max_notional:
        qty       = max_notional / entry_price
        notional  = max_notional
        risk_usdt = qty * sl_dist   # actual risk after cap

    # Floor at Binance min notional ($50)
    if notional < MIN_NOTIONAL_USDT:
        qty       = MIN_NOTIONAL_USDT / entry_price
        notional  = MIN_NOTIONAL_USDT
        risk_usdt = qty * sl_dist

    # Pick leverage so margin is reasonable. With notional already capped at
    # max_position_pct of balance, leverage=1 would use that full amount as
    # margin. Default DEFAULT_LEVERAGE keeps margin = notional / lev, freeing
    # capital for the other 2 concurrent positions + buffer.
    margin = notional / leverage
    if margin > balance:   # safety: never exceed wallet (unlikely with notional cap)
        leverage = min(MAX_LEVERAGE, max(1, math.ceil(notional / balance)))

    return round(qty, 6), round(risk_usdt, 2), int(leverage)


# ── Open trade ────────────────────────────────
def open_trade(exchange: ccxt.binance,
               fvg: FVG,
               balance: float,
               dry_run: bool = True,
               tp_override: Optional[float] = None) -> Optional[Trade]:
    """
    Open a new trade from an FVG signal.
    dry_run=True logs the intended order without touching the exchange.

    tp_override: when set, places ONE TP at this price (structure mode).
                 when None, falls back to TP_MULTIPLIERS scale-out (partial mode).
    """
    direction = "long" if fvg.direction == "bullish" else "short"
    entry     = fvg.gap_mid
    sl        = fvg.sl_price
    tp_prices = (
        [tp_override] if tp_override is not None
        else [fvg.tp_price(entry, mult) for mult in TP_MULTIPLIERS]
    )

    qty, risk, leverage = calculate_position_size(balance, entry, sl)
    if qty == 0:
        logger.warning(f"Position size 0 for {fvg.symbol} — skipping")
        return None

    trade = Trade(
        symbol      = fvg.symbol,
        direction   = direction,
        entry_price = entry,
        sl_price    = sl,
        tp_prices   = tp_prices,
        qty         = qty,
        risk_amount = risk,
        fvg         = fvg,
    )

    if dry_run:
        logger.info(
            f"[DRY RUN] {direction.upper()} {fvg.symbol} | "
            f"entry={entry:.4f} SL={sl:.4f} "
            f"TP={[f'{t:.4f}' for t in tp_prices]} | "
            f"qty={qty:.6f} | risk=${risk:.2f} | lev={leverage}x"
        )
        return trade

    # ── Live order placement ──────────────────
    #
    # Order policy (matches backtest exactly):
    #   1. Market ENTRY   — fired on exchange
    #   2. Stop-market SL — fired on exchange (safety net if bot crashes)
    #   3. NO exchange-side TP — bot owns TP via manage_open_trades + close_trade
    #
    # Why no exchange TP:
    #   Previous version placed take_profit_market orders that often hit
    #   Binance's -2021 "would immediately trigger" when the structure TP was
    #   close to current price (typical on 5m). The exchange TP also raced
    #   the bot's own close logic, causing duplicate exits and bad P&L.
    #   The backtest never had exchange-side TP either — it just closes when
    #   price reaches the target. Live now matches.
    side    = "buy"  if direction == "long"  else "sell"
    sl_side = "sell" if direction == "long"  else "buy"

    # ── Step 0: Cancel ALL existing orders for this symbol ──
    # MUST happen BEFORE set_margin_mode — Binance error -4067 rejects margin
    # mode changes while open orders exist on the symbol.
    # Also clears orphaned SL/TP from previous trades that Binance Demo failed
    # to cancel via API. Without this, an old stop_market SL could trigger
    # against the NEW position and close it prematurely.
    try:
        exchange.cancel_all_orders(fvg.symbol)
        logger.info(f"Pre-trade cleanup: cancelled all orders on {fvg.symbol}")
    except Exception as e:
        logger.debug(f"Pre-trade cancel_all_orders({fvg.symbol}): {e}")

    # Set margin mode (isolated/cross) and leverage before opening the position.
    # Isolated mode limits loss to the position's margin only — prevents a single
    # bad trade from liquidating the entire account (which cross mode allows).
    try:
        exchange.set_margin_mode(MARGIN_MODE, fvg.symbol)
        logger.info(f"Margin mode set to {MARGIN_MODE.upper()} on {fvg.symbol}")
    except Exception as e:
        # Binance returns -4046 if margin mode is already set — safe to ignore.
        err = str(e)
        if "-4046" in err or "No need to change" in err:
            pass
        else:
            logger.warning(f"set_margin_mode({MARGIN_MODE}, {fvg.symbol}) failed: {e}")

    try:
        exchange.set_leverage(leverage, fvg.symbol)
        logger.info(f"Leverage set to {leverage}x on {fvg.symbol}")
    except Exception as e:
        logger.warning(f"set_leverage({leverage}, {fvg.symbol}) failed: {e}")

    # ── Pre-trade price check ──
    # A market order fills at current bid/ask, NOT at the historical gap_mid.
    # If the market has drifted more than 0.5% from the FVG entry zone, don't
    # even send the order — avoids the open→emergency-close churn that was
    # causing infinite rejection loops when price moved away from the FVG.
    current_price = get_ticker_price(exchange, fvg.symbol)
    if current_price > 0:
        drift_pct = abs(current_price - entry) / entry
        if drift_pct > 0.005:
            logger.warning(
                f"SKIPPED {fvg.symbol}: market price {current_price:.4f} has "
                f"drifted {drift_pct*100:.2f}% from FVG entry {entry:.4f}. "
                f"Not placing order."
            )
            return "SKIP"
    else:
        logger.warning(f"Could not fetch ticker for {fvg.symbol} — skipping pre-trade check")

    # ── Step 1: Market entry ──
    pre_order_price = current_price if current_price > 0 else entry
    try:
        entry_order = exchange.create_order(
            symbol=fvg.symbol, type="market", side=side, amount=qty
        )
    except ccxt.InsufficientFunds:
        logger.error(f"Insufficient funds to open {fvg.symbol} trade")
        return None
    except Exception as exc:
        logger.error(f"Entry order failed for {fvg.symbol}: {exc}")
        return None

    trade.order_ids.append(entry_order["id"])
    # Use the actual filled average price (may differ from gap_mid due to slippage)
    trade.entry_price = float(entry_order.get("average") or entry_order.get("price") or entry)

    # ── Slippage sanity check ──
    # If the actual fill puts the SL on the wrong side of entry (long with SL
    # ≥ entry, or short with SL ≤ entry), the trade is structurally broken —
    # the SL would behave like a take-profit. Reject and close immediately.
    bad_sl = (
        (direction == "long"  and sl >= trade.entry_price) or
        (direction == "short" and sl <= trade.entry_price)
    )
    if bad_sl:
        logger.error(
            f"REJECTED {fvg.symbol}: filled @ {trade.entry_price:.4f} but SL is "
            f"{sl:.4f} ({'above' if direction == 'long' else 'below'} entry). "
            f"Likely thin liquidity slipped through the FVG. Closing immediately."
        )
        _emergency_close(exchange, fvg.symbol, sl_side, qty)
        return None

    # Also reject if fill is too far outside the pre-order ticker price (>0.5%).
    # Compares against the FRESH ticker (not stale gap_mid) to avoid false
    # rejections when the market moved since the FVG formed.
    slip_pct = abs(trade.entry_price - pre_order_price) / pre_order_price if pre_order_price else 0
    if slip_pct > 0.005:
        logger.error(
            f"REJECTED {fvg.symbol}: execution slippage {slip_pct*100:.2f}% "
            f"(pre-order price {pre_order_price:.4f}, filled {trade.entry_price:.4f}). "
            f"Closing immediately."
        )
        _emergency_close(exchange, fvg.symbol, sl_side, qty)
        return None

    # ── Step 2: Stop-loss ──
    # If this fails, the position is naked. Close it immediately to avoid
    # an unprotected open trade.
    try:
        sl_order = exchange.create_order(
            symbol=fvg.symbol, type="stop_market", side=sl_side,
            amount=qty, params={"stopPrice": sl, "reduceOnly": True}
        )
        trade.order_ids.append(sl_order["id"])
        trade.sl_order_id = sl_order["id"]
    except Exception as exc:
        logger.error(
            f"SL placement failed for {fvg.symbol}: {exc}. "
            f"Closing entry to avoid orphaned position."
        )
        _emergency_close(exchange, fvg.symbol, sl_side, qty)
        return None

    # ── Step 3: Take-profit (LIMIT) ──
    # Uses a regular LIMIT order (not conditional TAKE_PROFIT) with reduceOnly.
    # Why: Binance Demo can't cancel conditional orders via API, causing orphans.
    # Regular limit orders CAN be cancelled, so when SL fires the bot can
    # always clean up the TP. When TP fills first, the SL orphan is harmless
    # (reduceOnly on zero position = can never trigger).
    # A limit sell at TP (longs) or limit buy at TP (shorts) sits on the book
    # and fills at exactly the TP price or better — same behaviour as TAKE_PROFIT.
    if len(tp_prices) == 1:
        tp_price = tp_prices[0]
        try:
            tp_order = exchange.create_order(
                symbol=fvg.symbol, type="limit", side=sl_side,
                amount=qty, price=tp_price,
                params={"reduceOnly": True}
            )
            trade.order_ids.append(tp_order["id"])
            trade.tp_order_id = tp_order["id"]
        except Exception as exc:
            err = str(exc)
            if "-2021" in err or "immediately trigger" in err.lower() \
               or "-4131" in err or "would immediately" in err.lower():
                # We're already AT or PAST the TP price. Take the win at market.
                logger.info(
                    f"TP {tp_price:.4f} already in range for {fvg.symbol} — "
                    f"closing position at market immediately."
                )
                _cancel_silent(exchange, trade.sl_order_id, fvg.symbol)
                actual_exit = _emergency_close(exchange, fvg.symbol, sl_side, qty)
                if actual_exit:
                    sign = 1 if direction == "long" else -1
                    trade.exit_price   = actual_exit
                    trade.status       = "closed"
                    trade.close_reason = "tp_immediate"
                    trade.closed_at    = datetime.now(timezone.utc)
                    trade.pnl_usdt     = round(
                        (actual_exit - trade.entry_price) * qty * sign, 4)
                    trade.pnl_pct      = round(
                        trade.pnl_usdt / risk * 100 if risk else 0, 2)
                    logger.info(
                        f"INSTANT FILL {trade.symbol} @ {actual_exit:.4f} "
                        f"| P&L=${trade.pnl_usdt:.2f}"
                    )
                return trade
            else:
                logger.warning(
                    f"TP placement failed for {fvg.symbol}: {exc}. "
                    f"Keeping position with SL only — bot will close at TP via fallback."
                )

    logger.info(f"OPENED {trade}")
    return trade


def _emergency_close(exchange, symbol, side, qty) -> Optional[float]:
    """Market close a position; return actual fill price or None on failure."""
    try:
        order = exchange.create_order(
            symbol=symbol, type="market", side=side,
            amount=qty, params={"reduceOnly": True}
        )
        fill = order.get("average") or order.get("price")
        return float(fill) if fill else None
    except Exception as exc:
        logger.critical(
            f"!! CRITICAL: Could not market-close {symbol}: {exc}. "
            f"MANUAL INTERVENTION REQUIRED."
        )
        return None


def _cancel_silent(exchange, order_id: str, symbol: str) -> None:
    """Cancel an order, trying both regular and algo (conditional) endpoints.

    On Binance Demo, conditional orders (stop_market, TAKE_PROFIT) are stored
    as 'algo orders' and CANNOT be cancelled via the regular cancel_order API
    (returns -2011 'Unknown order sent'). They require the algo-specific
    endpoint: fapiPrivateDeleteAlgoOrder(algoId=...).
    """
    if not order_id:
        return
    # Try 1: regular order cancel (works for limit orders)
    try:
        exchange.cancel_order(order_id, symbol)
        return
    except Exception:
        pass
    # Try 2: algo order cancel (works for conditional orders on demo)
    try:
        exchange.fapiPrivateDeleteAlgoOrder({"algoId": str(order_id)})
        logger.info(f"Cancelled algo order {order_id} on {symbol}")
    except Exception:
        pass


# ── Close trade ───────────────────────────────
def close_trade(exchange: ccxt.binance,
                trade: Trade,
                exit_price: float,
                reason: str = "manual",
                dry_run: bool = True) -> Trade:
    """
    Mark a trade as closed and compute final realised P&L.

    P&L accounts for:
      - The remaining position size (qty_remaining after any partial close)
      - The already-banked partial_pnl from the 50% partial exit

    Previously the code used trade.qty (full size) regardless of whether
    a partial close had already been executed, inflating the reported P&L.
    """
    trade.close_reason = reason
    trade.closed_at    = datetime.now(timezone.utc)
    trade.status       = "closed"

    # Use remaining qty only — partial fill already captured in partial_pnl
    remaining = trade.qty_remaining if trade.partial_done else trade.qty
    sign      = 1 if trade.direction == "long" else -1

    # Place the close order FIRST, then read the actual fill price.
    # Previously P&L was computed against the *target* exit_price (e.g. the
    # planned TP), but the real market fill often slipped — leading to
    # +$1.48 logged for a trade that actually realised -$1.72 on the exchange.
    actual_exit = exit_price   # fallback if we can't read the fill
    if not dry_run:
        # ── Cancel BOTH counterpart orders (SL + TP) ──
        # Must happen BEFORE the market close attempt AND regardless of
        # whether it succeeds.  When the exchange itself fills one side
        # (e.g. TP limit triggers), the position is already flat — the
        # market close will fail with "reduce only" error, but we still
        # need the other side cancelled to avoid orphaned orders.
        _cancel_silent(exchange, trade.sl_order_id, trade.symbol)
        _cancel_silent(exchange, trade.tp_order_id, trade.symbol)

        try:
            close_side = "sell" if trade.direction == "long" else "buy"
            close_order = exchange.create_order(
                symbol=trade.symbol, type="market", side=close_side,
                amount=remaining, params={"reduceOnly": True}
            )
            # Read actual filled price (more reliable than the planned exit)
            fill = close_order.get("average") or close_order.get("price")
            if fill:
                actual_exit = float(fill)
        except Exception as exc:
            # Position may already be closed by exchange-side TP/SL fill.
            # Log but don't abort — P&L is still computed against planned exit.
            logger.warning(
                f"Market close for {trade.symbol} failed (position may "
                f"already be flat from exchange fill): {exc}"
            )

    trade.exit_price = actual_exit
    leg_pnl          = (actual_exit - trade.entry_price) * remaining * sign

    total_pnl       = leg_pnl + trade.partial_pnl
    trade.pnl_usdt  = round(total_pnl, 4)
    trade.pnl_pct   = round(
        total_pnl / trade.risk_amount * 100 if trade.risk_amount else 0, 2
    )

    rr = trade.rr_achieved
    slip_note = ""
    if not dry_run and abs(actual_exit - exit_price) > 1e-9:
        slip_note = f" (planned {exit_price:.4f}, slipped {actual_exit - exit_price:+.4f})"
    logger.info(
        f"CLOSED {trade.symbol} {trade.direction.upper()} | "
        f"reason={reason} | exit={actual_exit:.4f}{slip_note} | "
        f"P&L=${trade.pnl_usdt:.2f} ({trade.pnl_pct:.1f}% of risk)"
        + (f" | R:R={rr:.2f}x" if rr is not None else "")
    )
    return trade


# ── Partial close ─────────────────────────────
def execute_partial_close(exchange: ccxt.binance,
                           trade: Trade,
                           exit_price: float,
                           dry_run: bool = True) -> None:
    """
    Close 50% of the position at exit_price and amend the SL order to
    cover only the remaining half.  Stores the realised P&L in trade.partial_pnl
    so close_trade() can add it to the final settlement.
    """
    half_qty = trade.qty / 2
    sign     = 1 if trade.direction == "long" else -1
    pnl_half = (exit_price - trade.entry_price) * half_qty * sign

    trade.partial_pnl   = pnl_half
    trade.partial_done  = True
    trade.qty_remaining = half_qty

    if dry_run:
        logger.info(
            f"[DRY RUN PARTIAL] {trade.symbol} 50% at {exit_price:.4f} "
            f"| P&L=${pnl_half:.2f}"
        )
        return

    # ── Live: close half, then reduce the SL order to half qty ──
    close_side = "sell" if trade.direction == "long" else "buy"
    try:
        exchange.create_order(
            symbol=trade.symbol, type="market", side=close_side,
            amount=half_qty, params={"reduceOnly": True}
        )
    except Exception as exc:
        logger.error(f"Partial close order failed for {trade.symbol}: {exc}")
        # Revert state so the bot retries next cycle
        trade.partial_pnl   = 0.0
        trade.partial_done  = False
        trade.qty_remaining = trade.qty
        return

    # Amend SL: cancel old full-qty SL and replace with half-qty SL
    if trade.sl_order_id:
        try:
            exchange.cancel_order(trade.sl_order_id, trade.symbol)
        except Exception:
            pass

        sl_side = "sell" if trade.direction == "long" else "buy"
        try:
            new_sl = exchange.create_order(
                symbol=trade.symbol, type="stop_market", side=sl_side,
                amount=half_qty,
                params={"stopPrice": trade.current_sl, "reduceOnly": True}
            )
            trade.sl_order_id = new_sl["id"]
            trade.order_ids.append(new_sl["id"])
        except Exception as exc:
            logger.error(f"SL amendment failed for {trade.symbol}: {exc}")

    # Amend TP: cancel old full-qty TP and replace with half-qty TP
    if trade.tp_order_id:
        try:
            exchange.cancel_order(trade.tp_order_id, trade.symbol)
        except Exception:
            pass

        tp_side = "sell" if trade.direction == "long" else "buy"
        tp_price = trade.tp_prices[0] if trade.tp_prices else None
        if tp_price is not None:
            try:
                new_tp = exchange.create_order(
                    symbol=trade.symbol, type="limit", side=tp_side,
                    amount=half_qty, price=tp_price,
                    params={"reduceOnly": True}
                )
                trade.tp_order_id = new_tp["id"]
                trade.order_ids.append(new_tp["id"])
            except Exception as exc:
                logger.warning(
                    f"TP amendment failed for {trade.symbol}: {exc}. "
                    f"Bot will close at TP via fallback."
                )
                trade.tp_order_id = ""

    logger.info(
        f"[PARTIAL] {trade.symbol} 50% closed at {exit_price:.4f} "
        f"| P&L=${pnl_half:.2f} | Remaining qty={half_qty:.6f}"
    )
