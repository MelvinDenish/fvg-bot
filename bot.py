#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  FVG Trading Bot — REST Polling Mode
#  Run: python bot.py
# ─────────────────────────────────────────────

import time
import logging
import signal
import sys
from datetime import datetime, timezone, date

import pandas as pd

from config import (
    SYMBOLS, PRIMARY_TF, ENTRY_TF, HTF_TF,
    POLL_INTERVAL_SEC, FVG_MIN_SIZE_PCT, FVG_LOOKBACK,
    FVG_EXPIRY_CANDLES, FVG_MAX_ACTIVE, FVG_SCORE_MIN,
    SL_BUFFER_PCT, MIN_RR, TP_MODE, USE_HTF_FILTER,
    RISK_PCT, MAX_TRADES_DAY, MAX_OPEN_TRADES, MAX_CORRELATED,
    HTF_EMA_PERIOD, ATR_PERIOD, TRAIL_ATR_MULT,
    DEFAULT_LEVERAGE, MAX_POSITION_PCT,
    TESTNET, BINANCE_DEMO, DRY_RUN, STATE_FILE, LOG_FILE, LOG_LEVEL,
)
from exchange import get_exchange, fetch_ohlcv, get_account_balance, fetch_positions_safe, get_ticker_price
from fvg_detector import (
    detect_fvgs, check_retest, is_fvg_invalidated, fvg_quality_score,
)
from order_manager import (
    open_trade, close_trade, execute_partial_close,
    calc_atr, find_structure_tp, Trade,
)
from state import save_state, load_state

# ── Logging setup ─────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 → utf-8 for ≥ × ─ etc.
except Exception:
    pass

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Graceful shutdown ─────────────────────────
running = True

def shutdown(sig, frame):
    global running
    logger.info("Shutdown signal received — stopping bot...")
    running = False

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── Position size helper ──────────────────────
def _read_position_size(positions: list, symbol: str) -> float:
    """
    Safely extract position size from Binance fetch_positions response.

    CRITICAL: Must NOT use `p.get('contracts') or p.get('contractSize') or 0`
    because when contracts=0 (position closed), Python treats 0 as falsy and
    falls through to contractSize=1, making the bot think the position is
    still open. This bug prevented _poll_exit from detecting exits for hours.
    """
    for p in positions:
        psym = p.get("symbol", "")
        if psym == symbol or psym.replace(":USDT", "") == symbol:
            contracts = p.get("contracts")
            if contracts is not None:
                return abs(float(contracts))
            # Some exchange responses use 'info.positionAmt' instead
            info = p.get("info", {})
            pos_amt = info.get("positionAmt")
            if pos_amt is not None:
                return abs(float(pos_amt))
            return 0.0
    return 0.0


def _cancel_all_for_symbol(exchange, symbol: str, sl_id=None, tp_id=None) -> None:
    """
    Cancel ALL orders for a symbol — both regular and algo (conditional).

    On Binance Demo, conditional orders (stop_market, TAKE_PROFIT) are stored
    as 'algo orders' and can only be cancelled via fapiPrivateDeleteAlgoOrder.
    Regular cancel_order / cancel_all_orders silently fails for these.
    """
    # 1. Cancel regular orders (limit TP, etc.)
    try:
        exchange.cancel_all_orders(symbol)
    except Exception:
        pass

    # 2. Cancel individual orders by ID (try both regular + algo)
    for oid in (sl_id, tp_id):
        if not oid:
            continue
        try:
            exchange.cancel_order(str(oid), symbol)
        except Exception:
            pass
        try:
            exchange.fapiPrivateDeleteAlgoOrder({"algoId": str(oid)})
        except Exception:
            pass

    # 3. Nuclear: list ALL open algo orders and cancel any for this symbol
    bsym = symbol.replace("/", "")
    try:
        algo_orders = exchange.fapiPrivateGetOpenAlgoOrders()
        for ao in algo_orders:
            if ao.get("symbol") == bsym:
                try:
                    exchange.fapiPrivateDeleteAlgoOrder({"algoId": ao["algoId"]})
                    logger.info(f"Cancelled algo order {ao['algoId']} ({ao.get('orderType')}) on {symbol}")
                except Exception:
                    pass
    except Exception:
        pass


class FVGBot:
    def __init__(self):
        self.exchange = get_exchange(TESTNET)

        # Restored from state file on startup; empty defaults otherwise
        state = load_state(STATE_FILE)
        self.open_trades:     list[Trade]       = state["open_trades"]
        self.closed_trades:   list[Trade]       = []
        self.active_fvgs:     dict[str, list]   = state["active_fvgs"]
        self.trades_today:    dict[str, int]    = state["trades_today"]
        self.last_trade_date: date              = state["last_trade_date"]

        # Recover from stale state: any restored "open" trade whose position
        # has already been closed on the exchange must be marked closed here,
        # otherwise _poll_exit will fabricate a fake P&L from the planned
        # SL/TP price.
        self._reconcile_restored_trades()

        # Cancel any orphan conditional orders left from a previous run that
        # don't match a tracked open trade. Without this, repeated restarts
        # accumulate untriggered SL/TP orders on the exchange.
        self._cleanup_orphan_orders()

        # Track the last PRIMARY_TF candle timestamp per symbol so we
        # increment candles_since_formed exactly once per candle close,
        # not once per 10-second polling tick (the original bug).
        self._last_primary_candle: dict[str, pd.Timestamp] = {}

        # Cache candle fetches within a single scan cycle to avoid
        # redundant API calls when the same symbol appears in multiple
        # steps (e.g. both check_entries and manage_open_trades).
        self._cycle_cache: dict[tuple, pd.DataFrame] = {}

        # HTF bias cache — refreshed every hour per symbol
        self._htf_cache: dict[str, tuple] = {}  # symbol → (bias, fetch_time)

        # Balance cache — refreshed once per scan cycle (not per FVG check)
        self._cached_balance: float = 0.0

        # Dedup: FVG formation timestamps that already finished their lifecycle
        # (invalidated, expired, or retested). Prevents re-detecting the same
        # historical 3-candle pattern every scan cycle.
        from collections import defaultdict
        self.seen_fvgs: dict[str, set] = defaultdict(set)
        # Seed with anything already restored from state so restarts stay clean
        for sym, fvgs in self.active_fvgs.items():
            for f in fvgs:
                if f.status != "waiting":
                    self.seen_fvgs[sym].add(f.formed_at)

        logger.info("=" * 60)
        logger.info("  FVG Trading Bot starting up")
        mode = "DEMO TRADING" if BINANCE_DEMO else ("TESTNET" if TESTNET else "⚠ LIVE")
        logger.info(f"  Mode:      {mode}  (dry_run={DRY_RUN})")
        logger.info(f"  TP mode:   {TP_MODE.upper()}")
        logger.info(f"  Symbols:   {', '.join(SYMBOLS)}")
        logger.info(f"  Timeframe: {PRIMARY_TF} (entry: {ENTRY_TF}, HTF: {HTF_TF})")
        logger.info(f"  Risk:      {RISK_PCT*100:.2f}%/trade  |  "
                    f"Notional cap {MAX_POSITION_PCT*100:.0f}%  |  "
                    f"Default lev {DEFAULT_LEVERAGE}x")
        score_str = f"Score ≥ {FVG_SCORE_MIN}" if FVG_SCORE_MIN > 0 else "Score OFF"
        htf_str   = "HTF ON" if USE_HTF_FILTER else "HTF OFF"
        logger.info(f"  Filters:   Gap ≥ {FVG_MIN_SIZE_PCT*100:.2f}%  |  "
                    f"R:R ≥ {MIN_RR}  |  {score_str}  |  {htf_str}")
        logger.info("=" * 60)

    # ── Candle cache (within one cycle) ──────
    def _fetch(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        """Fetch candles, hitting the in-cycle cache first."""
        key = (symbol, tf)
        if key not in self._cycle_cache:
            self._cycle_cache[key] = fetch_ohlcv(self.exchange, symbol, tf, limit=limit)
        return self._cycle_cache[key]

    # ── Daily reset ────────────────────────────
    def reset_daily_counters(self):
        today = date.today()
        if today != self.last_trade_date:
            self.trades_today.clear()
            self.last_trade_date = today
            logger.info("Daily trade counters reset")

    # ── Trade eligibility ──────────────────────
    def can_trade(self, symbol: str) -> bool:
        if len(self.open_trades) >= MAX_OPEN_TRADES:
            return False
        if self.trades_today.get(symbol, 0) >= MAX_TRADES_DAY:
            return False
        if any(t.symbol == symbol for t in self.open_trades):
            return False
        # Correlation guard: cap simultaneous altcoin positions.
        # BTC/USDT is excluded because it tends to lead rather than follow.
        altcoins = [s for s in SYMBOLS if "BTC" not in s]
        if symbol in altcoins:
            open_alts = sum(1 for t in self.open_trades if t.symbol in altcoins)
            if open_alts >= MAX_CORRELATED:
                logger.debug(f"Correlation cap reached — skipping {symbol}")
                return False
        return True

    # ── HTF trend bias ─────────────────────────
    def get_htf_bias(self, symbol: str) -> str:
        """
        Returns 'bullish', 'bearish', or 'neutral' based on whether the
        closing price sits above or below the EMA-50 on the 1-hour chart.
        Result is cached for one hour to avoid extra API calls.
        """
        cached = self._htf_cache.get(symbol)
        if cached and time.time() - cached[1] < 3600:
            return cached[0]

        df = fetch_ohlcv(self.exchange, symbol, HTF_TF, limit=HTF_EMA_PERIOD + 20)
        if df.empty or len(df) < HTF_EMA_PERIOD:
            return "neutral"

        ema     = df["close"].ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
        price   = float(df["close"].iloc[-1])
        ema_val = float(ema.iloc[-1])

        if price > ema_val * 1.001:
            bias = "bullish"
        elif price < ema_val * 0.999:
            bias = "bearish"
        else:
            bias = "neutral"

        self._htf_cache[symbol] = (bias, time.time())
        logger.debug(f"HTF bias {symbol}: {bias} (price={price:.2f} EMA={ema_val:.2f})")
        return bias

    # ── FVG scan ───────────────────────────────
    def scan_fvgs(self, symbol: str):
        """
        Detect new FVGs on the primary timeframe, score them, and maintain
        a capped list of the best active FVGs per symbol.

        Candle-accurate expiry fix: candles_since_formed is incremented
        once per PRIMARY_TF candle close, not once per 10-second loop tick.

        Performance: FVG detection (detect_fvgs + scoring) only runs when
        a new candle closes.  Between closes we only prune/expire existing
        FVGs — saves ~95% of detection calls on 5m with 10s polling.
        """
        df = self._fetch(symbol, PRIMARY_TF, FVG_LOOKBACK)
        if df.empty or len(df) < 3:
            return

        latest_candle_time = df.index[-1]
        prev_candle_time   = self._last_primary_candle.get(symbol)
        new_candle_closed  = (
            prev_candle_time is None or latest_candle_time > prev_candle_time
        )

        if new_candle_closed:
            self._last_primary_candle[symbol] = latest_candle_time
            for fvg in self.active_fvgs[symbol]:
                fvg.candles_since_formed += 1
                fvg.last_candle_time      = latest_candle_time

        # ── Prune expired / invalidated FVGs (every cycle) ──
        seen = self.seen_fvgs[symbol]
        for f in self.active_fvgs[symbol]:
            if f.status != "waiting" or f.candles_since_formed >= FVG_EXPIRY_CANDLES:
                seen.add(f.formed_at)

        waiting = sorted(
            [
                f for f in self.active_fvgs[symbol]
                if f.status == "waiting"
                and f.candles_since_formed < FVG_EXPIRY_CANDLES
            ],
            key=lambda x: x.formed_at, reverse=True
        )
        for dropped in waiting[FVG_MAX_ACTIVE:]:
            seen.add(dropped.formed_at)
        self.active_fvgs[symbol] = waiting[:FVG_MAX_ACTIVE]

        # ── Detect new FVGs (only on candle close) ──
        if not new_candle_closed:
            return

        new_fvgs = detect_fvgs(df, symbol, PRIMARY_TF,
                                min_size_pct=FVG_MIN_SIZE_PCT,
                                sl_buffer_pct=SL_BUFFER_PCT)

        atr            = calc_atr(df, ATR_PERIOD)
        existing_times = {f.formed_at for f in self.active_fvgs[symbol]}
        current_close  = float(df["close"].iloc[-1])

        for fvg in new_fvgs[:10]:
            if fvg.formed_at in existing_times or fvg.formed_at in seen:
                continue
            # Pre-filter: skip FVGs already invalidated by current price.
            # Without this, FVGs from deep in the lookback window get detected
            # and invalidated in the same cycle — wasted log noise.
            if is_fvg_invalidated(fvg, current_close):
                seen.add(fvg.formed_at)
                continue
            fvg.score            = fvg_quality_score(fvg, df, atr)
            fvg.last_candle_time = latest_candle_time
            self.active_fvgs[symbol].append(fvg)
            logger.info(f"New FVG: {fvg}")

        # Re-cap after adding new FVGs
        waiting2 = sorted(
            [
                f for f in self.active_fvgs[symbol]
                if f.status == "waiting"
                and f.candles_since_formed < FVG_EXPIRY_CANDLES
            ],
            key=lambda x: x.formed_at, reverse=True
        )
        for dropped in waiting2[FVG_MAX_ACTIVE:]:
            seen.add(dropped.formed_at)
        self.active_fvgs[symbol] = waiting2[:FVG_MAX_ACTIVE]

    # ── Entry check ────────────────────────────
    def check_entries(self, symbol: str):
        """
        For each active FVG, check whether the latest ENTRY_TF candle
        represents a valid retest.  Applies three gates before entering:
          1. HTF trend alignment
          2. FVG quality score
          3. Minimum R:R
        """
        if not self.can_trade(symbol):
            return

        df = self._fetch(symbol, ENTRY_TF, 10)
        if df.empty:
            return

        latest  = df.iloc[-1]
        c_high  = float(latest["high"])
        c_low   = float(latest["low"])
        c_close = float(latest["close"])

        # HTF bias only computed when the filter is enabled (saves an API call
        # per cycle when disabled). Backtest showed the filter cost more profit
        # than it saved in drawdown, so it's off by default.
        htf_bias = self.get_htf_bias(symbol) if USE_HTF_FILTER else "neutral"

        for fvg in list(self.active_fvgs.get(symbol, [])):
            # Invalidation check uses CLOSE, not wick — a wick that pokes through
            # the gap and recovers leaves the FVG tradable for the next retest.
            if is_fvg_invalidated(fvg, c_close):
                fvg.status = "invalidated"
                self.seen_fvgs[symbol].add(fvg.formed_at)
                logger.info(f"FVG invalidated: {fvg}")
                continue

            # Gate 1 — HTF trend filter (skipped when USE_HTF_FILTER is False)
            if USE_HTF_FILTER:
                if htf_bias == "bullish" and fvg.direction == "bearish":
                    continue
                if htf_bias == "bearish" and fvg.direction == "bullish":
                    continue

            # Gate 2 — quality score
            if fvg.score < FVG_SCORE_MIN:
                logger.debug(f"FVG score {fvg.score:.2f} below threshold — skip")
                continue

            if not check_retest(fvg, c_high, c_low, c_close):
                continue

            entry   = fvg.gap_mid
            sl_dist = fvg.sl_distance(entry)
            if sl_dist <= 0:
                continue

            # Compute the ACTUAL TP that will be used and gate R:R against it.
            # Matches backtest exactly: if R:R fails this cycle, the FVG stays
            # waiting and we re-evaluate on the next cycle (a new candle may
            # have formed a swing pivot that pushes the structure target out).
            #
            # Log INFO only the first time this FVG fails the gate, then DEBUG
            # on subsequent attempts so we don't spam the log every 10s.
            tp_override = None
            if TP_MODE == "structure":
                df_primary  = self._fetch(symbol, PRIMARY_TF, FVG_LOOKBACK)
                tp_override = find_structure_tp(
                    df_primary, fvg.direction, entry, sl_dist, MIN_RR
                )
                actual_rr = abs(tp_override - entry) / sl_dist
                if actual_rr + 1e-6 < MIN_RR:
                    if not getattr(fvg, "_logged_no_target", False):
                        logger.info(
                            f"No structure target ≥ {MIN_RR}R for {fvg.symbol} "
                            f"(best={actual_rr:.2f}R) — will retry"
                        )
                        fvg._logged_no_target = True
                    else:
                        logger.debug(
                            f"R:R {actual_rr:.2f} < {MIN_RR} for {fvg.symbol}"
                        )
                    continue

            balance = self._cached_balance
            if balance <= 0:
                logger.warning("Zero balance — cannot open trade")
                continue

            trade = open_trade(self.exchange, fvg, balance,
                               dry_run=DRY_RUN, tp_override=tp_override)
            if isinstance(trade, Trade) and trade:
                # If open_trade already closed the trade (instant fill path
                # when TP was -2021), put it in closed_trades. Otherwise the
                # next _poll_exit cycle would see position=0 and fabricate
                # ANOTHER fake exit on top of the real one.
                if trade.status == "closed":
                    self.closed_trades.append(trade)
                else:
                    self.open_trades.append(trade)
                self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
                fvg.status = "retested"
                self.seen_fvgs[symbol].add(fvg.formed_at)
                save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                           self.trades_today, self.last_trade_date)
                break  # one entry per symbol per cycle
            elif trade == "SKIP":
                # Price drifted away from FVG zone — no order was placed.
                # FVG stays "waiting" so price can come back and retry later.
                logger.debug(
                    f"FVG {fvg.symbol} skipped (price drift) — "
                    f"will retry if price returns"
                )
                break  # don't try other FVGs this cycle
            else:
                # open_trade returned None — real rejection (order was placed
                # and emergency-closed due to bad SL or execution slippage).
                # Mark FVG consumed to prevent fee-burning retry loops.
                fvg.status = "retested"
                self.seen_fvgs[symbol].add(fvg.formed_at)
                logger.info(
                    f"FVG {fvg.symbol} marked consumed after failed entry "
                    f"(will not retry)"
                )
                break

    # ── Startup state reconciliation ─────────
    def _reconcile_restored_trades(self) -> None:
        """
        For each restored open trade, check exchange reality:
          - If position is gone AND orders are gone → trade was closed during
            downtime. We can't recover the actual fill price, so mark it
            closed with P&L=0 (don't fabricate losses from stale planned prices).
          - If position is open → leave as-is, normal polling will handle it.
        Runs once at startup before the main loop.
        """
        if not self.open_trades:
            return

        recovered = []
        kept      = []

        for trade in self.open_trades:
            try:
                positions = self.exchange.fetch_positions([trade.symbol])
                pos_size  = _read_position_size(positions, trade.symbol)
            except Exception as e:
                logger.warning(
                    f"Reconcile: fetch_positions failed for {trade.symbol}, "
                    f"keeping trade open: {e}"
                )
                kept.append(trade)
                continue

            if pos_size >= trade.qty * 0.5:
                kept.append(trade)
                continue

            # Position is gone — trade closed sometime during downtime.
            trade.status       = "closed"
            trade.close_reason = "stale_recovered"
            trade.exit_price   = trade.entry_price   # neutral; actual P&L unknown
            trade.pnl_usdt     = 0.0
            trade.pnl_pct      = 0.0
            trade.closed_at    = datetime.now(timezone.utc)
            recovered.append(trade)

            # Best-effort cleanup of any leftover SL/TP orders on the exchange
            for oid in (trade.sl_order_id, trade.tp_order_id):
                if oid:
                    try:
                        self.exchange.cancel_order(oid, trade.symbol)
                    except Exception:
                        pass

        self.open_trades = kept
        self.closed_trades.extend(recovered)
        if recovered:
            logger.info(
                f"Reconciled {len(recovered)} stale trade(s) at startup: "
                f"{[t.symbol for t in recovered]} "
                f"(positions already closed on exchange — P&L unknown, set to 0)"
            )

    # ── Startup orphan order cleanup ─────────
    def _cleanup_orphan_orders(self) -> None:
        """
        On startup, list open orders for every configured symbol. Cancel any
        order whose ID is NOT referenced by a currently-tracked open trade.
        Catches leftovers from crashed/killed bot sessions.
        """
        tracked = set()
        for t in self.open_trades:
            if t.sl_order_id: tracked.add(t.sl_order_id)
            if t.tp_order_id: tracked.add(t.tp_order_id)

        cancelled = 0
        for symbol in SYMBOLS:
            try:
                # Fetch both regular and conditional/stop orders. Binance returns
                # them via fetch_open_orders when called per-symbol on futures.
                orders = self.exchange.fetch_open_orders(symbol)
            except Exception as e:
                logger.warning(f"Orphan cleanup: fetch_open_orders({symbol}) failed: {e}")
                continue

            for o in orders:
                oid = str(o.get("id", ""))
                if not oid or oid in tracked:
                    continue
                # Only cancel reduceOnly orders — those are bot-placed exits.
                # Leaves alone any user-placed manual orders.
                params = o.get("info", {})
                reduce_only = (
                    o.get("reduceOnly") is True
                    or params.get("reduceOnly") is True
                    or str(params.get("reduceOnly", "")).lower() == "true"
                )
                if not reduce_only:
                    continue
                try:
                    self.exchange.cancel_order(oid, symbol)
                    cancelled += 1
                except Exception as e:
                    logger.warning(f"Failed to cancel orphan {oid} on {symbol}: {e}")

        if cancelled:
            logger.info(f"Cancelled {cancelled} orphan reduceOnly order(s) at startup")

    # ── Exchange-side exit polling (STRUCTURE mode) ──
    def _poll_exit(self, trade: Trade):
        """
        Check if either the exchange-side TP (limit) or SL (stop_market) order
        has filled. Returns (actual_exit_price, reason) if closed, None if still open.

        Three detection methods (all run every cycle):
          0. SOFTWARE SL/TP FAILSAFE — if current price has breached SL or TP,
             market-close immediately.  Catches Binance Demo bug where exchange
             orders get silently evicted/cancelled.
          1. ORDER STATUS — check if the SL or TP order itself shows as filled.
          2. POSITION SIZE — if position is zero/gone, one of the orders fired.
        """
        # ── Method 0: Software SL/TP failsafe ──
        # On Binance Demo, stop_market and limit orders get silently evicted.
        # Without this check, price can run far beyond SL with no protection.
        # Uses cached candle data (already fetched this cycle) for zero API cost.
        current_price = 0.0
        cached_df = self._cycle_cache.get(f"{trade.symbol}_{PRIMARY_TF}")
        if cached_df is not None and not cached_df.empty:
            current_price = float(cached_df["close"].iloc[-1])
        else:
            # No cached data — fetch ticker as fallback
            current_price = get_ticker_price(self.exchange, trade.symbol)

        if current_price > 0:
            sl_breached = (
                (trade.direction == "long"  and current_price <= trade.sl_price) or
                (trade.direction == "short" and current_price >= trade.sl_price)
            )
            tp_breached = (
                (trade.direction == "long"  and current_price >= trade.tp_prices[0]) or
                (trade.direction == "short" and current_price <= trade.tp_prices[0])
            ) if trade.tp_prices else False

            if sl_breached or tp_breached:
                reason = "stop_loss" if sl_breached else "take_profit_structure"
                level  = trade.sl_price if sl_breached else trade.tp_prices[0]
                logger.warning(
                    f"SOFTWARE {'SL' if sl_breached else 'TP'} TRIGGERED for "
                    f"{trade.symbol}: price {current_price:.4f} breached "
                    f"{'SL' if sl_breached else 'TP'} {level:.4f}. "
                    f"Market-closing now."
                )
                # Cancel all exchange orders and close at market
                try:
                    self.exchange.cancel_all_orders(trade.symbol)
                except Exception:
                    pass
                close_side = "sell" if trade.direction == "long" else "buy"
                try:
                    close_order = self.exchange.create_order(
                        symbol=trade.symbol, type="market",
                        side=close_side, amount=trade.qty_remaining,
                        params={"reduceOnly": True}
                    )
                    fill = float(
                        close_order.get("average")
                        or close_order.get("price")
                        or current_price
                    )
                    return fill, reason
                except Exception as e:
                    logger.error(f"Software {reason} market close failed: {e}")
                    # Position may already be closed — fall through to normal checks
        tp_filled_by_order, sl_filled_by_order = False, False
        order_fill_price = None

        if trade.tp_order_id:
            try:
                o = self.exchange.fetch_order(trade.tp_order_id, trade.symbol)
                if o.get("status") in ("closed", "filled") or \
                   float(o.get("filled") or 0) >= trade.qty * 0.5:
                    tp_filled_by_order = True
                    order_fill_price = float(
                        o.get("average") or o.get("price") or trade.tp_prices[0])
                    logger.info(
                        f"TP order FILLED for {trade.symbol} @ {order_fill_price:.4f}")
            except Exception as e:
                err_msg = str(e).lower()
                if "-2013" not in err_msg and "does not exist" not in err_msg:
                    logger.debug(f"fetch_order(tp) for {trade.symbol}: {e}")

        if not tp_filled_by_order and trade.sl_order_id:
            try:
                o = self.exchange.fetch_order(trade.sl_order_id, trade.symbol)
                if o.get("status") in ("closed", "filled") or \
                   float(o.get("filled") or 0) >= trade.qty * 0.5:
                    sl_filled_by_order = True
                    order_fill_price = float(
                        o.get("average") or o.get("price") or trade.sl_price)
                    logger.info(
                        f"SL order FILLED for {trade.symbol} @ {order_fill_price:.4f}")
            except Exception as e:
                err_msg = str(e).lower()
                if "-2013" not in err_msg and "does not exist" not in err_msg:
                    logger.debug(f"fetch_order(sl) for {trade.symbol}: {e}")

        if tp_filled_by_order or sl_filled_by_order:
            # Order filled — cancel the counterpart
            _cancel_all_for_symbol(
                self.exchange, trade.symbol,
                sl_id=trade.sl_order_id, tp_id=trade.tp_order_id
            )
            logger.info(f"Cleanup done for {trade.symbol} (order fill detected)")

            reason = "take_profit_structure" if tp_filled_by_order else "stop_loss"
            return float(order_fill_price), reason

        # ── Method 2: Position-based fallback ──
        try:
            positions = fetch_positions_safe(self.exchange, trade.symbol)
            # On Binance Demo, fetch_positions returns an EMPTY list when no
            # position exists (unlike live which returns [{contracts: 0}]).
            # Empty list = position is gone = size 0.
            pos_size = _read_position_size(positions, trade.symbol)
        except Exception as e:
            logger.warning(f"fetch_positions failed for {trade.symbol}: {e}")
            return None

        # Position is gone → trade closed. Determine reason and clean up the
        # leftover order (the one that DIDN'T fire).
        if pos_size < trade.qty * 0.01:
            tp_filled, fill = False, None

            if trade.tp_order_id:
                try:
                    o = self.exchange.fetch_order(trade.tp_order_id, trade.symbol)
                    if o.get("status") in ("closed", "filled") or \
                       float(o.get("filled") or 0) >= trade.qty * 0.99:
                        tp_filled = True
                        fill = float(o.get("average") or o.get("price") or trade.tp_prices[0])
                except Exception as e:
                    err_msg = str(e).lower()
                    if "-2013" in err_msg or "does not exist" in err_msg:
                        logger.debug(f"fetch_order(tp) for {trade.symbol}: order already gone")
                    else:
                        logger.warning(f"fetch_order(tp) failed for {trade.symbol}: {e}")

            # If TP didn't fire, the SL did (position is gone, so it must be one of them)
            if not tp_filled and trade.sl_order_id:
                try:
                    o = self.exchange.fetch_order(trade.sl_order_id, trade.symbol)
                    if o.get("status") in ("closed", "filled") or \
                       float(o.get("filled") or 0) >= trade.qty * 0.99:
                        fill = float(o.get("average") or o.get("price") or trade.sl_price)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "-2013" in err_msg or "does not exist" in err_msg:
                        logger.debug(f"fetch_order(sl) for {trade.symbol}: order already gone")
                    else:
                        logger.warning(f"fetch_order(sl) failed for {trade.symbol}: {e}")

            # Cancel ALL open orders for this symbol — nuclear but reliable.
            # On Binance Demo, fetch_order returns -2013 ("does not exist")
            # but the orders are STILL live on the exchange. Individual
            # cancel-by-ID fails silently while orphans accumulate.
            # cancel_all_orders is the only approach that guarantees cleanup.
            # Safe because we only allow one trade per symbol at a time.
            _cancel_all_for_symbol(
                self.exchange, trade.symbol,
                sl_id=trade.sl_order_id, tp_id=trade.tp_order_id
            )
            logger.info(f"Cleanup done for {trade.symbol} (position closed)")

            if fill is None:
                # Position closed but couldn't determine which order fired.
                # Fetch current market price to infer direction instead of
                # using the planned SL/TP price (which may be wrong on demo
                # where orders get silently evicted).
                current_price = get_ticker_price(self.exchange, trade.symbol)
                if current_price > 0:
                    # If current price is closer to TP than SL, assume TP
                    tp_dist = abs(current_price - trade.tp_prices[0])
                    sl_dist = abs(current_price - trade.sl_price)
                    if tp_dist < sl_dist:
                        tp_filled = True
                        fill = current_price
                    else:
                        fill = current_price
                    logger.warning(
                        f"Position {trade.symbol} closed, orders gone. "
                        f"Using market price {fill:.4f} as exit estimate."
                    )
                else:
                    fill = trade.tp_prices[0] if tp_filled else trade.sl_price
                    logger.warning(
                        f"Position {trade.symbol} closed but couldn't read fill price; "
                        f"using planned price {fill:.4f}"
                    )

            reason = "take_profit_structure" if tp_filled else "stop_loss"
            return float(fill), reason

        return None

    def _mark_closed(self, trade: Trade, actual_exit: float, reason: str) -> None:
        """Update Trade fields and log the exit using the actual fill price."""
        sign = 1 if trade.direction == "long" else -1
        leg_pnl = (actual_exit - trade.entry_price) * trade.qty * sign

        trade.exit_price   = actual_exit
        trade.close_reason = reason
        trade.closed_at    = datetime.now(timezone.utc)
        trade.status       = "closed"
        trade.pnl_usdt     = round(leg_pnl + trade.partial_pnl, 4)
        trade.pnl_pct      = round(
            trade.pnl_usdt / trade.risk_amount * 100 if trade.risk_amount else 0, 2)

        rr = trade.rr_achieved
        logger.info(
            f"CLOSED {trade.symbol} {trade.direction.upper()} | "
            f"reason={reason} | exit={actual_exit:.4f} | "
            f"P&L=${trade.pnl_usdt:.2f} ({trade.pnl_pct:.1f}% of risk)"
            + (f" | R:R={rr:.2f}x" if rr is not None else "")
        )

    # ── Trade management ───────────────────────
    def manage_open_trades(self):
        """
        For every open trade, on the latest ENTRY_TF candle:
          - Move SL to breakeven at 1R profit
          - Trail SL by ATR after 1.5R profit
          - Execute 50% partial close at TP1 (2R) — actually places the order
          - Check final SL / TP hit

        Candle data is fetched once per symbol (cache) to avoid redundant
        API calls when multiple trades share the same symbol.
        """
        still_open = []
        state_dirty = False

        for trade in self.open_trades:
            # ── STRUCTURE mode: poll exchange order status FIRST ──
            # Exit detection only needs fetch_order on the SL/TP IDs — no
            # candle data required. Running this before the kline fetch means
            # exit detection still works during klines outages (which happen
            # often on Binance demo). Previously a klines failure made the
            # bot skip the whole trade iteration → SL fills weren't detected.
            if TP_MODE == "structure":
                exit_info = self._poll_exit(trade)
                if exit_info is not None:
                    actual_exit, reason = exit_info
                    self._mark_closed(trade, actual_exit, reason)
                    self.closed_trades.append(trade)
                    state_dirty = True
                    continue
                still_open.append(trade)
                continue

            # PARTIAL mode below — needs candle data for active management
            df = self._fetch(trade.symbol, ENTRY_TF, 50)
            if df.empty:
                still_open.append(trade)
                continue

            latest  = df.iloc[-1]
            c_high  = float(latest["high"])
            c_low   = float(latest["low"])
            c_close = float(latest["close"])
            is_long = trade.direction == "long"
            sl_dist = abs(trade.entry_price - trade.sl_price)
            atr     = calc_atr(df, ATR_PERIOD)

            profit_r = (
                (c_close - trade.entry_price) / sl_dist if is_long
                else (trade.entry_price - c_close) / sl_dist
            )

            if TP_MODE == "partial":
                # ── Breakeven at 1R ─────────────────
                if profit_r >= 1.0 and not trade.be_moved:
                    be = trade.entry_price
                    trade.current_sl = (
                        max(trade.current_sl, be) if is_long
                        else min(trade.current_sl, be)
                    )
                    trade.be_moved = True
                    state_dirty = True
                    logger.info(f"[BE] {trade.symbol} SL → breakeven {be:.4f}")

                # ── ATR trail at 1.5R ────────────────
                if profit_r >= 1.5:
                    if is_long:
                        new_trail = c_low - atr * TRAIL_ATR_MULT
                        if new_trail > trade.current_sl:
                            trade.current_sl = new_trail
                            state_dirty = True
                    else:
                        new_trail = c_high + atr * TRAIL_ATR_MULT
                        if new_trail < trade.current_sl:
                            trade.current_sl = new_trail
                            state_dirty = True

                # ── Partial exit at TP1 (2R) ─────────
                tp1         = trade.tp_prices[0]
                partial_hit = (
                    (is_long  and c_high >= tp1) or
                    (not is_long and c_low  <= tp1)
                )

                if partial_hit and not trade.partial_done:
                    execute_partial_close(self.exchange, trade, tp1, dry_run=DRY_RUN)
                    state_dirty = True

                    # Extend remaining TP to nearest structural level
                    struct_tp = find_structure_tp(
                        df, "bullish" if is_long else "bearish",
                        trade.entry_price, sl_dist
                    )
                    trade.tp_prices = [struct_tp]
                    logger.info(f"[PARTIAL] TP extended to structure {struct_tp:.4f}")

            # PARTIAL mode exit checks (active management)
            active_sl = trade.current_sl
            tp_final  = trade.tp_prices[0]

            sl_hit = (is_long  and c_low  <= active_sl) or \
                     (not is_long and c_high >= active_sl)
            tp_hit = (is_long  and c_high >= tp_final)  or \
                     (not is_long and c_low  <= tp_final)

            if sl_hit and tp_hit:
                sl_hit, tp_hit = True, False   # conservative: SL wins

            if sl_hit:
                reason = "trailing_sl" if trade.be_moved else "stop_loss"
                trade  = close_trade(self.exchange, trade, active_sl, reason,
                                     dry_run=DRY_RUN)
                self.closed_trades.append(trade)
                state_dirty = True
                continue

            if tp_hit:
                trade = close_trade(self.exchange, trade, tp_final,
                                    "take_profit_structure", dry_run=DRY_RUN)
                self.closed_trades.append(trade)
                state_dirty = True
                continue

            still_open.append(trade)

        self.open_trades = still_open
        if state_dirty:
            save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                       self.trades_today, self.last_trade_date)

    # ── Summary ────────────────────────────────
    def print_summary(self):
        wins  = [t for t in self.closed_trades if t.pnl_usdt and t.pnl_usdt > 0]
        total = len(self.closed_trades)
        pnl   = sum(t.pnl_usdt or 0 for t in self.closed_trades)
        wr    = len(wins) / total * 100 if total else 0
        logger.info(
            f"SUMMARY | Closed={total} | Wins={len(wins)} | "
            f"Win%={wr:.1f}% | Total P&L=${pnl:.2f}"
        )

    # ── Main loop ──────────────────────────────
    def run(self):
        logger.info("Bot loop started")
        cycle = 0

        while running:
            cycle += 1
            self._cycle_cache.clear()   # fresh per-cycle candle cache
            self._cached_balance = get_account_balance(self.exchange)  # once per cycle
            self.reset_daily_counters()

            for symbol in SYMBOLS:
                try:
                    self.scan_fvgs(symbol)
                    self.check_entries(symbol)
                except Exception as exc:
                    logger.error(f"Error processing {symbol}: {exc}", exc_info=True)

            try:
                self.manage_open_trades()
            except Exception as exc:
                logger.error(f"Error managing trades: {exc}", exc_info=True)

            if cycle % 60 == 0:
                self.print_summary()

            # Periodic state save every ~5 minutes (30 cycles at 10s)
            # Protects FVG tracking against crashes between trade events.
            if cycle % 30 == 0:
                save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                           self.trades_today, self.last_trade_date)

            time.sleep(POLL_INTERVAL_SEC)

        logger.info("Bot stopped cleanly")
        self.print_summary()


if __name__ == "__main__":
    bot = FVGBot()
    bot.run()
