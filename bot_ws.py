#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  FVG Trading Bot — WebSocket / Event-Driven Mode
#  Requires:  pip install "ccxt[pro]"
#  Run:       python bot_ws.py
#
#  Instead of polling every 10 s, this bot subscribes to Binance
#  kline WebSocket streams.  Each PRIMARY_TF candle close triggers
#  FVG detection and entry evaluation for that symbol immediately
#  (~50 ms latency vs 10 s polling).  Multiple symbols run in
#  parallel asyncio tasks so no symbol blocks another.
#
#  IMPORTANT: This module mirrors bot.py logic exactly — same gates,
#  same R:R checks, same drift/SL-breach/post-fill guards.
# ─────────────────────────────────────────────

import asyncio
import logging
import signal
import sys
import time
from datetime import date, datetime, timezone, timedelta

import pandas as pd

try:
    import ccxt.pro as ccxtpro
except ImportError:
    print(
        "ERROR: ccxt[pro] is not installed.\n"
        "Run:  pip install \"ccxt[pro]\"\n"
        "Or use the REST polling bot instead:  python bot.py"
    )
    sys.exit(1)

from config import (
    API_KEY, API_SECRET, TESTNET, BINANCE_DEMO, DRY_RUN, MARKET_TYPE,
    SYMBOLS, PRIMARY_TF, ENTRY_TF, HTF_TF,
    FVG_MIN_SIZE_PCT, FVG_LOOKBACK, FVG_EXPIRY_CANDLES,
    FVG_MAX_ACTIVE, FVG_SCORE_MIN, SL_BUFFER_PCT, MIN_RR,
    TP_MODE, USE_HTF_FILTER,
    RISK_PCT, MAX_TRADES_DAY, MAX_OPEN_TRADES, MAX_CORRELATED,
    HTF_EMA_PERIOD, ATR_PERIOD, TRAIL_ATR_MULT,
    DEFAULT_LEVERAGE, MAX_POSITION_PCT, MARGIN_MODE,
    STATE_FILE, LOG_FILE, LOG_LEVEL,
)
from fvg_detector import (
    detect_fvgs, check_retest, is_fvg_invalidated, fvg_quality_score,
)
from order_manager import (
    open_trade, close_trade, execute_partial_close,
    calc_atr, find_structure_tp, Trade,
)
from exchange import fetch_positions_safe, get_ticker_price
from state import save_state, load_state
from trade_logger import log_trade

# ── Logging ───────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
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

running = True


def _shutdown(sig, frame):
    global running
    running = False
    logger.info("Shutdown signal — stopping WebSocket bot...")


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Helpers ───────────────────────────────────

def _raw_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


def _read_position_size(positions: list, symbol: str) -> float:
    """Safely extract position size (matches bot.py exactly)."""
    for p in positions:
        psym = p.get("symbol", "")
        if psym == symbol or psym.replace(":USDT", "") == symbol:
            contracts = p.get("contracts")
            if contracts is not None:
                return abs(float(contracts))
            info = p.get("info", {})
            pos_amt = info.get("positionAmt")
            if pos_amt is not None:
                return abs(float(pos_amt))
            return 0.0
    return 0.0


async def _cancel_all_for_symbol(exchange, symbol: str, sl_id=None, tp_id=None) -> None:
    """Cancel ALL orders for a symbol — regular + algo (matches bot.py)."""
    try:
        await exchange.cancel_all_orders(symbol)
    except Exception:
        pass
    for oid in (sl_id, tp_id):
        if not oid:
            continue
        try:
            await exchange.cancel_order(str(oid), symbol)
        except Exception:
            pass
        try:
            await exchange.fapiPrivateDeleteAlgoOrder({"algoId": str(oid)})
        except Exception:
            pass
    bsym = symbol.replace("/", "")
    try:
        algo_orders = await exchange.fapiPrivateGetOpenAlgoOrders()
        for ao in algo_orders:
            if ao.get("symbol") == bsym:
                try:
                    await exchange.fapiPrivateDeleteAlgoOrder({"algoId": ao["algoId"]})
                    logger.info(f"Cancelled algo order {ao['algoId']} on {symbol}")
                except Exception:
                    pass
    except Exception:
        pass


# ── Bot class ─────────────────────────────────

class FVGBotWS:
    def __init__(self):
        opts = {
            "apiKey":  API_KEY,
            "secret":  API_SECRET,
            "options": {
                "defaultType": MARKET_TYPE,
                "adjustForTimeDifference": True,
                "recvWindow": 30000,
                "fetchCurrencies": False,
                "fetchMarkets":    ["linear"],
            },
            "enableRateLimit": True,
        }
        self.exchange: ccxtpro.binance = ccxtpro.binance(opts)

        if BINANCE_DEMO:
            DEMO_HOST = "https://demo-fapi.binance.com"
            for k in ("fapiPublic", "fapiPublicV2", "fapiPublicV3",
                       "fapiPrivate", "fapiPrivateV2", "fapiPrivateV3",
                       "fapiData"):
                old = self.exchange.urls["api"].get(k)
                if old:
                    suffix = old.split(".com", 1)[-1]
                    self.exchange.urls["api"][k] = DEMO_HOST + suffix
            logger.info(f"Connected to Binance {MARKET_TYPE.upper()} DEMO TRADING ({DEMO_HOST})")
        elif TESTNET:
            self.exchange.set_sandbox_mode(True)
            logger.info("Connected to Binance TESTNET")

        state = load_state(STATE_FILE)
        self.open_trades:     list[Trade]     = state["open_trades"]
        self.closed_trades:   list[Trade]     = []
        self.active_fvgs:     dict[str, list] = state["active_fvgs"]
        self.trades_today:    dict[str, int]  = state["trades_today"]
        self.last_trade_date: date            = state["last_trade_date"]
        self.seen_fvgs:       dict[str, set]  = {s: set() for s in SYMBOLS}

        # Per-symbol cooldown after SL (matches bot.py _entry_cooldown)
        self._entry_cooldown: dict[str, datetime] = {}

        # Per-symbol buffers of raw OHLCV rows (maintained as rolling window)
        self._buffers:  dict[str, list]          = {s: [] for s in SYMBOLS}
        # HTF DataFrames — refreshed hourly
        self._htf_dfs:  dict[str, pd.DataFrame]  = {}
        self._htf_ts:   dict[str, float]         = {}
        # Cached balance (refreshed periodically)
        self._cached_balance: float = 0.0
        self._balance_ts: float = 0.0
        # Locks prevent concurrent entry/manage calls for the same symbol
        self._locks:    dict[str, asyncio.Lock]  = {s: asyncio.Lock() for s in SYMBOLS}

        logger.info("=" * 60)
        logger.info("  FVG WebSocket Bot starting")
        logger.info(f"  Mode:    {'DRY RUN' if DRY_RUN else 'DEMO' if BINANCE_DEMO else '⚠ LIVE'}")
        logger.info(f"  Symbols: {', '.join(SYMBOLS)}")
        logger.info(f"  Streams: {PRIMARY_TF} kline per symbol")
        logger.info(f"  R:R min: {MIN_RR}  |  TP mode: {TP_MODE}")
        logger.info(f"  Risk:    {RISK_PCT*100:.1f}%  |  Max open: {MAX_OPEN_TRADES}")
        logger.info("=" * 60)

    # ── Initialisation ────────────────────────
    async def _boot(self):
        """Seed candle buffers and HTF data before opening streams."""
        # Fetch initial balance
        await self._refresh_balance()

        # Reconcile state, clean orphans, verify orders (matches bot.py startup)
        await self._reconcile_startup()

        for symbol in SYMBOLS:
            try:
                raw = await self.exchange.fetch_ohlcv(
                    symbol, PRIMARY_TF, limit=FVG_LOOKBACK
                )
                self._buffers[symbol] = list(raw)
                df  = _raw_to_df(raw)
                atr = calc_atr(df, ATR_PERIOD)
                self._refresh_fvgs(symbol, df, atr)
                await self._refresh_htf(symbol)
                logger.info(f"Booted {symbol} — {len(raw)} seed candles")
            except Exception as exc:
                logger.error(f"Boot failed for {symbol}: {exc}")

    # ── Balance ───────────────────────────────
    async def _refresh_balance(self) -> None:
        now = time.time()
        if now - self._balance_ts < 30:  # cache for 30s
            return
        try:
            bal_data = await self.exchange.fetch_balance()
            total = bal_data.get("total", {}).get("USDT", 0)
            free  = bal_data.get("free", {}).get("USDT", 0)
            self._cached_balance = float(free if free else total)
            self._balance_ts = now
        except Exception as exc:
            logger.warning(f"Balance fetch failed: {exc}")

    # ── HTF bias ──────────────────────────────
    async def _refresh_htf(self, symbol: str) -> None:
        """Fetch 1-hour candles for trend bias; cached for 1 hour."""
        now = time.time()
        if symbol in self._htf_ts and now - self._htf_ts[symbol] < 3600:
            return
        try:
            raw = await self.exchange.fetch_ohlcv(
                symbol, HTF_TF, limit=HTF_EMA_PERIOD + 20
            )
            if raw:
                self._htf_dfs[symbol] = _raw_to_df(raw)
                self._htf_ts[symbol]  = now
        except Exception as exc:
            logger.warning(f"HTF refresh failed for {symbol}: {exc}")

    def _htf_bias(self, symbol: str) -> str:
        """Matches bot.py get_htf_bias exactly."""
        if not USE_HTF_FILTER:
            return "neutral"
        df = self._htf_dfs.get(symbol)
        if df is None or len(df) < HTF_EMA_PERIOD:
            return "neutral"
        ema   = df["close"].ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
        price = float(df["close"].iloc[-1])
        ema_v = float(ema.iloc[-1])
        if price > ema_v * 1.001:
            return "bullish"
        if price < ema_v * 0.999:
            return "bearish"
        return "neutral"

    # ── FVG state management ──────────────────
    def _refresh_fvgs(self, symbol: str, df: pd.DataFrame, atr: float) -> None:
        """Detect new FVGs, score them, prune stale ones, enforce cap."""
        seen = self.seen_fvgs.get(symbol, set())
        new_fvgs = detect_fvgs(df, symbol, PRIMARY_TF,
                               min_size_pct=FVG_MIN_SIZE_PCT,
                               sl_buffer_pct=SL_BUFFER_PCT)
        existing_times = {f.formed_at for f in self.active_fvgs.get(symbol, [])}

        for fvg in new_fvgs[:10]:
            if fvg.formed_at not in existing_times and fvg.formed_at not in seen:
                fvg.score            = fvg_quality_score(fvg, df, atr)
                fvg.last_candle_time = df.index[-1]
                self.active_fvgs[symbol].append(fvg)
                logger.info(f"New FVG: {fvg}")

        # Prune, sort newest-first, cap
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

    def _can_trade(self, symbol: str) -> bool:
        """Matches bot.py can_trade exactly."""
        if len(self.open_trades) >= MAX_OPEN_TRADES:
            return False
        if self.trades_today.get(symbol, 0) >= MAX_TRADES_DAY:
            return False
        if any(t.symbol == symbol for t in self.open_trades):
            return False
        altcoins   = [s for s in SYMBOLS if "BTC" not in s]
        open_alts  = sum(1 for t in self.open_trades if t.symbol in altcoins)
        if symbol in altcoins and open_alts >= MAX_CORRELATED:
            return False
        return True

    def _reset_daily(self) -> None:
        today = date.today()
        if today != self.last_trade_date:
            self.trades_today.clear()
            self.last_trade_date = today
            logger.info("Daily trade counters reset")

    # ── Candle-close handler ──────────────────
    async def _on_candle_close(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Called once per PRIMARY_TF candle close.  Runs under the per-symbol
        lock so concurrent stream events never race each other.
        """
        self._reset_daily()
        await self._refresh_htf(symbol)
        await self._refresh_balance()

        atr = calc_atr(df, ATR_PERIOD)

        # Advance expiry counters by exactly one candle
        for fvg in self.active_fvgs.get(symbol, []):
            fvg.candles_since_formed += 1
            fvg.last_candle_time      = df.index[-1]

        self._refresh_fvgs(symbol, df, atr)

        if self._can_trade(symbol):
            await self._check_entries(symbol, df, atr)

        await self._manage_trades(symbol, df, atr)
        save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                   self.trades_today, self.last_trade_date)

    # ── Entry evaluation (matches bot.py check_entries exactly) ──
    async def _check_entries(self, symbol: str,
                              df: pd.DataFrame, atr: float) -> None:
        # Cooldown gate: skip if this symbol had a recent SL loss
        cooldown_until = self._entry_cooldown.get(symbol)
        if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
            remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds()
            logger.debug(f"{symbol} cooldown: no re-entry for {remaining:.0f}s")
            return

        latest  = df.iloc[-1]
        c_high  = float(latest["high"])
        c_low   = float(latest["low"])
        c_close = float(latest["close"])
        bias    = self._htf_bias(symbol)

        for fvg in list(self.active_fvgs.get(symbol, [])):
            # Invalidation check uses CLOSE, not wick
            if is_fvg_invalidated(fvg, c_close):
                fvg.status = "invalidated"
                self.seen_fvgs[symbol].add(fvg.formed_at)
                logger.info(f"FVG invalidated: {fvg}")
                continue

            # Gate 1 — HTF trend filter (skipped when USE_HTF_FILTER is False)
            if USE_HTF_FILTER:
                if bias == "bullish" and fvg.direction == "bearish":
                    continue
                if bias == "bearish" and fvg.direction == "bullish":
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

            # Early drift check: skip if current price drifted > 0.5% from entry
            drift_pct = abs(c_close - entry) / entry if entry > 0 else 0
            if drift_pct > 0.005:
                logger.debug(
                    f"SKIP {fvg.symbol}: price {c_close:.4f} drifted "
                    f"{drift_pct*100:.2f}% from entry {entry:.4f}"
                )
                continue

            # Structure TP with NET R:R gate (matches bot.py + backtest_net.py)
            tp_override = None
            if TP_MODE == "structure":
                realistic_entry = c_close
                realistic_sl_dist = abs(realistic_entry - fvg.sl_price)
                if realistic_sl_dist <= 0:
                    continue

                balance = self._cached_balance
                if balance <= 0:
                    continue

                from order_manager import calculate_position_size, find_structure_tps
                est_qty, _, _ = calculate_position_size(
                    balance, realistic_entry, fvg.sl_price
                )
                if est_qty <= 0:
                    continue

                fee_rate = 0.0005  # 0.05% taker
                notional = est_qty * realistic_entry
                entry_fee = notional * fee_rate
                exit_fee_sl = est_qty * fvg.sl_price * fee_rate
                total_fee_loss = entry_fee + exit_fee_sl
                sign = 1 if fvg.direction == "bullish" else -1

                # Get ALL swing pivots sorted nearest-to-farthest
                all_tps = find_structure_tps(
                    df, fvg.direction, realistic_entry,
                    realistic_sl_dist, min_rr=0.0
                )

                best_net_rr = -999.0
                for tp_candidate in all_tps:
                    tp_dist = (tp_candidate - realistic_entry) * sign
                    if tp_dist <= 0:
                        continue
                    exit_fee_tp = est_qty * tp_candidate * fee_rate
                    total_fee_win = entry_fee + exit_fee_tp
                    net_win  = est_qty * tp_dist - total_fee_win
                    net_loss = est_qty * realistic_sl_dist + total_fee_loss
                    if net_loss <= 0:
                        net_loss = 0.001
                    net_rr = net_win / net_loss
                    best_net_rr = max(best_net_rr, net_rr)
                    if net_rr + 1e-6 >= MIN_RR:
                        tp_override = tp_candidate
                        logger.debug(
                            f"Structure TP found for {fvg.symbol}: "
                            f"tp={tp_candidate:.4f}, net_rr={net_rr:.2f}"
                        )
                        break

                if tp_override is None:
                    if not getattr(fvg, "_logged_no_target", False):
                        logger.info(
                            f"No structure target ≥ {MIN_RR}R NET for {fvg.symbol} "
                            f"(best_net_rr={best_net_rr:.2f}, price={c_close:.4f}) — will retry"
                        )
                        fvg._logged_no_target = True
                    else:
                        logger.debug(
                            f"Best net R:R {best_net_rr:.2f} < {MIN_RR} for {fvg.symbol}"
                        )
                    continue

            balance = self._cached_balance
            if balance <= 0:
                logger.warning("Zero balance — cannot open trade")
                continue

            trade = open_trade(self.exchange, fvg, balance,
                               dry_run=DRY_RUN, tp_override=tp_override)
            if isinstance(trade, Trade) and trade:
                if trade.status == "closed":
                    self.closed_trades.append(trade)
                else:
                    self.open_trades.append(trade)
                self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
                fvg.status = "retested"
                self.seen_fvgs[symbol].add(fvg.formed_at)
                save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                           self.trades_today, self.last_trade_date)
                break  # one entry per candle close per symbol
            elif trade == "SKIP":
                logger.debug(
                    f"FVG {fvg.symbol} skipped (price drift) — "
                    f"will retry if price returns"
                )
                break
            else:
                # open_trade returned None — real rejection
                fvg.status = "retested"
                self.seen_fvgs[symbol].add(fvg.formed_at)
                logger.info(
                    f"FVG {fvg.symbol} marked consumed after failed entry "
                    f"(will not retry)"
                )
                break

    # ── Trade management (matches bot.py manage_open_trades exactly) ──
    async def _manage_trades(self, symbol: str,
                              df: pd.DataFrame, atr: float) -> None:
        still_open = []

        for trade in self.open_trades:
            if trade.symbol != symbol:
                still_open.append(trade)
                continue

            # STRUCTURE mode: poll exchange orders (matches bot.py exactly)
            if TP_MODE == "structure":
                exit_info = await self._poll_exit(trade, df)
                if exit_info is not None:
                    actual_exit, reason = exit_info
                    self._mark_closed(trade, actual_exit, reason)
                    self.closed_trades.append(trade)
                    continue
                still_open.append(trade)
                continue

            # PARTIAL mode: candle-based management
            latest  = df.iloc[-1]
            c_high  = float(latest["high"])
            c_low   = float(latest["low"])
            c_close = float(latest["close"])
            is_long = trade.direction == "long"
            sl_dist = abs(trade.entry_price - trade.sl_price)

            if sl_dist == 0:
                still_open.append(trade)
                continue

            profit_r = (
                (c_close - trade.entry_price) / sl_dist if is_long
                else (trade.entry_price - c_close) / sl_dist
            )

            if profit_r >= 1.0 and not trade.be_moved:
                be = trade.entry_price
                trade.current_sl = (
                    max(trade.current_sl, be) if is_long
                    else min(trade.current_sl, be)
                )
                trade.be_moved = True
                logger.info(f"[BE] {symbol} SL → breakeven {be:.4f}")

            if profit_r >= 1.5:
                if is_long:
                    new_trail = c_low - atr * TRAIL_ATR_MULT
                    if new_trail > trade.current_sl:
                        trade.current_sl = new_trail
                else:
                    new_trail = c_high + atr * TRAIL_ATR_MULT
                    if new_trail < trade.current_sl:
                        trade.current_sl = new_trail

            tp1 = trade.tp_prices[0]
            partial_hit = (is_long and c_high >= tp1) or \
                          (not is_long and c_low <= tp1)
            if partial_hit and not trade.partial_done:
                execute_partial_close(self.exchange, trade, tp1, dry_run=DRY_RUN)
                struct_tp = find_structure_tp(
                    df, "bullish" if is_long else "bearish",
                    trade.entry_price, sl_dist, MIN_RR
                )
                trade.tp_prices = [struct_tp]

            active_sl = trade.current_sl
            tp_final  = trade.tp_prices[0]
            sl_hit    = (is_long  and c_low  <= active_sl) or \
                        (not is_long and c_high >= active_sl)
            tp_hit    = (is_long  and c_high >= tp_final)  or \
                        (not is_long and c_low  <= tp_final)

            if sl_hit and tp_hit:
                sl_hit, tp_hit = True, False

            if sl_hit:
                reason = "trailing_sl" if trade.be_moved else "stop_loss"
                trade  = close_trade(self.exchange, trade, active_sl, reason,
                                     dry_run=DRY_RUN)
                self.closed_trades.append(trade)
                log_trade(trade)
                self._entry_cooldown[symbol] = (
                    datetime.now(timezone.utc) + timedelta(seconds=300)
                )
                logger.info(f"{symbol} cooldown: no re-entry for 300s")
                continue

            if tp_hit:
                trade = close_trade(self.exchange, trade, tp_final,
                                    "take_profit_structure", dry_run=DRY_RUN)
                self.closed_trades.append(trade)
                log_trade(trade)
                continue

            still_open.append(trade)

        self.open_trades = still_open

    # ── Exchange-aware exit polling (matches bot.py _poll_exit) ──
    async def _poll_exit(self, trade: Trade, df: pd.DataFrame = None):
        """Check exchange order fills, software failsafe, position size."""
        current_price = 0.0
        if df is not None and not df.empty:
            current_price = float(df["close"].iloc[-1])
        else:
            current_price = get_ticker_price(self.exchange, trade.symbol)

        tp_filled_by_order, sl_filled_by_order = False, False
        order_fill_price = None
        sl_order_alive, tp_order_alive = False, False

        # Check TP order
        if trade.tp_order_id:
            try:
                o = await self.exchange.fetch_order(trade.tp_order_id, trade.symbol)
                status = o.get("status")
                if status in ("closed", "filled") or \
                   float(o.get("filled") or 0) >= trade.qty * 0.5:
                    tp_filled_by_order = True
                    order_fill_price = float(
                        o.get("average") or o.get("price") or trade.tp_prices[0])
                    logger.info(f"TP order FILLED for {trade.symbol} @ {order_fill_price:.4f}")
                elif status == "open":
                    tp_order_alive = True
            except Exception as e:
                err_msg = str(e).lower()
                if "-2013" not in err_msg and "does not exist" not in err_msg:
                    logger.debug(f"fetch_order(tp) for {trade.symbol}: {e}")

        # Check SL order
        if not tp_filled_by_order and trade.sl_order_id:
            try:
                o = await self.exchange.fetch_order(trade.sl_order_id, trade.symbol)
                status = o.get("status")
                if status in ("closed", "filled") or \
                   float(o.get("filled") or 0) >= trade.qty * 0.5:
                    sl_filled_by_order = True
                    order_fill_price = float(
                        o.get("average") or o.get("price") or trade.sl_price)
                    logger.info(f"SL order FILLED for {trade.symbol} @ {order_fill_price:.4f}")
                elif status == "open":
                    sl_order_alive = True
            except Exception as e:
                err_msg = str(e).lower()
                if "-2013" not in err_msg and "does not exist" not in err_msg:
                    logger.debug(f"fetch_order(sl) for {trade.symbol}: {e}")

        if tp_filled_by_order or sl_filled_by_order:
            await _cancel_all_for_symbol(
                self.exchange, trade.symbol,
                sl_id=trade.sl_order_id, tp_id=trade.tp_order_id
            )
            logger.info(f"Cleanup done for {trade.symbol} (order fill detected)")
            reason = "take_profit_structure" if tp_filled_by_order else "stop_loss"
            return float(order_fill_price), reason

        # Software SL/TP failsafe (only if exchange orders are missing)
        if current_price > 0 and not (sl_order_alive and tp_order_alive):
            sl_breached = (
                (trade.direction == "long"  and current_price <= trade.sl_price) or
                (trade.direction == "short" and current_price >= trade.sl_price)
            )
            tp_breached = (
                (trade.direction == "long"  and current_price >= trade.tp_prices[0]) or
                (trade.direction == "short" and current_price <= trade.tp_prices[0])
            ) if trade.tp_prices else False

            if sl_breached or tp_breached:
                order_missing = (sl_breached and not sl_order_alive) or \
                                (tp_breached and not tp_order_alive)
                if order_missing:
                    reason = "software_sl" if sl_breached else "software_tp"
                    level = trade.sl_price if sl_breached else trade.tp_prices[0]
                    logger.warning(
                        f"SOFTWARE {'SL' if sl_breached else 'TP'} TRIGGERED for "
                        f"{trade.symbol}: price {current_price:.4f} breached "
                        f"{'SL' if sl_breached else 'TP'} {level:.4f} "
                        f"(exchange order MISSING). Market-closing now."
                    )
                    try:
                        await self.exchange.cancel_all_orders(trade.symbol)
                    except Exception:
                        pass
                    close_side = "sell" if trade.direction == "long" else "buy"
                    try:
                        close_order = await self.exchange.create_order(
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
                        logger.warning(f"Software {reason} market close failed: {e}")

        # Position-based fallback
        try:
            positions = fetch_positions_safe(self.exchange, trade.symbol)
            pos_size = _read_position_size(positions, trade.symbol)
        except Exception:
            return None

        if pos_size < trade.qty * 0.01:
            await _cancel_all_for_symbol(
                self.exchange, trade.symbol,
                sl_id=trade.sl_order_id, tp_id=trade.tp_order_id
            )
            logger.info(f"Cleanup done for {trade.symbol} (position closed)")
            fill = current_price if current_price > 0 else trade.sl_price
            if current_price > 0:
                tp_dist = abs(current_price - trade.tp_prices[0])
                sl_dist = abs(current_price - trade.sl_price)
                tp_filled = tp_dist < sl_dist
                logger.warning(
                    f"Position {trade.symbol} closed, orders gone. "
                    f"Using market price {fill:.4f} as exit estimate."
                )
            else:
                tp_filled = False
            reason = "take_profit_structure" if tp_filled else "stop_loss"
            return float(fill), reason

        return None

    def _mark_closed(self, trade: Trade, actual_exit: float, reason: str) -> None:
        """Update Trade fields and log exit (matches bot.py exactly)."""
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
        if reason in ("stop_loss", "software_sl"):
            self._entry_cooldown[trade.symbol] = (
                datetime.now(timezone.utc) + timedelta(seconds=300)
            )
            logger.info(f"{trade.symbol} cooldown: no re-entry for 300s")
        try:
            log_trade(trade)
        except Exception:
            pass

    # ── Startup reconciliation (matches bot.py) ──
    async def _reconcile_startup(self) -> None:
        """Reconcile state, clean orphans, verify orders on boot."""
        # 1. Reconcile stale trades
        kept = []
        for trade in self.open_trades:
            try:
                positions = fetch_positions_safe(self.exchange, trade.symbol)
                pos_size = _read_position_size(positions, trade.symbol)
            except Exception:
                kept.append(trade)
                continue
            if pos_size >= trade.qty * 0.5:
                kept.append(trade)
            else:
                trade.status = "closed"
                trade.close_reason = "stale_recovered"
                trade.exit_price = trade.entry_price
                trade.pnl_usdt = 0.0
                trade.closed_at = datetime.now(timezone.utc)
                self.closed_trades.append(trade)
                for oid in (trade.sl_order_id, trade.tp_order_id):
                    if oid:
                        try:
                            await self.exchange.cancel_order(str(oid), trade.symbol)
                        except Exception:
                            pass
                logger.info(f"Reconciled stale trade: {trade.symbol}")
        self.open_trades = kept

        # 2. Close orphan positions
        tracked_symbols = {t.symbol for t in self.open_trades}
        for sym in SYMBOLS:
            try:
                positions = fetch_positions_safe(self.exchange, sym)
                for pos in positions:
                    psym = pos.get("symbol", "")
                    clean_sym = psym.split(":")[0] if ":" in psym else psym
                    contracts = pos.get("contracts")
                    amt = abs(float(contracts)) if contracts is not None else 0.0
                    if amt == 0 or clean_sym in tracked_symbols:
                        continue
                    side = pos.get("side")
                    close_side = "sell" if side == "long" else "buy"
                    logger.warning(f"ORPHAN POSITION: {clean_sym} {side} qty={amt}")
                    try:
                        await self.exchange.create_order(
                            symbol=clean_sym, type="market", side=close_side,
                            amount=amt, params={"reduceOnly": True}
                        )
                        await _cancel_all_for_symbol(self.exchange, clean_sym)
                    except Exception as e:
                        logger.error(f"Failed to close orphan {clean_sym}: {e}")
            except Exception:
                pass

        # 3. Verify SL/TP orders for open trades
        for trade in self.open_trades:
            sl_side = "sell" if trade.direction == "long" else "buy"
            sl_ok = False
            if trade.sl_order_id:
                try:
                    o = await self.exchange.fetch_order(str(trade.sl_order_id), trade.symbol)
                    sl_ok = o.get("status") == "open"
                except Exception:
                    pass
            if not sl_ok:
                await _cancel_all_for_symbol(self.exchange, trade.symbol)
                logger.warning(f"SL order missing for {trade.symbol} — re-placing stop_market @ {trade.sl_price:.6f}")
                try:
                    sl_order = await self.exchange.create_order(
                        symbol=trade.symbol, type="stop_market",
                        side=sl_side, amount=trade.qty_remaining,
                        params={"stopPrice": trade.sl_price, "reduceOnly": True}
                    )
                    trade.sl_order_id = sl_order["id"]
                except Exception as e:
                    logger.error(f"Failed to re-place SL for {trade.symbol}: {e}")

            tp_ok = False
            if trade.tp_order_id:
                try:
                    o = await self.exchange.fetch_order(str(trade.tp_order_id), trade.symbol)
                    tp_ok = o.get("status") == "open"
                except Exception:
                    pass
            if not tp_ok and trade.tp_prices:
                tp_price = trade.tp_prices[0]
                logger.warning(f"TP order missing for {trade.symbol} — re-placing limit @ {tp_price:.6f}")
                try:
                    tp_order = await self.exchange.create_order(
                        symbol=trade.symbol, type="limit",
                        side=sl_side, amount=trade.qty_remaining,
                        price=tp_price, params={"reduceOnly": True}
                    )
                    trade.tp_order_id = tp_order["id"]
                except Exception as e:
                    logger.error(f"Failed to re-place TP for {trade.symbol}: {e}")

        if self.open_trades:
            save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                       self.trades_today, self.last_trade_date)

    # ── Per-symbol WebSocket loop ─────────────
    async def _watch_symbol(self, symbol: str) -> None:
        """
        Subscribe to kline updates for one symbol.
        Detects candle close by watching for a timestamp change in the
        last row: when a new candle opens, the previous one is closed.
        Reconnects automatically on network errors.
        """
        last_ts: int | None = None
        logger.info(f"Subscribing to {symbol} {PRIMARY_TF} kline stream")

        while running:
            try:
                # watch_ohlcv returns the full rolling window on each update
                candles = await self.exchange.watch_ohlcv(
                    symbol, PRIMARY_TF, limit=FVG_LOOKBACK
                )
                if not candles:
                    continue

                current_ts = candles[-1][0]   # timestamp of the forming candle

                if last_ts is not None and current_ts != last_ts:
                    # The previous candle just closed — use everything up to
                    # (but not including) the newly forming candle.
                    closed_raw = candles[:-1]
                    if len(closed_raw) >= 3:
                        df = _raw_to_df(closed_raw)
                        async with self._locks[symbol]:
                            await self._on_candle_close(symbol, df)

                last_ts = current_ts

            except ccxtpro.NetworkError as exc:
                logger.warning(
                    f"WS network error on {symbol}: {exc} — reconnecting in 5 s"
                )
                await asyncio.sleep(5)
            except Exception as exc:
                logger.error(f"WS error on {symbol}: {exc}", exc_info=True)
                await asyncio.sleep(2)

    # ── Periodic summary ──────────────────────
    async def _summary_loop(self) -> None:
        while running:
            await asyncio.sleep(600)   # every 10 minutes
            wins  = [t for t in self.closed_trades if t.pnl_usdt and t.pnl_usdt > 0]
            total = len(self.closed_trades)
            pnl   = sum(t.pnl_usdt or 0 for t in self.closed_trades)
            wr    = len(wins) / total * 100 if total else 0
            logger.info(
                f"SUMMARY | Closed={total} | Wins={len(wins)} | "
                f"Win%={wr:.1f}% | Total P&L=${pnl:.2f}"
            )

    # ── Entry point ───────────────────────────
    async def run(self) -> None:
        await self._boot()

        tasks = [self._watch_symbol(sym) for sym in SYMBOLS]
        tasks.append(self._summary_loop())

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as exc:
            logger.error(f"Top-level task error: {exc}", exc_info=True)
        finally:
            await self.exchange.close()
            logger.info("WebSocket bot stopped cleanly")
            wins  = [t for t in self.closed_trades if t.pnl_usdt and t.pnl_usdt > 0]
            total = len(self.closed_trades)
            pnl   = sum(t.pnl_usdt or 0 for t in self.closed_trades)
            wr    = len(wins) / total * 100 if total else 0
            logger.info(
                f"FINAL SUMMARY | Closed={total} | Wins={len(wins)} | "
                f"Win%={wr:.1f}% | Total P&L=${pnl:.2f}"
            )


if __name__ == "__main__":
    bot = FVGBotWS()
    asyncio.run(bot.run())
