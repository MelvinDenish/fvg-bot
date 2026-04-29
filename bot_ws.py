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
# ─────────────────────────────────────────────

import asyncio
import logging
import signal
import sys
import time
from datetime import date

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
    API_KEY, API_SECRET, TESTNET,
    SYMBOLS, PRIMARY_TF, HTF_TF,
    FVG_MIN_SIZE_PCT, FVG_LOOKBACK, FVG_EXPIRY_CANDLES,
    FVG_MAX_ACTIVE, FVG_SCORE_MIN, SL_BUFFER_PCT, MIN_RR,
    MAX_TRADES_DAY, MAX_OPEN_TRADES, MAX_CORRELATED,
    HTF_EMA_PERIOD, ATR_PERIOD, TRAIL_ATR_MULT,
    STATE_FILE, LOG_FILE, LOG_LEVEL,
)
from fvg_detector import (
    detect_fvgs, check_retest, is_fvg_invalidated, fvg_quality_score,
)
from order_manager import (
    open_trade, close_trade, execute_partial_close,
    calc_atr, find_structure_tp, Trade,
)
from state import save_state, load_state

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
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


# ── Bot class ─────────────────────────────────

class FVGBotWS:
    def __init__(self):
        opts = {
            "apiKey":  API_KEY,
            "secret":  API_SECRET,
            "options": {"defaultType": "spot"},
            "enableRateLimit": True,
        }
        self.exchange: ccxtpro.binance = ccxtpro.binance(opts)
        if TESTNET:
            self.exchange.set_sandbox_mode(True)

        state = load_state(STATE_FILE)
        self.open_trades:     list[Trade]     = state["open_trades"]
        self.closed_trades:   list[Trade]     = []
        self.active_fvgs:     dict[str, list] = state["active_fvgs"]
        self.trades_today:    dict[str, int]  = state["trades_today"]
        self.last_trade_date: date            = state["last_trade_date"]

        # Per-symbol buffers of raw OHLCV rows (maintained as rolling window)
        self._buffers:  dict[str, list]          = {s: [] for s in SYMBOLS}
        # HTF DataFrames — refreshed hourly
        self._htf_dfs:  dict[str, pd.DataFrame]  = {}
        self._htf_ts:   dict[str, float]         = {}
        # Locks prevent concurrent entry/manage calls for the same symbol
        self._locks:    dict[str, asyncio.Lock]  = {s: asyncio.Lock() for s in SYMBOLS}

        logger.info("=" * 60)
        logger.info("  FVG WebSocket Bot starting")
        logger.info(f"  Mode:    {'TESTNET' if TESTNET else '⚠ LIVE'}")
        logger.info(f"  Symbols: {', '.join(SYMBOLS)}")
        logger.info(f"  Streams: {PRIMARY_TF} kline per symbol")
        logger.info("=" * 60)

    # ── Initialisation ────────────────────────
    async def _boot(self):
        """Seed candle buffers and HTF data before opening streams."""
        for symbol in SYMBOLS:
            try:
                raw = await self.exchange.fetch_ohlcv(
                    symbol, PRIMARY_TF, limit=FVG_LOOKBACK
                )
                self._buffers[symbol] = list(raw)
                # Detect FVGs from historical data
                df  = _raw_to_df(raw)
                atr = calc_atr(df, ATR_PERIOD)
                self._refresh_fvgs(symbol, df, atr)
                await self._refresh_htf(symbol)
                logger.info(f"Booted {symbol} — {len(raw)} seed candles")
            except Exception as exc:
                logger.error(f"Boot failed for {symbol}: {exc}")

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
        new_fvgs       = detect_fvgs(df, symbol, PRIMARY_TF,
                                     min_size_pct=FVG_MIN_SIZE_PCT,
                                     sl_buffer_pct=SL_BUFFER_PCT)
        existing_times = {f.formed_at for f in self.active_fvgs[symbol]}

        for fvg in new_fvgs[:10]:
            if fvg.formed_at not in existing_times:
                fvg.score            = fvg_quality_score(fvg, df, atr)
                fvg.last_candle_time = df.index[-1]
                self.active_fvgs[symbol].append(fvg)
                logger.info(f"New FVG: {fvg}")

        # Prune, sort newest-first, cap
        self.active_fvgs[symbol] = sorted(
            [
                f for f in self.active_fvgs[symbol]
                if f.status == "waiting"
                and f.candles_since_formed < FVG_EXPIRY_CANDLES
            ],
            key=lambda x: x.formed_at, reverse=True
        )[:FVG_MAX_ACTIVE]

    def _can_trade(self, symbol: str) -> bool:
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

        atr = calc_atr(df, ATR_PERIOD)

        # Advance expiry counters by exactly one candle
        for fvg in self.active_fvgs[symbol]:
            fvg.candles_since_formed += 1
            fvg.last_candle_time      = df.index[-1]

        self._refresh_fvgs(symbol, df, atr)

        if self._can_trade(symbol):
            await self._check_entries(symbol, df, atr)

        await self._manage_trades(symbol, df, atr)
        save_state(STATE_FILE, self.open_trades, self.active_fvgs,
                   self.trades_today, self.last_trade_date)

    # ── Entry evaluation ──────────────────────
    async def _check_entries(self, symbol: str,
                              df: pd.DataFrame, atr: float) -> None:
        latest  = df.iloc[-1]
        c_high  = float(latest["high"])
        c_low   = float(latest["low"])
        c_close = float(latest["close"])
        bias    = self._htf_bias(symbol)

        for fvg in list(self.active_fvgs.get(symbol, [])):
            if is_fvg_invalidated(fvg, c_close):
                fvg.status = "invalidated"
                logger.info(f"FVG invalidated: {fvg}")
                continue

            if bias == "bullish" and fvg.direction == "bearish":
                continue
            if bias == "bearish" and fvg.direction == "bullish":
                continue
            if fvg.score < FVG_SCORE_MIN:
                continue
            if not check_retest(fvg, c_high, c_low, c_close):
                continue

            entry   = fvg.gap_mid
            sl_dist = fvg.sl_distance(entry)
            if sl_dist <= 0:
                continue
            if abs(fvg.tp_price(entry, MIN_RR) - entry) / sl_dist < MIN_RR:
                continue

            # Fetch balance via thread executor (sync CCXT call)
            loop = asyncio.get_event_loop()
            try:
                bal_data = await loop.run_in_executor(
                    None, self.exchange.fetch_balance
                )
                balance = float(bal_data.get("free", {}).get("USDT", 0))
            except Exception as exc:
                logger.warning(f"Balance fetch failed: {exc}")
                continue

            if balance <= 0:
                continue

            trade = open_trade(self.exchange, fvg, balance, dry_run=TESTNET)
            if trade:
                self.open_trades.append(trade)
                self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
                fvg.status = "retested"
                logger.info(f"ENTERED: {trade}")
                break  # one entry per candle close per symbol

    # ── Trade management ──────────────────────
    async def _manage_trades(self, symbol: str,
                              df: pd.DataFrame, atr: float) -> None:
        still_open = []

        for trade in self.open_trades:
            if trade.symbol != symbol:
                still_open.append(trade)
                continue

            latest  = df.iloc[-1]
            c_high  = float(latest["high"])
            c_low   = float(latest["low"])
            c_close = float(latest["close"])
            is_long = trade.direction == "long"
            sl_dist = abs(trade.entry_price - trade.sl_price)

            profit_r = (
                (c_close - trade.entry_price) / sl_dist if is_long
                else (trade.entry_price - c_close) / sl_dist
            )

            # Breakeven at 1R
            if profit_r >= 1.0 and not trade.be_moved:
                be = trade.entry_price
                trade.current_sl = (
                    max(trade.current_sl, be) if is_long
                    else min(trade.current_sl, be)
                )
                trade.be_moved = True
                logger.info(f"[BE] {symbol} SL → breakeven {be:.4f}")

            # ATR trail at 1.5R
            if profit_r >= 1.5:
                if is_long:
                    new_trail = c_low - atr * TRAIL_ATR_MULT
                    if new_trail > trade.current_sl:
                        trade.current_sl = new_trail
                else:
                    new_trail = c_high + atr * TRAIL_ATR_MULT
                    if new_trail < trade.current_sl:
                        trade.current_sl = new_trail

            # Partial exit at TP1 (2R)
            tp1         = trade.tp_prices[0]
            partial_hit = (is_long and c_high >= tp1) or \
                          (not is_long and c_low <= tp1)

            if partial_hit and not trade.partial_done:
                execute_partial_close(self.exchange, trade, tp1, dry_run=TESTNET)
                struct_tp = find_structure_tp(
                    df, "bullish" if is_long else "bearish",
                    trade.entry_price, sl_dist
                )
                trade.tp_prices = [struct_tp]
                logger.info(f"[PARTIAL] TP extended to structure {struct_tp:.4f}")

            # SL / TP check
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
                                     dry_run=TESTNET)
                self.closed_trades.append(trade)
                continue

            if tp_hit:
                trade = close_trade(self.exchange, trade, tp_final,
                                    "take_profit_structure", dry_run=TESTNET)
                self.closed_trades.append(trade)
                continue

            still_open.append(trade)

        self.open_trades = still_open

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
                f"SUMMARY | Closed={total} | Win%={wr:.1f}% | P&L=${pnl:.2f}"
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
                f"FINAL SUMMARY | Closed={total} | Win%={wr:.1f}% | P&L=${pnl:.2f}"
            )


if __name__ == "__main__":
    bot = FVGBotWS()
    asyncio.run(bot.run())
