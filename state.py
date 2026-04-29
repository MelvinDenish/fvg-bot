# ─────────────────────────────────────────────
#  FVG Trading Bot — State Persistence Helpers
#  Shared by bot.py (polling) and bot_ws.py (WebSocket)
# ─────────────────────────────────────────────
"""
Serialise / deserialise FVG and Trade objects to plain dicts so the
bot can write bot_state.json on every trade change and restore its
full state after a crash or restart — including open exchange orders.
"""

import json
import logging
from datetime import date, datetime, timezone
from collections import defaultdict
from typing import Any

import pandas as pd

from fvg_detector import FVG
# Trade is imported lazily inside functions to avoid a circular import
# (order_manager imports fvg_detector; state imports order_manager).

logger = logging.getLogger(__name__)


# ── FVG ──────────────────────────────────────

def serialize_fvg(fvg: FVG) -> dict:
    return {
        "symbol":               fvg.symbol,
        "timeframe":            fvg.timeframe,
        "direction":            fvg.direction,
        "gap_high":             fvg.gap_high,
        "gap_low":              fvg.gap_low,
        "sl_price":             fvg.sl_price,
        "formed_at":            fvg.formed_at.isoformat() if fvg.formed_at is not None else None,
        "formed_index":         fvg.formed_index,
        "status":               fvg.status,
        "entry_price":          fvg.entry_price,
        "candles_since_formed": fvg.candles_since_formed,
        "last_candle_time":     fvg.last_candle_time.isoformat() if fvg.last_candle_time else None,
        "score":                fvg.score,
    }


def deserialize_fvg(d: dict) -> FVG:
    formed_at = (
        pd.Timestamp(d["formed_at"])
        if d.get("formed_at") else pd.Timestamp.now(tz="UTC")
    )
    last_candle_time = (
        pd.Timestamp(d["last_candle_time"])
        if d.get("last_candle_time") else None
    )
    return FVG(
        symbol               = d["symbol"],
        timeframe            = d["timeframe"],
        direction            = d["direction"],
        gap_high             = d["gap_high"],
        gap_low              = d["gap_low"],
        sl_price             = d["sl_price"],
        formed_at            = formed_at,
        formed_index         = d.get("formed_index", 0),
        status               = d.get("status", "waiting"),
        entry_price          = d.get("entry_price"),
        candles_since_formed = d.get("candles_since_formed", 0),
        last_candle_time     = last_candle_time,
        score                = d.get("score", 0.0),
    )


# ── Trade ─────────────────────────────────────

def serialize_trade(trade: Any) -> dict:
    return {
        "symbol":        trade.symbol,
        "direction":     trade.direction,
        "entry_price":   trade.entry_price,
        "sl_price":      trade.sl_price,
        "tp_prices":     trade.tp_prices,
        "qty":           trade.qty,
        "risk_amount":   trade.risk_amount,
        "opened_at":     trade.opened_at.isoformat() if trade.opened_at else None,
        "status":        trade.status,
        "order_ids":     trade.order_ids,
        "current_sl":    trade.current_sl,
        "be_moved":      trade.be_moved,
        "partial_done":  trade.partial_done,
        "qty_remaining": trade.qty_remaining,
        "sl_order_id":   trade.sl_order_id,
        "tp_order_id":   trade.tp_order_id,
        "partial_pnl":   trade.partial_pnl,
        "fvg":           serialize_fvg(trade.fvg) if trade.fvg is not None else None,
    }


def deserialize_trade(d: dict) -> Any:
    from order_manager import Trade

    opened_at_raw = d.get("opened_at")
    opened_at = (
        datetime.fromisoformat(opened_at_raw)
        if opened_at_raw else datetime.now(timezone.utc)
    )
    fvg = deserialize_fvg(d["fvg"]) if d.get("fvg") else None

    return Trade(
        symbol        = d["symbol"],
        direction     = d["direction"],
        entry_price   = d["entry_price"],
        sl_price      = d["sl_price"],
        tp_prices     = d["tp_prices"],
        qty           = d["qty"],
        risk_amount   = d["risk_amount"],
        fvg           = fvg,
        opened_at     = opened_at,
        status        = d.get("status", "open"),
        order_ids     = d.get("order_ids", []),
        current_sl    = d.get("current_sl", d["sl_price"]),
        be_moved      = d.get("be_moved", False),
        partial_done  = d.get("partial_done", False),
        qty_remaining = d.get("qty_remaining", d["qty"]),
        sl_order_id   = d.get("sl_order_id", ""),
        tp_order_id   = d.get("tp_order_id", ""),
        partial_pnl   = d.get("partial_pnl", 0.0),
    )


# ── File I/O ──────────────────────────────────

def save_state(path: str,
               open_trades: list,
               active_fvgs: dict,
               trades_today: dict,
               last_trade_date: date) -> None:
    try:
        state = {
            "open_trades":     [serialize_trade(t) for t in open_trades],
            "active_fvgs":     {
                sym: [serialize_fvg(f) for f in fvgs]
                for sym, fvgs in active_fvgs.items()
            },
            "trades_today":    dict(trades_today),
            "last_trade_date": last_trade_date.isoformat(),
        }
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)
    except Exception as exc:
        logger.error(f"State save failed: {exc}")


def load_state(path: str) -> dict:
    """
    Returns a dict with keys:
      open_trades, active_fvgs, trades_today, last_trade_date
    Returns empty defaults on missing file or parse error.
    """
    empty = {
        "open_trades":     [],
        "active_fvgs":     defaultdict(list),
        "trades_today":    defaultdict(int),
        "last_trade_date": date.today(),
    }
    try:
        with open(path) as fh:
            raw = json.load(fh)

        open_trades = [deserialize_trade(t) for t in raw.get("open_trades", [])]

        active_fvgs: dict = defaultdict(list)
        for sym, fvgs in raw.get("active_fvgs", {}).items():
            active_fvgs[sym] = [deserialize_fvg(f) for f in fvgs]

        trades_today: dict = defaultdict(int, raw.get("trades_today", {}))

        date_str = raw.get("last_trade_date")
        last_trade_date = date.fromisoformat(date_str) if date_str else date.today()

        if open_trades:
            logger.info(f"Restored {len(open_trades)} open trade(s) from {path}")

        return {
            "open_trades":     open_trades,
            "active_fvgs":     active_fvgs,
            "trades_today":    trades_today,
            "last_trade_date": last_trade_date,
        }

    except FileNotFoundError:
        return empty
    except Exception as exc:
        logger.error(f"State load failed ({path}): {exc}")
        return empty
