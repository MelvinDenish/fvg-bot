#!/usr/bin/env python3
"""Debug: what does the exchange ACTUALLY return for positions and orders?"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")

from exchange import get_exchange
from state import load_state
from config import STATE_FILE

exchange = get_exchange()
state = load_state(STATE_FILE)

print("=== OPEN TRADES IN STATE ===")
for t in state["open_trades"]:
    print(f"  {t.symbol} {t.direction} | qty={t.qty} | sl_id={t.sl_order_id} | tp_id={t.tp_order_id}")

for t in state["open_trades"]:
    sym = t.symbol
    print(f"\n=== {sym} ===")

    # 1. fetch_positions
    print("\n  -- fetch_positions --")
    try:
        positions = exchange.fetch_positions([sym])
        print(f"  Got {len(positions)} position entries")
        for p in positions:
            psym = p.get("symbol")
            contracts = p.get("contracts")
            contractSize = p.get("contractSize")
            side = p.get("side")
            print(f"    symbol={psym} | contracts={contracts} (type={type(contracts).__name__}) | "
                  f"contractSize={contractSize} | side={side}")
            # Also check info
            info = p.get("info", {})
            print(f"    info.positionAmt={info.get('positionAmt')} | info.symbol={info.get('symbol')}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. fetch_order for TP
    if t.tp_order_id:
        print(f"\n  -- fetch_order(TP={t.tp_order_id}) --")
        try:
            o = exchange.fetch_order(t.tp_order_id, sym)
            print(f"    status={o.get('status')} | filled={o.get('filled')} | "
                  f"average={o.get('average')} | price={o.get('price')} | type={o.get('type')}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # 3. fetch_order for SL
    if t.sl_order_id:
        print(f"\n  -- fetch_order(SL={t.sl_order_id}) --")
        try:
            o = exchange.fetch_order(t.sl_order_id, sym)
            print(f"    status={o.get('status')} | filled={o.get('filled')} | "
                  f"average={o.get('average')} | price={o.get('price')} | type={o.get('type')}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # 4. _read_position_size result
    from bot import _read_position_size
    try:
        positions = exchange.fetch_positions([sym])
        ps = _read_position_size(positions, sym)
        print(f"\n  _read_position_size = {ps}")
        print(f"  trade.qty * 0.01 = {t.qty * 0.01}")
        print(f"  pos_size < threshold? {ps < t.qty * 0.01}")
    except Exception as e:
        print(f"\n  _read_position_size ERROR: {e}")

print("\nDone.")
