#!/usr/bin/env python3
"""Full test: place SL + TP, then cancel both using _cancel_all_for_symbol."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from exchange import get_exchange
exchange = get_exchange()

symbol = "BNB/USDT"
bsym = "BNBUSDT"

print("=" * 60)
print("TEST: Place SL + TP, then cancel both")
print("=" * 60)

# Step 1: Place a small BUY position first (need position for reduceOnly)
print("\n1. Opening small BNB LONG position...")
try:
    entry = exchange.create_order(symbol=symbol, type="market", side="buy", amount=0.1)
    print(f"   Opened @ {entry.get('average', entry.get('price'))}")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

time.sleep(1)

# Step 2: Place SL (stop_market) and TP (limit)
print("\n2. Placing SL (stop_market @ 600) and TP (limit @ 650)...")
sl_id, tp_id = None, None

try:
    sl = exchange.create_order(
        symbol=symbol, type="stop_market", side="sell",
        amount=0.1, params={"stopPrice": 600.0, "reduceOnly": True}
    )
    sl_id = sl["id"]
    print(f"   SL placed: id={sl_id}")
except Exception as e:
    print(f"   SL FAILED: {e}")

try:
    tp = exchange.create_order(
        symbol=symbol, type="limit", side="sell",
        amount=0.1, price=650.0, params={"reduceOnly": True}
    )
    tp_id = tp["id"]
    print(f"   TP placed: id={tp_id}")
except Exception as e:
    print(f"   TP FAILED: {e}")

time.sleep(1)

# Step 3: Check what's open
print("\n3. Checking open orders BEFORE cancel...")
try:
    algo = exchange.fapiPrivateGetOpenAlgoOrders()
    bnb_algo = [a for a in algo if a.get("symbol") == bsym]
    print(f"   Algo orders (SL): {len(bnb_algo)}")
    for a in bnb_algo:
        print(f"     {a['algoId']} | {a['orderType']} {a['side']} | status={a['algoStatus']}")
except Exception as e:
    print(f"   Algo list failed: {e}")

try:
    regular = exchange.fetch_open_orders(symbol)
    print(f"   Regular orders (TP): {len(regular)}")
    for o in regular:
        print(f"     {o['id']} | {o.get('type')} {o.get('side')} | status={o.get('status')}")
except Exception as e:
    print(f"   Regular list failed: {e}")

# Step 4: Cancel using _cancel_all_for_symbol
print("\n4. Running _cancel_all_for_symbol...")
from bot import _cancel_all_for_symbol
_cancel_all_for_symbol(exchange, symbol, sl_id=sl_id, tp_id=tp_id)
print("   Done!")

time.sleep(1)

# Step 5: Verify everything is gone
print("\n5. Checking open orders AFTER cancel...")
try:
    algo = exchange.fapiPrivateGetOpenAlgoOrders()
    bnb_algo = [a for a in algo if a.get("symbol") == bsym]
    print(f"   Algo orders (SL): {len(bnb_algo)}")
    for a in bnb_algo:
        print(f"     STILL OPEN: {a['algoId']} | {a['orderType']}")
except Exception as e:
    print(f"   Algo list failed: {e}")

try:
    regular = exchange.fetch_open_orders(symbol)
    print(f"   Regular orders (TP): {len(regular)}")
    for o in regular:
        print(f"     STILL OPEN: {o['id']} | {o.get('type')}")
except Exception as e:
    print(f"   Regular list failed: {e}")

# Step 6: Close the position
print("\n6. Closing test position...")
try:
    exchange.create_order(symbol=symbol, type="market", side="sell",
                          amount=0.1, params={"reduceOnly": True})
    print("   Position closed!")
except Exception as e:
    print(f"   Close failed: {e}")

# Final result
if bnb_algo or regular:
    print("\n FAIL - Orders still open!")
else:
    print("\n SUCCESS - ALL orders cancelled! SL + TP both cleaned up!")
