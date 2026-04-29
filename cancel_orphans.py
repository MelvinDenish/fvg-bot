#!/usr/bin/env python3
"""Force cancel all conditional orders using different API methods."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from exchange import get_exchange
from config import SYMBOLS

exchange = get_exchange()

# Method 1: cancel_all_orders (DELETE /fapi/v1/allOpenOrders)
for symbol in SYMBOLS:
    bsym = symbol.replace("/", "")
    print(f"\n--- {symbol} ---")
    
    # Try cancel_all_orders
    try:
        result = exchange.cancel_all_orders(symbol)
        print(f"  cancel_all_orders: {result}")
    except Exception as e:
        print(f"  cancel_all_orders: {e}")
    
    # Try raw DELETE endpoint
    try:
        result = exchange.fapiPrivateDeleteAllOpenOrders({"symbol": bsym})
        print(f"  fapiPrivateDeleteAllOpenOrders: {result}")
    except Exception as e:
        print(f"  raw DELETE: {e}")

    # Try listing via different endpoints
    for method_name in ["fapiPrivateGetOpenOrders", "fapiPrivateGetAllOrders"]:
        try:
            method = getattr(exchange, method_name, None)
            if method:
                result = method({"symbol": bsym, "limit": 20})
                open_ones = [o for o in result if o.get("status") == "NEW"]
                print(f"  {method_name}: {len(result)} total, {len(open_ones)} NEW")
                for o in open_ones[-5:]:
                    print(f"    id={o.get('orderId')} type={o.get('type')} side={o.get('side')} qty={o.get('origQty')} stop={o.get('stopPrice')}")
        except Exception as e:
            print(f"  {method_name}: {e}")

print("\nDone. Check the exchange UI to see if orders are gone.")
