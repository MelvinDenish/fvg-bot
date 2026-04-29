#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Sanity check — places a tiny market order on Binance futures TESTNET,
#  reads it back, then closes it. Proves API auth + endpoint + order flow
#  before letting bot.py loose with live signals.
#
#  Run: python test_order.py [SYMBOL] [USD_NOTIONAL]
#       python test_order.py BTC/USDT 20
# ─────────────────────────────────────────────

import sys
import time
from exchange import get_exchange, get_account_balance, get_ticker_price
from config import TESTNET, BINANCE_DEMO, MARKET_TYPE


def main():
    symbol   = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    notional = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0   # $ notional (Binance min = $50)

    if BINANCE_DEMO:
        mode = "DEMO TRADING"
    elif TESTNET:
        mode = "TESTNET"
    else:
        mode = "LIVE"

    print(f"\n  Symbol:  {symbol}")
    print(f"  Mode:    {mode}  ({MARKET_TYPE})")
    print(f"  Test $:  ${notional:.2f} notional\n")

    if not (TESTNET or BINANCE_DEMO):
        confirm = input("  ⚠ LIVE mode — type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("  Aborted.")
            return

    ex = get_exchange(TESTNET)

    # Load markets so symbol → exchange-id mapping works
    print("  Loading markets...")
    ex.load_markets()

    # ── 1. Balance check ─────────────────────
    bal = get_account_balance(ex, "USDT")
    print(f"  USDT balance: ${bal:.2f}")
    if bal < notional:
        print(f"  ✗ Need at least ${notional} free. Top up the testnet account.")
        return

    # ── 2. Current price + qty calc ─────────
    price = get_ticker_price(ex, symbol)
    if price <= 0:
        print(f"  ✗ Could not fetch price for {symbol}")
        return
    qty = round(notional / price, 6)
    print(f"  Price: ${price:,.2f}  →  qty: {qty}\n")

    # ── 3. Place a market BUY ────────────────
    print("  Placing market BUY...")
    try:
        order = ex.create_order(symbol=symbol, type="market", side="buy", amount=qty)
        oid   = order.get("id")
        print(f"  ✓ Order placed   id={oid}")
    except Exception as e:
        print(f"  ✗ Order failed: {e}")
        return

    time.sleep(1.5)   # let exchange settle

    # ── 4. Fetch the order back ─────────────
    try:
        filled = ex.fetch_order(oid, symbol)
        avg    = filled.get("average") or filled.get("price")
        amt    = filled.get("filled")  or filled.get("amount")
        status = filled.get("status")
        print(f"  ✓ Fetched back   status={status}  avg=${avg}  filled={amt}")
    except Exception as e:
        print(f"  ⚠ fetch_order failed (may still be working): {e}")
        avg = price
        amt = qty

    # ── 5. Verify open position ─────────────
    try:
        positions = ex.fetch_positions([symbol])
        pos = next((p for p in positions if float(p.get("contracts") or 0) > 0), None)
        if pos:
            print(f"  ✓ Position open  size={pos['contracts']}  entry=${pos.get('entryPrice')}")
        else:
            print("  ⚠ No open position reported (may have already closed)")
    except Exception as e:
        print(f"  ⚠ fetch_positions failed: {e}")

    # ── 6. Close it (market SELL with reduceOnly) ──
    print("\n  Closing position...")
    try:
        close = ex.create_order(
            symbol=symbol, type="market", side="sell",
            amount=amt or qty, params={"reduceOnly": True}
        )
        print(f"  ✓ Close placed   id={close.get('id')}")
    except Exception as e:
        print(f"  ✗ Close failed: {e}")
        return

    time.sleep(1.5)

    # ── 7. Final balance ────────────────────
    bal_after = get_account_balance(ex, "USDT")
    delta     = bal_after - bal
    print(f"\n  USDT before: ${bal:.4f}")
    print(f"  USDT after:  ${bal_after:.4f}")
    print(f"  Delta:       ${delta:+.4f}  (fees + tiny price drift)\n")
    print("  ✓ END-TO-END OK — orders hit the exchange and settled.\n")


if __name__ == "__main__":
    main()
