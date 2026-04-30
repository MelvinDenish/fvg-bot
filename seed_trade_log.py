"""Fetch actual trade history from Binance Demo and rebuild trade_log.csv."""
import sys, csv
sys.stdout.reconfigure(encoding="utf-8")

from exchange import get_exchange
from config import SYMBOLS
from pathlib import Path
from datetime import datetime, timezone

exchange = get_exchange()
CSV_FILE = Path(__file__).parent / "trade_log.csv"

HEADERS = [
    "id", "opened_at", "closed_at", "symbol", "direction",
    "entry_price", "exit_price", "sl_price", "tp_price",
    "qty", "risk_usdt", "pnl_usdt", "pnl_pct", "rr",
    "close_reason", "duration_min",
]

print("Fetching trade history from Binance Demo...\n")

all_trades = []
for symbol in SYMBOLS:
    bsym = symbol.replace("/", "")
    print(f"── {symbol} ──")
    
    # Fetch all trades (fills) from API
    try:
        # Sync exchange time first
        exchange.load_time_difference()
        raw = exchange.fetch_my_trades(symbol, limit=500)
        print(f"  {len(raw)} fills found")
        for t in raw:
            all_trades.append({
                "symbol": symbol,
                "time": t["timestamp"],
                "side": t["side"].upper(),
                "price": float(t["price"]),
                "qty": float(t["amount"]),
                "realizedPnl": float(t["info"].get("realizedPnl", 0)),
                "commission": float(t["info"].get("commission", 0)),
                "orderId": t["order"],
            })
    except Exception as e:
        print(f"  Error: {e}")
        continue

all_trades.sort(key=lambda x: x["time"])

print(f"\nTotal fills across all symbols: {len(all_trades)}")
print()

# Group fills into trades (entry + exit pairs)
# Strategy: entry has realizedPnl=0 (opening), exit has realizedPnl!=0 (closing)
open_positions = {}  # symbol -> entry info
closed_trades = []
trade_id = 1

for fill in all_trades:
    sym = fill["symbol"]
    pnl = fill["realizedPnl"]
    fee = fill["commission"]
    ts  = datetime.fromtimestamp(fill["time"] / 1000, tz=timezone.utc)
    
    if abs(pnl) < 0.0001 and sym not in open_positions:
        # This is an entry fill
        direction = "long" if fill["side"] == "BUY" else "short"
        open_positions[sym] = {
            "opened_at": ts,
            "direction": direction,
            "entry_price": fill["price"],
            "qty": fill["qty"],
            "entry_fee": fee,
        }
    elif sym in open_positions:
        # This is an exit fill (or additional fill on existing position)
        if abs(pnl) > 0.0001:
            entry = open_positions.pop(sym)
            net_pnl = pnl - fee - entry["entry_fee"]
            duration = (ts - entry["opened_at"]).total_seconds() / 60
            
            # Determine reason from P&L direction
            is_win = net_pnl > 0
            reason = "take_profit" if is_win else "stop_loss"
            
            closed_trades.append({
                "id": trade_id,
                "opened_at": str(entry["opened_at"])[:19],
                "closed_at": str(ts)[:19],
                "symbol": sym,
                "direction": entry["direction"],
                "entry_price": f"{entry['entry_price']:.4f}",
                "exit_price": f"{fill['price']:.4f}",
                "sl_price": "",
                "tp_price": "",
                "qty": f"{entry['qty']:.6f}",
                "risk_usdt": "",
                "pnl_usdt": f"{net_pnl:.4f}",
                "pnl_pct": "",
                "rr": "",
                "close_reason": reason,
                "duration_min": f"{duration:.1f}",
            })
            trade_id += 1

# Write CSV
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=HEADERS)
    w.writeheader()
    w.writerows(closed_trades)

print(f"Written {len(closed_trades)} closed trades to {CSV_FILE.name}")
if open_positions:
    print(f"Still open: {list(open_positions.keys())}")

# Print summary table
print()
print("═" * 110)
print("  BINANCE DEMO — ACTUAL TRADE HISTORY")
print("═" * 110)
print(f"  {'#':<4} {'Opened':<20} {'Closed':<20} {'Symbol':<12} {'Dir':<6} "
      f"{'Entry':>10} {'Exit':>10} {'P&L':>10} {'Reason':<15} {'Dur':<8}")
print("─" * 110)

total_pnl = 0
wins = 0
losses = 0
for t in closed_trades:
    pnl = float(t["pnl_usdt"])
    total_pnl += pnl
    if pnl >= 0:
        wins += 1
        pnl_str = f"+${pnl:.2f}"
    else:
        losses += 1
        pnl_str = f"-${abs(pnl):.2f}"
    
    dur = float(t["duration_min"]) if t["duration_min"] else 0
    print(f"  {t['id']:<4} {t['opened_at']:<20} {t['closed_at']:<20} {t['symbol']:<12} "
          f"{t['direction']:<6} {t['entry_price']:>10} {t['exit_price']:>10} "
          f"{pnl_str:>10} {t['close_reason']:<15} {dur:.0f}m")

total = wins + losses
wr = wins / total * 100 if total else 0
print("─" * 110)
print()
print(f"  Total: {total} trades | Wins: {wins} | Losses: {losses} | "
      f"Win Rate: {wr:.1f}% | Net P&L: ${total_pnl:.2f}")
print("═" * 110)
