"""
Analyze backtest trade patterns across all top symbols.
Run all backtests first, then this script reads the CSVs.
"""
import sys, csv, os
import pandas as pd
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

# Run backtests for all symbols and collect results
from exchange import get_exchange, fetch_historical_ohlcv
from backtest import Backtester

SYMBOLS = ["XRP/USDT", "DOGE/USDT", "LINK/USDT", "SOL/USDT", "ETH/USDT", "BNB/USDT"]
all_trades = []

print("Running backtests for all symbols...\n")

for sym in SYMBOLS:
    print(f"  {sym}...", end=" ", flush=True)
    exchange = get_exchange()
    df = fetch_historical_ohlcv(exchange, sym, "5m", 400)
    bt = Backtester(
        symbol=sym, timeframe="5m", days=400, initial_balance=100,
        risk_pct=0.005, min_rr=3.1, tp_mode="structure",
        leverage=5.0, fee_rate=0.0005, max_notional_pct=0.20
    )
    bt.run(df)
    
    for t in bt.trades:
        if t.result == "open":
            continue
        open_time = pd.Timestamp(t.open_time)
        close_time = pd.Timestamp(t.close_time) if t.close_time else open_time
        duration = (close_time - open_time).total_seconds() / 60
        
        all_trades.append({
            "symbol": sym,
            "direction": t.direction,
            "result": t.result,
            "open_hour": open_time.hour,
            "open_dow": open_time.dayofweek,  # 0=Mon, 6=Sun
            "pnl": t.pnl_usdt,
            "pnl_r": t.pnl_r,
            "duration_min": duration,
            "entry": t.entry_price,
            "sl": t.sl_price,
            "tp": t.tp_price,
            "sl_dist_pct": abs(t.entry_price - t.sl_price) / t.entry_price * 100 if t.entry_price else 0,
        })
    
    wins = sum(1 for t in bt.trades if t.result == "win")
    total = sum(1 for t in bt.trades if t.result != "open")
    print(f"{total} trades, {wins} wins ({wins/total*100:.1f}%)")

df = pd.DataFrame(all_trades)
print(f"\nTotal trades across all symbols: {len(df)}")

# ═══════════════════════════════════════════════
# ANALYSIS 1: Direction (Long vs Short)
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  1. DIRECTION ANALYSIS (Long vs Short)")
print("═" * 70)

for direction in ["long", "short"]:
    sub = df[df["direction"] == direction]
    wins = sub[sub["result"] == "win"]
    losses = sub[sub["result"] == "loss"]
    wr = len(wins) / len(sub) * 100 if len(sub) else 0
    pnl = sub["pnl"].sum()
    gw = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else 0
    print(f"  {direction.upper():<6} | {len(sub):>5} trades | WR {wr:>5.1f}% | "
          f"PF {pf:.2f} | P&L ${pnl:>10.2f} | AvgWin ${wins['pnl'].mean():>7.2f} | AvgLoss ${losses['pnl'].mean():>7.2f}")

# Per symbol direction
print("\n  Per-Symbol Direction:")
print(f"  {'Symbol':<12} {'L-Trades':>8} {'L-WR%':>7} {'L-PF':>6} {'S-Trades':>8} {'S-WR%':>7} {'S-PF':>6} {'Better':>8}")
print("  " + "─" * 65)
for sym in SYMBOLS:
    s = df[df["symbol"] == sym]
    for d, label in [("long", "L"), ("short", "S")]:
        sub = s[s["direction"] == d]
        w = sub[sub["result"] == "win"]
        l = sub[sub["result"] == "loss"]
        locals()[f"{label}_n"] = len(sub)
        locals()[f"{label}_wr"] = len(w) / len(sub) * 100 if len(sub) else 0
        gw = w["pnl"].sum(); gl = abs(l["pnl"].sum())
        locals()[f"{label}_pf"] = gw / gl if gl > 0 else 0
    better = "LONG" if L_pf > S_pf else "SHORT"
    print(f"  {sym:<12} {L_n:>8} {L_wr:>6.1f}% {L_pf:>6.2f} {S_n:>8} {S_wr:>6.1f}% {S_pf:>6.2f} {better:>8}")

# ═══════════════════════════════════════════════
# ANALYSIS 2: Hour of Day
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  2. HOUR-OF-DAY ANALYSIS (UTC)")
print("═" * 70)

print(f"  {'Hour':>4} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PF':>6} {'P&L':>12} {'Verdict':>10}")
print("  " + "─" * 55)

best_hours = []
for h in range(24):
    sub = df[df["open_hour"] == h]
    if len(sub) < 20:
        continue
    wins = sub[sub["result"] == "win"]
    losses = sub[sub["result"] == "loss"]
    wr = len(wins) / len(sub) * 100
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else 0
    pnl = sub["pnl"].sum()
    verdict = "✅ GOOD" if pf > 1.5 else ("⚠️ OK" if pf > 1.0 else "❌ BAD")
    best_hours.append((h, pf, pnl))
    print(f"  {h:>4} {len(sub):>7} {len(wins):>6} {wr:>6.1f}% {pf:>6.2f} ${pnl:>10.2f} {verdict:>10}")

# ═══════════════════════════════════════════════
# ANALYSIS 3: Day of Week
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  3. DAY-OF-WEEK ANALYSIS")
print("═" * 70)

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print(f"  {'Day':>4} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PF':>6} {'P&L':>12} {'Verdict':>10}")
print("  " + "─" * 55)
for dow in range(7):
    sub = df[df["open_dow"] == dow]
    if len(sub) < 10:
        continue
    wins = sub[sub["result"] == "win"]
    losses = sub[sub["result"] == "loss"]
    wr = len(wins) / len(sub) * 100
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else 0
    pnl = sub["pnl"].sum()
    verdict = "✅ GOOD" if pf > 1.5 else ("⚠️ OK" if pf > 1.0 else "❌ BAD")
    print(f"  {days[dow]:>4} {len(sub):>7} {len(wins):>6} {wr:>6.1f}% {pf:>6.2f} ${pnl:>10.2f} {verdict:>10}")

# ═══════════════════════════════════════════════
# ANALYSIS 4: Trade Duration
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  4. TRADE DURATION ANALYSIS")
print("═" * 70)

bins = [(0, 5, "0-5m"), (5, 15, "5-15m"), (15, 60, "15-60m"), (60, 240, "1-4h"), (240, 1440, "4-24h"), (1440, 99999, "24h+")]
print(f"  {'Duration':>8} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PF':>6} {'P&L':>12}")
print("  " + "─" * 50)
for lo, hi, label in bins:
    sub = df[(df["duration_min"] >= lo) & (df["duration_min"] < hi)]
    if len(sub) < 10:
        continue
    wins = sub[sub["result"] == "win"]
    losses = sub[sub["result"] == "loss"]
    wr = len(wins) / len(sub) * 100
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else 0
    pnl = sub["pnl"].sum()
    print(f"  {label:>8} {len(sub):>7} {len(wins):>6} {wr:>6.1f}% {pf:>6.2f} ${pnl:>10.2f}")

# ═══════════════════════════════════════════════
# ANALYSIS 5: SL Distance
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  5. STOP-LOSS DISTANCE ANALYSIS")
print("═" * 70)

sl_bins = [(0, 0.1, "<0.1%"), (0.1, 0.2, "0.1-0.2%"), (0.2, 0.5, "0.2-0.5%"), (0.5, 1.0, "0.5-1%"), (1.0, 2.0, "1-2%"), (2.0, 100, "2%+")]
print(f"  {'SL Dist':>8} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PF':>6} {'P&L':>12}")
print("  " + "─" * 50)
for lo, hi, label in sl_bins:
    sub = df[(df["sl_dist_pct"] >= lo) & (df["sl_dist_pct"] < hi)]
    if len(sub) < 10:
        continue
    wins = sub[sub["result"] == "win"]
    losses = sub[sub["result"] == "loss"]
    wr = len(wins) / len(sub) * 100
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else 0
    pnl = sub["pnl"].sum()
    print(f"  {label:>8} {len(sub):>7} {len(wins):>6} {wr:>6.1f}% {pf:>6.2f} ${pnl:>10.2f}")

# ═══════════════════════════════════════════════
# ANALYSIS 6: Consecutive Losses (Losing Streaks)
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  6. LOSING STREAK ANALYSIS")
print("═" * 70)

streaks = []
current = 0
for _, row in df.iterrows():
    if row["result"] == "loss":
        current += 1
    else:
        if current > 0:
            streaks.append(current)
        current = 0
if current > 0:
    streaks.append(current)

if streaks:
    print(f"  Total losing streaks: {len(streaks)}")
    print(f"  Avg streak length:    {np.mean(streaks):.1f}")
    print(f"  Max streak:           {max(streaks)}")
    print(f"  Streaks of 5+:        {sum(1 for s in streaks if s >= 5)}")
    print(f"  Streaks of 10+:       {sum(1 for s in streaks if s >= 10)}")
    print(f"  Streaks of 15+:       {sum(1 for s in streaks if s >= 15)}")

# ═══════════════════════════════════════════════
# ANALYSIS 7: Win R:R Distribution
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  7. WIN R:R DISTRIBUTION")
print("═" * 70)

wins_df = df[df["result"] == "win"]
rr_bins = [(3, 4, "3-4R"), (4, 6, "4-6R"), (6, 10, "6-10R"), (10, 20, "10-20R"), (20, 100, "20R+")]
print(f"  {'R:R':>8} {'Count':>7} {'% of wins':>10} {'Avg P&L':>10}")
print("  " + "─" * 40)
for lo, hi, label in rr_bins:
    sub = wins_df[(wins_df["pnl_r"] >= lo) & (wins_df["pnl_r"] < hi)]
    pct = len(sub) / len(wins_df) * 100 if len(wins_df) else 0
    avg = sub["pnl"].mean() if len(sub) else 0
    print(f"  {label:>8} {len(sub):>7} {pct:>9.1f}% ${avg:>9.2f}")

# ═══════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════
print("\n" + "═" * 70)
print("  STRATEGY IMPROVEMENT RECOMMENDATIONS")
print("═" * 70)

# Best/worst hours
best_hours.sort(key=lambda x: x[1], reverse=True)
good_h = [h for h, pf, _ in best_hours if pf > 1.5]
bad_h  = [h for h, pf, _ in best_hours if pf < 1.0]

if good_h:
    print(f"\n  ✅ Best hours (UTC):  {', '.join(f'{h}:00' for h in sorted(good_h))}")
if bad_h:
    print(f"  ❌ Worst hours (UTC): {', '.join(f'{h}:00' for h in sorted(bad_h))}")

# Direction recommendation
long_sub = df[df["direction"] == "long"]
short_sub = df[df["direction"] == "short"]
l_pf = long_sub[long_sub["result"]=="win"]["pnl"].sum() / abs(long_sub[long_sub["result"]=="loss"]["pnl"].sum()) if abs(long_sub[long_sub["result"]=="loss"]["pnl"].sum()) > 0 else 0
s_pf = short_sub[short_sub["result"]=="win"]["pnl"].sum() / abs(short_sub[short_sub["result"]=="loss"]["pnl"].sum()) if abs(short_sub[short_sub["result"]=="loss"]["pnl"].sum()) > 0 else 0
print(f"\n  Direction: Long PF={l_pf:.2f} vs Short PF={s_pf:.2f}")
if abs(l_pf - s_pf) > 0.3:
    better = "LONG" if l_pf > s_pf else "SHORT"
    print(f"  💡 Consider {better}-only mode for stronger edge")

# Duration
short_trades = df[df["duration_min"] < 5]
if len(short_trades) > 50:
    short_wr = len(short_trades[short_trades["result"]=="win"]) / len(short_trades) * 100
    print(f"\n  ⚠️  {len(short_trades)} trades close in <5 min (WR {short_wr:.1f}%)")
    if short_wr < 25:
        print(f"  💡 Consider minimum hold time to filter instant SL hits")

print()
