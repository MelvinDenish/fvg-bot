#!/usr/bin/env python3
"""Analyze XRP backtest_trades_structure.csv for session/pattern insights."""
import pandas as pd

df = pd.read_csv("backtest_trades_structure.csv", parse_dates=["open_time", "close_time"])
print(f"Total trades: {len(df)}")
print(f"Date range: {df.open_time.min()} to {df.open_time.max()}")
print()

# Win/Loss breakdown
wins = df[df.result == "win"]
losses = df[df.result == "loss"]
print(f"Wins: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
print()

# Direction breakdown
for d in ["long", "short"]:
    sub = df[df.direction == d]
    w = sub[sub.result == "win"]
    wr = len(w) / len(sub) * 100 if len(sub) else 0
    avg = sub.pnl_usdt.mean()
    print(f"{d.upper()}: {len(sub)} trades, {len(w)} wins ({wr:.1f}% WR), avg PnL=${avg:.4f}")
print()

# Monthly breakdown
df["month"] = df.open_time.dt.to_period("M")
monthly = df.groupby("month").agg(
    trades=("pnl_usdt", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    pnl=("pnl_usdt", "sum"),
    fees=("fees_paid", "sum"),
).reset_index()
monthly["wr"] = (monthly.wins / monthly.trades * 100).round(1)

print("=== MONTHLY BREAKDOWN ===")
for _, r in monthly.iterrows():
    pnl_str = f"${r.pnl:12.2f}"
    fee_str = f"${r.fees:.2f}"
    print(f"  {r.month}  trades={int(r.trades):4d}  wins={int(r.wins):3d}  WR={r.wr:5.1f}%  PnL={pnl_str}  fees={fee_str}")
print()

# Hour of day analysis (UTC)
df["hour"] = df.open_time.dt.hour
hourly = df.groupby("hour").agg(
    trades=("pnl_usdt", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    pnl=("pnl_usdt", "sum"),
).reset_index()
hourly["wr"] = (hourly.wins / hourly.trades * 100).round(1)

print("=== TOP 5 HOURS by PnL (UTC) ===")
for _, r in hourly.nlargest(5, "pnl").iterrows():
    print(f"  {int(r.hour):02d}:00 UTC  trades={int(r.trades):3d}  WR={r.wr:.1f}%  PnL=${r.pnl:.2f}")
print()
print("=== WORST 3 HOURS (UTC) ===")
for _, r in hourly.nsmallest(3, "pnl").iterrows():
    print(f"  {int(r.hour):02d}:00 UTC  trades={int(r.trades):3d}  WR={r.wr:.1f}%  PnL=${r.pnl:.2f}")
print()

# Trade duration analysis
df["duration_min"] = (df.close_time - df.open_time).dt.total_seconds() / 60
w_dur = df.loc[df.result == "win", "duration_min"].mean()
l_dur = df.loc[df.result == "loss", "duration_min"].mean()
print("=== TRADE DURATION ===")
print(f"  Mean: {df.duration_min.mean():.1f} min")
print(f"  Median: {df.duration_min.median():.1f} min")
print(f"  Max: {df.duration_min.max():.1f} min")
print(f"  Avg WIN duration: {w_dur:.1f} min")
print(f"  Avg LOSS duration: {l_dur:.1f} min")
print()

# R:R distribution of wins
print("=== WIN R:R DISTRIBUTION ===")
if len(wins):
    for pct in [25, 50, 75, 90, 95]:
        val = wins.pnl_r.quantile(pct / 100)
        print(f"  P{pct}: {val:.2f}R")
    print(f"  Max win: {wins.pnl_r.max():.2f}R (${wins.pnl_usdt.max():.2f})")
print()

# Streak analysis
streaks = []
current = 0
for r in df.result:
    if r == "loss":
        current += 1
    else:
        if current > 0:
            streaks.append(current)
        current = 0
if current > 0:
    streaks.append(current)
print("=== LOSING STREAKS ===")
print(f"  Max: {max(streaks)} consecutive losses")
print(f"  Avg: {sum(streaks)/len(streaks):.1f} losses in a row")
print()

# Day of week
df["dow"] = df.open_time.dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print("=== DAY OF WEEK ===")
for d in dow_order:
    sub = df[df.dow == d]
    if len(sub) == 0:
        continue
    w = (sub.result == "win").sum()
    wr = w / len(sub) * 100
    pnl = sub.pnl_usdt.sum()
    print(f"  {d:10s}  trades={len(sub):3d}  WR={wr:.1f}%  PnL=${pnl:.2f}")
print()

# Session analysis (Asian/London/NY)
def get_session(hour):
    if 0 <= hour < 8:
        return "Asia (00-08 UTC)"
    elif 8 <= hour < 16:
        return "London (08-16 UTC)"
    else:
        return "New York (16-24 UTC)"

df["session"] = df.hour.apply(get_session)
print("=== SESSION BREAKDOWN ===")
for sess in ["Asia (00-08 UTC)", "London (08-16 UTC)", "New York (16-24 UTC)"]:
    sub = df[df.session == sess]
    if len(sub) == 0:
        continue
    w = (sub.result == "win").sum()
    wr = w / len(sub) * 100
    pnl = sub.pnl_usdt.sum()
    avg_rr = sub[sub.result == "win"].pnl_r.mean() if w > 0 else 0
    print(f"  {sess}:  trades={len(sub):4d}  WR={wr:.1f}%  PnL=${pnl:.2f}  avg_win_R={avg_rr:.1f}")
