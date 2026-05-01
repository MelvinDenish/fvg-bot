"""Compare MIN_SL_PCT = 0.1% vs 0.2% across all symbols."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from exchange import get_exchange, fetch_historical_ohlcv
from backtest import Backtester

exchange = get_exchange()
symbols = ["XRP/USDT", "DOGE/USDT", "LINK/USDT", "SOL/USDT", "ETH/USDT", "BNB/USDT"]

print("=" * 85)
print("  MIN_SL_PCT COMPARISON: 0.1% (current) vs 0.2% (proposed)")
print("=" * 85)
print(f"  {'Symbol':<12} {'SL%':<6} {'Trades':>7} {'WR%':>7} {'PF':>6} {'Final$':>10} {'Sharpe':>7} {'MaxDD':>7}")
print("-" * 85)

totals = {0.001: {"final": 0.0, "trades": 0}, 0.002: {"final": 0.0, "trades": 0}}

for sym in symbols:
    print(f"  Fetching {sym}...", end=" ", flush=True)
    df = fetch_historical_ohlcv(exchange, sym, "5m", 400)
    print("OK")
    for sl_pct in [0.001, 0.002]:
        bt = Backtester(
            symbol=sym, timeframe="5m", days=400, initial_balance=100,
            risk_pct=0.005, min_rr=3.1, tp_mode="structure",
            leverage=5.0, fee_rate=0.0005, max_notional_pct=0.20,
            min_sl_pct=sl_pct
        )
        result = bt.run(df)  # returns BacktestResult
        label = "0.1%" if sl_pct == 0.001 else "0.2%"
        n_closed = len(result.closed_trades)
        final = result.final_balance
        totals[sl_pct]["final"] += final
        totals[sl_pct]["trades"] += n_closed
        print(f"  {sym:<12} {label:<6} {n_closed:>7} {result.win_rate:>6.1f}% "
              f"{result.profit_factor:>6.2f} ${final:>9.2f} {result.sharpe_ratio:>7.2f} "
              f"{result.max_drawdown:>6.2f}%")
    print()

print("=" * 85)
print("  TOTALS (combined across all symbols, starting $100 each)")
print("-" * 85)
for sl_pct in [0.001, 0.002]:
    label = "0.1%" if sl_pct == 0.001 else "0.2%"
    t = totals[sl_pct]
    pnl = t["final"] - 600  # 6 symbols x $100
    print(f"  {label:<6} Trades: {t['trades']:>6}  |  Final: ${t['final']:>12.2f}  |  Net P/L: ${pnl:>10.2f}")

better = "0.2%" if totals[0.002]["final"] > totals[0.001]["final"] else "0.1% (KEEP CURRENT)"
print(f"\n  VERDICT: {better}")
print("=" * 85)
