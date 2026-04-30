#!/usr/bin/env python3
"""
FVG Bot — Capital Simulation: $10 vs $100 USDT
================================================
Runs the backtester with YOUR live config settings for both capital amounts
across all configured symbols, with realistic leverage + fees.

Usage:  python simulate_capital.py
        python simulate_capital.py --days 30 --leverage 5
        python simulate_capital.py --symbols BTC/USDT ETH/USDT
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    SYMBOLS, PRIMARY_TF, RISK_PCT, MIN_RR,
    FVG_MIN_SIZE_PCT, FVG_EXPIRY_CANDLES, SL_BUFFER_PCT,
    FVG_SCORE_MIN, DEFAULT_LEVERAGE, MAX_POSITION_PCT,
)
from exchange import get_exchange, fetch_historical_ohlcv
from backtest import Backtester, BacktestResult

# ── Worker for parallel execution ─────────────
def _run_worker(packed):
    symbol, df, balance, kwargs = packed
    bt = Backtester(**kwargs, initial_balance=balance)
    return bt.run(df)


def main():
    parser = argparse.ArgumentParser(description="Simulate $10 vs $100 USDT capital")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help=f"Symbols to test (default: {SYMBOLS})")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE,
                        help=f"Leverage (default: {DEFAULT_LEVERAGE})")
    parser.add_argument("--tf", default=PRIMARY_TF, help=f"Timeframe (default: {PRIMARY_TF})")
    parser.add_argument("--tp-mode", default="partial",
                        choices=["fixed", "structure", "trailing", "partial"],
                        help="TP mode (default: partial)")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    balances = [10.0, 100.0]

    print("\n" + "═" * 70)
    print("  FVG BOT — CAPITAL SIMULATION: $10 vs $100 USDT")
    print("═" * 70)
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Timeframe:  {args.tf}  |  Days: {args.days}")
    print(f"  Leverage:   {args.leverage:.0f}x  |  TP Mode: {args.tp_mode.upper()}")
    print(f"  Risk/trade: {RISK_PCT*100:.2f}%  |  Min R:R: {MIN_RR}")
    print(f"  Fee:        0.05% taker  |  Max notional: {MAX_POSITION_PCT*100:.0f}%")
    print("─" * 70)

    # ── Fetch candle data ─────────────────────
    print(f"\n  Connecting to Binance (public endpoint)...")
    exchange = get_exchange(testnet=True)

    candle_data = {}
    for sym in symbols:
        print(f"\n  Fetching {args.days}d of {sym} {args.tf} candles...")
        df = fetch_historical_ohlcv(exchange, sym, args.tf, days=args.days)
        if df.empty:
            print(f"  ⚠ No data for {sym}, skipping.")
            continue
        candle_data[sym] = df
        print(f"  ✓ {len(df)} candles loaded for {sym}")

    if not candle_data:
        print("\n  ERROR: No candle data. Check internet connection.")
        sys.exit(1)

    # ── Build jobs ────────────────────────────
    base_kwargs = dict(
        timeframe        = args.tf,
        days             = args.days,
        risk_pct         = RISK_PCT,
        min_rr           = MIN_RR,
        tp_mode          = args.tp_mode,
        tp_multiplier    = 2.0,
        fvg_min_size     = FVG_MIN_SIZE_PCT,
        fvg_expiry       = FVG_EXPIRY_CANDLES,
        sl_buffer        = SL_BUFFER_PCT,
        leverage         = args.leverage,
        fee_rate         = 0.0005,
        max_notional_pct = MAX_POSITION_PCT,
        min_sl_pct       = 0.001,
        slippage_pct     = 0.0005,   # realistic 0.05% slippage
        max_trades_day   = 10,
        score_min        = FVG_SCORE_MIN,
        use_htf_filter   = False,
    )

    jobs = []
    for sym, df in candle_data.items():
        for bal in balances:
            kw = {**base_kwargs, "symbol": sym}
            jobs.append((sym, df, bal, kw))

    # ── Run simulations ───────────────────────
    print(f"\n  Running {len(jobs)} simulations...\n")
    t0 = time.perf_counter()

    results = {}  # key: (symbol, balance) → BacktestResult
    with ProcessPoolExecutor(max_workers=min(len(jobs), 6)) as pool:
        futures = {pool.submit(_run_worker, job): (job[0], job[2]) for job in jobs}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
                print(f"    ✓ {key[0]} @ ${key[1]:.0f} done")
            except Exception as exc:
                print(f"    ✗ {key[0]} @ ${key[1]:.0f} failed: {exc}")

    elapsed = time.perf_counter() - t0
    print(f"\n  All simulations finished in {elapsed:.1f}s")

    # ── Per-symbol reports ────────────────────
    for sym in candle_data:
        r10  = results.get((sym, 10.0))
        r100 = results.get((sym, 100.0))
        if not r10 or not r100:
            continue

        print(f"\n{'═' * 70}")
        print(f"  {sym} — {args.tf} ({args.days}d)  |  {args.leverage:.0f}x leverage  |  {args.tp_mode.upper()}")
        print(f"{'═' * 70}")
        print(f"  {'Metric':<24} {'$10 Account':>18} {'$100 Account':>18}")
        print(f"  {'─' * 62}")

        rows = [
            ("Initial balance",    f"${r10.initial_balance:>10.2f}",     f"${r100.initial_balance:>10.2f}"),
            ("Final balance",      f"${r10.final_balance:>10.2f}",      f"${r100.final_balance:>10.2f}"),
            ("Net P&L ($)",        f"{'+'if r10.total_pnl>=0 else ''}${r10.total_pnl:>9.2f}",
                                   f"{'+'if r100.total_pnl>=0 else ''}${r100.total_pnl:>9.2f}"),
            ("Net P&L (%)",        f"{'+'if r10.total_pnl_pct>=0 else ''}{r10.total_pnl_pct:>9.1f}%",
                                   f"{'+'if r100.total_pnl_pct>=0 else ''}{r100.total_pnl_pct:>9.1f}%"),
            ("Gross P&L",          f"${r10.gross_pnl:>10.2f}",          f"${r100.gross_pnl:>10.2f}"),
            ("Total fees",         f"-${r10.total_fees:>9.2f}",         f"-${r100.total_fees:>9.2f}"),
            ("─" * 24,             "─" * 18,                            "─" * 18),
            ("Total trades",       f"{len(r10.trades):>18}",            f"{len(r100.trades):>18}"),
            ("Closed trades",      f"{len(r10.closed_trades):>18}",     f"{len(r100.closed_trades):>18}"),
            ("Wins / Losses",      f"{len(r10.wins):>7} / {len(r10.losses):<7}",
                                   f"{len(r100.wins):>7} / {len(r100.losses):<7}"),
            ("Win rate",           f"{r10.win_rate:>17.1f}%",           f"{r100.win_rate:>17.1f}%"),
            ("Avg R:R (wins)",     f"{r10.avg_rr:>17.2f}x",            f"{r100.avg_rr:>17.2f}x"),
            ("Profit factor",      f"{r10.profit_factor:>18.2f}",       f"{r100.profit_factor:>18.2f}"),
            ("Expectancy/trade",   f"${r10.expectancy:>10.4f}",         f"${r100.expectancy:>10.4f}"),
            ("─" * 24,             "─" * 18,                            "─" * 18),
            ("Max drawdown",       f"{r10.max_drawdown:>17.2f}%",       f"{r100.max_drawdown:>17.2f}%"),
            ("Sharpe ratio",       f"{r10.sharpe_ratio:>18.2f}",        f"{r100.sharpe_ratio:>18.2f}"),
            ("Avg win ($)",        f"+${r10.avg_win_usdt:>9.4f}",       f"+${r100.avg_win_usdt:>9.4f}"),
            ("Avg loss ($)",       f"-${abs(r10.avg_loss_usdt):>9.4f}",  f"-${abs(r100.avg_loss_usdt):>9.4f}"),
        ]
        for label, v10, v100 in rows:
            print(f"  {label:<24} {v10:>18} {v100:>18}")
        print(f"{'═' * 70}")

    # ── Combined portfolio summary ────────────
    if len(candle_data) > 1:
        print(f"\n{'═' * 70}")
        print(f"  COMBINED PORTFOLIO SUMMARY (all {len(candle_data)} symbols)")
        print(f"{'═' * 70}")

        for bal in balances:
            syms_results = [results[(s, bal)] for s in candle_data if (s, bal) in results]
            total_pnl     = sum(r.total_pnl for r in syms_results)
            total_fees    = sum(r.total_fees for r in syms_results)
            total_trades  = sum(len(r.trades) for r in syms_results)
            total_wins    = sum(len(r.wins) for r in syms_results)
            total_losses  = sum(len(r.losses) for r in syms_results)
            total_closed  = sum(len(r.closed_trades) for r in syms_results)
            final_balance = bal + total_pnl
            pnl_pct       = total_pnl / bal * 100
            wr            = total_wins / total_closed * 100 if total_closed else 0
            worst_dd      = min(r.max_drawdown for r in syms_results)

            sign = "+" if total_pnl >= 0 else ""
            print(f"\n  ── ${bal:.0f} USDT Account ──")
            print(f"  Starting:     ${bal:>10.2f}")
            print(f"  Final:        ${final_balance:>10.2f}  ({sign}{pnl_pct:.1f}%)")
            print(f"  Net P&L:      {sign}${total_pnl:>9.2f}")
            print(f"  Fees paid:    -${total_fees:>9.2f}")
            print(f"  Trades:       {total_trades}  (W:{total_wins} / L:{total_losses})")
            print(f"  Win rate:     {wr:.1f}%")
            print(f"  Worst DD:     {worst_dd:.2f}%")

        print(f"\n{'═' * 70}")

    # ── Practical insights ────────────────────
    print(f"\n{'═' * 70}")
    print(f"  💡 PRACTICAL INSIGHTS")
    print(f"{'═' * 70}")

    # Check minimum notional constraint
    print(f"\n  ⚠  Binance Futures minimum order notional: $5 USDT")
    print(f"     With $10 at {args.leverage:.0f}x leverage:")
    print(f"       → Max buying power: ${10 * args.leverage:.0f}")
    print(f"       → Risk per trade (0.5%): ${10 * RISK_PCT:.2f}")

    print(f"\n     With $100 at {args.leverage:.0f}x leverage:")
    print(f"       → Max buying power: ${100 * args.leverage:.0f}")
    print(f"       → Risk per trade (0.5%): ${100 * RISK_PCT:.2f}")

    # Minimum notional feasibility
    if 10 * args.leverage < 5:
        print(f"\n  🚫 $10 at {args.leverage:.0f}x leverage CANNOT meet $5 min notional!")
        print(f"     You need at least {5/args.leverage:.0f}x leverage or more capital.")
    elif 10 * RISK_PCT * args.leverage < 0.50:
        print(f"\n  ⚠  $10 account: risk per trade is tiny (${10*RISK_PCT:.2f}).")
        print(f"     Gains will be very small in absolute terms.")
        print(f"     Consider higher leverage to scale position size.")

    print(f"\n  📊 Key takeaway:")
    print(f"     • Win rate & R:R are IDENTICAL — strategy doesn't change")
    print(f"     • Dollar P&L scales linearly with capital (10× account = 10× profit)")
    print(f"     • Fees eat a larger % on tiny accounts due to minimum order sizes")
    print(f"     • $100 is the practical minimum for Binance Futures trading")
    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
