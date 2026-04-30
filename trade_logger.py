"""
Persistent trade logger — writes every closed trade to trade_log.csv.
Import and call log_trade() from bot.py whenever a trade closes.
Call print_summary() to display a formatted report of all trades.
"""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).parent / "trade_log.csv"

HEADERS = [
    "id", "opened_at", "closed_at", "symbol", "direction",
    "entry_price", "exit_price", "sl_price", "tp_price",
    "qty", "risk_usdt", "pnl_usdt", "pnl_pct", "rr",
    "close_reason", "duration_min",
]


def _ensure_file():
    """Create the CSV with headers if it doesn't exist."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)


def log_trade(trade) -> None:
    """Append a closed trade to the CSV log."""
    _ensure_file()

    opened  = getattr(trade, "opened_at", None) or ""
    closed  = getattr(trade, "closed_at", None) or datetime.now(timezone.utc)
    tp0     = trade.tp_prices[0] if trade.tp_prices else 0.0
    rr      = getattr(trade, "rr_achieved", None) or 0.0

    # Duration in minutes
    dur = ""
    if opened and closed:
        try:
            delta = closed - opened
            dur = f"{delta.total_seconds() / 60:.1f}"
        except Exception:
            dur = ""

    # Auto-increment ID
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            trade_id = sum(1 for _ in f)  # header + rows → next id = count
    except Exception:
        trade_id = 1

    row = [
        trade_id,
        str(opened)[:19] if opened else "",
        str(closed)[:19] if closed else "",
        trade.symbol,
        trade.direction,
        f"{trade.entry_price:.4f}",
        f"{trade.exit_price:.4f}" if trade.exit_price else "",
        f"{trade.sl_price:.4f}",
        f"{tp0:.4f}",
        f"{trade.qty:.6f}",
        f"{trade.risk_amount:.4f}",
        f"{trade.pnl_usdt:.4f}",
        f"{trade.pnl_pct:.2f}",
        f"{rr:.2f}" if rr else "",
        trade.close_reason or "",
        dur,
    ]

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _calc_stats(pnl_list):
    """Compute stats dict from a list of P&L values."""
    wins   = [p for p in pnl_list if p >= 0]
    losses = [p for p in pnl_list if p < 0]
    total  = len(pnl_list)
    n_win  = len(wins)
    n_loss = len(losses)
    wr     = n_win / total * 100 if total else 0
    total_pnl = sum(pnl_list)
    avg_win   = sum(wins) / n_win if n_win else 0
    avg_loss  = sum(losses) / n_loss if n_loss else 0
    gross_w   = sum(wins)
    gross_l   = abs(sum(losses))
    pf        = gross_w / gross_l if gross_l > 0 else 0

    # Max drawdown
    peak = dd = running = 0.0
    for p in pnl_list:
        running += p
        if running > peak:
            peak = running
        this_dd = peak - running
        if this_dd > dd:
            dd = this_dd

    return {
        "total": total, "wins": n_win, "losses": n_loss,
        "wr": wr, "pnl": total_pnl, "avg_win": avg_win,
        "avg_loss": avg_loss, "pf": pf, "dd": dd,
        "exp": total_pnl / total if total else 0,
    }


def print_summary():
    """Print trade table, per-symbol breakdown, and overall summary."""
    _ensure_file()

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        trades = list(csv.DictReader(f))

    if not trades:
        print("\n  No trades recorded yet.\n")
        return

    # ── Trade Table ──
    all_pnl = []
    by_symbol = {}   # symbol → list of pnl values

    print()
    print("═" * 110)
    print("  DEMO TRADING LOG — ALL TRADES")
    print("═" * 110)
    print(f"  {'#':<4} {'Time':<18} {'Symbol':<12} {'Dir':<6} {'Entry':>10} {'Exit':>10} "
          f"{'P&L':>9} {'Risk%':>8} {'R:R':>7} {'Reason':<20} {'Dur':<8}")
    print("─" * 110)

    for t in trades:
        pnl     = float(t.get("pnl_usdt") or 0)
        pnl_pct = float(t.get("pnl_pct") or 0)
        rr      = t.get("rr") or ""
        reason  = t.get("close_reason") or ""
        dur     = t.get("duration_min") or ""
        dur_str = f"{float(dur):.0f}m" if dur else ""
        sym     = t.get("symbol") or "?"

        all_pnl.append(pnl)
        by_symbol.setdefault(sym, []).append(pnl)

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        closed_time = t.get("closed_at", "")[:16]

        print(f"  {t.get('id', ''):<4} {closed_time:<18} {sym:<12} "
              f"{t.get('direction', ''):<6} {t.get('entry_price', ''):>10} "
              f"{t.get('exit_price', ''):>10} {pnl_str:>9} "
              f"{pnl_pct:>7.1f}% {rr:>7} {reason:<20} {dur_str:<8}")

    print("─" * 110)

    # ── Per-Symbol Breakdown ──
    print()
    print("═" * 80)
    print("  PER-SYMBOL BREAKDOWN")
    print("═" * 80)
    print(f"  {'Symbol':<12} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'Win%':>7} "
          f"{'PF':>6} {'P&L':>10} {'AvgWin':>9} {'AvgLoss':>9} {'MaxDD':>9}")
    print("─" * 80)

    # Sort by P&L descending
    sorted_syms = sorted(by_symbol.items(), key=lambda x: sum(x[1]), reverse=True)

    for sym, pnl_list in sorted_syms:
        s = _calc_stats(pnl_list)
        sign = "+" if s["pnl"] >= 0 else ""
        print(f"  {sym:<12} {s['total']:>7} {s['wins']:>6} {s['losses']:>7} "
              f"{s['wr']:>6.1f}% {s['pf']:>6.2f} "
              f"{sign}${s['pnl']:>8.2f} "
              f"+${s['avg_win']:>7.2f} "
              f"-${abs(s['avg_loss']):>7.2f} "
              f"-${s['dd']:>7.2f}")

    print("─" * 80)

    # ── Overall Summary ──
    s = _calc_stats(all_pnl)
    print()
    print("═" * 60)
    print("  OVERALL SUMMARY")
    print("═" * 60)
    print(f"  Total Trades    {s['total']}")
    print(f"  Wins            {s['wins']}")
    print(f"  Losses          {s['losses']}")
    print(f"  Win Rate        {s['wr']:.1f}%")
    print(f"  Profit Factor   {s['pf']:.2f}")
    print("─" * 60)
    sign = "+" if s["pnl"] >= 0 else ""
    print(f"  Total P&L       {sign}${s['pnl']:.2f}")
    print(f"  Avg Win         +${s['avg_win']:.2f}" if s["wins"] else "  Avg Win         N/A")
    print(f"  Avg Loss        -${abs(s['avg_loss']):.2f}" if s["losses"] else "  Avg Loss        N/A")
    print(f"  Max Drawdown    -${s['dd']:.2f}")
    print(f"  Expectancy      ${s['exp']:.2f}/trade" if s["total"] else "")
    print("═" * 60)
    print()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print_summary()
