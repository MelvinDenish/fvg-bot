# FVG Trading Bot — Setup & Usage Guide

## Strategy
- **Entry**: Fair Value Gap (FVG) retest on 1m/5m candles
- **Stop Loss**: Just below the previous candle low of the FVG, max 0.5% account risk
- **Take Profit**: Minimum 1:2 R:R (configurable)
- **Exchange**: Binance (spot + futures testnet supported)

---

## File Structure
```
fvg_bot/
├── bot.py            ← Live trading bot (run this)
├── backtest.py       ← Backtester with real Binance candles
├── config.py         ← All settings
├── fvg_detector.py   ← FVG detection engine
├── exchange.py       ← Binance connection + data fetcher
├── order_manager.py  ← Position sizing + order placement
├── .env.example      ← API key template
└── requirements.txt  ← Python dependencies
```

---

## Installation

```bash
# 1. Install dependencies
pip install ccxt pandas numpy python-dotenv requests

# 2. Set up API keys
cp .env.example .env
# Edit .env and paste your Binance API key and secret

# 3. Get Binance Testnet keys (free):
#    https://testnet.binancefuture.com → Register → API Management
```

---

## Run the Backtest (no API key needed)

```bash
# Default: BTC/USDT, 5m, 90 days, $1000 balance
python backtest.py

# Custom settings
python backtest.py --symbol ETH/USDT --tf 1m --days 60 --balance 5000 --rr 2.5

# All options:
#  --symbol    BTC/USDT | ETH/USDT | SOL/USDT ...
#  --tf        1m | 5m | 15m
#  --days      Days of history (max ~200 for 1m, unlimited for 5m+)
#  --balance   Starting balance in USDT
#  --risk      Risk per trade in % (default 0.5)
#  --rr        Min R:R ratio (default 2.0)
#  --tp        TP multiplier (default 2.0)
#  --fvg-size  Min FVG size as % of price (default 0.1)
#  --no-save   Don't write CSV/JSON output files
```

### Backtest Output Files
- `backtest_trades.csv` — Every trade with entry/exit/P&L
- `backtest_summary.json` — Aggregate stats + equity curve

---

## Run the Live Bot

```bash
# Set TESTNET=true in .env first (paper trading)
python bot.py

# To go live: set TESTNET=false in .env (⚠ real money)
```

---

## Key Settings (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `RISK_PCT` | 0.005 | 0.5% account risk per trade |
| `MIN_RR` | 2.0 | Minimum reward-to-risk ratio |
| `MAX_TRADES_DAY` | 10 | Max trades per symbol per day |
| `MAX_OPEN_TRADES` | 3 | Max simultaneous open positions |
| `FVG_MIN_SIZE_PCT` | 0.001 | Minimum FVG gap size (filters noise) |
| `FVG_EXPIRY_CANDLES` | 20 | FVG expires if not retested within N candles |
| `TP_MULTIPLIERS` | [2.0, 3.0] | Scale out at 2R and 3R |
| `PRIMARY_TF` | 5m | Main timeframe for FVG detection |
| `ENTRY_TF` | 1m | Finer timeframe for entry confirmation |

---

## Safety Notes
1. Always test on **TESTNET** first
2. Start with a small balance (e.g. $100) when going live
3. Never risk money you can't afford to lose
4. The bot uses **market orders** for entry — adjust to limit orders in `order_manager.py` for lower slippage
5. Enable 2FA on your Binance account and use **IP-restricted API keys**
