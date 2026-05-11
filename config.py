# ─────────────────────────────────────────────
#  FVG Trading Bot — Configuration
# ─────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()

# ── Binance API keys (set in .env file) ──────
API_KEY     = os.getenv("BINANCE_API_KEY", "")
API_SECRET  = os.getenv("BINANCE_API_SECRET", "")
TESTNET      = os.getenv("TESTNET", "false").lower() == "true"   # spot-only now (futures testnet was deprecated)
BINANCE_DEMO = os.getenv("BINANCE_DEMO", "true").lower() == "true"  # futures Demo Trading — live endpoint + demo keys
DRY_RUN      = os.getenv("DRY_RUN", "true").lower() == "true"    # log orders without exchange calls
MARKET_TYPE  = os.getenv("MARKET_TYPE", "future")                 # "future" | "spot" — order_manager uses reduceOnly + stop_market (futures-only)

# ── Trading pairs & timeframes ───────────────
# Backtest-validated rankings (400d, 1m, structure TP, RR 5.0):
#   BNB  PF4.00 S6.25 DD-3.3% | XRP  PF3.14 S7.30 DD-4.7%
#   ETH  PF2.41 S6.36 DD-6.6% | DOGE PF2.59 S5.21 DD-6.0%
#   ADA  PF2.59 S5.18 DD-12%  | SOL  PF2.40 S4.17 DD-8.8%
SYMBOLS    = ["BNB/USDT", "XRP/USDT", "ETH/USDT", "DOGE/USDT", "ADA/USDT", "SOL/USDT"]
PRIMARY_TF = "1m"       # FVG detection timeframe (1m > 5m on all pairs)
ENTRY_TF   = "1m"       # entry precision / trade management
HTF_TF     = "1h"       # higher-timeframe trend filter

# ── Risk management ───────────────────────────
RISK_PCT        = 0.005   # 0.5% account risk per trade
MIN_RR          = 5.0    # minimum NET reward-to-risk ratio after fees (higher = fewer but better trades)
MAX_TRADES_DAY  = 10      # max trades per symbol per day
MAX_OPEN_TRADES = 3       # max simultaneously open positions
MAX_CORRELATED  = 4       # max open altcoin positions (limits correlated DD stacking)
                           # BTC/USDT is excluded from this cap

# ── Exchange minimums & leverage scaling ──────
MIN_NOTIONAL_USDT = 50.0   # Binance USDT-M futures min notional per opening order
MAX_LEVERAGE      = 20     # hard cap on auto-leverage
DEFAULT_LEVERAGE  = 5      # starting leverage; auto-bumps if needed to fit margin
MARGIN_MODE       = "isolated"  # "isolated" | "cross" — isolated limits loss per position
MAX_POSITION_PCT  = 0.20   # max margin per position as % of balance (caps drawdown
                            # if 3 positions all hit SL = 3 × 0.5% = 1.5% account loss)

# ── FVG detection ─────────────────────────────
FVG_MIN_SIZE_PCT   = 0.001   # min gap size as % of price (filters micro-gaps)
FVG_LOOKBACK       = 100     # candles fetched for PRIMARY_TF detection
FVG_EXPIRY_CANDLES = 20      # FVG expires after N PRIMARY_TF candles (not loop ticks)
FVG_MAX_ACTIVE     = 3       # hard cap on active FVGs stored per symbol
FVG_SCORE_MIN      = 0.0     # quality score gate (0 = disabled — backtest showed
                              # filter cost +200%+ profit to save 2-3% DD; bad trade)
USE_HTF_FILTER     = False   # 1h EMA-50 trend bias gate (disabled for same reason)

# ── Stop loss & take profit ───────────────────
SL_BUFFER_PCT  = 0.0002      # extra buffer below/above middle-candle low/high for SL
TP_MULTIPLIERS = [2.0, 3.0]  # partial mode only: 50% at 2R, 50% at 3R
TP_MODE        = os.getenv("TP_MODE", "structure")   # "structure" | "partial"
                              # structure = single TP at nearest swing, no BE/trail/partial
                              # partial   = 50% at 2R + BE move + ATR trail to structure

# ── Higher-timeframe trend filter ────────────
HTF_EMA_PERIOD = 50          # EMA period on HTF_TF for bias detection

# ── ATR ───────────────────────────────────────
ATR_PERIOD     = 14          # Wilder's ATR period (matches TradingView default)
TRAIL_ATR_MULT = 1.5         # trailing SL distance = ATR * this multiplier

# ── Bot loop ──────────────────────────────────
POLL_INTERVAL_SEC = 10      # seconds between scan cycles (REST polling mode)
                            # 5m candles → new data every 300s; 3s was wasting API quota

# ── State persistence ─────────────────────────
STATE_FILE = "bot_state.json"  # file for crash-recovery state

# ── Logging ───────────────────────────────────
LOG_FILE   = "fvg_bot.log"
LOG_LEVEL  = "INFO"
