# ─────────────────────────────────────────────
#  FVG Trading Bot — Binance Connection
# ─────────────────────────────────────────────

import ccxt
import pandas as pd
import time
import logging
from datetime import datetime, timezone
from config import API_KEY, API_SECRET, TESTNET, BINANCE_DEMO, MARKET_TYPE

logger = logging.getLogger(__name__)


def get_exchange(testnet: bool = TESTNET) -> ccxt.binance:
    """Create and return a Binance exchange instance."""
    exchange = ccxt.binance({
        "apiKey":  API_KEY,
        "secret":  API_SECRET,
        "options": {
            "defaultType": MARKET_TYPE,
            "adjustForTimeDifference": True,
            "recvWindow": 15000,  # wider window for demo endpoint latency
            # Skip endpoints that require permissions a futures-only / demo key lacks:
            #   - fetchCurrencies → sapi/capital/config (spot wallet)
            #   - fetchMarkets restricted to 'linear' → skips sapi/margin/allPairs
            "fetchCurrencies": False,
            "fetchMarkets":    ["linear"],   # USDT-M futures only
        },
        "enableRateLimit": True,
    })

    if BINANCE_DEMO:
        # Binance Futures Demo Trading lives at demo-fapi.binance.com.
        # ccxt defaults to fapi.binance.com; we re-point all USDT-M futures URLs.
        DEMO_HOST = "https://demo-fapi.binance.com"
        for k in ("fapiPublic", "fapiPublicV2", "fapiPublicV3",
                  "fapiPrivate", "fapiPrivateV2", "fapiPrivateV3",
                  "fapiData"):
            old = exchange.urls["api"].get(k)
            if old:
                # preserve the URL suffix (/fapi/v1, /fapi/v2, /futures/data, …)
                suffix = old.split(".com", 1)[-1]
                exchange.urls["api"][k] = DEMO_HOST + suffix
        logger.info(f"Connected to Binance {MARKET_TYPE.upper()} DEMO TRADING ({DEMO_HOST})")
    elif testnet:
        if MARKET_TYPE == "future":
            raise RuntimeError(
                "Binance futures testnet was deprecated. "
                "Set BINANCE_DEMO=true and use demo-account API keys, "
                "or switch MARKET_TYPE=spot to use the spot testnet."
            )
        exchange.set_sandbox_mode(True)
        logger.info("Connected to Binance SPOT TESTNET — orders go to sandbox")
    else:
        logger.warning(f"Connected to Binance {MARKET_TYPE.upper()} LIVE — real money at risk!")

    return exchange


def fetch_ohlcv(exchange: ccxt.binance,
                symbol: str,
                timeframe: str = "5m",
                limit: int = 200,
                since: int | None = None) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance and return as DataFrame.
    Retries up to 3 times on network error.
    """
    for attempt in range(3):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                        limit=limit, since=since)
            if not raw:
                logger.warning(f"Empty OHLCV response for {symbol} {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df

        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching {symbol} {timeframe} (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise

    return pd.DataFrame()


def fetch_historical_ohlcv_direct(symbol: str,
                                   timeframe: str = "5m",
                                   days: int = 90) -> pd.DataFrame:
    """
    Fallback: fetch candles directly via Binance public REST API (no auth needed).
    Use this if ccxt is blocked in your environment.
    """
    import requests

    tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf_ms  = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

    ms_per = tf_ms.get(timeframe, 300_000)
    total  = int(days * 86_400_000 / ms_per)
    binance_sym = symbol.replace("/", "")
    since_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - days * 86_400_000

    all_data = []
    url = "https://api.binance.com/api/v3/klines"

    while len(all_data) < total:
        params = {"symbol": binance_sym, "interval": tf_map[timeframe],
                  "startTime": since_ms, "limit": 1000}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        all_data.extend([[c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in batch])
        since_ms = batch[-1][0] + ms_per
        if len(batch) < 1000:
            break
        import time; time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def fetch_historical_ohlcv(exchange: ccxt.binance,
                            symbol: str,
                            timeframe: str = "5m",
                            days: int = 90) -> pd.DataFrame:
    """
    Fetch full historical OHLCV data for backtesting.
    Paginates automatically to get `days` worth of candles.
    """
    logger.info(f"Fetching {days} days of {symbol} {timeframe} candles from Binance...")

    tf_ms = {
        "1m":  60_000,
        "5m":  300_000,
        "15m": 900_000,
        "1h":  3_600_000,
        "4h":  14_400_000,
        "1d":  86_400_000,
    }
    ms_per_candle = tf_ms.get(timeframe, 300_000)
    total_candles = int((days * 24 * 60 * 60 * 1000) / ms_per_candle)
    batch_size    = 1000

    since_ms = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_data  = []
    fetched   = 0

    while fetched < total_candles:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                          since=since_ms, limit=batch_size)
            if not batch:
                break

            all_data.extend(batch)
            since_ms = batch[-1][0] + ms_per_candle
            fetched += len(batch)

            pct = min(100, int(fetched / total_candles * 100))
            print(f"\r  Downloading {symbol} {timeframe}: {pct}% ({fetched}/{total_candles} candles)    ", end="")

            if len(batch) < batch_size:
                break

            time.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            logger.warning(f"Network error, retrying... {e}")
            time.sleep(2)

    print()  # newline after progress

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe} ({df.index[0]} → {df.index[-1]})")
    return df


def get_account_balance(exchange: ccxt.binance, currency: str = "USDT") -> float:
    """Return available balance for the given currency. Retries up to 3 times."""
    for attempt in range(3):
        try:
            balance = exchange.fetch_balance()
            return float(balance["free"].get(currency, 0))
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching balance (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0
    return 0.0


def fetch_positions_safe(exchange: ccxt.binance, symbol: str) -> list:
    """Fetch positions with retry. Returns empty list on persistent failure."""
    for attempt in range(3):
        try:
            return exchange.fetch_positions([symbol])
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching positions for {symbol} (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Failed to fetch positions for {symbol}: {e}")
            return []
    return []


def get_ticker_price(exchange: ccxt.binance, symbol: str) -> float:
    """Return current market price for a symbol. Retries up to 2 times."""
    for attempt in range(2):
        try:
            ticker = exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching ticker for {symbol} (attempt {attempt+1}/2): {e}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return 0.0
    return 0.0
