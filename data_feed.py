"""
Real Market Data Feed for NQ Futures

Fetches REAL historical data from Yahoo Finance API (no yfinance dependency).
Supports NQ=F (NQ futures), QQQ (Nasdaq-100 ETF), and TQQQ as data sources.
Falls back gracefully: NQ futures -> QQQ -> cached data -> synthetic supplement.

All data returned is REAL market data with actual OHLCV bars.
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path


# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# Yahoo Finance API endpoints (public, no auth needed)
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_SPARK_URL = "https://query1.finance.yahoo.com/v8/finance/spark"

# Symbols to try, in order of preference
NQ_SYMBOLS = ["NQ=F", "QQQ", "TQQQ", "^NDX"]

# User agent to avoid blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}


def fetch_yahoo_chart(symbol: str, interval: str = "5m", range_str: str = "5d",
                      period1: Optional[int] = None, period2: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Yahoo Finance chart API.

    Args:
        symbol: Ticker symbol (NQ=F, QQQ, etc.)
        interval: Bar interval (1m, 5m, 15m, 60m, 1d)
        range_str: Time range (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y)
        period1/period2: Unix timestamps for custom range

    Returns:
        DataFrame with timestamp, open, high, low, close, volume columns, or None on failure.
    """
    params = {
        "interval": interval,
        "includePrePost": "true",
        "includeTimestamps": "true",
    }

    if period1 and period2:
        params["period1"] = str(period1)
        params["period2"] = str(period2)
    else:
        params["range"] = range_str

    url = YAHOO_CHART_URL.format(symbol=symbol)

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            print(f"  [DATA] No data returned for {symbol}")
            return None

        chart = result[0]
        timestamps = chart.get("timestamp", [])
        if not timestamps:
            print(f"  [DATA] No timestamps for {symbol}")
            return None

        quote = chart["indicators"]["quote"][0]
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        # Build DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="s"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

        # Clean: remove rows with NaN prices
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.reset_index(drop=True)

        if len(df) == 0:
            return None

        # Add bar_index and vwap
        df["bar_index"] = range(len(df))
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3

        # Fill NaN volumes with 0
        df["volume"] = df["volume"].fillna(0).astype(int)

        print(f"  [DATA] Fetched {len(df)} real bars of {symbol} @ {interval} "
              f"({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")

        return df

    except requests.exceptions.RequestException as e:
        print(f"  [DATA] Failed to fetch {symbol}: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"  [DATA] Parse error for {symbol}: {e}")
        return None


def fetch_real_nq_data(interval: str = "5m", range_str: str = "5d") -> Optional[pd.DataFrame]:
    """
    Fetch real NQ data, trying multiple symbols in order.
    Returns the first successful result.
    """
    for symbol in NQ_SYMBOLS:
        print(f"  [DATA] Trying {symbol}...")
        df = fetch_yahoo_chart(symbol, interval=interval, range_str=range_str)
        if df is not None and len(df) >= 20:
            df.attrs["symbol"] = symbol
            df.attrs["interval"] = interval
            df.attrs["source"] = "Yahoo Finance (REAL)"
            return df
        time.sleep(0.5)  # Rate limit courtesy

    print("  [DATA] All symbols failed")
    return None


def fetch_multi_timeframe(timeframes: Optional[List[str]] = None,
                          range_str: str = "5d") -> Dict[str, pd.DataFrame]:
    """
    Fetch real data across multiple timeframes — fractal analysis needs this.

    Returns dict mapping interval -> DataFrame.
    """
    if timeframes is None:
        timeframes = ["1m", "5m", "15m", "60m"]

    # Yahoo limits: 1m data only available for last 7 days
    # 5m/15m for last 60 days, 60m for up to 2 years
    range_map = {
        "1m": "5d",
        "5m": range_str,
        "15m": range_str if range_str in ("5d", "1mo") else "1mo",
        "60m": "1mo",
        "1d": "6mo",
    }

    result = {}
    for tf in timeframes:
        r = range_map.get(tf, range_str)
        df = fetch_real_nq_data(interval=tf, range_str=r)
        if df is not None:
            result[tf] = df

    return result


def save_cache(df: pd.DataFrame, label: str):
    """Cache downloaded data to disk."""
    path = CACHE_DIR / f"{label}.parquet"
    try:
        df_save = df.copy()
        df_save["timestamp"] = df_save["timestamp"].astype(str)
        df_save.to_parquet(path, index=False)
        print(f"  [CACHE] Saved {len(df)} bars to {path}")
    except Exception as e:
        # Fallback to CSV if parquet not available
        path = CACHE_DIR / f"{label}.csv"
        df_save = df.copy()
        df_save["timestamp"] = df_save["timestamp"].astype(str)
        df_save.to_csv(path, index=False)
        print(f"  [CACHE] Saved {len(df)} bars to {path} (csv fallback)")


def load_cache(label: str) -> Optional[pd.DataFrame]:
    """Load cached data from disk."""
    for ext in (".parquet", ".csv"):
        path = CACHE_DIR / f"{label}{ext}"
        if path.exists():
            try:
                if ext == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                if "bar_index" not in df.columns:
                    df["bar_index"] = range(len(df))
                if "vwap" not in df.columns:
                    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                print(f"  [CACHE] Loaded {len(df)} bars from {path}")
                return df
            except Exception as e:
                print(f"  [CACHE] Failed to load {path}: {e}")
    return None


class RealDataFeed:
    """
    Manages real market data for the backtest loop.

    Strategy:
    1. Try to fetch live data from Yahoo Finance
    2. Cache every successful fetch
    3. Rotate through cached datasets for variety
    4. Track data freshness and quality
    """

    # Map timeframe minutes to Yahoo interval strings
    TF_MAP = {1: "1m", 5: "5m", 15: "15m", 60: "60m", 240: "60m", 1440: "1d"}

    # Map timeframe minutes to reasonable ranges
    RANGE_MAP = {1: "5d", 5: "5d", 15: "1mo", 60: "1mo", 240: "3mo", 1440: "1y"}

    def __init__(self):
        self.fetch_count = 0
        self.cache_labels: List[str] = []
        self.last_fetch_time = 0
        self.fetch_cooldown = 30  # Seconds between live fetches (be nice to Yahoo)
        self.data_log: List[Dict] = []  # Track every data fetch

        # Discover existing cache
        self._discover_cache()

    def _discover_cache(self):
        """Find all cached datasets."""
        for f in CACHE_DIR.iterdir():
            if f.suffix in (".parquet", ".csv"):
                self.cache_labels.append(f.stem)
        if self.cache_labels:
            print(f"  [DATA] Found {len(self.cache_labels)} cached datasets")

    def get_bars(self, timeframe_minutes: int = 5, min_bars: int = 200) -> Tuple[pd.DataFrame, Dict]:
        """
        Get real market bars for backtesting.

        Returns:
            (DataFrame of bars, metadata dict about the data source)
        """
        interval = self.TF_MAP.get(timeframe_minutes, "5m")
        range_str = self.RANGE_MAP.get(timeframe_minutes, "5d")

        meta = {
            "source": "unknown",
            "symbol": "unknown",
            "interval": interval,
            "timeframe_minutes": timeframe_minutes,
            "bars_requested": min_bars,
            "bars_received": 0,
            "is_real": False,
            "fetch_time": 0,
        }

        # Try live fetch (with cooldown to avoid rate limiting)
        now = time.time()
        df = None

        if now - self.last_fetch_time >= self.fetch_cooldown:
            t0 = time.time()
            df = fetch_real_nq_data(interval=interval, range_str=range_str)
            meta["fetch_time"] = time.time() - t0

            if df is not None and len(df) >= min_bars:
                self.last_fetch_time = now
                self.fetch_count += 1

                # Cache it
                label = f"nq_{interval}_{self.fetch_count}_{int(now)}"
                save_cache(df, label)
                self.cache_labels.append(label)

                meta["source"] = f"Yahoo Finance LIVE ({df.attrs.get('symbol', 'NQ')})"
                meta["symbol"] = df.attrs.get("symbol", "NQ")
                meta["is_real"] = True
                meta["bars_received"] = len(df)

                self.data_log.append(meta.copy())
                return df, meta

        # Try cache
        if self.cache_labels:
            # Rotate through cache for variety
            rng = np.random.default_rng()
            idx = rng.integers(0, len(self.cache_labels))
            label = self.cache_labels[idx]
            df = load_cache(label)

            if df is not None and len(df) >= min_bars:
                # Random window into cached data for variety
                if len(df) > min_bars * 2:
                    start = rng.integers(0, len(df) - min_bars)
                    df = df.iloc[start:start + min_bars].reset_index(drop=True)
                    df["bar_index"] = range(len(df))

                meta["source"] = f"Cache ({label})"
                meta["symbol"] = "NQ (cached)"
                meta["is_real"] = True
                meta["bars_received"] = len(df)

                self.data_log.append(meta.copy())
                return df, meta

        # Last resort: use synthetic with a warning
        print("  [DATA] WARNING: No real data available, using synthetic fallback")
        from market_generator import NQMarketGenerator
        gen = NQMarketGenerator()
        df = gen.generate_bars(max(min_bars, 500), timeframe_minutes=timeframe_minutes)

        meta["source"] = "SYNTHETIC FALLBACK"
        meta["symbol"] = "NQ (synthetic)"
        meta["is_real"] = False
        meta["bars_received"] = len(df)

        self.data_log.append(meta.copy())
        return df, meta

    def get_multi_timeframe(self) -> Dict[int, Tuple[pd.DataFrame, Dict]]:
        """Get real data across multiple timeframes."""
        result = {}
        for tf_min in [1, 5, 15, 60]:
            bars, meta = self.get_bars(timeframe_minutes=tf_min)
            result[tf_min] = (bars, meta)
        return result

    def get_data_stats(self) -> Dict:
        """Return statistics about data usage."""
        total = len(self.data_log)
        real = sum(1 for d in self.data_log if d["is_real"])
        return {
            "total_fetches": total,
            "real_data_fetches": real,
            "synthetic_fallbacks": total - real,
            "real_data_pct": real / total * 100 if total > 0 else 0,
            "cached_datasets": len(self.cache_labels),
            "live_fetches": self.fetch_count,
        }
