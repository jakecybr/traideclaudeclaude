"""
Multi-Source Data Workers — Continuous Data Collection

Spawns background threads that continuously fetch market data from multiple
free APIs and write it to the data lake as parquet files.

Workers:
  - CryptoWorker: Binance API (BTC, ETH, SOL, etc.)
  - StockWorker: Yahoo Finance (QQQ, SPY, AAPL, NVDA, NQ=F, ES=F, etc.)
  - ForexWorker: Yahoo Finance (EUR/USD, GBP/USD, etc.)
  - MacroWorker: Yahoo Finance (VIX, TNX, Gold, Oil, DXY)
  - HistoricalBackfillWorker: Yahoo Finance daily data (5y backfill)
"""

import time
import threading
import pandas as pd
import numpy as np
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from data_feed import YahooSession, fetch_yahoo_chart


# Global Yahoo rate limiter — all Yahoo-based workers share this
_yahoo_lock = threading.Lock()
_yahoo_last_request = 0.0
YAHOO_MIN_INTERVAL = 2.0  # seconds between Yahoo requests


def _yahoo_rate_limit():
    """Enforce minimum interval between Yahoo Finance requests."""
    global _yahoo_last_request
    with _yahoo_lock:
        now = time.time()
        wait = YAHOO_MIN_INTERVAL - (now - _yahoo_last_request)
        if wait > 0:
            time.sleep(wait)
        _yahoo_last_request = time.time()


class BaseWorker(ABC):
    """Abstract base class for all data workers."""

    def __init__(self, name: str, data_lake_root: Path):
        self.name = name
        self.data_lake_root = data_lake_root
        self.running = False
        self.total_fetches = 0
        self.total_bars = 0
        self.last_fetch_time = 0.0
        self.errors = 0
        self.start_time = 0.0
        self._backoff = 5.0

    def run(self):
        """Main loop — fetch data continuously with exponential backoff on failure."""
        self.running = True
        self.start_time = time.time()
        print(f"  [WORKER] {self.name} started")

        while self.running:
            try:
                count = self._fetch_cycle()
                self.total_fetches += 1
                self.total_bars += count
                self.last_fetch_time = time.time()
                self._backoff = 5.0  # reset on success
                # Sleep between cycles
                time.sleep(self._cycle_sleep())
            except Exception as e:
                self.errors += 1
                print(f"  [WORKER] {self.name} error: {e}")
                time.sleep(min(self._backoff, 300))
                self._backoff = min(self._backoff * 2, 300)

    @abstractmethod
    def _fetch_cycle(self) -> int:
        """Fetch one batch of data. Return number of bars written."""
        pass

    def _cycle_sleep(self) -> float:
        """How long to sleep between fetch cycles."""
        return 60.0

    def _write_bars(self, source: str, instrument: str, timeframe: str, df: pd.DataFrame):
        """Write bars to data lake as parquet files."""
        if df is None or len(df) == 0:
            return

        # Ensure directory exists
        out_dir = self.data_lake_root / source / instrument / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filename: start_end timestamps
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            if hasattr(ts.iloc[0], 'timestamp'):
                start_ts = int(ts.iloc[0].timestamp())
                end_ts = int(ts.iloc[-1].timestamp())
            else:
                start_ts = int(ts.iloc[0])
                end_ts = int(ts.iloc[-1])
        else:
            start_ts = int(time.time())
            end_ts = start_ts

        filepath = out_dir / f"{start_ts}_{end_ts}.parquet"

        # Skip if identical file already exists
        if filepath.exists():
            return

        try:
            # Ensure timestamp is string for parquet compatibility
            df_save = df.copy()
            if "timestamp" in df_save.columns:
                df_save["timestamp"] = df_save["timestamp"].astype(str)
            df_save.to_parquet(filepath, index=False)
        except Exception:
            # Fallback to CSV
            csv_path = out_dir / f"{start_ts}_{end_ts}.csv"
            df_save = df.copy()
            if "timestamp" in df_save.columns:
                df_save["timestamp"] = df_save["timestamp"].astype(str)
            df_save.to_csv(csv_path, index=False)

    def get_stats(self) -> Dict:
        """Return worker statistics."""
        uptime = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            "name": self.name,
            "status": "running" if self.running else "stopped",
            "total_fetches": self.total_fetches,
            "total_bars": self.total_bars,
            "last_fetch_time": self.last_fetch_time,
            "errors": self.errors,
            "uptime": uptime,
        }

    def stop(self):
        """Stop the worker."""
        self.running = False
        print(f"  [WORKER] {self.name} stopping")


class CryptoWorker(BaseWorker):
    """Fetches crypto data from Binance public API."""

    TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
    TIMEFRAMES = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, data_lake_root: Path):
        super().__init__("CryptoWorker", data_lake_root)

    def _fetch_cycle(self) -> int:
        total_bars = 0
        for ticker in self.TICKERS:
            for tf_name, tf_api in self.TIMEFRAMES.items():
                if not self.running:
                    return total_bars
                try:
                    resp = requests.get(self.BASE_URL, params={
                        "symbol": ticker, "interval": tf_api, "limit": 1000
                    }, timeout=10)

                    if resp.status_code == 429 or resp.status_code == 418:
                        time.sleep(30)
                        continue
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    if not data:
                        continue

                    rows = []
                    for k in data:
                        rows.append({
                            "timestamp": pd.to_datetime(k[0], unit="ms"),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                        })

                    df = pd.DataFrame(rows)
                    df = df.dropna(subset=["open", "high", "low", "close"])
                    if len(df) > 0:
                        df["bar_index"] = range(len(df))
                        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                        df["volume"] = df["volume"].fillna(0)
                        self._write_bars("binance", ticker, tf_name, df)
                        total_bars += len(df)

                    time.sleep(0.1)  # Rate limit courtesy

                except Exception as e:
                    self.errors += 1
                    continue
        return total_bars

    def _cycle_sleep(self) -> float:
        return 120.0  # 2 minutes between full cycles


class StockWorker(BaseWorker):
    """Fetches stock/index/futures data from Yahoo Finance."""

    TICKERS = ["QQQ", "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
               "META", "TSLA", "AMD", "NFLX", "DIA", "IWM", "NQ=F", "ES=F"]
    TIMEFRAMES = {"5m": "5d", "15m": "5d", "60m": "1mo", "1d": "6mo"}

    def __init__(self, data_lake_root: Path):
        super().__init__("StockWorker", data_lake_root)

    def _fetch_cycle(self) -> int:
        total_bars = 0
        for ticker in self.TICKERS:
            for interval, range_str in self.TIMEFRAMES.items():
                if not self.running:
                    return total_bars
                try:
                    _yahoo_rate_limit()
                    df = fetch_yahoo_chart(ticker, interval=interval, range_str=range_str)
                    if df is not None and len(df) > 0:
                        self._write_bars("yahoo", ticker, interval, df)
                        total_bars += len(df)
                except Exception:
                    self.errors += 1
                    continue
        return total_bars

    def _cycle_sleep(self) -> float:
        return 180.0  # 3 minutes between full cycles


class ForexWorker(BaseWorker):
    """Fetches forex data from Yahoo Finance."""

    TICKERS = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"]
    TIMEFRAMES = {"5m": "5d", "15m": "5d", "60m": "1mo", "1d": "6mo"}

    def __init__(self, data_lake_root: Path):
        super().__init__("ForexWorker", data_lake_root)

    def _fetch_cycle(self) -> int:
        total_bars = 0
        for ticker in self.TICKERS:
            for interval, range_str in self.TIMEFRAMES.items():
                if not self.running:
                    return total_bars
                try:
                    _yahoo_rate_limit()
                    df = fetch_yahoo_chart(ticker, interval=interval, range_str=range_str)
                    if df is not None and len(df) > 0:
                        self._write_bars("yahoo", ticker, interval, df)
                        total_bars += len(df)
                except Exception:
                    self.errors += 1
                    continue
        return total_bars

    def _cycle_sleep(self) -> float:
        return 300.0  # 5 minutes


class MacroWorker(BaseWorker):
    """Fetches macro indicator data (VIX, bonds, gold, oil, DXY) from Yahoo."""

    TICKERS = ["^VIX", "^TNX", "^TYX", "GC=F", "CL=F", "DX-Y.NYB"]
    TIMEFRAMES = {"5m": "5d", "15m": "5d", "60m": "1mo", "1d": "6mo"}

    def __init__(self, data_lake_root: Path):
        super().__init__("MacroWorker", data_lake_root)

    def _fetch_cycle(self) -> int:
        total_bars = 0
        for ticker in self.TICKERS:
            for interval, range_str in self.TIMEFRAMES.items():
                if not self.running:
                    return total_bars
                try:
                    _yahoo_rate_limit()
                    df = fetch_yahoo_chart(ticker, interval=interval, range_str=range_str)
                    if df is not None and len(df) > 0:
                        self._write_bars("yahoo", ticker, interval, df)
                        total_bars += len(df)
                except Exception:
                    self.errors += 1
                    continue
        return total_bars

    def _cycle_sleep(self) -> float:
        return 300.0


class HistoricalBackfillWorker(BaseWorker):
    """Fetches deep historical daily data for all instruments."""

    ALL_TICKERS = (
        StockWorker.TICKERS + ForexWorker.TICKERS + MacroWorker.TICKERS
    )

    def __init__(self, data_lake_root: Path):
        super().__init__("HistoricalBackfill", data_lake_root)
        self._completed_tickers = set()

    def _fetch_cycle(self) -> int:
        total_bars = 0
        for ticker in self.ALL_TICKERS:
            if not self.running:
                return total_bars
            if ticker in self._completed_tickers:
                continue

            # Check if we already have daily data for this ticker
            daily_dir = self.data_lake_root / "yahoo" / ticker / "1d"
            if daily_dir.exists() and any(daily_dir.iterdir()):
                self._completed_tickers.add(ticker)
                continue

            try:
                _yahoo_rate_limit()
                df = fetch_yahoo_chart(ticker, interval="1d", range_str="5y")
                if df is not None and len(df) > 0:
                    self._write_bars("yahoo", ticker, "1d", df)
                    total_bars += len(df)
                    print(f"  [BACKFILL] {ticker}: {len(df)} daily bars (5y)")
                self._completed_tickers.add(ticker)
            except Exception:
                self.errors += 1
                continue

        return total_bars

    def _cycle_sleep(self) -> float:
        # After all tickers are done, check monthly for new data
        if len(self._completed_tickers) >= len(self.ALL_TICKERS):
            return 3600.0  # 1 hour
        return 5.0  # Still backfilling, go fast


def start_all_workers(data_lake_root: Path) -> List[BaseWorker]:
    """Create and start all data workers in daemon threads."""
    data_lake_root.mkdir(exist_ok=True)

    workers = [
        CryptoWorker(data_lake_root),
        StockWorker(data_lake_root),
        ForexWorker(data_lake_root),
        MacroWorker(data_lake_root),
        HistoricalBackfillWorker(data_lake_root),
    ]

    for worker in workers:
        thread = threading.Thread(target=worker.run, daemon=True, name=worker.name)
        thread.start()

    print(f"  [WORKERS] Started {len(workers)} data workers: "
          f"{', '.join(w.name for w in workers)}")
    return workers
