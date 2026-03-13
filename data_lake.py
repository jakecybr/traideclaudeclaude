"""
Data Lake — Organized Multi-Source Market Data Storage & Routing

Provides:
  - DataLakeIndex: scans and indexes all parquet/csv files in data_lake/
  - DataNormalizer: normalizes OHLCV to percentage scale for cross-instrument comparison
  - DataRouter: selects random instruments/windows, applies walk-forward train/test splits

Directory structure:
  data_lake/
    binance/BTCUSDT/5m/*.parquet
    yahoo/NQ=F/5m/*.parquet
    yahoo/QQQ/1d/*.parquet
    ...
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class DataLakeIndex:
    """Scans and indexes the data lake directory tree."""

    def __init__(self, root: Path):
        self.root = root
        # index: source -> instrument -> timeframe -> [(start_ts, end_ts, path)]
        self.index: Dict[str, Dict[str, Dict[str, List[Tuple[int, int, Path]]]]] = {}
        self._last_scan = 0.0
        self._scan()

    def _scan(self):
        """Walk the data lake directory tree and build an in-memory index."""
        self.index = {}
        if not self.root.exists():
            return

        for source_dir in self.root.iterdir():
            if not source_dir.is_dir():
                continue
            source = source_dir.name
            self.index[source] = {}

            for instrument_dir in source_dir.iterdir():
                if not instrument_dir.is_dir():
                    continue
                instrument = instrument_dir.name
                self.index[source][instrument] = {}

                for tf_dir in instrument_dir.iterdir():
                    if not tf_dir.is_dir():
                        continue
                    timeframe = tf_dir.name
                    files = []

                    for f in tf_dir.iterdir():
                        if f.suffix not in (".parquet", ".csv"):
                            continue
                        try:
                            parts = f.stem.split("_")
                            if len(parts) >= 2:
                                start_ts = int(parts[0])
                                end_ts = int(parts[1])
                                files.append((start_ts, end_ts, f))
                        except (ValueError, IndexError):
                            continue

                    if files:
                        files.sort(key=lambda x: x[0])
                        self.index[source][instrument][timeframe] = files

        self._last_scan = time.time()

    def refresh(self):
        """Re-scan the data lake (call periodically or on demand)."""
        self._scan()

    def get_instruments(self) -> List[str]:
        """List all instruments with data across all sources."""
        instruments = set()
        for source_data in self.index.values():
            for instrument in source_data:
                instruments.add(instrument)
        return sorted(instruments)

    def get_available(self, instrument: str, timeframe: str) -> List[Tuple[int, int, Path]]:
        """Return available file ranges for an instrument/timeframe."""
        for source_data in self.index.values():
            if instrument in source_data and timeframe in source_data[instrument]:
                return source_data[instrument][timeframe]
        return []

    def get_bars(self, instrument: str, timeframe: str,
                 start_ts: int = None, end_ts: int = None) -> Optional[pd.DataFrame]:
        """Load and concatenate bars for an instrument/timeframe."""
        files = self.get_available(instrument, timeframe)
        if not files:
            return None

        dfs = []
        for f_start, f_end, filepath in files:
            if start_ts and f_end < start_ts:
                continue
            if end_ts and f_start > end_ts:
                continue

            try:
                if filepath.suffix == ".parquet":
                    df = pd.read_parquet(filepath)
                else:
                    df = pd.read_csv(filepath)

                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=True)

        # Deduplicate on timestamp
        if "timestamp" in combined.columns:
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Ensure standard columns
        if "bar_index" not in combined.columns:
            combined["bar_index"] = range(len(combined))
        if "vwap" not in combined.columns and all(c in combined.columns for c in ["high", "low", "close"]):
            combined["vwap"] = (combined["high"] + combined["low"] + combined["close"]) / 3
        if "volume" in combined.columns:
            combined["volume"] = combined["volume"].fillna(0)

        return combined

    def get_freshness(self, instrument: str, timeframe: str) -> float:
        """Return seconds since last data point for an instrument."""
        files = self.get_available(instrument, timeframe)
        if not files:
            return float("inf")
        latest_end = max(f[1] for f in files)
        return time.time() - latest_end

    def get_stats(self) -> Dict:
        """Return summary statistics about the data lake."""
        total_instruments = len(self.get_instruments())
        total_files = 0
        total_bytes = 0

        for source_data in self.index.values():
            for inst_data in source_data.values():
                for tf_files in inst_data.values():
                    for _, _, filepath in tf_files:
                        total_files += 1
                        try:
                            total_bytes += filepath.stat().st_size
                        except OSError:
                            pass

        return {
            "total_instruments": total_instruments,
            "total_files": total_files,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "sources": list(self.index.keys()),
            "instruments": self.get_instruments(),
            "last_scan": self._last_scan,
        }


class DataNormalizer:
    """Normalizes OHLCV data for cross-instrument comparison."""

    @staticmethod
    def normalize_to_pct(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert OHLCV to percentage scale.
        Divide all price columns by the first close, multiply by 100.
        Volume is z-score normalized.
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        first_close = df["close"].iloc[0]
        if first_close == 0:
            return df

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col] / first_close * 100.0

        if "vwap" in df.columns:
            df["vwap"] = df["vwap"] / first_close * 100.0

        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            mean_v = vol.mean()
            std_v = vol.std()
            if std_v > 0:
                df["volume"] = (vol - mean_v) / std_v
            else:
                df["volume"] = 0.0

        return df

    @staticmethod
    def attach_regime(df: pd.DataFrame, vix_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Attach VIX-based regime classification to bars."""
        if vix_df is None or len(vix_df) == 0:
            df = df.copy()
            df["vix_level"] = np.nan
            df["vol_regime"] = "UNKNOWN"
            return df

        df = df.copy()
        # Merge by nearest timestamp
        if "timestamp" in df.columns and "timestamp" in vix_df.columns:
            df = df.sort_values("timestamp")
            vix_df = vix_df.sort_values("timestamp")
            df = pd.merge_asof(df, vix_df[["timestamp", "close"]].rename(columns={"close": "vix_level"}),
                               on="timestamp", direction="nearest")
        else:
            df["vix_level"] = np.nan

        # Classify regime based on VIX thresholds
        def classify_vix(v):
            if pd.isna(v):
                return "UNKNOWN"
            if v < 12:
                return "LOW"
            elif v < 18:
                return "NORMAL"
            elif v < 25:
                return "HIGH"
            elif v < 35:
                return "EXTREME"
            else:
                return "EXTREME"

        df["vol_regime"] = df["vix_level"].apply(classify_vix)
        return df

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        """Validate that a DataFrame has usable data."""
        if df is None or len(df) < 20:
            return False
        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return False
        # Check for excessive NaNs
        nan_pct = df[required].isna().sum().sum() / (len(df) * len(required))
        if nan_pct > 0.1:
            return False
        return True


class DataRouter:
    """
    Routes data from the lake to the optimizer.
    Selects instruments, applies normalization, and does walk-forward splits.
    """

    def __init__(self, lake: DataLakeIndex):
        self.lake = lake
        self.rng = np.random.default_rng()
        self.windows_served = 0
        self.instruments_used = set()

    def get_training_data(self, min_bars: int = 200,
                          instrument: str = None,
                          timeframe: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Get train/test data with walk-forward split.

        Returns (train_df, test_df, metadata).
        Train = first 70%, Test = last 30%. Never overlap.
        """
        # Refresh index periodically
        if time.time() - self.lake._last_scan > 60:
            self.lake.refresh()

        # Pick instrument and timeframe
        if instrument is None or timeframe is None:
            instrument, timeframe = self._pick_random_instrument(min_bars)

        if instrument is None:
            return None, None, {"error": "no_data"}

        df = self.lake.get_bars(instrument, timeframe)
        if df is None or len(df) < min_bars:
            return None, None, {"error": "insufficient_bars"}

        # Random window if data is much larger than needed
        if len(df) > min_bars * 3:
            max_start = len(df) - min_bars
            start = self.rng.integers(0, max_start)
            df = df.iloc[start:start + min_bars * 2].reset_index(drop=True)
            df["bar_index"] = range(len(df))

        # Get VIX data for regime classification
        vix_df = self.lake.get_bars("^VIX", "5m")
        if vix_df is None:
            vix_df = self.lake.get_bars("^VIX", "1d")

        # Normalize to percentage scale
        raw_first_close = df["close"].iloc[0] if len(df) > 0 else 0
        df_norm = DataNormalizer.normalize_to_pct(df)
        df_norm = DataNormalizer.attach_regime(df_norm, vix_df)

        # Walk-forward split: 70% train, 30% test
        split_idx = int(len(df_norm) * 0.7)
        train_df = df_norm.iloc[:split_idx].reset_index(drop=True)
        train_df["bar_index"] = range(len(train_df))
        test_df = df_norm.iloc[split_idx:].reset_index(drop=True)
        test_df["bar_index"] = range(len(test_df))

        self.windows_served += 1
        self.instruments_used.add(instrument)

        metadata = {
            "instrument": instrument,
            "timeframe": timeframe,
            "source": "data_lake",
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "total_bars": len(df),
            "is_normalized": True,
            "raw_first_close": raw_first_close,
            "is_real": True,
            "symbol": instrument,
        }

        return train_df, test_df, metadata

    def _pick_random_instrument(self, min_bars: int) -> Tuple[Optional[str], Optional[str]]:
        """Pick a random instrument that has enough data."""
        # Preferred timeframes for backtesting
        preferred_tfs = ["5m", "15m", "60m", "1h", "4h", "1d"]
        instruments = self.lake.get_instruments()

        if not instruments:
            return None, None

        # Shuffle to get variety
        self.rng.shuffle(instruments)

        for instrument in instruments:
            for tf in preferred_tfs:
                files = self.lake.get_available(instrument, tf)
                if files:
                    # Quick estimate of total bars
                    total_range = sum(f[1] - f[0] for f in files)
                    if total_range > 0 or len(files) > 0:
                        df = self.lake.get_bars(instrument, tf)
                        if df is not None and len(df) >= min_bars:
                            return instrument, tf

        return None, None

    def has_real_data(self) -> bool:
        """Check if the data lake has any usable data."""
        return len(self.lake.get_instruments()) > 0

    def wait_for_data(self, min_bars: int = 200, timeout: float = 300) -> bool:
        """Block until data is available or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            self.lake.refresh()
            if self.has_real_data():
                instruments = self.lake.get_instruments()
                for inst in instruments:
                    for tf in ["5m", "15m", "1h", "1d"]:
                        df = self.lake.get_bars(inst, tf)
                        if df is not None and len(df) >= min_bars:
                            return True
            print(f"  [ROUTER] Waiting for data... ({int(time.time() - start)}s)")
            time.sleep(10)
        return False

    def get_lake_stats(self) -> Dict:
        """Get data lake + router stats."""
        lake_stats = self.lake.get_stats()
        lake_stats["windows_served"] = self.windows_served
        lake_stats["instruments_used"] = list(self.instruments_used)
        return lake_stats
