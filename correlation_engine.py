"""
Cross-Instrument Correlation Engine

Computes rolling correlations between instruments, finds similar patterns,
and provides NQ-focused intelligence for the AI analyzer.
"""

import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from data_lake import DataLakeIndex


class CorrelationEngine:
    """Computes and caches cross-instrument correlations."""

    NQ_PROXIES = ["NQ=F", "QQQ", "TQQQ"]

    def __init__(self, lake: DataLakeIndex):
        self.lake = lake
        self._lock = threading.Lock()
        self._correlations: Dict[str, Dict[str, float]] = {}
        self._nq_correlations: List[Tuple[str, float]] = []
        self._pattern_matches: List[Dict] = []
        self._instrument_regimes: Dict[str, Dict] = {}
        self._last_compute = 0.0

    def compute_all_correlations(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Pearson correlation of daily returns for all instruments."""
        instruments = self.lake.get_instruments()
        if len(instruments) < 2:
            return {}

        # Load daily returns for each instrument
        returns_map = {}
        for inst in instruments:
            df = self.lake.get_bars(inst, "1d")
            if df is None:
                df = self.lake.get_bars(inst, "60m")
            if df is None:
                df = self.lake.get_bars(inst, "5m")
            if df is None or len(df) < 20:
                continue

            close = df["close"].values.astype(float)
            rets = np.diff(close) / (close[:-1] + 1e-10)
            returns_map[inst] = rets

        # Compute pairwise correlations
        corr_matrix = {}
        insts = list(returns_map.keys())
        for i in range(len(insts)):
            corr_matrix[insts[i]] = {}
            for j in range(len(insts)):
                if i == j:
                    corr_matrix[insts[i]][insts[j]] = 1.0
                    continue
                # Align lengths
                r1 = returns_map[insts[i]]
                r2 = returns_map[insts[j]]
                min_len = min(len(r1), len(r2))
                if min_len < 10:
                    corr_matrix[insts[i]][insts[j]] = 0.0
                    continue
                r1 = r1[-min_len:]
                r2 = r2[-min_len:]
                if np.std(r1) == 0 or np.std(r2) == 0:
                    corr_matrix[insts[i]][insts[j]] = 0.0
                else:
                    corr_matrix[insts[i]][insts[j]] = float(np.corrcoef(r1, r2)[0, 1])

        with self._lock:
            self._correlations = corr_matrix
            self._last_compute = time.time()

        return corr_matrix

    def get_nq_correlations(self) -> List[Tuple[str, float]]:
        """Return instruments sorted by correlation to NQ/QQQ."""
        if not self._correlations:
            self.compute_all_correlations()

        # Find NQ proxy in correlation matrix
        nq_key = None
        for proxy in self.NQ_PROXIES:
            if proxy in self._correlations:
                nq_key = proxy
                break

        if nq_key is None:
            return []

        nq_corrs = self._correlations.get(nq_key, {})
        sorted_corrs = sorted(nq_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

        with self._lock:
            self._nq_correlations = [(inst, corr) for inst, corr in sorted_corrs if inst != nq_key]

        return self._nq_correlations

    def find_similar_patterns(self, target_instrument: str, window: pd.DataFrame,
                              top_n: int = 5) -> List[Dict]:
        """Find historical windows in other instruments similar to the given window."""
        if window is None or len(window) < 20:
            return []

        # Get normalized returns of the target window
        close = window["close"].values.astype(float)
        target_rets = np.diff(close) / (close[:-1] + 1e-10)

        if len(target_rets) < 10:
            return []

        matches = []
        instruments = self.lake.get_instruments()

        for inst in instruments:
            if inst == target_instrument:
                continue

            for tf in ["5m", "15m", "1h", "1d"]:
                df = self.lake.get_bars(inst, tf)
                if df is None or len(df) < len(window) + 20:
                    continue

                close_all = df["close"].values.astype(float)
                rets_all = np.diff(close_all) / (close_all[:-1] + 1e-10)
                window_len = len(target_rets)

                # Sliding window correlation
                best_corr = 0.0
                best_idx = 0
                step = max(1, window_len // 4)

                for start in range(0, len(rets_all) - window_len, step):
                    chunk = rets_all[start:start + window_len]
                    if np.std(chunk) == 0:
                        continue
                    corr = float(np.corrcoef(target_rets, chunk)[0, 1])
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_idx = start

                if abs(best_corr) > 0.5:
                    # What happened after the similar window?
                    after_idx = best_idx + window_len
                    if after_idx + 10 < len(close_all):
                        subsequent_move = (close_all[after_idx + 10] - close_all[after_idx]) / close_all[after_idx] * 100
                    else:
                        subsequent_move = 0.0

                    matches.append({
                        "instrument": inst,
                        "timeframe": tf,
                        "similarity_score": round(best_corr, 3),
                        "subsequent_move_pct": round(subsequent_move, 2),
                        "window_start_idx": best_idx,
                    })

                if len(matches) >= top_n * 3:
                    break

        # Sort by similarity and return top_n
        matches.sort(key=lambda x: abs(x["similarity_score"]), reverse=True)
        result = matches[:top_n]

        with self._lock:
            self._pattern_matches = result

        return result

    def compute_instrument_regime(self, instrument: str) -> Dict:
        """Compute current trend, volatility, momentum for an instrument."""
        for tf in ["5m", "15m", "1h", "1d"]:
            df = self.lake.get_bars(instrument, tf)
            if df is not None and len(df) >= 50:
                break
        else:
            return {"trend": "UNKNOWN", "volatility": "UNKNOWN", "momentum": "UNKNOWN"}

        close = df["close"].values.astype(float)

        # Trend
        if len(close) >= 20:
            ma20 = np.mean(close[-20:])
            pct = (close[-1] - ma20) / ma20 * 100
            if pct > 1.5:
                trend = "STRONG_UP"
            elif pct > 0.3:
                trend = "UP"
            elif pct < -1.5:
                trend = "STRONG_DOWN"
            elif pct < -0.3:
                trend = "DOWN"
            else:
                trend = "FLAT"
        else:
            trend = "FLAT"

        # Volatility (ATR-based)
        if len(close) >= 15:
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
            atr14 = np.mean(tr[-14:])
            atr_pct = atr14 / close[-1] * 100 if close[-1] > 0 else 0
            if atr_pct > 2.0:
                volatility = "EXTREME"
            elif atr_pct > 1.0:
                volatility = "HIGH"
            elif atr_pct > 0.4:
                volatility = "NORMAL"
            else:
                volatility = "LOW"
        else:
            volatility = "NORMAL"

        # Momentum (simple RSI proxy)
        if len(close) >= 15:
            deltas = np.diff(close[-15:])
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.001
            rs = gains / losses
            rsi = 100 - 100 / (1 + rs)
            if rsi > 65:
                momentum = "BULLISH"
            elif rsi < 35:
                momentum = "BEARISH"
            else:
                momentum = "NEUTRAL"
        else:
            momentum = "NEUTRAL"

        regime = {"trend": trend, "volatility": volatility, "momentum": momentum}

        with self._lock:
            self._instrument_regimes[instrument] = regime

        return regime

    def run_continuous(self):
        """Background loop that recomputes correlations periodically."""
        while True:
            try:
                self.compute_all_correlations()
                self.get_nq_correlations()
                # Compute regimes for all instruments
                for inst in self.lake.get_instruments():
                    self.compute_instrument_regime(inst)
            except Exception as e:
                print(f"  [CORR] Error in continuous compute: {e}")
            time.sleep(300)  # 5 minutes

    def start_background(self):
        """Start the continuous computation in a daemon thread."""
        thread = threading.Thread(target=self.run_continuous, daemon=True, name="CorrelationEngine")
        thread.start()

    def get_state(self) -> Dict:
        """Return all correlation data for dashboard/analyzer consumption."""
        with self._lock:
            return {
                "nq_correlations": [
                    {"instrument": inst, "correlation": round(corr, 3)}
                    for inst, corr in self._nq_correlations[:20]
                ],
                "pattern_matches": self._pattern_matches[:10],
                "instrument_regimes": dict(self._instrument_regimes),
                "total_pairs_computed": sum(
                    len(v) for v in self._correlations.values()
                ),
                "instruments_tracked": len(self._instrument_regimes),
                "last_compute": self._last_compute,
            }
