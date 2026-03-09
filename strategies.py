"""
Trading Strategy Library for NQ Fractal Backtester

Each strategy operates on OHLCV bars and produces signals.
Strategies are parameterized — the optimizer mutates parameters
across cycles to find better configurations.

All strategies respect the fractal nature of markets:
  - Patterns repeat across timeframes
  - Support/resistance is fractal (pivots within pivots)
  - Volatility is self-similar
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class StrategyParams:
    """Mutable parameters that the optimizer tunes."""
    # EMA crossover
    fast_ema: int = 9
    slow_ema: int = 21
    trend_ema: int = 50

    # Fractal breakout
    fractal_lookback: int = 5
    fractal_confirmation_bars: int = 2
    breakout_atr_mult: float = 0.5

    # Mean reversion
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Momentum
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Volume
    volume_ma_period: int = 20
    volume_spike_mult: float = 1.5

    # Risk
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    max_position_size: int = 4
    trail_atr_mult: float = 1.0

    def mutate(self, rng: np.random.Generator, mutation_rate: float = 0.3) -> 'StrategyParams':
        """Create a mutated copy of these parameters."""
        new = StrategyParams(**self.__dict__)
        fields_to_mutate = [
            ('fast_ema', 3, 20, int), ('slow_ema', 10, 50, int),
            ('trend_ema', 30, 200, int), ('fractal_lookback', 3, 10, int),
            ('breakout_atr_mult', 0.1, 2.0, float),
            ('bollinger_period', 10, 50, int), ('bollinger_std', 1.0, 3.5, float),
            ('rsi_period', 7, 28, int), ('rsi_oversold', 15, 40, float),
            ('rsi_overbought', 60, 85, float),
            ('macd_fast', 6, 20, int), ('macd_slow', 18, 40, int),
            ('macd_signal', 5, 15, int),
            ('volume_ma_period', 10, 40, int), ('volume_spike_mult', 1.1, 3.0, float),
            ('atr_period', 7, 28, int), ('stop_atr_mult', 0.5, 3.0, float),
            ('target_atr_mult', 1.0, 5.0, float), ('trail_atr_mult', 0.5, 2.5, float),
        ]
        for name, lo, hi, dtype in fields_to_mutate:
            if rng.random() < mutation_rate:
                current = getattr(new, name)
                if dtype == int:
                    delta = rng.integers(-2, 3)
                    setattr(new, name, int(np.clip(current + delta, lo, hi)))
                else:
                    delta = rng.normal(0, (hi - lo) * 0.1)
                    setattr(new, name, float(np.clip(current + delta, lo, hi)))
        return new

    def crossover(self, other: 'StrategyParams', rng: np.random.Generator) -> 'StrategyParams':
        """Uniform crossover with another parameter set."""
        child = StrategyParams()
        for key in self.__dict__:
            if rng.random() < 0.5:
                setattr(child, key, getattr(self, key))
            else:
                setattr(child, key, getattr(other, key))
        return child


# ─── Indicator Functions ───────────────────────────────────────────

def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return result
    k = 2 / (period + 1)
    result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range — measures volatility."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[high[0] - low[0]], tr])
    return ema(tr, period)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    rs = np.where(avg_loss > 0, avg_gain / (avg_loss + 1e-10), 100.0)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_bands(close: np.ndarray, period: int, num_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands."""
    mid = np.full_like(close, np.nan, dtype=float)
    upper = np.full_like(close, np.nan, dtype=float)
    lower = np.full_like(close, np.nan, dtype=float)
    for i in range(period - 1, len(close)):
        window = close[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window)
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
    return upper, mid, lower


def macd(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD indicator."""
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    # Replace NaN with 0 for signal calculation
    macd_clean = np.nan_to_num(macd_line, nan=0.0)
    signal_line = ema(macd_clean, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def find_fractals(high: np.ndarray, low: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify Williams fractals — the building blocks of fractal market structure.

    A fractal high: bar with highest high in lookback window on both sides.
    A fractal low: bar with lowest low in lookback window on both sides.
    """
    n = len(high)
    fractal_highs = np.full(n, np.nan)
    fractal_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        # Fractal high
        if high[i] == np.max(high[i - lookback:i + lookback + 1]):
            fractal_highs[i] = high[i]
        # Fractal low
        if low[i] == np.min(low[i - lookback:i + lookback + 1]):
            fractal_lows[i] = low[i]

    return fractal_highs, fractal_lows


def volume_profile(volume: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Volume MA and relative volume."""
    vol_ma = ema(volume.astype(float), period)
    rel_vol = volume / (vol_ma + 1e-10)
    return vol_ma, rel_vol


# ─── Strategy Implementations ─────────────────────────────────────

class BaseStrategy:
    """Base class for all strategies."""

    def __init__(self, name: str, params: StrategyParams):
        self.name = name
        self.params = params

    def compute_signals(self, bars: pd.DataFrame) -> List[Dict]:
        """Compute signals for each bar. Returns list of signal dicts."""
        raise NotImplementedError


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover with trend filter.
    Long when fast crosses above slow AND price > trend EMA.
    Short when fast crosses below slow AND price < trend EMA.
    """

    def __init__(self, params: StrategyParams):
        super().__init__("EMA_Crossover", params)

    def compute_signals(self, bars: pd.DataFrame) -> List[Dict]:
        close = bars["close"].values
        p = self.params

        fast = ema(close, p.fast_ema)
        slow = ema(close, p.slow_ema)
        trend = ema(close, p.trend_ema)
        atr_vals = atr(bars["high"].values, bars["low"].values, close, p.atr_period)

        signals = []
        for i in range(1, len(close)):
            sig = {"bar_index": i, "signal": Signal.FLAT, "strategy": self.name,
                   "stop_distance": 0, "target_distance": 0, "reason": ""}

            if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(trend[i]) or np.isnan(atr_vals[i]):
                signals.append(sig)
                continue

            atr_v = atr_vals[i]

            # Bullish crossover
            if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1] and close[i] > trend[i]:
                sig["signal"] = Signal.LONG
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = atr_v * p.target_atr_mult
                sig["reason"] = f"Fast EMA({p.fast_ema}) crossed above Slow EMA({p.slow_ema}), above trend"

            # Bearish crossover
            elif fast[i] < slow[i] and fast[i - 1] >= slow[i - 1] and close[i] < trend[i]:
                sig["signal"] = Signal.SHORT
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = atr_v * p.target_atr_mult
                sig["reason"] = f"Fast EMA({p.fast_ema}) crossed below Slow EMA({p.slow_ema}), below trend"

            signals.append(sig)

        return signals


class FractalBreakoutStrategy(BaseStrategy):
    """
    Fractal Breakout — trades breaks of Williams fractals.

    This is inherently fractal: fractals are self-similar pivot points.
    A breakout above a fractal high = new structure forming.
    """

    def __init__(self, params: StrategyParams):
        super().__init__("Fractal_Breakout", params)

    def compute_signals(self, bars: pd.DataFrame) -> List[Dict]:
        high = bars["high"].values
        low = bars["low"].values
        close = bars["close"].values
        p = self.params

        frac_highs, frac_lows = find_fractals(high, low, p.fractal_lookback)
        atr_vals = atr(high, low, close, p.atr_period)

        # Track last confirmed fractal levels
        last_frac_high = np.nan
        last_frac_low = np.nan

        signals = []
        for i in range(1, len(close)):
            sig = {"bar_index": i, "signal": Signal.FLAT, "strategy": self.name,
                   "stop_distance": 0, "target_distance": 0, "reason": ""}

            # Update fractal levels (with confirmation delay)
            confirm_idx = i - p.fractal_confirmation_bars
            if confirm_idx >= 0 and not np.isnan(frac_highs[confirm_idx]):
                last_frac_high = frac_highs[confirm_idx]
            if confirm_idx >= 0 and not np.isnan(frac_lows[confirm_idx]):
                last_frac_low = frac_lows[confirm_idx]

            if np.isnan(last_frac_high) or np.isnan(last_frac_low) or np.isnan(atr_vals[i]):
                signals.append(sig)
                continue

            atr_v = atr_vals[i]
            threshold = atr_v * p.breakout_atr_mult

            # Breakout above fractal high
            if close[i] > last_frac_high + threshold and close[i - 1] <= last_frac_high + threshold:
                sig["signal"] = Signal.LONG
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = atr_v * p.target_atr_mult
                sig["reason"] = f"Breakout above fractal high {last_frac_high:.2f}"

            # Breakdown below fractal low
            elif close[i] < last_frac_low - threshold and close[i - 1] >= last_frac_low - threshold:
                sig["signal"] = Signal.SHORT
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = atr_v * p.target_atr_mult
                sig["reason"] = f"Breakdown below fractal low {last_frac_low:.2f}"

            signals.append(sig)

        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion using Bollinger Bands + RSI.

    Markets are fractal: they oscillate around a mean at every timeframe.
    This strategy captures the snap-back when price overextends.
    """

    def __init__(self, params: StrategyParams):
        super().__init__("Mean_Reversion", params)

    def compute_signals(self, bars: pd.DataFrame) -> List[Dict]:
        close = bars["close"].values
        p = self.params

        upper, mid, lower = bollinger_bands(close, p.bollinger_period, p.bollinger_std)
        rsi_vals = rsi(close, p.rsi_period)
        atr_vals = atr(bars["high"].values, bars["low"].values, close, p.atr_period)

        signals = []
        for i in range(1, len(close)):
            sig = {"bar_index": i, "signal": Signal.FLAT, "strategy": self.name,
                   "stop_distance": 0, "target_distance": 0, "reason": ""}

            if np.isnan(upper[i]) or np.isnan(rsi_vals[i]) or np.isnan(atr_vals[i]):
                signals.append(sig)
                continue

            atr_v = atr_vals[i]

            # Oversold bounce
            if close[i] <= lower[i] and rsi_vals[i] < p.rsi_oversold:
                sig["signal"] = Signal.LONG
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = mid[i] - close[i]  # Target the mean
                sig["reason"] = f"BB lower touch + RSI {rsi_vals[i]:.1f} < {p.rsi_oversold}"

            # Overbought fade
            elif close[i] >= upper[i] and rsi_vals[i] > p.rsi_overbought:
                sig["signal"] = Signal.SHORT
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = close[i] - mid[i]  # Target the mean
                sig["reason"] = f"BB upper touch + RSI {rsi_vals[i]:.1f} > {p.rsi_overbought}"

            signals.append(sig)

        return signals


class MomentumStrategy(BaseStrategy):
    """
    MACD + Volume momentum strategy.

    Trades strong directional moves confirmed by volume expansion.
    Fractal principle: momentum is self-similar across timeframes.
    """

    def __init__(self, params: StrategyParams):
        super().__init__("Momentum", params)

    def compute_signals(self, bars: pd.DataFrame) -> List[Dict]:
        close = bars["close"].values
        volume = bars["volume"].values
        p = self.params

        macd_line, signal_line, histogram = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
        vol_ma, rel_vol = volume_profile(volume, p.volume_ma_period)
        atr_vals = atr(bars["high"].values, bars["low"].values, close, p.atr_period)

        signals = []
        for i in range(1, len(close)):
            sig = {"bar_index": i, "signal": Signal.FLAT, "strategy": self.name,
                   "stop_distance": 0, "target_distance": 0, "reason": ""}

            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or np.isnan(atr_vals[i]):
                signals.append(sig)
                continue

            atr_v = atr_vals[i]
            vol_confirmed = rel_vol[i] > p.volume_spike_mult

            # MACD bullish cross with volume
            if (histogram[i] > 0 and histogram[i - 1] <= 0 and vol_confirmed):
                sig["signal"] = Signal.LONG
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = atr_v * p.target_atr_mult
                sig["reason"] = f"MACD bullish cross, volume {rel_vol[i]:.1f}x avg"

            # MACD bearish cross with volume
            elif (histogram[i] < 0 and histogram[i - 1] >= 0 and vol_confirmed):
                sig["signal"] = Signal.SHORT
                sig["stop_distance"] = atr_v * p.stop_atr_mult
                sig["target_distance"] = atr_v * p.target_atr_mult
                sig["reason"] = f"MACD bearish cross, volume {rel_vol[i]:.1f}x avg"

            signals.append(sig)

        return signals


class MultiTimeframeFractalStrategy(BaseStrategy):
    """
    Multi-Timeframe Fractal Confluence — the crown jewel.

    Combines signals from multiple timeframes using the fractal principle
    that structure at higher timeframes constrains lower timeframes.

    Uses: EMA trend on higher TF, fractal levels on mid TF, entry on lower TF.
    """

    def __init__(self, params: StrategyParams):
        super().__init__("MTF_Fractal", params)
        self.sub_strategies = [
            EMACrossoverStrategy(params),
            FractalBreakoutStrategy(params),
            MomentumStrategy(params),
        ]

    def compute_signals(self, bars: pd.DataFrame) -> List[Dict]:
        """Combine signals — require confluence from 2+ sub-strategies."""
        all_signals = [s.compute_signals(bars) for s in self.sub_strategies]
        close = bars["close"].values
        atr_vals = atr(bars["high"].values, bars["low"].values, close, self.params.atr_period)

        combined = []
        n = min(len(sigs) for sigs in all_signals)

        for i in range(n):
            long_votes = sum(1 for sigs in all_signals if sigs[i]["signal"] == Signal.LONG)
            short_votes = sum(1 for sigs in all_signals if sigs[i]["signal"] == Signal.SHORT)

            bar_idx = all_signals[0][i]["bar_index"]
            atr_v = atr_vals[bar_idx] if bar_idx < len(atr_vals) and not np.isnan(atr_vals[bar_idx]) else 10.0

            sig = {"bar_index": bar_idx, "signal": Signal.FLAT, "strategy": self.name,
                   "stop_distance": 0, "target_distance": 0, "reason": ""}

            if long_votes >= 2:
                reasons = [sigs[i]["reason"] for sigs in all_signals if sigs[i]["signal"] == Signal.LONG]
                sig["signal"] = Signal.LONG
                sig["stop_distance"] = atr_v * self.params.stop_atr_mult
                sig["target_distance"] = atr_v * self.params.target_atr_mult * 1.2  # Wider target for confluence
                sig["reason"] = f"Confluence ({long_votes}/3): " + " | ".join(reasons)

            elif short_votes >= 2:
                reasons = [sigs[i]["reason"] for sigs in all_signals if sigs[i]["signal"] == Signal.SHORT]
                sig["signal"] = Signal.SHORT
                sig["stop_distance"] = atr_v * self.params.stop_atr_mult
                sig["target_distance"] = atr_v * self.params.target_atr_mult * 1.2
                sig["reason"] = f"Confluence ({short_votes}/3): " + " | ".join(reasons)

            combined.append(sig)

        return combined


def get_all_strategies(params: StrategyParams) -> List[BaseStrategy]:
    """Return all available strategies."""
    return [
        EMACrossoverStrategy(params),
        FractalBreakoutStrategy(params),
        MeanReversionStrategy(params),
        MomentumStrategy(params),
        MultiTimeframeFractalStrategy(params),
    ]
