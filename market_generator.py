"""
Fractal NQ (Nasdaq-100 Futures) Market Data Generator

Generates realistic price bars using fractal geometry principles:
- Self-similar patterns across timeframes (1m, 5m, 15m, 1h, 4h, daily)
- Volatility clustering (GARCH-like behavior)
- Mean-reversion at macro scale, momentum at micro scale
- Realistic session structure (overnight, RTH open, midday, close)
- Volume profile that mirrors real NQ behavior
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class NQSessionConfig:
    """NQ futures session parameters based on real market structure."""
    base_price: float = 18500.0          # Starting NQ price level
    tick_size: float = 0.25              # NQ tick size
    point_value: float = 20.0            # $20 per point
    avg_daily_range: float = 250.0       # Average daily range in points
    avg_daily_volume: int = 450_000      # Average daily contracts
    overnight_volatility_ratio: float = 0.6
    rth_open_spike: float = 1.8          # Volatility multiplier at RTH open
    midday_lull: float = 0.5             # Volatility drops midday
    close_spike: float = 1.4             # Volatility picks up near close


class FractalNoiseGenerator:
    """
    Generates fractal Brownian motion for realistic price movement.

    Uses the Cholesky decomposition method to create correlated noise
    with a specified Hurst exponent. H > 0.5 = trending, H < 0.5 = mean-reverting.
    """

    def __init__(self, hurst: float = 0.55, seed: Optional[int] = None):
        self.hurst = hurst
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int) -> np.ndarray:
        """Generate fractal Brownian motion increments."""
        if n <= 0:
            return np.array([])

        # Build covariance matrix for fractional Brownian motion
        gamma = np.zeros(n)
        for k in range(n):
            gamma[k] = 0.5 * (
                abs(k - 1) ** (2 * self.hurst)
                - 2 * abs(k) ** (2 * self.hurst)
                + abs(k + 1) ** (2 * self.hurst)
            )

        # Use circulant embedding for efficient generation
        row = np.concatenate([gamma, gamma[-2:0:-1]])
        eigenvalues = np.fft.fft(row).real
        eigenvalues = np.maximum(eigenvalues, 0)

        sqrt_eigen = np.sqrt(eigenvalues)
        z = self.rng.standard_normal(len(row)) + 1j * self.rng.standard_normal(len(row))
        w = np.fft.ifft(sqrt_eigen * z).real[:n]

        return w / (np.std(w) + 1e-10)

    def set_hurst(self, h: float):
        self.hurst = np.clip(h, 0.01, 0.99)


class VolatilityEngine:
    """
    GARCH(1,1)-inspired volatility clustering.

    Produces realistic volatility that clusters — high vol begets high vol,
    low vol begets low vol — just like real NQ.
    """

    def __init__(self, omega: float = 0.05, alpha: float = 0.10, beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.current_var = 1.0
        self.last_return = 0.0

    def step(self, innovation: float) -> float:
        """Update and return current volatility level."""
        self.current_var = (
            self.omega
            + self.alpha * innovation ** 2
            + self.beta * self.current_var
        )
        self.last_return = innovation
        return np.sqrt(self.current_var)

    def reset(self, variance: float = 1.0):
        self.current_var = variance
        self.last_return = 0.0


class NQMarketGenerator:
    """
    Generates realistic NQ futures bars using fractal market principles.

    The market is fractal: patterns that appear on a 1-minute chart also
    appear on daily charts. This generator creates self-similar structure
    across all timeframes by layering multiple fractal noise processes.
    """

    def __init__(self, config: Optional[NQSessionConfig] = None, seed: Optional[int] = None):
        self.config = config or NQSessionConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Multi-scale fractal generators (self-similarity across timeframes)
        self.fractal_micro = FractalNoiseGenerator(hurst=0.45, seed=self._next_seed())   # Mean-reverting scalp
        self.fractal_intra = FractalNoiseGenerator(hurst=0.52, seed=self._next_seed())   # Slight trend intraday
        self.fractal_swing = FractalNoiseGenerator(hurst=0.58, seed=self._next_seed())   # Trending swing
        self.fractal_macro = FractalNoiseGenerator(hurst=0.55, seed=self._next_seed())   # Macro trend

        self.vol_engine = VolatilityEngine()
        self.current_price = self.config.base_price
        self.bar_count = 0

    def _next_seed(self) -> Optional[int]:
        if self.seed is not None:
            return self.rng.integers(0, 2**31)
        return None

    def _session_volatility_multiplier(self, bar_of_day: int, bars_per_day: int) -> float:
        """
        Model intraday volatility seasonality.
        NQ has high vol at open, drops midday, picks up into close.
        """
        pct_of_day = bar_of_day / max(bars_per_day, 1)

        if pct_of_day < 0.05:
            return self.config.rth_open_spike
        elif pct_of_day < 0.15:
            return 1.2
        elif pct_of_day < 0.5:
            return self.config.midday_lull + 0.3 * (0.5 - pct_of_day)
        elif pct_of_day < 0.85:
            return 0.7 + 0.5 * (pct_of_day - 0.5)
        else:
            return self.config.close_spike

    def _generate_volume(self, volatility: float, bar_of_day: int, bars_per_day: int) -> int:
        """Generate realistic volume correlated with volatility."""
        base_vol_per_bar = self.config.avg_daily_volume / max(bars_per_day, 1)
        session_mult = self._session_volatility_multiplier(bar_of_day, bars_per_day)
        noise = self.rng.lognormal(0, 0.4)
        return max(1, int(base_vol_per_bar * session_mult * noise * (0.5 + volatility)))

    def generate_bars(self, num_bars: int, timeframe_minutes: int = 5) -> pd.DataFrame:
        """
        Generate a sequence of realistic NQ price bars.

        Args:
            num_bars: Number of bars to generate
            timeframe_minutes: Bar timeframe (1, 5, 15, 60, 240, 1440)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        """
        bars_per_day = int(1440 / timeframe_minutes)
        scale_factor = np.sqrt(timeframe_minutes / 1440) * self.config.avg_daily_range

        # Generate multi-scale fractal noise
        noise_micro = self.fractal_micro.generate(num_bars)
        noise_intra = self.fractal_intra.generate(num_bars)
        noise_swing = self.fractal_swing.generate(num_bars)
        noise_macro = self.fractal_macro.generate(num_bars)

        # Blend scales — this creates self-similar fractal structure
        weights = self._scale_weights(timeframe_minutes)
        blended = (
            weights[0] * noise_micro
            + weights[1] * noise_intra
            + weights[2] * noise_swing
            + weights[3] * noise_macro
        )

        records = []
        base_ts = pd.Timestamp("2024-01-02 18:00:00")  # NQ globex open

        for i in range(num_bars):
            bar_of_day = (self.bar_count + i) % bars_per_day
            session_mult = self._session_volatility_multiplier(bar_of_day, bars_per_day)

            # Volatility clustering
            vol = self.vol_engine.step(blended[i])

            # Price change for this bar
            bar_move = blended[i] * scale_factor * session_mult * vol

            # Build OHLC from the move
            open_price = self.current_price
            direction = np.sign(bar_move) if bar_move != 0 else 1

            # Intrabar structure: wick ratios based on real NQ behavior
            body = abs(bar_move)
            upper_wick = abs(self.rng.normal(0, body * 0.3))
            lower_wick = abs(self.rng.normal(0, body * 0.3))

            if direction > 0:
                close_price = open_price + body
                high_price = close_price + upper_wick
                low_price = open_price - lower_wick
            else:
                close_price = open_price - body
                high_price = open_price + upper_wick
                low_price = close_price - lower_wick

            # Snap to tick size
            open_price = self._snap(open_price)
            high_price = self._snap(high_price)
            low_price = self._snap(low_price)
            close_price = self._snap(close_price)

            # Ensure OHLC consistency
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)

            volume = self._generate_volume(vol, bar_of_day, bars_per_day)
            vwap = (high_price + low_price + close_price) / 3

            ts = base_ts + pd.Timedelta(minutes=timeframe_minutes * (self.bar_count + i))

            records.append({
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "vwap": vwap,
                "bar_index": self.bar_count + i,
            })

            self.current_price = close_price

        self.bar_count += num_bars

        return pd.DataFrame(records)

    def _scale_weights(self, tf_minutes: int) -> List[float]:
        """Weight fractal scales based on timeframe — this is key to fractal self-similarity."""
        if tf_minutes <= 1:
            return [0.50, 0.25, 0.15, 0.10]
        elif tf_minutes <= 5:
            return [0.30, 0.35, 0.20, 0.15]
        elif tf_minutes <= 15:
            return [0.15, 0.35, 0.30, 0.20]
        elif tf_minutes <= 60:
            return [0.10, 0.20, 0.40, 0.30]
        else:
            return [0.05, 0.15, 0.30, 0.50]

    def _snap(self, price: float) -> float:
        """Snap price to NQ tick size."""
        return round(price / self.config.tick_size) * self.config.tick_size

    def reset(self, price: Optional[float] = None):
        """Reset generator to fresh state."""
        self.current_price = price or self.config.base_price
        self.bar_count = 0
        self.vol_engine.reset()


def resample_bars(bars: pd.DataFrame, target_tf_minutes: int, source_tf_minutes: int) -> pd.DataFrame:
    """
    Resample bars to a higher timeframe — fractal structure is preserved.
    """
    ratio = target_tf_minutes // source_tf_minutes
    if ratio <= 1:
        return bars.copy()

    groups = []
    for start in range(0, len(bars), ratio):
        chunk = bars.iloc[start:start + ratio]
        if len(chunk) == 0:
            continue
        groups.append({
            "timestamp": chunk.iloc[0]["timestamp"],
            "open": chunk.iloc[0]["open"],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk.iloc[-1]["close"],
            "volume": chunk["volume"].sum(),
            "vwap": (chunk["vwap"] * chunk["volume"]).sum() / max(chunk["volume"].sum(), 1),
            "bar_index": chunk.iloc[0]["bar_index"],
        })

    return pd.DataFrame(groups)
