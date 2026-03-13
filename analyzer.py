"""
AI Trade Analyzer — The Learning Brain

This is the intelligence layer. After every backtest cycle it:
  1. Analyzes every trade (why it won/lost, market context at entry/exit)
  2. Detects patterns across trades (time-of-day, volatility regime, trend state)
  3. Builds a statistical model of what works and what doesn't
  4. Generates concrete "insights" — actionable conclusions
  5. Translates insights into parameter adjustments for the optimizer
  6. Tracks which insights held up over time vs which were noise

The analyzer thinks in fractal terms: patterns at one timeframe
inform expectations at others.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time

from strategies import (
    StrategyParams, ema, atr, rsi, bollinger_bands, find_fractals
)
from backtester import BacktestResult, Trade


@dataclass
class MarketRegime:
    """Classified market state at a point in time."""
    trend: str          # "STRONG_UP", "UP", "FLAT", "DOWN", "STRONG_DOWN"
    volatility: str     # "LOW", "NORMAL", "HIGH", "EXTREME"
    momentum: str       # "BULLISH", "NEUTRAL", "BEARISH"
    session: str        # "OVERNIGHT", "OPEN", "MIDDAY", "CLOSE"
    fractal_position: str  # "NEAR_HIGH", "MID_RANGE", "NEAR_LOW"


@dataclass
class TradeAnalysis:
    """Deep analysis of a single trade."""
    trade: Trade
    regime_at_entry: MarketRegime
    regime_at_exit: MarketRegime
    entry_atr: float
    risk_reward_actual: float       # Actual R:R achieved
    risk_reward_planned: float      # Planned R:R (target/stop)
    efficiency: float               # How much of MFE was captured (0-1)
    heat: float                     # MAE as fraction of stop distance
    held_too_long: bool             # Would earlier exit have been better?
    context_score: float            # How favorable was the context? (-1 to 1)


@dataclass
class Insight:
    """A concrete conclusion drawn from analyzing trades."""
    id: int
    created_cycle: int
    category: str           # "REGIME", "TIMING", "RISK", "STRATEGY", "FRACTAL"
    conclusion: str         # Human-readable insight
    confidence: float       # 0-1, how confident are we
    sample_size: int        # How many trades support this
    supporting_stats: Dict  # Statistical evidence
    param_adjustments: Dict # Suggested parameter changes
    validated: bool = False # Has this held up in subsequent cycles?
    validation_count: int = 0
    invalidation_count: int = 0
    active: bool = True


class MarketContextEngine:
    """
    Classifies market conditions from OHLCV bars.
    Used to understand WHAT the market was doing when trades were taken.
    """

    @staticmethod
    def classify_regime(bars: pd.DataFrame, bar_index: int) -> MarketRegime:
        """Classify the market regime at a specific bar."""
        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        n = len(close)

        # Need enough history
        lookback = min(bar_index, 50)
        if lookback < 10:
            return MarketRegime("FLAT", "NORMAL", "NEUTRAL", "MIDDAY", "MID_RANGE")

        window = close[max(0, bar_index - lookback):bar_index + 1]
        h_window = high[max(0, bar_index - lookback):bar_index + 1]
        l_window = low[max(0, bar_index - lookback):bar_index + 1]

        # TREND: based on price vs moving averages and slope
        if len(window) >= 20:
            ma20 = np.mean(window[-20:])
            ma50 = np.mean(window[-min(50, len(window)):])
            price = close[bar_index]
            pct_from_ma20 = (price - ma20) / ma20 * 100

            if pct_from_ma20 > 1.5:
                trend = "STRONG_UP"
            elif pct_from_ma20 > 0.3:
                trend = "UP"
            elif pct_from_ma20 < -1.5:
                trend = "STRONG_DOWN"
            elif pct_from_ma20 < -0.3:
                trend = "DOWN"
            else:
                trend = "FLAT"
        else:
            trend = "FLAT"

        # VOLATILITY: based on ATR relative to price
        atr_vals = atr(high[:bar_index + 1], low[:bar_index + 1], close[:bar_index + 1], 14)
        current_atr = atr_vals[bar_index] if bar_index < len(atr_vals) and not np.isnan(atr_vals[bar_index]) else 0
        atr_pct = current_atr / close[bar_index] * 100 if close[bar_index] > 0 else 0

        if atr_pct > 2.0:
            volatility = "EXTREME"
        elif atr_pct > 1.0:
            volatility = "HIGH"
        elif atr_pct > 0.4:
            volatility = "NORMAL"
        else:
            volatility = "LOW"

        # MOMENTUM: based on RSI
        rsi_vals = rsi(close[:bar_index + 1], 14)
        current_rsi = rsi_vals[bar_index] if bar_index < len(rsi_vals) and not np.isnan(rsi_vals[bar_index]) else 50

        if current_rsi > 65:
            momentum = "BULLISH"
        elif current_rsi < 35:
            momentum = "BEARISH"
        else:
            momentum = "NEUTRAL"

        # SESSION: based on bar position in day
        bars_per_day = 288  # Approximate for 5m bars
        bar_of_day = bar_index % bars_per_day
        day_pct = bar_of_day / bars_per_day

        if day_pct < 0.1:
            session = "OPEN"
        elif day_pct < 0.4:
            session = "MORNING"
        elif day_pct < 0.7:
            session = "MIDDAY"
        else:
            session = "CLOSE"

        # FRACTAL POSITION: where are we in the recent range?
        recent_high = np.max(h_window)
        recent_low = np.min(l_window)
        rng = recent_high - recent_low
        if rng > 0:
            position_pct = (close[bar_index] - recent_low) / rng
            if position_pct > 0.75:
                fractal_pos = "NEAR_HIGH"
            elif position_pct < 0.25:
                fractal_pos = "NEAR_LOW"
            else:
                fractal_pos = "MID_RANGE"
        else:
            fractal_pos = "MID_RANGE"

        return MarketRegime(trend, volatility, momentum, session, fractal_pos)


class AITradeAnalyzer:
    """
    The AI brain that learns from every trade.

    Maintains a growing knowledge base of insights derived from
    statistical analysis of trade outcomes across different market conditions.
    Now includes cross-instrument correlation analysis, NQ directional opinion,
    and overfit detection.
    """

    MAX_ANALYSES = 5000  # Cap to prevent unbounded memory growth

    def __init__(self):
        self.all_analyses: List[TradeAnalysis] = []
        self.insights: List[Insight] = []
        self.next_insight_id = 1
        self.cycle_count = 0

        # Accumulated statistics
        self.stats_by_regime = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        self.stats_by_strategy = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        self.stats_by_direction = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        self.stats_by_session = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        self.stats_by_volatility = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        self.stats_by_exit_reason = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})

        # Pattern detection accumulators
        self.mfe_mae_ratios: List[float] = []
        self.holding_periods: List[int] = []
        self.efficiency_scores: List[float] = []

        # Cross-instrument correlation data
        self.correlation_data: Dict = {}

    def analyze_cycle(self, result: BacktestResult, bars: pd.DataFrame,
                      correlations: Optional[Dict] = None) -> List[Insight]:
        """
        Analyze all trades from a backtest cycle.
        Returns new insights generated.
        """
        self.cycle_count += 1
        new_insights = []

        # Store correlation data if provided
        if correlations:
            self.correlation_data = correlations

        if not result.trades or len(result.trades) < 3:
            return new_insights

        # Deep-analyze each trade
        cycle_analyses = []
        for trade in result.trades:
            analysis = self._analyze_trade(trade, bars)
            if analysis:
                cycle_analyses.append(analysis)
                self.all_analyses.append(analysis)
                self._update_stats(analysis)

        # Cap memory growth
        if len(self.all_analyses) > self.MAX_ANALYSES:
            self.all_analyses = self.all_analyses[-self.MAX_ANALYSES:]

        if len(cycle_analyses) < 3:
            return new_insights

        # Generate insights from accumulated data
        new_insights.extend(self._detect_regime_patterns())
        new_insights.extend(self._detect_timing_patterns())
        new_insights.extend(self._detect_risk_patterns())
        new_insights.extend(self._detect_strategy_patterns())
        new_insights.extend(self._detect_fractal_patterns(bars))
        new_insights.extend(self._detect_cross_instrument_patterns())
        new_insights.extend(self._detect_overfit_patterns(result))

        # Validate existing insights against this cycle
        self._validate_insights(cycle_analyses)

        # Prune low-confidence or invalidated insights
        self._prune_insights()

        return new_insights

    def _analyze_trade(self, trade: Trade, bars: pd.DataFrame) -> Optional[TradeAnalysis]:
        """Deep analysis of a single trade."""
        if trade.entry_bar >= len(bars) or trade.exit_bar >= len(bars):
            return None

        regime_entry = MarketContextEngine.classify_regime(bars, trade.entry_bar)
        regime_exit = MarketContextEngine.classify_regime(bars, trade.exit_bar)

        # ATR at entry
        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        atr_vals = atr(high, low, close, 14)
        entry_atr = atr_vals[trade.entry_bar] if trade.entry_bar < len(atr_vals) and not np.isnan(atr_vals[trade.entry_bar]) else 10.0

        # Risk:reward
        stop_dist = abs(trade.entry_price - trade.stop_price) if trade.stop_price else entry_atr
        target_dist = abs(trade.target_price - trade.entry_price) if trade.target_price else entry_atr * 2

        rr_planned = target_dist / stop_dist if stop_dist > 0 else 1.0
        rr_actual = trade.pnl_points / stop_dist if stop_dist > 0 else 0.0

        # Efficiency: how much of the max favorable excursion did we capture?
        efficiency = trade.pnl_points / trade.max_favorable if trade.max_favorable > 0 else 0.0
        efficiency = max(-1.0, min(1.0, efficiency))

        # Heat: how close did we get to stop (MAE / stop distance)
        heat = trade.max_adverse / stop_dist if stop_dist > 0 else 0.0

        # Would earlier exit have been better?
        held_too_long = (trade.max_favorable > abs(trade.pnl_points) * 2 and trade.pnl_dollars <= 0)

        # Context score: combine regime factors
        context_score = 0.0
        if trade.direction == "LONG":
            if regime_entry.trend in ("STRONG_UP", "UP"):
                context_score += 0.3
            elif regime_entry.trend in ("STRONG_DOWN", "DOWN"):
                context_score -= 0.3
            if regime_entry.momentum == "BULLISH":
                context_score += 0.2
            elif regime_entry.momentum == "BEARISH":
                context_score -= 0.2
        else:
            if regime_entry.trend in ("STRONG_DOWN", "DOWN"):
                context_score += 0.3
            elif regime_entry.trend in ("STRONG_UP", "UP"):
                context_score -= 0.3
            if regime_entry.momentum == "BEARISH":
                context_score += 0.2
            elif regime_entry.momentum == "BULLISH":
                context_score -= 0.2

        if regime_entry.volatility == "EXTREME":
            context_score -= 0.2
        elif regime_entry.volatility == "LOW":
            context_score -= 0.1

        self.mfe_mae_ratios.append(trade.max_favorable / (trade.max_adverse + 0.01))
        self.holding_periods.append(trade.bars_held)
        self.efficiency_scores.append(efficiency)

        return TradeAnalysis(
            trade=trade,
            regime_at_entry=regime_entry,
            regime_at_exit=regime_exit,
            entry_atr=entry_atr,
            risk_reward_actual=rr_actual,
            risk_reward_planned=rr_planned,
            efficiency=efficiency,
            heat=heat,
            held_too_long=held_too_long,
            context_score=context_score,
        )

    def _update_stats(self, analysis: TradeAnalysis):
        """Update accumulated statistics."""
        t = analysis.trade
        won = t.pnl_dollars > 0

        # By regime
        regime_key = f"{analysis.regime_at_entry.trend}_{analysis.regime_at_entry.volatility}"
        s = self.stats_by_regime[regime_key]
        s["wins" if won else "losses"] += 1
        s["pnl"] += t.pnl_dollars
        s["count"] += 1

        # By strategy
        s = self.stats_by_strategy[t.strategy]
        s["wins" if won else "losses"] += 1
        s["pnl"] += t.pnl_dollars
        s["count"] += 1

        # By direction
        s = self.stats_by_direction[t.direction]
        s["wins" if won else "losses"] += 1
        s["pnl"] += t.pnl_dollars
        s["count"] += 1

        # By session
        s = self.stats_by_session[analysis.regime_at_entry.session]
        s["wins" if won else "losses"] += 1
        s["pnl"] += t.pnl_dollars
        s["count"] += 1

        # By volatility
        s = self.stats_by_volatility[analysis.regime_at_entry.volatility]
        s["wins" if won else "losses"] += 1
        s["pnl"] += t.pnl_dollars
        s["count"] += 1

        # By exit reason
        s = self.stats_by_exit_reason[t.exit_reason]
        s["wins" if won else "losses"] += 1
        s["pnl"] += t.pnl_dollars
        s["count"] += 1

    def _detect_regime_patterns(self) -> List[Insight]:
        """Detect which market regimes are profitable vs dangerous."""
        insights = []

        for regime, stats in self.stats_by_regime.items():
            if stats["count"] < 10:
                continue

            wr = stats["wins"] / stats["count"]
            avg_pnl = stats["pnl"] / stats["count"]

            # Strong regime patterns
            if wr > 0.65 and avg_pnl > 50:
                insights.append(self._create_insight(
                    "REGIME",
                    f"Regime '{regime}' is highly favorable: {wr:.0%} win rate, ${avg_pnl:.0f}/trade avg",
                    confidence=min(0.9, 0.5 + stats["count"] / 100),
                    sample_size=stats["count"],
                    stats={"win_rate": wr, "avg_pnl": avg_pnl, "regime": regime},
                    adjustments={}  # Regime insights inform strategy selection, not params
                ))

            elif wr < 0.35 and avg_pnl < -50:
                insights.append(self._create_insight(
                    "REGIME",
                    f"AVOID regime '{regime}': {wr:.0%} win rate, ${avg_pnl:.0f}/trade avg — destructive",
                    confidence=min(0.9, 0.5 + stats["count"] / 100),
                    sample_size=stats["count"],
                    stats={"win_rate": wr, "avg_pnl": avg_pnl, "regime": regime},
                    adjustments={}
                ))

        return insights

    def _detect_timing_patterns(self) -> List[Insight]:
        """Detect time-of-day and session patterns."""
        insights = []

        best_session = None
        best_wr = 0
        worst_session = None
        worst_wr = 1.0

        for session, stats in self.stats_by_session.items():
            if stats["count"] < 10:
                continue
            wr = stats["wins"] / stats["count"]
            if wr > best_wr:
                best_wr = wr
                best_session = session
            if wr < worst_wr:
                worst_wr = wr
                worst_session = session

        if best_session and best_wr > 0.55:
            stats = self.stats_by_session[best_session]
            insights.append(self._create_insight(
                "TIMING",
                f"Best session is '{best_session}': {best_wr:.0%} win rate across {stats['count']} trades",
                confidence=min(0.85, 0.4 + stats["count"] / 80),
                sample_size=stats["count"],
                stats={"session": best_session, "win_rate": best_wr},
                adjustments={}
            ))

        if worst_session and worst_wr < 0.40:
            stats = self.stats_by_session[worst_session]
            insights.append(self._create_insight(
                "TIMING",
                f"Worst session is '{worst_session}': {worst_wr:.0%} win rate — consider avoiding",
                confidence=min(0.85, 0.4 + stats["count"] / 80),
                sample_size=stats["count"],
                stats={"session": worst_session, "win_rate": worst_wr},
                adjustments={}
            ))

        return insights

    def _detect_risk_patterns(self) -> List[Insight]:
        """Detect risk management patterns (stops, targets, position sizing)."""
        insights = []

        if len(self.all_analyses) < 20:
            return insights

        recent = self.all_analyses[-200:]  # Look at recent trades

        # Efficiency analysis: are we capturing enough of MFE?
        efficiencies = [a.efficiency for a in recent]
        avg_eff = np.mean(efficiencies)

        if avg_eff < 0.3:
            insights.append(self._create_insight(
                "RISK",
                f"Trade efficiency is low ({avg_eff:.0%}). Capturing only {avg_eff:.0%} of max favorable moves. "
                f"Consider tighter targets or trailing stops.",
                confidence=0.75,
                sample_size=len(recent),
                stats={"avg_efficiency": avg_eff, "median_efficiency": float(np.median(efficiencies))},
                adjustments={"target_atr_mult": -0.3, "trail_atr_mult": -0.2}
            ))
        elif avg_eff > 0.7:
            insights.append(self._create_insight(
                "RISK",
                f"Excellent trade efficiency ({avg_eff:.0%}). Current risk params are working well.",
                confidence=0.7,
                sample_size=len(recent),
                stats={"avg_efficiency": avg_eff},
                adjustments={}
            ))

        # Heat analysis: are stops too tight?
        heats = [a.heat for a in recent]
        avg_heat = np.mean(heats)
        stopped_out = [a for a in recent if a.trade.exit_reason == "STOP"]
        stop_rate = len(stopped_out) / len(recent) if recent else 0

        if stop_rate > 0.5 and avg_heat > 0.8:
            insights.append(self._create_insight(
                "RISK",
                f"Stops too tight: {stop_rate:.0%} of trades stopped out, avg heat {avg_heat:.0%}. "
                f"Widen stops to give trades room.",
                confidence=0.8,
                sample_size=len(recent),
                stats={"stop_rate": stop_rate, "avg_heat": avg_heat},
                adjustments={"stop_atr_mult": 0.3}
            ))
        elif stop_rate < 0.2:
            insights.append(self._create_insight(
                "RISK",
                f"Low stop rate ({stop_rate:.0%}): stops may be too wide, risking too much per trade.",
                confidence=0.6,
                sample_size=len(recent),
                stats={"stop_rate": stop_rate, "avg_heat": avg_heat},
                adjustments={"stop_atr_mult": -0.2}
            ))

        # Holding period analysis
        winners = [a for a in recent if a.trade.pnl_dollars > 0]
        losers = [a for a in recent if a.trade.pnl_dollars <= 0]

        if winners and losers:
            avg_win_bars = np.mean([a.trade.bars_held for a in winners])
            avg_loss_bars = np.mean([a.trade.bars_held for a in losers])

            if avg_loss_bars > avg_win_bars * 1.5:
                insights.append(self._create_insight(
                    "RISK",
                    f"Losers held too long: avg {avg_loss_bars:.0f} bars vs winners {avg_win_bars:.0f} bars. "
                    f"Cut losses faster.",
                    confidence=0.7,
                    sample_size=len(recent),
                    stats={"avg_win_bars": avg_win_bars, "avg_loss_bars": avg_loss_bars},
                    adjustments={"stop_atr_mult": -0.2}
                ))

        # Held-too-long detection
        held_too_long = [a for a in recent if a.held_too_long]
        htl_rate = len(held_too_long) / len(recent) if recent else 0
        if htl_rate > 0.3:
            insights.append(self._create_insight(
                "RISK",
                f"{htl_rate:.0%} of trades had large MFE but ended as losses — need better exit management. "
                f"Tighten trailing stop.",
                confidence=0.8,
                sample_size=len(recent),
                stats={"held_too_long_rate": htl_rate},
                adjustments={"trail_atr_mult": -0.3}
            ))

        return insights

    def _detect_strategy_patterns(self) -> List[Insight]:
        """Detect which strategies work best and worst."""
        insights = []

        if len(self.stats_by_strategy) < 2:
            return insights

        best_strat = None
        best_pnl = float('-inf')
        worst_strat = None
        worst_pnl = float('inf')

        for strat, stats in self.stats_by_strategy.items():
            if stats["count"] < 10:
                continue
            avg = stats["pnl"] / stats["count"]
            if avg > best_pnl:
                best_pnl = avg
                best_strat = strat
            if avg < worst_pnl:
                worst_pnl = avg
                worst_strat = strat

        if best_strat and best_pnl > 0:
            stats = self.stats_by_strategy[best_strat]
            wr = stats["wins"] / stats["count"]
            insights.append(self._create_insight(
                "STRATEGY",
                f"Best performing strategy: {best_strat} (${best_pnl:.0f}/trade, {wr:.0%} WR, {stats['count']} trades)",
                confidence=min(0.9, 0.5 + stats["count"] / 100),
                sample_size=stats["count"],
                stats={"strategy": best_strat, "avg_pnl": best_pnl, "win_rate": wr},
                adjustments={}
            ))

        if worst_strat and worst_pnl < -20:
            stats = self.stats_by_strategy[worst_strat]
            wr = stats["wins"] / stats["count"]
            insights.append(self._create_insight(
                "STRATEGY",
                f"Worst strategy: {worst_strat} (${worst_pnl:.0f}/trade, {wr:.0%} WR) — needs parameter rework",
                confidence=min(0.85, 0.5 + stats["count"] / 100),
                sample_size=stats["count"],
                stats={"strategy": worst_strat, "avg_pnl": worst_pnl, "win_rate": wr},
                adjustments={}
            ))

        # Direction bias
        for direction, stats in self.stats_by_direction.items():
            if stats["count"] < 20:
                continue
            wr = stats["wins"] / stats["count"]
            avg = stats["pnl"] / stats["count"]
            if abs(wr - 0.5) > 0.1:
                insights.append(self._create_insight(
                    "STRATEGY",
                    f"{direction} trades: {wr:.0%} WR, ${avg:.0f}/trade across {stats['count']} trades "
                    f"— {'favored' if wr > 0.5 else 'avoid'}",
                    confidence=min(0.8, 0.4 + stats["count"] / 100),
                    sample_size=stats["count"],
                    stats={"direction": direction, "win_rate": wr, "avg_pnl": avg},
                    adjustments={}
                ))

        return insights

    def _detect_fractal_patterns(self, bars: pd.DataFrame) -> List[Insight]:
        """Detect fractal-specific patterns in the trade data."""
        insights = []

        if len(self.all_analyses) < 20:
            return insights

        recent = self.all_analyses[-200:]

        # Fractal position analysis: do we win more near highs, lows, or mid-range?
        position_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        for a in recent:
            pos = a.regime_at_entry.fractal_position
            won = a.trade.pnl_dollars > 0
            position_stats[pos]["wins" if won else "losses"] += 1
            position_stats[pos]["pnl"] += a.trade.pnl_dollars
            position_stats[pos]["count"] += 1

        for pos, stats in position_stats.items():
            if stats["count"] < 8:
                continue
            wr = stats["wins"] / stats["count"]
            avg_pnl = stats["pnl"] / stats["count"]

            if abs(wr - 0.5) > 0.12:
                insights.append(self._create_insight(
                    "FRACTAL",
                    f"Fractal position '{pos}': {wr:.0%} WR, ${avg_pnl:.0f}/trade — "
                    f"{'sweet spot' if wr > 0.5 else 'danger zone'} for entries",
                    confidence=min(0.75, 0.4 + stats["count"] / 60),
                    sample_size=stats["count"],
                    stats={"position": pos, "win_rate": wr, "avg_pnl": avg_pnl},
                    adjustments={}
                ))

        # Volatility regime analysis
        for vol, stats in self.stats_by_volatility.items():
            if stats["count"] < 10:
                continue
            wr = stats["wins"] / stats["count"]
            avg_pnl = stats["pnl"] / stats["count"]

            if vol == "EXTREME" and wr < 0.45:
                insights.append(self._create_insight(
                    "FRACTAL",
                    f"Extreme volatility hurts: {wr:.0%} WR in extreme vol. "
                    f"Widen stops in high-vol regimes or reduce size.",
                    confidence=0.75,
                    sample_size=stats["count"],
                    stats={"volatility": vol, "win_rate": wr, "avg_pnl": avg_pnl},
                    adjustments={"stop_atr_mult": 0.5}
                ))
            elif vol == "NORMAL" and wr > 0.55:
                insights.append(self._create_insight(
                    "FRACTAL",
                    f"Normal volatility is the sweet spot: {wr:.0%} WR. Optimize for this regime.",
                    confidence=0.7,
                    sample_size=stats["count"],
                    stats={"volatility": vol, "win_rate": wr, "avg_pnl": avg_pnl},
                    adjustments={}
                ))

        return insights

    def _validate_insights(self, cycle_analyses: List[TradeAnalysis]):
        """Check if existing insights held up in this cycle's data."""
        for insight in self.insights:
            if not insight.active:
                continue

            # Check regime insights
            if insight.category == "REGIME" and "regime" in insight.supporting_stats:
                regime = insight.supporting_stats["regime"]
                matching = [a for a in cycle_analyses
                            if f"{a.regime_at_entry.trend}_{a.regime_at_entry.volatility}" == regime]
                if len(matching) >= 3:
                    wr = sum(1 for a in matching if a.trade.pnl_dollars > 0) / len(matching)
                    expected_good = insight.supporting_stats.get("win_rate", 0.5) > 0.5

                    if (wr > 0.5) == expected_good:
                        insight.validation_count += 1
                    else:
                        insight.invalidation_count += 1

            # Validate risk insights by checking if adjustments helped
            if insight.category == "RISK" and insight.param_adjustments:
                # If we made the adjustment and performance improved, validate
                if len(cycle_analyses) > 5:
                    cycle_wr = sum(1 for a in cycle_analyses if a.trade.pnl_dollars > 0) / len(cycle_analyses)
                    if cycle_wr > 0.5:
                        insight.validation_count += 1
                    else:
                        insight.invalidation_count += 1

            # Update confidence based on validation
            total_checks = insight.validation_count + insight.invalidation_count
            if total_checks > 0:
                validation_rate = insight.validation_count / total_checks
                insight.confidence = 0.5 * insight.confidence + 0.5 * validation_rate
                insight.validated = validation_rate > 0.6

    def _prune_insights(self):
        """Remove insights that have been invalidated."""
        for insight in self.insights:
            total = insight.validation_count + insight.invalidation_count
            if total >= 5 and insight.confidence < 0.3:
                insight.active = False

        # Keep only active + last 20 inactive for history
        active = [i for i in self.insights if i.active]
        inactive = [i for i in self.insights if not i.active][-20:]
        self.insights = active + inactive

    def _create_insight(self, category: str, conclusion: str, confidence: float,
                        sample_size: int, stats: Dict, adjustments: Dict) -> Insight:
        """Create a new insight, deduplicating against existing ones."""
        # Check if we already have a similar insight
        for existing in self.insights:
            if existing.category == category and existing.active:
                # Check for similar stats
                if self._insights_similar(existing.supporting_stats, stats):
                    # Update existing instead of creating new
                    existing.confidence = max(existing.confidence, confidence)
                    existing.sample_size = max(existing.sample_size, sample_size)
                    existing.supporting_stats = stats
                    existing.conclusion = conclusion
                    return existing

        insight = Insight(
            id=self.next_insight_id,
            created_cycle=self.cycle_count,
            category=category,
            conclusion=conclusion,
            confidence=confidence,
            sample_size=sample_size,
            supporting_stats=stats,
            param_adjustments=adjustments,
        )
        self.next_insight_id += 1
        self.insights.append(insight)
        return insight

    def _insights_similar(self, stats1: Dict, stats2: Dict) -> bool:
        """Check if two stat dicts are about the same thing."""
        # Same keys that are strings match
        for key in ("regime", "session", "strategy", "direction", "position", "volatility"):
            if key in stats1 and key in stats2:
                return stats1[key] == stats2[key]
        return False

    def _detect_cross_instrument_patterns(self) -> List[Insight]:
        """Detect cross-instrument correlation patterns."""
        insights = []
        if not self.correlation_data:
            return insights

        nq_corrs = self.correlation_data.get("nq_correlations", [])
        if not nq_corrs:
            return insights

        # Highlight highly correlated instruments
        high_corr = [c for c in nq_corrs if abs(c.get("correlation", 0)) > 0.7]
        if high_corr:
            top = high_corr[:3]
            names = ", ".join(f"{c['instrument']} ({c['correlation']:.2f})" for c in top)
            insights.append(self._create_insight(
                "CROSS_INSTRUMENT",
                f"Top NQ-correlated instruments: {names} — patterns in these may predict NQ moves",
                confidence=0.65,
                sample_size=len(high_corr),
                stats={"top_correlated": [c["instrument"] for c in top]},
                adjustments={}
            ))

        # Pattern match insights
        matches = self.correlation_data.get("pattern_matches", [])
        if matches:
            bullish = [m for m in matches if m.get("subsequent_move_pct", 0) > 0.5]
            bearish = [m for m in matches if m.get("subsequent_move_pct", 0) < -0.5]
            if len(bullish) > len(bearish) and bullish:
                insights.append(self._create_insight(
                    "CROSS_INSTRUMENT",
                    f"Cross-instrument pattern matching: {len(bullish)} similar patterns had bullish outcomes vs {len(bearish)} bearish",
                    confidence=min(0.7, 0.4 + len(bullish) / 10),
                    sample_size=len(matches),
                    stats={"bullish_matches": len(bullish), "bearish_matches": len(bearish)},
                    adjustments={}
                ))
            elif len(bearish) > len(bullish) and bearish:
                insights.append(self._create_insight(
                    "CROSS_INSTRUMENT",
                    f"Cross-instrument pattern matching: {len(bearish)} similar patterns had bearish outcomes vs {len(bullish)} bullish",
                    confidence=min(0.7, 0.4 + len(bearish) / 10),
                    sample_size=len(matches),
                    stats={"bullish_matches": len(bullish), "bearish_matches": len(bearish)},
                    adjustments={}
                ))

        return insights

    def _detect_overfit_patterns(self, result: BacktestResult) -> List[Insight]:
        """Flag potential overfitting patterns."""
        insights = []

        if result.win_rate >= 0.95 and result.total_trades >= 5:
            insights.append(self._create_insight(
                "OVERFIT",
                f"Possible overfitting: {result.win_rate:.0%} win rate across {result.total_trades} trades is unrealistic",
                confidence=0.85,
                sample_size=result.total_trades,
                stats={"win_rate": result.win_rate, "trades": result.total_trades},
                adjustments={}
            ))

        # All trades same direction
        if result.trades and len(result.trades) >= 5:
            directions = set(t.direction for t in result.trades)
            if len(directions) == 1 and result.win_rate > 0.8:
                insights.append(self._create_insight(
                    "OVERFIT",
                    f"All {len(result.trades)} trades are {list(directions)[0]} with {result.win_rate:.0%} WR — may be fitting to trend direction",
                    confidence=0.7,
                    sample_size=result.total_trades,
                    stats={"direction": list(directions)[0], "win_rate": result.win_rate},
                    adjustments={}
                ))

        # MFE/MAE suspiciously high
        if result.avg_mfe > 0 and result.avg_mae > 0:
            ratio = result.avg_mfe / (result.avg_mae + 0.01)
            if ratio > 10 and result.total_trades >= 5:
                insights.append(self._create_insight(
                    "OVERFIT",
                    f"Suspiciously high MFE/MAE ratio ({ratio:.1f}x) — trades may be fitting to known price movements",
                    confidence=0.75,
                    sample_size=result.total_trades,
                    stats={"mfe_mae_ratio": ratio},
                    adjustments={}
                ))

        return insights

    def get_nq_opinion(self) -> Dict:
        """Synthesize a directional opinion on NQ."""
        # Gather signal components
        direction_signals = []
        reasoning = []
        best_strategies = []

        # From strategy stats
        for strat, stats in self.stats_by_strategy.items():
            if stats["count"] >= 10:
                wr = stats["wins"] / stats["count"]
                avg_pnl = stats["pnl"] / stats["count"]
                if avg_pnl > 0:
                    best_strategies.append({"name": strat, "win_rate": round(wr, 2), "avg_pnl": round(avg_pnl, 0)})

        best_strategies.sort(key=lambda x: x["avg_pnl"], reverse=True)

        # From direction stats
        for direction, stats in self.stats_by_direction.items():
            if stats["count"] >= 10:
                wr = stats["wins"] / stats["count"]
                if direction == "LONG" and wr > 0.55:
                    direction_signals.append(1)
                    reasoning.append(f"LONG trades winning {wr:.0%}")
                elif direction == "LONG" and wr < 0.45:
                    direction_signals.append(-1)
                    reasoning.append(f"LONG trades losing {wr:.0%}")
                elif direction == "SHORT" and wr > 0.55:
                    direction_signals.append(-1)
                    reasoning.append(f"SHORT trades winning {wr:.0%}")
                elif direction == "SHORT" and wr < 0.45:
                    direction_signals.append(1)
                    reasoning.append(f"SHORT trades losing {wr:.0%}")

        # From correlation data
        regimes = self.correlation_data.get("instrument_regimes", {})
        nq_regime = regimes.get("NQ=F") or regimes.get("QQQ") or {}
        if nq_regime:
            if nq_regime.get("trend") in ("STRONG_UP", "UP"):
                direction_signals.append(1)
                reasoning.append(f"NQ trend: {nq_regime.get('trend')}")
            elif nq_regime.get("trend") in ("STRONG_DOWN", "DOWN"):
                direction_signals.append(-1)
                reasoning.append(f"NQ trend: {nq_regime.get('trend')}")
            if nq_regime.get("momentum") == "BULLISH":
                direction_signals.append(0.5)
            elif nq_regime.get("momentum") == "BEARISH":
                direction_signals.append(-0.5)

        # Compute aggregate
        if direction_signals:
            avg_signal = np.mean(direction_signals)
            confidence = min(0.9, abs(avg_signal) * 0.7 + len(direction_signals) * 0.05)
        else:
            avg_signal = 0
            confidence = 0

        if avg_signal > 0.2:
            direction = "BULLISH"
        elif avg_signal < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Correlated signals
        corr_signals = []
        nq_corrs = self.correlation_data.get("nq_correlations", [])
        for c in nq_corrs[:5]:
            inst = c.get("instrument", "")
            regime = regimes.get(inst, {})
            if regime:
                corr_signals.append(f"{inst}: {regime.get('trend', '?')} ({c.get('correlation', 0):.2f})")

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "reasoning": reasoning[:5],
            "best_strategies": best_strategies[:5],
            "regime": nq_regime if nq_regime else {"trend": "UNKNOWN", "volatility": "UNKNOWN", "momentum": "UNKNOWN"},
            "correlated_signals": corr_signals[:5],
        }

    def get_param_adjustments(self) -> Dict[str, float]:
        """
        Aggregate parameter adjustments from all active, high-confidence insights.
        This is how the AI's learnings feed back into the optimizer.
        """
        adjustments = defaultdict(float)
        weights_sum = defaultdict(float)

        for insight in self.insights:
            if not insight.active or insight.confidence < 0.5:
                continue
            if not insight.param_adjustments:
                continue

            weight = insight.confidence * min(1.0, insight.sample_size / 20)
            for param, delta in insight.param_adjustments.items():
                adjustments[param] += delta * weight
                weights_sum[param] += weight

        # Normalize
        result = {}
        for param in adjustments:
            if weights_sum[param] > 0:
                result[param] = adjustments[param] / weights_sum[param]

        return result

    def get_state(self) -> Dict:
        """Get current analyzer state for the dashboard."""
        active_insights = [i for i in self.insights if i.active]
        active_insights.sort(key=lambda i: i.confidence, reverse=True)

        return {
            "total_trades_analyzed": len(self.all_analyses),
            "total_insights": len(self.insights),
            "active_insights": len(active_insights),
            "insights": [
                {
                    "id": i.id,
                    "cycle": i.created_cycle,
                    "category": i.category,
                    "conclusion": i.conclusion,
                    "confidence": round(i.confidence, 2),
                    "sample_size": i.sample_size,
                    "validated": i.validated,
                    "validations": i.validation_count,
                    "invalidations": i.invalidation_count,
                    "active": i.active,
                    "adjustments": i.param_adjustments,
                }
                for i in active_insights[:30]  # Top 30 by confidence
            ],
            "regime_stats": {k: {**v, "win_rate": round(v["wins"] / v["count"], 3) if v["count"] > 0 else 0}
                            for k, v in self.stats_by_regime.items() if v["count"] >= 5},
            "strategy_stats": {k: {**v, "win_rate": round(v["wins"] / v["count"], 3) if v["count"] > 0 else 0}
                              for k, v in self.stats_by_strategy.items()},
            "session_stats": {k: {**v, "win_rate": round(v["wins"] / v["count"], 3) if v["count"] > 0 else 0}
                             for k, v in self.stats_by_session.items()},
            "volatility_stats": {k: {**v, "win_rate": round(v["wins"] / v["count"], 3) if v["count"] > 0 else 0}
                                for k, v in self.stats_by_volatility.items()},
            "exit_reason_stats": {k: {**v, "win_rate": round(v["wins"] / v["count"], 3) if v["count"] > 0 else 0}
                                 for k, v in self.stats_by_exit_reason.items()},
            "param_adjustments": self.get_param_adjustments(),
            "avg_efficiency": round(np.mean(self.efficiency_scores[-200:]), 3) if self.efficiency_scores else 0,
            "avg_mfe_mae_ratio": round(np.mean(self.mfe_mae_ratios[-200:]), 2) if self.mfe_mae_ratios else 0,
            "nq_opinion": self.get_nq_opinion(),
            "cross_instrument_insights": [
                {
                    "id": i.id,
                    "conclusion": i.conclusion,
                    "confidence": round(i.confidence, 2),
                }
                for i in active_insights if i.category == "CROSS_INSTRUMENT"
            ],
            "overfit_warnings": [
                {
                    "id": i.id,
                    "conclusion": i.conclusion,
                    "confidence": round(i.confidence, 2),
                }
                for i in active_insights if i.category == "OVERFIT"
            ],
        }
