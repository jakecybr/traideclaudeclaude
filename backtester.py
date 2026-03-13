"""
Core Backtest Engine for NQ Fractal Trading System

Simulates trade execution with realistic fills, slippage, commissions,
and position management. Tracks every trade with full detail.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from strategies import Signal, StrategyParams, BaseStrategy
import time


@dataclass
class Trade:
    """A completed round-trip trade."""
    trade_id: int
    strategy: str
    direction: str              # "LONG" or "SHORT"
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    size: int                   # Number of contracts
    pnl_points: float          # P&L in points
    pnl_dollars: float         # P&L in dollars
    stop_price: float
    target_price: float
    exit_reason: str            # "TARGET", "STOP", "TRAIL", "SIGNAL", "EOD"
    entry_reason: str           # Why the trade was entered
    max_favorable: float        # Max favorable excursion (MFE) in points
    max_adverse: float          # Max adverse excursion (MAE) in points
    bars_held: int
    commission: float


@dataclass
class OpenPosition:
    """A currently open position."""
    trade_id: int
    strategy: str
    direction: str
    entry_bar: int
    entry_price: float
    entry_time: str
    size: int
    stop_price: float
    target_price: float
    trail_stop: float
    entry_reason: str
    max_favorable: float = 0.0
    max_adverse: float = 0.0


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    trades: List[Trade]
    equity_curve: List[float]
    bars_processed: int
    strategy_name: str
    params: StrategyParams
    start_time: float
    end_time: float

    # Performance metrics
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_bars_held: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    expectancy: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    score: float = 0.0          # Composite fitness score

    def compute_metrics(self):
        """Calculate all performance metrics from trade list."""
        if not self.trades:
            return

        pnls = [t.pnl_dollars for t in self.trades]
        self.total_pnl = sum(pnls)
        self.total_trades = len(self.trades)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0
        self.largest_win = max(wins) if wins else 0
        self.largest_loss = min(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        self.avg_bars_held = np.mean([t.bars_held for t in self.trades])
        self.avg_mfe = np.mean([t.max_favorable for t in self.trades])
        self.avg_mae = np.mean([t.max_adverse for t in self.trades])

        # Equity curve and drawdown
        equity = [100_000.0]  # Starting equity
        for p in pnls:
            equity.append(equity[-1] + p)
        self.equity_curve = equity

        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = peak - e
            max_dd = max(max_dd, dd)
        self.max_drawdown = max_dd
        self.max_drawdown_pct = max_dd / equity[0] * 100 if equity[0] > 0 else 0

        # Sharpe ratio (annualized, assuming ~252 trading days)
        if len(pnls) > 1:
            returns = np.array(pnls) / equity[0]
            if np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Expectancy per trade
        self.expectancy = self.total_pnl / self.total_trades if self.total_trades > 0 else 0

        # Consecutive streaks
        streak = 0
        max_w_streak = 0
        max_l_streak = 0
        for p in pnls:
            if p > 0:
                if streak > 0:
                    streak += 1
                else:
                    streak = 1
                max_w_streak = max(max_w_streak, streak)
            else:
                if streak < 0:
                    streak -= 1
                else:
                    streak = -1
                max_l_streak = max(max_l_streak, abs(streak))
        self.consecutive_wins = max_w_streak
        self.consecutive_losses = max_l_streak

        # Composite fitness score for the optimizer
        self.score = self._compute_score()

    def _compute_score(self) -> float:
        """
        Composite fitness score balancing multiple objectives.
        Higher is better. Penalizes drawdown, rewards consistency.
        """
        if self.total_trades < 5:
            return -1000  # Not enough trades to evaluate

        score = 0.0

        # Profitability (40% weight)
        score += self.total_pnl * 0.01

        # Win rate bonus (15% weight)
        score += (self.win_rate - 0.5) * 500

        # Profit factor (15% weight)
        pf = min(self.profit_factor, 5.0)  # Cap extreme values
        score += pf * 100

        # Sharpe ratio (15% weight)
        score += self.sharpe_ratio * 200

        # Drawdown penalty (15% weight)
        score -= self.max_drawdown_pct * 20

        # Trade frequency bonus (small)
        if self.total_trades > 20:
            score += 50
        elif self.total_trades < 10:
            score -= 100

        # Overfit penalties
        # Penalize suspiciously high win rates (100% is unrealistic)
        if self.win_rate >= 0.99:
            score -= 500
        elif self.win_rate >= 0.95:
            score -= 200

        # Penalize if too few unique exit prices (sign of overfitting to specific bars)
        if self.trades:
            unique_exits = len(set(round(t.exit_price, 2) for t in self.trades))
            if unique_exits < self.total_trades * 0.3:
                score -= 300

        return score

    def summary(self) -> str:
        """Human-readable performance summary."""
        return (
            f"═══ {self.strategy_name} ═══\n"
            f"  Trades: {self.total_trades} ({self.winning_trades}W / {self.losing_trades}L)\n"
            f"  Win Rate: {self.win_rate:.1%}\n"
            f"  Total P&L: ${self.total_pnl:,.2f}\n"
            f"  Avg Win: ${self.avg_win:,.2f}  |  Avg Loss: ${self.avg_loss:,.2f}\n"
            f"  Largest Win: ${self.largest_win:,.2f}  |  Largest Loss: ${self.largest_loss:,.2f}\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1f}%)\n"
            f"  Expectancy: ${self.expectancy:,.2f}/trade\n"
            f"  Avg MFE: {self.avg_mfe:.1f} pts  |  Avg MAE: {self.avg_mae:.1f} pts\n"
            f"  Avg Bars Held: {self.avg_bars_held:.1f}\n"
            f"  Score: {self.score:.1f}\n"
        )


class BacktestEngine:
    """
    Simulates trading on NQ bars with realistic execution.

    Features:
    - Slippage modeling (0.25-0.50 tick per side)
    - Commission ($2.50 per side per contract)
    - Stop and target management
    - Trailing stops
    - One position per strategy at a time
    - MFE/MAE tracking
    """

    COMMISSION_PER_SIDE = 2.50  # Per contract
    POINT_VALUE = 20.0          # NQ: $20/point
    SLIPPAGE_TICKS = 1          # 1 tick slippage per side
    TICK_SIZE = 0.25

    def __init__(self, point_value: float = 20.0, tick_size: float = 0.25,
                 commission_per_side: float = 2.50):
        self.open_positions: List[OpenPosition] = []
        self.closed_trades: List[Trade] = []
        self.next_trade_id = 1
        # Allow overrides for normalized/percentage data
        self.POINT_VALUE = point_value
        self.TICK_SIZE = tick_size
        self.COMMISSION_PER_SIDE = commission_per_side

    def run(self, bars: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """Run a full backtest of a strategy on the given bars."""
        t0 = time.time()
        self.open_positions = []
        self.closed_trades = []
        self.next_trade_id = 1

        signals = strategy.compute_signals(bars)

        for sig in signals:
            bar_idx = sig["bar_index"]
            if bar_idx >= len(bars):
                continue

            bar = bars.iloc[bar_idx]

            # Update open positions (check stops/targets)
            self._update_positions(bar, bar_idx)

            # Process new signal
            if sig["signal"] in (Signal.LONG, Signal.SHORT):
                # Only enter if no existing position for this strategy
                has_position = any(p.strategy == strategy.name for p in self.open_positions)
                if not has_position:
                    self._open_position(sig, bar, strategy.name)

        # Close any remaining positions at last bar
        if len(bars) > 0:
            last_bar = bars.iloc[-1]
            for pos in list(self.open_positions):
                self._close_position(pos, last_bar, len(bars) - 1, "EOD")

        t1 = time.time()
        result = BacktestResult(
            trades=self.closed_trades,
            equity_curve=[],
            bars_processed=len(bars),
            strategy_name=strategy.name,
            params=strategy.params,
            start_time=t0,
            end_time=t1,
        )
        result.compute_metrics()
        return result

    def _open_position(self, sig: Dict, bar: pd.Series, strategy_name: str):
        """Open a new position from a signal."""
        direction = "LONG" if sig["signal"] == Signal.LONG else "SHORT"
        entry_price = bar["close"]

        # Apply slippage
        slippage = self.SLIPPAGE_TICKS * self.TICK_SIZE
        if direction == "LONG":
            entry_price += slippage
            stop_price = entry_price - sig["stop_distance"]
            target_price = entry_price + sig["target_distance"]
        else:
            entry_price -= slippage
            stop_price = entry_price + sig["stop_distance"]
            target_price = entry_price - sig["target_distance"]

        pos = OpenPosition(
            trade_id=self.next_trade_id,
            strategy=strategy_name,
            direction=direction,
            entry_bar=sig["bar_index"],
            entry_price=entry_price,
            entry_time=str(bar["timestamp"]),
            size=1,
            stop_price=stop_price,
            target_price=target_price,
            trail_stop=stop_price,
            entry_reason=sig.get("reason", ""),
        )
        self.open_positions.append(pos)
        self.next_trade_id += 1

    def _update_positions(self, bar: pd.Series, bar_idx: int):
        """Check stops, targets, and trailing stops for all open positions."""
        for pos in list(self.open_positions):
            high = bar["high"]
            low = bar["low"]

            if pos.direction == "LONG":
                # Update MFE/MAE
                excursion_up = high - pos.entry_price
                excursion_down = pos.entry_price - low
                pos.max_favorable = max(pos.max_favorable, excursion_up)
                pos.max_adverse = max(pos.max_adverse, excursion_down)

                # Stop hit
                if low <= pos.trail_stop:
                    self._close_position(pos, bar, bar_idx, "STOP",
                                         exit_price=pos.trail_stop - self.SLIPPAGE_TICKS * self.TICK_SIZE)
                # Target hit
                elif high >= pos.target_price:
                    self._close_position(pos, bar, bar_idx, "TARGET",
                                         exit_price=pos.target_price - self.SLIPPAGE_TICKS * self.TICK_SIZE)
                else:
                    # Update trailing stop
                    new_trail = high - (pos.entry_price - pos.stop_price)
                    pos.trail_stop = max(pos.trail_stop, new_trail)

            else:  # SHORT
                excursion_up = pos.entry_price - low
                excursion_down = high - pos.entry_price
                pos.max_favorable = max(pos.max_favorable, excursion_up)
                pos.max_adverse = max(pos.max_adverse, excursion_down)

                # Stop hit
                if high >= pos.trail_stop:
                    self._close_position(pos, bar, bar_idx, "STOP",
                                         exit_price=pos.trail_stop + self.SLIPPAGE_TICKS * self.TICK_SIZE)
                # Target hit
                elif low <= pos.target_price:
                    self._close_position(pos, bar, bar_idx, "TARGET",
                                         exit_price=pos.target_price + self.SLIPPAGE_TICKS * self.TICK_SIZE)
                else:
                    new_trail = low + (pos.stop_price - pos.entry_price)
                    pos.trail_stop = min(pos.trail_stop, new_trail)

    def _close_position(self, pos: OpenPosition, bar: pd.Series, bar_idx: int,
                        reason: str, exit_price: Optional[float] = None):
        """Close a position and record the trade."""
        if exit_price is None:
            exit_price = bar["close"]

        if pos.direction == "LONG":
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price

        commission = self.COMMISSION_PER_SIDE * 2 * pos.size
        pnl_dollars = pnl_points * self.POINT_VALUE * pos.size - commission

        trade = Trade(
            trade_id=pos.trade_id,
            strategy=pos.strategy,
            direction=pos.direction,
            entry_bar=pos.entry_bar,
            exit_bar=bar_idx,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=str(bar["timestamp"]),
            size=pos.size,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            exit_reason=reason,
            entry_reason=pos.entry_reason,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
            bars_held=bar_idx - pos.entry_bar,
            commission=commission,
        )
        self.closed_trades.append(trade)
        self.open_positions.remove(pos)
