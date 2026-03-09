"""
AI Strategy Optimizer — Continuous Refinement Loop

Uses evolutionary optimization to continuously improve trading strategies.
Each cycle:
  1. Generate fresh fractal NQ market data
  2. Run all strategies on the data
  3. Score results
  4. Evolve: keep winners, mutate, crossover
  5. Log everything
  6. Repeat forever

The optimizer treats the strategy parameter space as a population
of organisms competing for survival — Darwinian selection on trading fitness.
"""

import numpy as np
import pandas as pd
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from market_generator import NQMarketGenerator, NQSessionConfig, resample_bars
from strategies import (
    StrategyParams, get_all_strategies, BaseStrategy,
    EMACrossoverStrategy, FractalBreakoutStrategy, MeanReversionStrategy,
    MomentumStrategy, MultiTimeframeFractalStrategy,
)
from backtester import BacktestEngine, BacktestResult, Trade


@dataclass
class CycleResult:
    """Results from one optimization cycle."""
    cycle_number: int
    timestamp: float
    duration_seconds: float
    bars_generated: int
    timeframe: int
    num_strategies_tested: int
    best_strategy: str
    best_score: float
    best_pnl: float
    best_win_rate: float
    best_sharpe: float
    best_max_dd_pct: float
    best_trades: int
    avg_score: float
    population_diversity: float  # How different the parameter sets are
    all_results: List[Dict]     # Summary of every strategy tested
    best_params: Dict           # The winning parameter set
    best_trades_detail: List[Dict]  # Full trade log of winner


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer with elitism, crossover, and mutation.

    Population: multiple parameter sets
    Selection: top N survive
    Crossover: combine winning parameter sets
    Mutation: random perturbation
    """

    def __init__(self, population_size: int = 12, elite_count: int = 3,
                 mutation_rate: float = 0.3, seed: Optional[int] = None):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

        # Initialize population with diverse parameter sets
        self.population: List[StrategyParams] = []
        base = StrategyParams()
        self.population.append(base)  # Always include defaults
        for _ in range(population_size - 1):
            self.population.append(base.mutate(self.rng, mutation_rate=0.6))

    def evolve(self, scores: List[float]):
        """
        Evolve the population based on fitness scores.
        Uses elitism + tournament selection + crossover + mutation.
        """
        # Pair scores with parameter sets
        paired = list(zip(scores, self.population))
        paired.sort(key=lambda x: x[0], reverse=True)

        new_population = []

        # Elitism: keep top performers unchanged
        for i in range(min(self.elite_count, len(paired))):
            new_population.append(paired[i][1])

        # Fill rest with crossover + mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(paired)
            parent2 = self._tournament_select(paired)

            # Crossover
            child = parent1.crossover(parent2, self.rng)

            # Mutation
            child = child.mutate(self.rng, self.mutation_rate)

            new_population.append(child)

        self.population = new_population

    def _tournament_select(self, paired: List[Tuple[float, StrategyParams]],
                           tournament_size: int = 3) -> StrategyParams:
        """Select a parent via tournament selection."""
        contestants = self.rng.choice(len(paired), size=min(tournament_size, len(paired)), replace=False)
        best = max(contestants, key=lambda idx: paired[idx][0])
        return paired[best][1]

    def compute_diversity(self) -> float:
        """Measure population diversity (0-1 scale)."""
        if len(self.population) < 2:
            return 0.0

        # Compare each pair of parameter sets
        diffs = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                d = self._param_distance(self.population[i], self.population[j])
                diffs.append(d)

        return np.mean(diffs) if diffs else 0.0

    def _param_distance(self, a: StrategyParams, b: StrategyParams) -> float:
        """Normalized distance between two parameter sets."""
        fields = [
            ('fast_ema', 3, 20), ('slow_ema', 10, 50), ('trend_ema', 30, 200),
            ('fractal_lookback', 3, 10), ('bollinger_period', 10, 50),
            ('bollinger_std', 1.0, 3.5), ('rsi_period', 7, 28),
            ('rsi_oversold', 15, 40), ('rsi_overbought', 60, 85),
            ('stop_atr_mult', 0.5, 3.0), ('target_atr_mult', 1.0, 5.0),
        ]
        dist = 0
        for name, lo, hi in fields:
            va = getattr(a, name)
            vb = getattr(b, name)
            dist += abs(va - vb) / (hi - lo + 1e-10)
        return dist / len(fields)


class BacktestLoop:
    """
    The main continuous backtest-and-refine loop.

    Runs forever, generating market data, testing strategies,
    evolving parameters, and reporting results.
    """

    STRATEGY_CLASSES = [
        ("EMA_Crossover", EMACrossoverStrategy),
        ("Fractal_Breakout", FractalBreakoutStrategy),
        ("Mean_Reversion", MeanReversionStrategy),
        ("Momentum", MomentumStrategy),
        ("MTF_Fractal", MultiTimeframeFractalStrategy),
    ]

    TIMEFRAMES = [1, 5, 15, 60]  # Minutes — test across fractal timeframes

    def __init__(self, population_size: int = 12, seed: Optional[int] = None):
        self.optimizer = EvolutionaryOptimizer(population_size=population_size, seed=seed)
        self.engine = BacktestEngine()
        self.cycle_count = 0
        self.all_cycle_results: List[CycleResult] = []
        self.best_ever_score = float('-inf')
        self.best_ever_params: Optional[StrategyParams] = None
        self.best_ever_strategy: str = ""
        self.running = False
        self._lock = threading.Lock()

        # Live state for the dashboard
        self.current_bars: Optional[pd.DataFrame] = None
        self.current_trades: List[Trade] = []
        self.current_result: Optional[BacktestResult] = None
        self.status = "IDLE"

    def run_cycle(self) -> CycleResult:
        """Execute one full optimization cycle."""
        t0 = time.time()
        self.cycle_count += 1
        self.status = f"CYCLE {self.cycle_count} — Generating market data"

        # Pick a random timeframe for this cycle (fractal: test all scales)
        tf = self.TIMEFRAMES[self.cycle_count % len(self.TIMEFRAMES)]
        num_bars = max(500, 2000 // max(tf, 1))

        # Generate fresh fractal market data
        gen = NQMarketGenerator(seed=None)  # Different data each cycle
        bars = gen.generate_bars(num_bars, timeframe_minutes=tf)

        with self._lock:
            self.current_bars = bars.copy()

        self.status = f"CYCLE {self.cycle_count} — Testing {len(self.optimizer.population)} parameter sets x {len(self.STRATEGY_CLASSES)} strategies"

        # Test every strategy with every parameter set in the population
        all_results: List[BacktestResult] = []
        scores: List[float] = []
        param_scores: List[float] = []  # Aggregate score per param set

        for param_idx, params in enumerate(self.optimizer.population):
            param_total_score = 0
            for strat_name, strat_class in self.STRATEGY_CLASSES:
                strategy = strat_class(params)
                result = self.engine.run(bars, strategy)
                all_results.append(result)
                param_total_score += result.score

            param_scores.append(param_total_score)

        # Find the single best result this cycle
        best_result = max(all_results, key=lambda r: r.score)

        with self._lock:
            self.current_result = best_result
            self.current_trades = best_result.trades

        # Evolve population based on aggregate scores
        self.optimizer.evolve(param_scores)

        # Track global best
        if best_result.score > self.best_ever_score:
            self.best_ever_score = best_result.score
            self.best_ever_params = best_result.params
            self.best_ever_strategy = best_result.strategy_name

        # Build cycle result
        duration = time.time() - t0
        diversity = self.optimizer.compute_diversity()

        cycle_result = CycleResult(
            cycle_number=self.cycle_count,
            timestamp=time.time(),
            duration_seconds=duration,
            bars_generated=num_bars,
            timeframe=tf,
            num_strategies_tested=len(all_results),
            best_strategy=best_result.strategy_name,
            best_score=best_result.score,
            best_pnl=best_result.total_pnl,
            best_win_rate=best_result.win_rate,
            best_sharpe=best_result.sharpe_ratio,
            best_max_dd_pct=best_result.max_drawdown_pct,
            best_trades=best_result.total_trades,
            avg_score=np.mean([r.score for r in all_results]),
            population_diversity=diversity,
            all_results=[
                {
                    "strategy": r.strategy_name,
                    "score": round(r.score, 1),
                    "pnl": round(r.total_pnl, 2),
                    "trades": r.total_trades,
                    "win_rate": round(r.win_rate, 3),
                    "sharpe": round(r.sharpe_ratio, 2),
                    "profit_factor": round(r.profit_factor, 2),
                }
                for r in all_results
            ],
            best_params=best_result.params.__dict__,
            best_trades_detail=[
                {
                    "id": t.trade_id,
                    "dir": t.direction,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "pnl": round(t.pnl_dollars, 2),
                    "reason_in": t.entry_reason,
                    "reason_out": t.exit_reason,
                    "bars": t.bars_held,
                    "mfe": round(t.max_favorable, 1),
                    "mae": round(t.max_adverse, 1),
                }
                for t in best_result.trades
            ],
        )

        with self._lock:
            self.all_cycle_results.append(cycle_result)

        self.status = f"CYCLE {self.cycle_count} DONE — Best: {best_result.strategy_name} score={best_result.score:.0f}"
        return cycle_result

    def run_forever(self, callback=None):
        """Run the backtest loop continuously."""
        self.running = True
        self.status = "RUNNING"

        while self.running:
            try:
                result = self.run_cycle()

                # Print cycle summary to console
                print(f"\n{'='*70}")
                print(f"  CYCLE {result.cycle_number}  |  TF={result.timeframe}m  |  "
                      f"{result.duration_seconds:.1f}s  |  "
                      f"Tested {result.num_strategies_tested} configs")
                print(f"  BEST: {result.best_strategy}  "
                      f"Score={result.best_score:.0f}  "
                      f"P&L=${result.best_pnl:,.0f}  "
                      f"WR={result.best_win_rate:.0%}  "
                      f"Sharpe={result.best_sharpe:.2f}  "
                      f"MaxDD={result.best_max_dd_pct:.1f}%")
                print(f"  GLOBAL BEST: {self.best_ever_strategy} "
                      f"Score={self.best_ever_score:.0f}")
                print(f"  Population diversity: {result.population_diversity:.3f}")
                print(f"{'='*70}")

                if callback:
                    callback(result)

            except Exception as e:
                print(f"  [ERROR] Cycle {self.cycle_count}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def stop(self):
        """Stop the loop."""
        self.running = False
        self.status = "STOPPED"

    def get_state(self) -> Dict:
        """Get current state for the dashboard."""
        with self._lock:
            bars_data = None
            if self.current_bars is not None:
                bars_data = self.current_bars.to_dict(orient="records")
                for b in bars_data:
                    b["timestamp"] = str(b["timestamp"])

            trades_data = []
            if self.current_trades:
                trades_data = [
                    {
                        "id": t.trade_id,
                        "strategy": t.strategy,
                        "direction": t.direction,
                        "entry_bar": t.entry_bar,
                        "exit_bar": t.exit_bar,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "pnl_points": round(t.pnl_points, 2),
                        "pnl_dollars": round(t.pnl_dollars, 2),
                        "exit_reason": t.exit_reason,
                        "entry_reason": t.entry_reason,
                        "bars_held": t.bars_held,
                        "mfe": round(t.max_favorable, 1),
                        "mae": round(t.max_adverse, 1),
                    }
                    for t in self.current_trades
                ]

            result_data = None
            if self.current_result:
                r = self.current_result
                result_data = {
                    "strategy": r.strategy_name,
                    "total_pnl": round(r.total_pnl, 2),
                    "win_rate": round(r.win_rate, 3),
                    "profit_factor": round(r.profit_factor, 2),
                    "sharpe": round(r.sharpe_ratio, 2),
                    "max_dd": round(r.max_drawdown, 2),
                    "max_dd_pct": round(r.max_drawdown_pct, 1),
                    "total_trades": r.total_trades,
                    "expectancy": round(r.expectancy, 2),
                    "score": round(r.score, 1),
                    "equity_curve": r.equity_curve,
                    "avg_mfe": round(r.avg_mfe, 1),
                    "avg_mae": round(r.avg_mae, 1),
                    "consecutive_wins": r.consecutive_wins,
                    "consecutive_losses": r.consecutive_losses,
                }

            cycle_history = [
                {
                    "cycle": c.cycle_number,
                    "tf": c.timeframe,
                    "best_strategy": c.best_strategy,
                    "score": round(c.best_score, 1),
                    "pnl": round(c.best_pnl, 2),
                    "win_rate": round(c.best_win_rate, 3),
                    "sharpe": round(c.best_sharpe, 2),
                    "diversity": round(c.population_diversity, 3),
                    "duration": round(c.duration_seconds, 1),
                }
                for c in self.all_cycle_results[-50:]  # Last 50 cycles
            ]

            return {
                "status": self.status,
                "cycle_count": self.cycle_count,
                "best_ever_score": round(self.best_ever_score, 1) if self.best_ever_score > float('-inf') else 0,
                "best_ever_strategy": self.best_ever_strategy,
                "bars": bars_data,
                "trades": trades_data,
                "result": result_data,
                "cycle_history": cycle_history,
                "best_params": self.best_ever_params.__dict__ if self.best_ever_params else {},
            }
