"""
AI Strategy Optimizer — Continuous Refinement Loop

Uses evolutionary optimization to continuously improve trading strategies.
Each cycle:
  1. Fetch REAL multi-source market data from the data lake
  2. Walk-forward split: 70% train, 30% out-of-sample test
  3. Run all strategies on training bars
  4. AI analyzer examines every trade — learns what worked, what didn't
  5. Out-of-sample validation — detect overfitting
  6. Evolve: keep winners, mutate, crossover, APPLY AI LEARNINGS
  7. Cross-instrument correlation analysis
  8. Repeat forever — getting smarter every cycle

The optimizer treats the strategy parameter space as a population
of organisms competing for survival — Darwinian selection on trading fitness,
guided by the AI analyzer's accumulated intelligence.
"""

import numpy as np
import pandas as pd
import time
import json
import threading
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from strategies import (
    StrategyParams, get_all_strategies, BaseStrategy,
    EMACrossoverStrategy, FractalBreakoutStrategy, MeanReversionStrategy,
    MomentumStrategy, MultiTimeframeFractalStrategy,
)
from backtester import BacktestEngine, BacktestResult, Trade
from data_lake import DataLakeIndex, DataRouter, DataNormalizer
from correlation_engine import CorrelationEngine
from analyzer import AITradeAnalyzer


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
    data_source: str = ""           # Where the bars came from
    data_symbol: str = ""           # What symbol was traded
    ai_insights_count: int = 0      # New insights generated this cycle
    ai_adjustments: Dict = field(default_factory=dict)  # AI parameter adjustments applied
    # Walk-forward / OOS fields
    oos_score: float = 0.0
    oos_win_rate: float = 0.0
    oos_pnl: float = 0.0
    overfit_flag: bool = False
    instrument: str = ""
    train_bars: int = 0
    test_bars: int = 0


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

    def evolve(self, scores: List[float], ai_adjustments: Optional[Dict] = None):
        """
        Evolve the population based on fitness scores.
        Uses elitism + tournament selection + crossover + mutation.
        AI adjustments are applied to children — this is how the analyzer's
        learnings feed back into the population.
        """
        # Pair scores with parameter sets
        paired = list(zip(scores, self.population))
        paired.sort(key=lambda x: x[0], reverse=True)

        new_population = []

        # Elitism: keep top performers unchanged
        for i in range(min(self.elite_count, len(paired))):
            new_population.append(paired[i][1])

        # Fill rest with crossover + mutation + AI adjustments
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(paired)
            parent2 = self._tournament_select(paired)

            # Crossover
            child = parent1.crossover(parent2, self.rng)

            # Mutation
            child = child.mutate(self.rng, self.mutation_rate)

            # Apply AI analyzer's learned adjustments (if any)
            if ai_adjustments:
                child = self._apply_ai_adjustments(child, ai_adjustments)

            new_population.append(child)

        self.population = new_population

    def _apply_ai_adjustments(self, params: StrategyParams, adjustments: Dict) -> StrategyParams:
        """
        Apply AI-derived parameter adjustments to a parameter set.
        The analyzer tells us things like 'widen stops' or 'tighten targets'
        and we translate that into actual parameter changes.
        """
        for param_name, delta in adjustments.items():
            if hasattr(params, param_name):
                current = getattr(params, param_name)
                if isinstance(current, int):
                    new_val = int(current + delta)
                else:
                    new_val = current + delta
                setattr(params, param_name, new_val)
        return params

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

    def reset_population(self):
        """Reset population with fresh random params (keep best elite)."""
        if not self.population:
            return
        best = self.population[0]  # Assume sorted by evolve()
        self.population = [best]
        base = StrategyParams()
        for _ in range(self.population_size - 1):
            self.population.append(base.mutate(self.rng, mutation_rate=0.6))
        print("  [OPTIMIZER] Population diversity collapsed — reset with fresh params")


class BacktestLoop:
    """
    The main continuous backtest-and-refine loop.

    Uses multi-source data lake, walk-forward validation, OOS scoring,
    and cross-instrument correlation analysis.
    """

    STRATEGY_CLASSES = [
        ("EMA_Crossover", EMACrossoverStrategy),
        ("Fractal_Breakout", FractalBreakoutStrategy),
        ("Mean_Reversion", MeanReversionStrategy),
        ("Momentum", MomentumStrategy),
        ("MTF_Fractal", MultiTimeframeFractalStrategy),
    ]

    TIMEFRAMES = [1, 5, 15, 60]  # Minutes — test across fractal timeframes

    def __init__(self, population_size: int = 12, seed: Optional[int] = None,
                 workers: Optional[list] = None):
        self.optimizer = EvolutionaryOptimizer(population_size=population_size, seed=seed)
        # Use normalized-data engine: point_value=1 for percentage data, small commission
        self.engine = BacktestEngine(point_value=1.0, tick_size=0.01, commission_per_side=0.0)
        self.analyzer = AITradeAnalyzer()      # AI learning brain
        self.workers = workers or []

        # Data lake and router
        self.lake = DataLakeIndex(Path(__file__).parent / "data_lake")
        self.router = DataRouter(self.lake)
        self.correlation_engine = CorrelationEngine(self.lake)
        self.correlation_engine.start_background()

        # Fallback data feed for when lake is empty
        self._data_feed_fallback = None

        self.cycle_count = 0
        self.all_cycle_results: List[CycleResult] = []
        self.best_ever_score = float('-inf')
        self.best_ever_params: Optional[StrategyParams] = None
        self.best_ever_strategy: str = ""
        self.running = False
        self._lock = threading.Lock()

        # OOS tracking
        self.oos_scores: List[float] = []
        self.overfit_warnings: List[str] = []

        # Live state for the dashboard
        self.current_bars: Optional[pd.DataFrame] = None
        self.current_trades: List[Trade] = []
        self.current_result: Optional[BacktestResult] = None
        self.current_data_meta: Dict = {}
        self.status = "IDLE"

    def _get_fallback_feed(self):
        """Lazy-load the old RealDataFeed as fallback."""
        if self._data_feed_fallback is None:
            from data_feed import RealDataFeed
            self._data_feed_fallback = RealDataFeed()
        return self._data_feed_fallback

    def run_cycle(self) -> CycleResult:
        """Execute one full optimization cycle with walk-forward validation."""
        t0 = time.time()
        self.cycle_count += 1
        self.status = f"CYCLE {self.cycle_count} — Fetching multi-source data"

        # Pick a random timeframe for this cycle
        tf = self.TIMEFRAMES[self.cycle_count % len(self.TIMEFRAMES)]
        num_bars = max(200, 2000 // max(tf, 1))

        # ── STEP 1: Get data from data lake with walk-forward split ──
        train_df, test_df, data_meta = None, None, None

        if self.router.has_real_data():
            train_df, test_df, data_meta = self.router.get_training_data(min_bars=num_bars)

        # Fallback to old data feed if lake is empty
        if train_df is None or test_df is None:
            self.status = f"CYCLE {self.cycle_count} — Lake empty, using Yahoo fallback"
            feed = self._get_fallback_feed()
            bars, meta = feed.get_bars(timeframe_minutes=tf, min_bars=num_bars)

            # Manual walk-forward split
            split_idx = int(len(bars) * 0.7)
            train_df = bars.iloc[:split_idx].reset_index(drop=True)
            train_df["bar_index"] = range(len(train_df))
            test_df = bars.iloc[split_idx:].reset_index(drop=True)
            test_df["bar_index"] = range(len(test_df))

            data_meta = {
                "source": meta.get("source", "Yahoo Fallback"),
                "symbol": meta.get("symbol", "NQ"),
                "instrument": meta.get("symbol", "NQ"),
                "is_real": meta.get("is_real", True),
                "is_normalized": False,
                "train_bars": len(train_df),
                "test_bars": len(test_df),
            }

        with self._lock:
            self.current_bars = train_df.copy()
            self.current_data_meta = data_meta

        data_label = "MULTI-SRC" if data_meta.get("is_normalized") else "REAL"
        instrument = data_meta.get("instrument", data_meta.get("symbol", "NQ"))

        self.status = (f"CYCLE {self.cycle_count} — [{data_label}] {instrument} "
                       f"Testing {len(self.optimizer.population)} params x "
                       f"{len(self.STRATEGY_CLASSES)} strategies")

        # ── STEP 2: Test every strategy x every parameter set on TRAIN data ──
        all_results: List[BacktestResult] = []
        param_scores: List[float] = []

        for param_idx, params in enumerate(self.optimizer.population):
            param_total_score = 0
            for strat_name, strat_class in self.STRATEGY_CLASSES:
                strategy = strat_class(params)
                result = self.engine.run(train_df, strategy)
                all_results.append(result)
                param_total_score += result.score

            param_scores.append(param_total_score)

        # Find the single best result this cycle
        best_result = max(all_results, key=lambda r: r.score)

        with self._lock:
            self.current_result = best_result
            self.current_trades = best_result.trades

        # ── STEP 3: AI ANALYZER — Learn from every trade ──
        self.status = f"CYCLE {self.cycle_count} — AI analyzing {best_result.total_trades} trades..."
        new_insights = self.analyzer.analyze_cycle(
            best_result, train_df,
            correlations=self.correlation_engine.get_state()
        )

        for result in all_results:
            if result is not best_result and result.total_trades >= 3:
                self.analyzer.analyze_cycle(result, train_df)

        ai_adjustments = self.analyzer.get_param_adjustments()

        # ── STEP 3.5: OUT-OF-SAMPLE VALIDATION ──
        oos_score = 0.0
        oos_win_rate = 0.0
        oos_pnl = 0.0
        overfit_flag = False

        if test_df is not None and len(test_df) >= 20:
            # Run the best strategy+params on test data
            best_strat_class = None
            for name, cls in self.STRATEGY_CLASSES:
                if name == best_result.strategy_name:
                    best_strat_class = cls
                    break
            if best_strat_class is None:
                best_strat_class = self.STRATEGY_CLASSES[0][1]

            oos_strategy = best_strat_class(best_result.params)
            oos_result = self.engine.run(test_df, oos_strategy)
            oos_score = oos_result.score
            oos_win_rate = oos_result.win_rate
            oos_pnl = oos_result.total_pnl

            self.oos_scores.append(oos_score)

            # Overfit detection
            if best_result.score > 0 and oos_score < best_result.score * 0.3:
                overfit_flag = True
                msg = (f"Cycle {self.cycle_count}: OOS score ({oos_score:.0f}) is <30% of "
                       f"in-sample ({best_result.score:.0f}) — likely overfit")
                self.overfit_warnings.append(msg)
                # Penalize the score
                best_result.score *= 0.5

            if best_result.win_rate >= 0.99 and oos_win_rate < 0.5:
                overfit_flag = True
                msg = (f"Cycle {self.cycle_count}: 100% in-sample WR but "
                       f"{oos_win_rate:.0%} OOS — severe overfit")
                self.overfit_warnings.append(msg)
                best_result.score = -500

        # ── STEP 4: Evolve population WITH AI learnings ──
        self.optimizer.evolve(param_scores, ai_adjustments=ai_adjustments if ai_adjustments else None)

        # Diversity check
        diversity = self.optimizer.compute_diversity()
        if diversity < 0.05:
            self.optimizer.reset_population()

        # Track global best
        if best_result.score > self.best_ever_score:
            self.best_ever_score = best_result.score
            self.best_ever_params = best_result.params
            self.best_ever_strategy = best_result.strategy_name

        # ── STEP 5: Build cycle result ──
        duration = time.time() - t0

        cycle_result = CycleResult(
            cycle_number=self.cycle_count,
            timestamp=time.time(),
            duration_seconds=duration,
            bars_generated=len(train_df) + (len(test_df) if test_df is not None else 0),
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
            data_source=data_meta.get("source", "unknown"),
            data_symbol=data_meta.get("symbol", "unknown"),
            ai_insights_count=len(new_insights),
            ai_adjustments=ai_adjustments,
            oos_score=oos_score,
            oos_win_rate=oos_win_rate,
            oos_pnl=oos_pnl,
            overfit_flag=overfit_flag,
            instrument=instrument,
            train_bars=len(train_df),
            test_bars=len(test_df) if test_df is not None else 0,
        )

        with self._lock:
            self.all_cycle_results.append(cycle_result)

        self.status = (f"CYCLE {self.cycle_count} DONE [{data_label}] {instrument} — "
                       f"Best: {best_result.strategy_name} score={best_result.score:.0f} "
                       f"OOS={oos_score:.0f} | "
                       f"AI: {len(new_insights)} new insights, "
                       f"{self.analyzer.get_state()['active_insights']} active"
                       f"{' [OVERFIT]' if overfit_flag else ''}")
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
                print(f"  DATA: {result.data_source} ({result.data_symbol}) "
                      f"[{result.instrument}]")
                print(f"  TRAIN: {result.train_bars} bars  |  "
                      f"TEST: {result.test_bars} bars  |  "
                      f"Walk-Forward 70/30")
                print(f"  BEST: {result.best_strategy}  "
                      f"Score={result.best_score:.0f}  "
                      f"P&L=${result.best_pnl:,.0f}  "
                      f"WR={result.best_win_rate:.0%}  "
                      f"Sharpe={result.best_sharpe:.2f}  "
                      f"MaxDD={result.best_max_dd_pct:.1f}%")
                print(f"  OOS: Score={result.oos_score:.0f}  "
                      f"WR={result.oos_win_rate:.0%}  "
                      f"P&L=${result.oos_pnl:,.0f}"
                      f"{'  *** OVERFIT DETECTED ***' if result.overfit_flag else ''}")
                print(f"  GLOBAL BEST: {self.best_ever_strategy} "
                      f"Score={self.best_ever_score:.0f}")
                print(f"  Population diversity: {result.population_diversity:.3f}")

                # AI insights summary
                ai_state = self.analyzer.get_state()
                print(f"  AI BRAIN: {ai_state['total_trades_analyzed']} trades analyzed | "
                      f"{ai_state['active_insights']} active insights | "
                      f"{result.ai_insights_count} new this cycle")
                if result.ai_adjustments:
                    adj_str = ", ".join(f"{k}:{v:+.2f}" for k, v in result.ai_adjustments.items())
                    print(f"  AI ADJUSTMENTS: {adj_str}")

                # Print top insights
                for ins in ai_state["insights"][:3]:
                    conf_bar = "█" * int(ins["confidence"] * 10)
                    print(f"  [{ins['category']}] {ins['conclusion'][:80]} "
                          f"(conf={ins['confidence']:.0%} {conf_bar})")

                # Data lake stats
                lake_stats = self.router.get_lake_stats()
                print(f"  DATA LAKE: {lake_stats['total_instruments']} instruments, "
                      f"{lake_stats['total_files']} files, "
                      f"{lake_stats.get('total_mb', 0):.1f} MB")

                # Worker stats
                if self.workers:
                    active = sum(1 for w in self.workers if w.running)
                    total_bars = sum(w.total_bars for w in self.workers)
                    print(f"  WORKERS: {active}/{len(self.workers)} active, "
                          f"{total_bars} total bars fetched")

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

    def worker_stats(self) -> List[Dict]:
        """Get stats from all data workers."""
        return [w.get_stats() for w in self.workers]

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
                    "data_source": c.data_source,
                    "data_symbol": c.data_symbol,
                    "ai_insights": c.ai_insights_count,
                    "oos_score": round(c.oos_score, 1),
                    "oos_win_rate": round(c.oos_win_rate, 3),
                    "overfit_flag": c.overfit_flag,
                    "instrument": c.instrument,
                }
                for c in self.all_cycle_results[-50:]
            ]

            # AI analyzer state
            ai_state = self.analyzer.get_state()

            # Data lake stats
            data_stats = self.router.get_lake_stats()

            # Correlation state
            corr_state = self.correlation_engine.get_state()

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
                "data_meta": self.current_data_meta,
                "data_stats": data_stats,
                "ai": ai_state,
                "correlations": corr_state,
                "oos_scores": self.oos_scores[-50:],
                "overfit_warnings": self.overfit_warnings[-20:],
                "workers": self.worker_stats(),
            }
