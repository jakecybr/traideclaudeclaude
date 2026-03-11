#!/usr/bin/env python3
"""
NQ Fractal AI Trading Backtester — Main Entry Point

Launches the continuous backtest-and-refine loop with a live web dashboard.

Usage:
    python main.py                    # Run with defaults
    python main.py --port 8080        # Custom dashboard port
    python main.py --pop 20           # Larger population
    python main.py --no-dashboard     # Headless mode (console only)

The system will:
  1. Fetch REAL NQ market data from Yahoo Finance (NQ=F, QQQ fallback)
  2. Test 5 strategies x N parameter sets each cycle on REAL bars
  3. AI analyzer examines every trade — learns patterns, draws conclusions
  4. Analyzer feeds learnings back into evolutionary optimizer
  5. Evolve parameters using genetic optimization + AI adjustments
  6. Display everything on a live web dashboard (charts, AI insights, regime analysis)
  7. Repeat forever — getting smarter every cycle
"""

import argparse
import signal
import sys
import time
from optimizer import BacktestLoop
from dashboard import start_dashboard


def main():
    parser = argparse.ArgumentParser(description="NQ Fractal AI Trading Backtester")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard port (default: 5000)")
    parser.add_argument("--pop", type=int, default=12, help="Population size (default: 12)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-dashboard", action="store_true", help="Run without web dashboard")
    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          NQ FRACTAL AI TRADING BACKTESTER                   ║
    ║                                                              ║
    ║   ALL DATA IS REAL — Yahoo Finance (NQ=F / QQQ)             ║
    ║   Markets are fractal. Patterns repeat across all scales.    ║
    ║                                                              ║
    ║   5 Strategies:                                              ║
    ║     * EMA Crossover (trend following)                        ║
    ║     * Fractal Breakout (structure breaks)                    ║
    ║     * Mean Reversion (Bollinger + RSI)                       ║
    ║     * Momentum (MACD + Volume)                               ║
    ║     * Multi-TF Fractal Confluence (combined)                 ║
    ║                                                              ║
    ║   AI BRAIN: Analyzes every trade, learns patterns,           ║
    ║   draws conclusions, feeds learnings back into optimizer.    ║
    ║                                                              ║
    ║   Population: {pop:>3} parameter sets evolving via GA + AI      ║
    ║   Press Ctrl+C to stop                                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """.format(pop=args.pop))

    # Create the backtest loop
    loop = BacktestLoop(population_size=args.pop, seed=args.seed)

    # Start dashboard
    if not args.no_dashboard:
        start_dashboard(loop, port=args.port)
        print(f"  Open your browser to http://localhost:{args.port}")
    print()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n  Stopping backtest loop...")
        loop.stop()
        time.sleep(0.5)

        if loop.best_ever_params:
            print(f"\n  === FINAL RESULTS ===")
            print(f"  Cycles completed: {loop.cycle_count}")
            print(f"  Best strategy: {loop.best_ever_strategy}")
            print(f"  Best score: {loop.best_ever_score:.1f}")
            print(f"  Best parameters:")
            for k, v in loop.best_ever_params.__dict__.items():
                print(f"    {k}: {v}")

            # Print AI conclusions
            ai_state = loop.analyzer.get_state()
            print(f"\n  === AI BRAIN SUMMARY ===")
            print(f"  Total trades analyzed: {ai_state['total_trades_analyzed']}")
            print(f"  Active insights: {ai_state['active_insights']}")
            for ins in ai_state["insights"][:10]:
                v_mark = "[VALIDATED]" if ins["validated"] else ""
                print(f"    [{ins['category']}] {ins['conclusion'][:90]} "
                      f"(conf={ins['confidence']:.0%}) {v_mark}")

            # Data stats
            data_stats = loop.data_feed.get_data_stats()
            print(f"\n  === DATA USAGE ===")
            print(f"  Real data: {data_stats['real_data_pct']:.0f}%")
            print(f"  Cached datasets: {data_stats['cached_datasets']}")
            print(f"  Live fetches: {data_stats['live_fetches']}")

        print("\n  Goodbye.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the loop forever
    loop.run_forever()


if __name__ == "__main__":
    main()
