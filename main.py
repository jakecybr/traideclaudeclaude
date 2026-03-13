#!/usr/bin/env python3
"""
NQ Fractal AI Trading Backtester — Main Entry Point

Launches multi-source data workers, the continuous backtest-and-refine loop,
and a live web dashboard.

Usage:
    python main.py                    # Run with defaults
    python main.py --port 8080        # Custom dashboard port
    python main.py --pop 20           # Larger population
    python main.py --no-dashboard     # Headless mode (console only)

The system will:
  1. Start 5 data workers (Crypto, Stock, Forex, Macro, Historical Backfill)
  2. Collect multi-source data into data_lake/ (parquet files)
  3. Test 5 strategies x N parameter sets on random instruments from the lake
  4. Walk-forward validation: 70% train, 30% out-of-sample test
  5. AI analyzer examines every trade — learns patterns, detects overfitting
  6. Cross-instrument correlation engine finds NQ-related signals
  7. Evolve parameters using genetic optimization + AI adjustments
  8. Display everything on a live web dashboard
  9. Repeat forever — getting smarter every cycle
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from optimizer import BacktestLoop
from dashboard import start_dashboard
from data_workers import start_all_workers


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
    ║                  MULTI-SOURCE EDITION                        ║
    ║                                                              ║
    ║   Data Workers: 5 active (Crypto, Stock, Forex, Macro, BF)  ║
    ║   Data Lake: data_lake/ (multi-source, multi-instrument)     ║
    ║   Walk-Forward Validation: ON (70/30 train/test split)       ║
    ║   Overfit Detection: ON (OOS scoring + win rate penalties)   ║
    ║                                                              ║
    ║   5 Strategies:                                              ║
    ║     * EMA Crossover (trend following)                        ║
    ║     * Fractal Breakout (structure breaks)                    ║
    ║     * Mean Reversion (Bollinger + RSI)                       ║
    ║     * Momentum (MACD + Volume)                               ║
    ║     * Multi-TF Fractal Confluence (combined)                 ║
    ║                                                              ║
    ║   AI BRAIN: Analyzes every trade, learns patterns,           ║
    ║   cross-instrument correlation, NQ directional opinion.      ║
    ║                                                              ║
    ║   Population: {pop:>3} parameter sets evolving via GA + AI      ║
    ║   Press Ctrl+C to stop                                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """.format(pop=args.pop))

    # Start data workers
    data_lake_root = Path(__file__).parent / "data_lake"
    data_lake_root.mkdir(exist_ok=True)
    workers = start_all_workers(data_lake_root)
    print(f"  Started {len(workers)} data workers")

    # Create the backtest loop with workers
    loop = BacktestLoop(population_size=args.pop, seed=args.seed, workers=workers)

    # Start dashboard
    if not args.no_dashboard:
        start_dashboard(loop, port=args.port)
        print(f"  Open your browser to http://localhost:{args.port}")
    print()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n  Stopping backtest loop...")

        # Stop workers
        for w in workers:
            w.stop()
        print("  Workers stopped.")

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

            # NQ opinion
            nq = ai_state.get("nq_opinion", {})
            if nq:
                print(f"\n  === NQ AI OPINION ===")
                print(f"  Direction: {nq.get('direction', '?')} "
                      f"(confidence: {nq.get('confidence', 0):.0%})")

            # Data lake stats
            data_stats = loop.router.get_lake_stats()
            print(f"\n  === DATA LAKE ===")
            print(f"  Instruments: {data_stats.get('total_instruments', 0)}")
            print(f"  Files: {data_stats.get('total_files', 0)}")
            print(f"  Size: {data_stats.get('total_mb', 0):.1f} MB")
            print(f"  Windows served: {data_stats.get('windows_served', 0)}")

            # Worker stats
            print(f"\n  === WORKER STATS ===")
            for w in workers:
                s = w.get_stats()
                print(f"  {s['name']}: {s['total_fetches']} fetches, "
                      f"{s['total_bars']} bars, {s['errors']} errors")

            # OOS summary
            if loop.oos_scores:
                print(f"\n  === OUT-OF-SAMPLE VALIDATION ===")
                print(f"  OOS scores: {len(loop.oos_scores)} tests")
                print(f"  Avg OOS score: {sum(loop.oos_scores)/len(loop.oos_scores):.1f}")
                print(f"  Overfit warnings: {len(loop.overfit_warnings)}")

        print("\n  Goodbye.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the loop forever
    loop.run_forever()


if __name__ == "__main__":
    main()
