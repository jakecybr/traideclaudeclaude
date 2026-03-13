"""
Web Dashboard for NQ Fractal AI Trading Backtester

Real-time visualization of:
- Multi-source data lake status and worker monitoring
- NQ AI Opinion panel with directional arrow and confidence
- All-Markets monitor with correlation heatmap
- Candlestick chart with trade entries/exits
- Equity curve and optimization progress
- AI Brain: insights, conclusions, learning progress
- Bias mitigation: in-sample vs OOS, overfit detection
- Regime analysis heatmap
- Strategy performance comparison
- Trade log with full details
- Parameter evolution
"""

from flask import Flask, jsonify, render_template_string
import threading

app = Flask(__name__)

# Global reference to the backtest loop — set by main.py
loop = None

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NQ Fractal AI Backtester</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0e17;
            color: #c8d6e5;
            font-family: 'Courier New', monospace;
            overflow-x: hidden;
        }
        .header {
            background: linear-gradient(135deg, #0d1321 0%, #1a1f36 100%);
            border-bottom: 1px solid #2a3a5c;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 18px;
            color: #00d4aa;
            letter-spacing: 2px;
        }
        .header .status {
            font-size: 13px;
            color: #ffd700;
            animation: pulse 2s infinite;
        }
        .data-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .data-real { background: #00d4aa33; color: #00d4aa; border: 1px solid #00d4aa; }
        .data-synthetic { background: #ff475733; color: #ff4757; border: 1px solid #ff4757; }
        .data-multi { background: #a78bfa33; color: #a78bfa; border: 1px solid #a78bfa; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .stats-bar {
            display: flex;
            gap: 2px;
            padding: 8px 24px;
            background: #0d1321;
            border-bottom: 1px solid #1a2744;
            flex-wrap: wrap;
        }
        .stat-card {
            flex: 1;
            min-width: 100px;
            background: #111827;
            border: 1px solid #1e2d4a;
            border-radius: 4px;
            padding: 6px 10px;
            text-align: center;
        }
        .stat-card .label {
            font-size: 9px;
            color: #5a6e8a;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-card .value {
            font-size: 16px;
            font-weight: bold;
            margin-top: 2px;
        }
        .positive { color: #00d4aa; }
        .negative { color: #ff4757; }
        .neutral { color: #ffd700; }
        .ai-color { color: #a78bfa; }
        .overfit-color { color: #ff6b6b; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            padding: 2px 24px;
        }
        .grid-full {
            grid-column: 1 / -1;
        }
        .chart-container {
            background: #111827;
            border: 1px solid #1e2d4a;
            border-radius: 4px;
            margin-top: 8px;
            padding: 8px;
        }
        .chart-container h3 {
            font-size: 12px;
            color: #5a6e8a;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 4px 8px;
            border-bottom: 1px solid #1e2d4a;
            margin-bottom: 4px;
        }
        .chart-container h3 .ai-badge {
            background: #a78bfa22;
            color: #a78bfa;
            border: 1px solid #a78bfa55;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 9px;
            margin-left: 8px;
        }
        .chart-container h3 .overfit-badge {
            background: #ff6b6b22;
            color: #ff6b6b;
            border: 1px solid #ff6b6b55;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 9px;
            margin-left: 8px;
        }

        /* NQ Opinion panel */
        .nq-opinion-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 8px;
            padding: 8px;
        }
        .opinion-box {
            background: #0d1321;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }
        .opinion-arrow {
            font-size: 40px;
            line-height: 1;
        }
        .opinion-label {
            font-size: 9px;
            color: #5a6e8a;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }
        .opinion-value {
            font-size: 14px;
            font-weight: bold;
        }
        .opinion-list {
            list-style: none;
            padding: 0;
            font-size: 10px;
            text-align: left;
            max-height: 120px;
            overflow-y: auto;
        }
        .opinion-list li {
            padding: 2px 0;
            border-bottom: 1px solid #1a2744;
        }

        /* Markets monitor grid */
        .markets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 4px;
            padding: 8px;
            max-height: 200px;
            overflow-y: auto;
        }
        .market-cell {
            background: #0d1321;
            border-radius: 3px;
            padding: 6px;
            text-align: center;
            font-size: 10px;
        }
        .market-cell .ticker { font-weight: bold; font-size: 11px; }
        .market-cell .trend-arrow { font-size: 14px; }
        .market-cell .corr-val { font-size: 9px; color: #5a6e8a; }

        /* Worker status */
        .workers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 4px;
            padding: 8px;
        }
        .worker-card {
            background: #0d1321;
            border-radius: 3px;
            padding: 8px;
            font-size: 10px;
        }
        .worker-card .wname { font-weight: bold; color: #00d4aa; font-size: 11px; }
        .worker-card .wstat { color: #5a6e8a; }

        /* Insights panel */
        .insights-panel {
            max-height: 380px;
            overflow-y: auto;
            padding: 8px;
        }
        .insight-card {
            background: #0d1321;
            border-left: 3px solid #a78bfa;
            padding: 8px 12px;
            margin-bottom: 6px;
            border-radius: 0 4px 4px 0;
        }
        .insight-card.validated { border-left-color: #00d4aa; }
        .insight-card.low-conf { border-left-color: #5a6e8a; opacity: 0.7; }
        .insight-card.overfit { border-left-color: #ff6b6b; }
        .insight-category {
            font-size: 9px;
            color: #a78bfa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 3px;
        }
        .insight-text {
            font-size: 11px;
            color: #c8d6e5;
            line-height: 1.4;
        }
        .insight-meta {
            font-size: 9px;
            color: #5a6e8a;
            margin-top: 4px;
            display: flex;
            gap: 12px;
        }
        .conf-bar {
            display: inline-block;
            height: 4px;
            background: #a78bfa;
            border-radius: 2px;
            vertical-align: middle;
        }

        /* Regime heatmap */
        .regime-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 4px;
            padding: 8px;
            max-height: 250px;
            overflow-y: auto;
        }
        .regime-cell {
            padding: 6px 8px;
            border-radius: 3px;
            font-size: 10px;
            text-align: center;
        }
        .regime-cell .rname { font-size: 9px; color: #5a6e8a; margin-bottom: 2px; }
        .regime-cell .rwr { font-size: 14px; font-weight: bold; }
        .regime-cell .rcount { font-size: 9px; color: #5a6e8a; }

        .trade-log {
            max-height: 350px;
            overflow-y: auto;
            margin-top: 8px;
            padding: 0 24px 24px;
        }
        .trade-log table, .cycle-history table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        .trade-log th, .cycle-history th {
            background: #1a1f36;
            color: #5a6e8a;
            padding: 5px 6px;
            text-align: left;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 10px;
            position: sticky;
            top: 0;
        }
        .trade-log td, .cycle-history td {
            padding: 4px 6px;
            border-bottom: 1px solid #1a2744;
        }
        .trade-log tr:hover, .cycle-history tr:hover {
            background: #1a2744;
        }
        .cycle-history {
            max-height: 300px;
            overflow-y: auto;
        }
        .params-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 4px;
            padding: 8px;
            max-height: 200px;
            overflow-y: auto;
        }
        .param-item {
            background: #0d1321;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
        }
        .param-item .pname { color: #5a6e8a; }
        .param-item .pval { color: #00d4aa; font-weight: bold; }
        .param-item .padj { color: #a78bfa; font-size: 9px; }
        #loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 20px;
            color: #ffd700;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>NQ FRACTAL AI BACKTESTER — MULTI-SOURCE</h1>
        <span id="dataBadge" class="data-badge data-multi">MULTI-SOURCE</span>
        <div class="status" id="status">INITIALIZING...</div>
    </div>

    <div class="stats-bar" id="statsBar">
        <div class="stat-card">
            <div class="label">Cycle</div>
            <div class="value neutral" id="statCycle">0</div>
        </div>
        <div class="stat-card">
            <div class="label">Instrument</div>
            <div class="value neutral" id="statSymbol">--</div>
        </div>
        <div class="stat-card">
            <div class="label">Best Strategy</div>
            <div class="value neutral" id="statStrategy">&mdash;</div>
        </div>
        <div class="stat-card">
            <div class="label">Score (IS)</div>
            <div class="value neutral" id="statScore">0</div>
        </div>
        <div class="stat-card">
            <div class="label">Score (OOS)</div>
            <div class="value neutral" id="statOOS">0</div>
        </div>
        <div class="stat-card">
            <div class="label">P&amp;L</div>
            <div class="value" id="statPnl">$0</div>
        </div>
        <div class="stat-card">
            <div class="label">Win Rate</div>
            <div class="value" id="statWinRate">0%</div>
        </div>
        <div class="stat-card">
            <div class="label">Sharpe</div>
            <div class="value" id="statSharpe">0</div>
        </div>
        <div class="stat-card">
            <div class="label">Profit Factor</div>
            <div class="value" id="statPF">0</div>
        </div>
        <div class="stat-card">
            <div class="label">Max DD</div>
            <div class="value negative" id="statDD">0%</div>
        </div>
        <div class="stat-card">
            <div class="label">Trades</div>
            <div class="value neutral" id="statTrades">0</div>
        </div>
        <div class="stat-card">
            <div class="label">AI Insights</div>
            <div class="value ai-color" id="statInsights">0</div>
        </div>
        <div class="stat-card">
            <div class="label">All-Time Best</div>
            <div class="value positive" id="statBestEver">0</div>
        </div>
        <div class="stat-card">
            <div class="label">Data Lake</div>
            <div class="value ai-color" id="statLake">0 inst</div>
        </div>
    </div>

    <div class="grid">
        <!-- NQ AI OPINION -->
        <div class="chart-container grid-full">
            <h3>NQ/SP500 AI Trading Intelligence <span class="ai-badge">AI OPINION</span></h3>
            <div class="nq-opinion-grid" id="nqOpinionPanel">
                <div class="opinion-box"><p style="color:#5a6e8a;font-size:11px;">Accumulating data...</p></div>
            </div>
        </div>

        <!-- ALL-MARKETS MONITOR -->
        <div class="chart-container grid-full">
            <h3>All-Markets Monitor <span class="ai-badge">LIVE</span></h3>
            <div class="markets-grid" id="marketsMonitor">
                <p style="color:#5a6e8a;padding:8px;font-size:11px;">Waiting for correlation data...</p>
            </div>
        </div>

        <!-- DATA LAKE & WORKERS -->
        <div class="chart-container grid-full">
            <h3>Data Lake & Workers <span class="ai-badge" id="workerBadge">0 workers</span></h3>
            <div class="workers-grid" id="workersPanel"></div>
        </div>

        <div class="chart-container grid-full">
            <h3>Price Action & Trades <span id="chartDataLabel" class="ai-badge">REAL DATA</span></h3>
            <div id="priceChart" style="height:420px;"></div>
        </div>

        <div class="chart-container">
            <h3>Equity Curve</h3>
            <div id="equityChart" style="height:250px;"></div>
        </div>

        <div class="chart-container">
            <h3>Optimization Progress</h3>
            <div id="progressChart" style="height:250px;"></div>
        </div>

        <!-- BIAS MITIGATION -->
        <div class="chart-container grid-full">
            <h3>Bias Mitigation &mdash; Walk-Forward Validation <span class="overfit-badge" id="overfitBadge"></span></h3>
            <div style="display:grid; grid-template-columns: 2fr 1fr; gap: 8px;">
                <div id="oosChart" style="height:250px;"></div>
                <div id="overfitWarnings" style="max-height:250px;overflow-y:auto;padding:8px;">
                    <p style="color:#5a6e8a;font-size:11px;">No overfit warnings yet.</p>
                </div>
            </div>
        </div>

        <!-- AI BRAIN PANEL -->
        <div class="chart-container grid-full">
            <h3>AI Brain &mdash; Insights & Conclusions <span class="ai-badge">LEARNING</span></h3>
            <div style="display:grid; grid-template-columns: 2fr 1fr 1fr; gap: 8px;">
                <div class="insights-panel" id="insightsPanel">
                    <p style="color:#5a6e8a;">Waiting for analysis data...</p>
                </div>
                <div>
                    <div style="font-size:10px;color:#5a6e8a;text-transform:uppercase;padding:4px 8px;letter-spacing:1px;border-bottom:1px solid #1e2d4a;">
                        Regime Performance
                    </div>
                    <div class="regime-grid" id="regimeGrid"></div>
                </div>
                <div>
                    <div style="font-size:10px;color:#5a6e8a;text-transform:uppercase;padding:4px 8px;letter-spacing:1px;border-bottom:1px solid #1e2d4a;">
                        Session / Volatility Stats
                    </div>
                    <div class="regime-grid" id="sessionGrid"></div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h3>Cycle History</h3>
            <div class="cycle-history" id="cycleHistory"></div>
        </div>

        <div class="chart-container">
            <h3>Best Parameters <span class="ai-badge" id="adjBadge"></span></h3>
            <div class="params-grid" id="paramsGrid"></div>
        </div>
    </div>

    <div class="chart-container" style="margin: 8px 24px;">
        <h3>Trade Log (Current Cycle Winner)</h3>
    </div>
    <div class="trade-log" id="tradeLog"></div>

    <div id="loading">Waiting for first cycle...</div>

    <script>
    const REFRESH_MS = 2000;
    let lastCycle = 0;

    function formatMoney(v) {
        if (v === undefined || v === null) return '$0';
        const sign = v >= 0 ? '' : '-';
        return sign + '$' + Math.abs(v).toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
    }

    function colorClass(v) {
        if (v > 0) return 'positive';
        if (v < 0) return 'negative';
        return 'neutral';
    }

    function wrColor(wr) {
        if (wr >= 0.6) return '#00d4aa';
        if (wr >= 0.5) return '#7bed9f';
        if (wr >= 0.4) return '#ffd700';
        return '#ff4757';
    }

    function wrBg(wr) {
        if (wr >= 0.6) return '#00d4aa22';
        if (wr >= 0.5) return '#7bed9f18';
        if (wr >= 0.4) return '#ffd70018';
        return '#ff475722';
    }

    function corrColor(c) {
        if (c > 0.7) return '#00d4aa';
        if (c > 0.3) return '#7bed9f';
        if (c > -0.3) return '#5a6e8a';
        if (c > -0.7) return '#ffd700';
        return '#ff4757';
    }

    function trendArrow(t) {
        if (t === 'STRONG_UP') return '<span style="color:#00d4aa;">&#9650;&#9650;</span>';
        if (t === 'UP') return '<span style="color:#7bed9f;">&#9650;</span>';
        if (t === 'STRONG_DOWN') return '<span style="color:#ff4757;">&#9660;&#9660;</span>';
        if (t === 'DOWN') return '<span style="color:#ffd700;">&#9660;</span>';
        return '<span style="color:#5a6e8a;">&#9644;</span>';
    }

    async function refresh() {
        try {
            const resp = await fetch('/api/state');
            const data = await resp.json();

            if (!data || !data.result) {
                setTimeout(refresh, REFRESH_MS);
                return;
            }

            document.getElementById('loading').style.display = 'none';
            document.getElementById('status').textContent = data.status;
            document.getElementById('statCycle').textContent = data.cycle_count;
            document.getElementById('statStrategy').textContent = data.result.strategy;
            document.getElementById('statScore').textContent = data.result.score.toFixed(0);

            // OOS score from latest cycle
            const latestCycle = data.cycle_history && data.cycle_history.length > 0 ?
                data.cycle_history[data.cycle_history.length - 1] : null;
            if (latestCycle) {
                const oosEl = document.getElementById('statOOS');
                oosEl.textContent = (latestCycle.oos_score || 0).toFixed(0);
                oosEl.className = 'value ' + (latestCycle.overfit_flag ? 'negative' : 'positive');
            }

            // Data source indicator
            const dm = data.data_meta || {};
            const badge = document.getElementById('dataBadge');
            if (dm.is_normalized) {
                badge.textContent = 'MULTI-SOURCE';
                badge.className = 'data-badge data-multi';
            } else if (dm.is_real !== false) {
                badge.textContent = 'REAL DATA';
                badge.className = 'data-badge data-real';
            } else {
                badge.textContent = 'SYNTHETIC';
                badge.className = 'data-badge data-synthetic';
            }

            const sym = dm.instrument || dm.symbol || '--';
            document.getElementById('statSymbol').textContent = sym;

            const chartLabel = document.getElementById('chartDataLabel');
            chartLabel.textContent = (dm.source || 'unknown').substring(0, 40);

            const pnlEl = document.getElementById('statPnl');
            pnlEl.textContent = formatMoney(data.result.total_pnl);
            pnlEl.className = 'value ' + colorClass(data.result.total_pnl);

            const wrEl = document.getElementById('statWinRate');
            wrEl.textContent = (data.result.win_rate * 100).toFixed(1) + '%';
            wrEl.className = 'value ' + (data.result.win_rate >= 0.5 ? 'positive' : 'negative');

            document.getElementById('statSharpe').textContent = data.result.sharpe.toFixed(2);
            document.getElementById('statPF').textContent = data.result.profit_factor.toFixed(2);
            document.getElementById('statDD').textContent = data.result.max_dd_pct.toFixed(1) + '%';
            document.getElementById('statTrades').textContent = data.result.total_trades;
            document.getElementById('statBestEver').textContent = data.best_ever_strategy + ' (' + data.best_ever_score + ')';

            // AI stats
            const ai = data.ai || {};
            document.getElementById('statInsights').textContent = (ai.active_insights || 0) + ' active';

            // Data lake stats
            const ds = data.data_stats || {};
            document.getElementById('statLake').textContent = (ds.total_instruments || 0) + ' inst / ' + (ds.total_files || 0) + ' files';

            // Only redraw charts if cycle changed
            if (data.cycle_count !== lastCycle) {
                lastCycle = data.cycle_count;
                drawNQOpinion(data);
                drawMarketsMonitor(data);
                drawWorkersPanel(data);
                drawPriceChart(data);
                drawEquityChart(data);
                drawProgressChart(data);
                drawOOSChart(data);
                drawOverfitWarnings(data);
                drawAIInsights(data);
                drawRegimeGrid(data);
                drawSessionGrid(data);
                drawCycleHistory(data);
                drawParams(data);
                drawTradeLog(data);
            }

        } catch (e) {
            console.error('Refresh error:', e);
        }
        setTimeout(refresh, REFRESH_MS);
    }

    function drawNQOpinion(data) {
        const ai = data.ai || {};
        const opinion = ai.nq_opinion || {};
        const dir = opinion.direction || 'NEUTRAL';
        const conf = opinion.confidence || 0;

        let arrowHtml, arrowColor;
        if (dir === 'BULLISH') { arrowHtml = '&#9650;'; arrowColor = '#00d4aa'; }
        else if (dir === 'BEARISH') { arrowHtml = '&#9660;'; arrowColor = '#ff4757'; }
        else { arrowHtml = '&#9644;'; arrowColor = '#ffd700'; }

        const regime = opinion.regime || {};
        const reasoning = opinion.reasoning || [];
        const bestStrats = opinion.best_strategies || [];
        const corrSignals = opinion.correlated_signals || [];

        let html = `
        <div class="opinion-box">
            <div class="opinion-label">AI NQ Direction</div>
            <div class="opinion-arrow" style="color:${arrowColor};">${arrowHtml}</div>
            <div class="opinion-value" style="color:${arrowColor};">${dir}</div>
            <div style="margin-top:4px;font-size:10px;color:#5a6e8a;">Confidence: ${(conf*100).toFixed(0)}%</div>
            <div style="width:100%;height:4px;background:#1a2744;border-radius:2px;margin-top:4px;">
                <div style="width:${conf*100}%;height:100%;background:${arrowColor};border-radius:2px;"></div>
            </div>
        </div>
        <div class="opinion-box">
            <div class="opinion-label">Best Strategies</div>
            <ul class="opinion-list">
                ${bestStrats.map(s => `<li>${s.name}: ${(s.win_rate*100).toFixed(0)}% WR, $${s.avg_pnl}/t</li>`).join('')}
                ${bestStrats.length === 0 ? '<li style="color:#5a6e8a;">Accumulating...</li>' : ''}
            </ul>
        </div>
        <div class="opinion-box">
            <div class="opinion-label">NQ Regime</div>
            <div style="font-size:12px;margin:4px 0;">
                Trend: ${trendArrow(regime.trend || 'FLAT')} <b>${regime.trend || '?'}</b><br>
                Vol: <b>${regime.volatility || '?'}</b><br>
                Mom: <b>${regime.momentum || '?'}</b>
            </div>
        </div>
        <div class="opinion-box">
            <div class="opinion-label">Reasoning</div>
            <ul class="opinion-list">
                ${reasoning.map(r => `<li>${r}</li>`).join('')}
                ${corrSignals.map(s => `<li style="color:#a78bfa;">${s}</li>`).join('')}
                ${reasoning.length === 0 && corrSignals.length === 0 ? '<li style="color:#5a6e8a;">Building opinion...</li>' : ''}
            </ul>
        </div>`;
        document.getElementById('nqOpinionPanel').innerHTML = html;
    }

    function drawMarketsMonitor(data) {
        const corr = data.correlations || {};
        const regimes = corr.instrument_regimes || {};
        const nqCorrs = corr.nq_correlations || [];

        if (Object.keys(regimes).length === 0 && nqCorrs.length === 0) {
            document.getElementById('marketsMonitor').innerHTML =
                '<p style="color:#5a6e8a;padding:8px;font-size:11px;">Waiting for multi-instrument data...</p>';
            return;
        }

        // Build a map of NQ correlations
        const corrMap = {};
        for (const c of nqCorrs) { corrMap[c.instrument] = c.correlation; }

        let html = '';
        // Sort instruments: NQ-correlated first
        const allInsts = Object.keys(regimes);
        allInsts.sort((a, b) => Math.abs(corrMap[b] || 0) - Math.abs(corrMap[a] || 0));

        for (const inst of allInsts) {
            const r = regimes[inst];
            const c = corrMap[inst] || 0;
            const borderColor = corrColor(c);
            html += `<div class="market-cell" style="border:1px solid ${borderColor}44;">
                <div class="ticker" style="color:${borderColor};">${inst}</div>
                <div class="trend-arrow">${trendArrow(r.trend)}</div>
                <div style="font-size:9px;">${r.volatility} vol | ${r.momentum}</div>
                <div class="corr-val">NQ corr: <b style="color:${borderColor};">${c.toFixed(2)}</b></div>
            </div>`;
        }
        document.getElementById('marketsMonitor').innerHTML = html || '<p style="color:#5a6e8a;padding:8px;">No instruments.</p>';
    }

    function drawWorkersPanel(data) {
        const workers = data.workers || [];
        const ds = data.data_stats || {};

        let html = `<div class="worker-card">
            <div class="wname">DATA LAKE</div>
            <div class="wstat">Instruments: <b>${ds.total_instruments || 0}</b></div>
            <div class="wstat">Files: <b>${ds.total_files || 0}</b></div>
            <div class="wstat">Size: <b>${(ds.total_mb || 0).toFixed(1)} MB</b></div>
            <div class="wstat">Sources: ${(ds.sources || []).join(', ') || 'none'}</div>
        </div>`;

        for (const w of workers) {
            const statusColor = w.status === 'running' ? '#00d4aa' : '#ff4757';
            const uptime = w.uptime ? (w.uptime / 60).toFixed(0) + 'm' : '0m';
            html += `<div class="worker-card">
                <div class="wname">${w.name} <span style="color:${statusColor};">&#9679;</span></div>
                <div class="wstat">Fetches: ${w.total_fetches} | Bars: ${w.total_bars}</div>
                <div class="wstat">Errors: ${w.errors} | Up: ${uptime}</div>
            </div>`;
        }
        document.getElementById('workersPanel').innerHTML = html;
        document.getElementById('workerBadge').textContent = workers.length + ' workers';
    }

    function drawPriceChart(data) {
        if (!data.bars || data.bars.length === 0) return;

        const bars = data.bars;
        const x = bars.map((b, i) => i);
        const opens = bars.map(b => b.open);
        const highs = bars.map(b => b.high);
        const lows = bars.map(b => b.low);
        const closes = bars.map(b => b.close);
        const volumes = bars.map(b => b.volume);

        const colors = closes.map((c, i) => c >= opens[i] ? '#00d4aa' : '#ff4757');

        const traces = [{
            type: 'candlestick',
            x: x,
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            increasing: {line: {color: '#00d4aa'}, fillcolor: '#00d4aa44'},
            decreasing: {line: {color: '#ff4757'}, fillcolor: '#ff475744'},
            name: 'Price',
            yaxis: 'y2',
        }, {
            type: 'bar',
            x: x,
            y: volumes,
            marker: {color: colors, opacity: 0.3},
            name: 'Volume',
            yaxis: 'y',
        }];

        if (data.trades && data.trades.length > 0) {
            const longEntries = data.trades.filter(t => t.direction === 'LONG');
            const shortEntries = data.trades.filter(t => t.direction === 'SHORT');
            const winners = data.trades.filter(t => t.pnl_dollars > 0);
            const losers = data.trades.filter(t => t.pnl_dollars <= 0);

            if (longEntries.length > 0) {
                traces.push({
                    type: 'scatter', mode: 'markers',
                    x: longEntries.map(t => t.entry_bar),
                    y: longEntries.map(t => t.entry_price),
                    marker: {symbol: 'triangle-up', size: 12, color: '#00d4aa', line: {width: 1, color: '#fff'}},
                    name: 'Long Entry', yaxis: 'y2',
                    hovertext: longEntries.map(t => t.entry_reason),
                });
            }
            if (shortEntries.length > 0) {
                traces.push({
                    type: 'scatter', mode: 'markers',
                    x: shortEntries.map(t => t.entry_bar),
                    y: shortEntries.map(t => t.entry_price),
                    marker: {symbol: 'triangle-down', size: 12, color: '#ff4757', line: {width: 1, color: '#fff'}},
                    name: 'Short Entry', yaxis: 'y2',
                    hovertext: shortEntries.map(t => t.entry_reason),
                });
            }
            if (winners.length > 0) {
                traces.push({
                    type: 'scatter', mode: 'markers',
                    x: winners.map(t => t.exit_bar),
                    y: winners.map(t => t.exit_price),
                    marker: {symbol: 'diamond', size: 9, color: '#00d4aa', line: {width: 1, color: '#fff'}},
                    name: 'Win Exit', yaxis: 'y2',
                    hovertext: winners.map(t => t.exit_reason + ' $' + t.pnl_dollars.toFixed(0)),
                });
            }
            if (losers.length > 0) {
                traces.push({
                    type: 'scatter', mode: 'markers',
                    x: losers.map(t => t.exit_bar),
                    y: losers.map(t => t.exit_price),
                    marker: {symbol: 'diamond', size: 9, color: '#ff4757', line: {width: 1, color: '#fff'}},
                    name: 'Loss Exit', yaxis: 'y2',
                    hovertext: losers.map(t => t.exit_reason + ' $' + t.pnl_dollars.toFixed(0)),
                });
            }
            for (const t of data.trades) {
                traces.push({
                    type: 'scatter', mode: 'lines',
                    x: [t.entry_bar, t.exit_bar],
                    y: [t.entry_price, t.exit_price],
                    line: {color: t.pnl_dollars > 0 ? '#00d4aa55' : '#ff475755', width: 1, dash: 'dot'},
                    showlegend: false, yaxis: 'y2', hoverinfo: 'skip',
                });
            }
        }

        const layout = {
            paper_bgcolor: '#111827', plot_bgcolor: '#0a0e17',
            font: {color: '#5a6e8a', size: 10},
            margin: {l: 50, r: 20, t: 10, b: 30},
            xaxis: {gridcolor: '#1a2744', rangeslider: {visible: false}},
            yaxis: {domain: [0, 0.15], gridcolor: '#1a2744', title: 'Vol'},
            yaxis2: {domain: [0.18, 1], gridcolor: '#1a2744', title: 'Price'},
            legend: {orientation: 'h', y: 1.02, x: 0, font: {size: 10}},
            dragmode: 'pan',
        };

        Plotly.newPlot('priceChart', traces, layout, {responsive: true, scrollZoom: true});
    }

    function drawEquityChart(data) {
        if (!data.result || !data.result.equity_curve) return;

        const eq = data.result.equity_curve;
        const x = eq.map((_, i) => i);
        let peak = eq[0];
        const dd = eq.map(e => { peak = Math.max(peak, e); return ((e - peak) / peak) * 100; });

        const traces = [{
            type: 'scatter', x: x, y: eq, mode: 'lines',
            fill: 'tozeroy', fillcolor: '#00d4aa11',
            line: {color: '#00d4aa', width: 2}, name: 'Equity', yaxis: 'y2',
        }, {
            type: 'scatter', x: x, y: dd, mode: 'lines',
            fill: 'tozeroy', fillcolor: '#ff475722',
            line: {color: '#ff4757', width: 1}, name: 'Drawdown %', yaxis: 'y',
        }];

        const layout = {
            paper_bgcolor: '#111827', plot_bgcolor: '#0a0e17',
            font: {color: '#5a6e8a', size: 10},
            margin: {l: 50, r: 20, t: 10, b: 30},
            showlegend: false,
            yaxis: {domain: [0, 0.25], gridcolor: '#1a2744', title: 'DD%'},
            yaxis2: {domain: [0.3, 1], gridcolor: '#1a2744', title: '$'},
            xaxis: {gridcolor: '#1a2744', title: 'Trade #'},
        };
        Plotly.newPlot('equityChart', traces, layout, {responsive: true});
    }

    function drawProgressChart(data) {
        if (!data.cycle_history || data.cycle_history.length === 0) return;

        const hist = data.cycle_history;
        const x = hist.map(h => h.cycle);

        const traces = [{
            type: 'scatter', x: x, y: hist.map(h => h.score),
            mode: 'lines+markers', line: {color: '#ffd700', width: 2},
            marker: {size: 4}, name: 'In-Sample Score', yaxis: 'y2',
        }, {
            type: 'scatter', x: x, y: hist.map(h => h.oos_score || 0),
            mode: 'lines+markers', line: {color: '#a78bfa', width: 2, dash: 'dash'},
            marker: {size: 4}, name: 'OOS Score', yaxis: 'y2',
        }, {
            type: 'scatter', x: x, y: hist.map(h => h.pnl),
            mode: 'lines', line: {color: '#00d4aa', width: 1},
            name: 'P&L', yaxis: 'y',
        }];

        const layout = {
            paper_bgcolor: '#111827', plot_bgcolor: '#0a0e17',
            font: {color: '#5a6e8a', size: 10},
            margin: {l: 50, r: 50, t: 10, b: 30},
            legend: {orientation: 'h', y: 1.05, font: {size: 9}},
            yaxis: {gridcolor: '#1a2744', title: 'P&L $', side: 'left'},
            yaxis2: {gridcolor: '#1a2744', title: 'Score', side: 'right', overlaying: 'y'},
            xaxis: {gridcolor: '#1a2744', title: 'Cycle'},
        };
        Plotly.newPlot('progressChart', traces, layout, {responsive: true});
    }

    function drawOOSChart(data) {
        const hist = data.cycle_history || [];
        if (hist.length === 0) return;

        const x = hist.map(h => h.cycle);
        const isScores = hist.map(h => h.score);
        const oosScores = hist.map(h => h.oos_score || 0);
        const overfits = hist.map(h => h.overfit_flag ? h.score : null);

        const traces = [{
            type: 'scatter', x: x, y: isScores,
            mode: 'lines', line: {color: '#ffd700', width: 2},
            name: 'In-Sample',
        }, {
            type: 'scatter', x: x, y: oosScores,
            mode: 'lines', line: {color: '#a78bfa', width: 2},
            name: 'Out-of-Sample',
        }, {
            type: 'scatter', x: x, y: overfits,
            mode: 'markers', marker: {color: '#ff6b6b', size: 10, symbol: 'x'},
            name: 'Overfit Flag',
        }];

        const layout = {
            paper_bgcolor: '#111827', plot_bgcolor: '#0a0e17',
            font: {color: '#5a6e8a', size: 10},
            margin: {l: 50, r: 20, t: 10, b: 30},
            legend: {orientation: 'h', y: 1.05, font: {size: 9}},
            yaxis: {gridcolor: '#1a2744', title: 'Score'},
            xaxis: {gridcolor: '#1a2744', title: 'Cycle'},
        };
        Plotly.newPlot('oosChart', traces, layout, {responsive: true});

        // Update overfit badge
        const overfitCount = hist.filter(h => h.overfit_flag).length;
        const badge = document.getElementById('overfitBadge');
        badge.textContent = overfitCount > 0 ? overfitCount + ' overfit detections' : 'CLEAN';
        badge.style.color = overfitCount > 0 ? '#ff6b6b' : '#00d4aa';
    }

    function drawOverfitWarnings(data) {
        const warnings = data.overfit_warnings || [];
        if (warnings.length === 0) {
            document.getElementById('overfitWarnings').innerHTML =
                '<p style="color:#5a6e8a;font-size:11px;padding:4px;">No overfit warnings. Walk-forward validation passing.</p>';
            return;
        }
        let html = '<div style="font-size:10px;color:#ff6b6b;margin-bottom:8px;">OVERFIT ALERTS</div>';
        for (const w of warnings.slice().reverse()) {
            html += `<div style="background:#ff6b6b11;border-left:2px solid #ff6b6b;padding:6px 8px;margin-bottom:4px;border-radius:0 3px 3px 0;font-size:10px;">
                ${w}
            </div>`;
        }
        document.getElementById('overfitWarnings').innerHTML = html;
    }

    function drawAIInsights(data) {
        const ai = data.ai;
        if (!ai || !ai.insights || ai.insights.length === 0) {
            document.getElementById('insightsPanel').innerHTML =
                '<p style="color:#5a6e8a;padding:8px;">AI is accumulating data... insights will appear after enough trades are analyzed.</p>' +
                '<p style="color:#5a6e8a;padding:0 8px;font-size:11px;">Trades analyzed: ' + (ai ? ai.total_trades_analyzed : 0) + '</p>';
            return;
        }

        let html = '<div style="font-size:10px;color:#a78bfa;padding:4px 0 8px;border-bottom:1px solid #1e2d4a;margin-bottom:8px;">' +
            'Trades Analyzed: <b>' + ai.total_trades_analyzed + '</b> | ' +
            'Active Insights: <b>' + ai.active_insights + '</b> | ' +
            'Avg Efficiency: <b>' + (ai.avg_efficiency * 100).toFixed(0) + '%</b> | ' +
            'MFE/MAE: <b>' + ai.avg_mfe_mae_ratio + '</b></div>';

        for (const ins of ai.insights) {
            const confPx = Math.round(ins.confidence * 80);
            let cardClass = ins.validated ? 'validated' : (ins.confidence < 0.5 ? 'low-conf' : '');
            if (ins.category === 'OVERFIT') cardClass = 'overfit';
            html += `<div class="insight-card ${cardClass}">
                <div class="insight-category" ${ins.category === 'OVERFIT' ? 'style="color:#ff6b6b;"' : ''}>${ins.category} ${ins.validated ? '&#10003; VALIDATED' : ''}</div>
                <div class="insight-text">${ins.conclusion}</div>
                <div class="insight-meta">
                    <span>Confidence: ${(ins.confidence*100).toFixed(0)}% <span class="conf-bar" style="width:${confPx}px;${ins.category === 'OVERFIT' ? 'background:#ff6b6b;' : ''}"></span></span>
                    <span>Sample: ${ins.sample_size}</span>
                    <span>+${ins.validations} / -${ins.invalidations}</span>
                    ${Object.keys(ins.adjustments).length > 0 ?
                        '<span style="color:#a78bfa;">Adj: ' + Object.entries(ins.adjustments).map(([k,v]) => k+'='+v.toFixed(2)).join(', ') + '</span>' : ''}
                </div>
            </div>`;
        }
        document.getElementById('insightsPanel').innerHTML = html;
    }

    function drawRegimeGrid(data) {
        const ai = data.ai;
        if (!ai || !ai.regime_stats) return;

        let html = '';
        const entries = Object.entries(ai.regime_stats).sort((a, b) => b[1].count - a[1].count);
        for (const [name, stats] of entries) {
            const wr = stats.win_rate;
            html += `<div class="regime-cell" style="background:${wrBg(wr)};border:1px solid ${wrColor(wr)}33;">
                <div class="rname">${name.replace('_', ' ')}</div>
                <div class="rwr" style="color:${wrColor(wr)};">${(wr*100).toFixed(0)}%</div>
                <div class="rcount">${stats.count} trades | $${(stats.pnl/stats.count).toFixed(0)}/t</div>
            </div>`;
        }
        document.getElementById('regimeGrid').innerHTML = html || '<p style="color:#5a6e8a;padding:8px;font-size:10px;">Accumulating regime data...</p>';
    }

    function drawSessionGrid(data) {
        const ai = data.ai;
        if (!ai) return;

        let html = '';
        if (ai.session_stats) {
            html += '<div style="font-size:9px;color:#5a6e8a;padding:2px 4px;grid-column:1/-1;">SESSIONS</div>';
            for (const [name, stats] of Object.entries(ai.session_stats)) {
                const wr = stats.win_rate;
                html += `<div class="regime-cell" style="background:${wrBg(wr)};border:1px solid ${wrColor(wr)}33;">
                    <div class="rname">${name}</div>
                    <div class="rwr" style="color:${wrColor(wr)};">${(wr*100).toFixed(0)}%</div>
                    <div class="rcount">${stats.count}t $${(stats.pnl/stats.count).toFixed(0)}/t</div>
                </div>`;
            }
        }
        if (ai.volatility_stats) {
            html += '<div style="font-size:9px;color:#5a6e8a;padding:2px 4px;grid-column:1/-1;margin-top:8px;">VOLATILITY</div>';
            for (const [name, stats] of Object.entries(ai.volatility_stats)) {
                const wr = stats.win_rate;
                html += `<div class="regime-cell" style="background:${wrBg(wr)};border:1px solid ${wrColor(wr)}33;">
                    <div class="rname">${name}</div>
                    <div class="rwr" style="color:${wrColor(wr)};">${(wr*100).toFixed(0)}%</div>
                    <div class="rcount">${stats.count}t $${(stats.pnl/stats.count).toFixed(0)}/t</div>
                </div>`;
            }
        }
        if (ai.exit_reason_stats) {
            html += '<div style="font-size:9px;color:#5a6e8a;padding:2px 4px;grid-column:1/-1;margin-top:8px;">EXIT REASONS</div>';
            for (const [name, stats] of Object.entries(ai.exit_reason_stats)) {
                const wr = stats.win_rate;
                html += `<div class="regime-cell" style="background:${wrBg(wr)};border:1px solid ${wrColor(wr)}33;">
                    <div class="rname">${name}</div>
                    <div class="rwr" style="color:${wrColor(wr)};">${(wr*100).toFixed(0)}%</div>
                    <div class="rcount">${stats.count}t $${(stats.pnl/stats.count).toFixed(0)}/t</div>
                </div>`;
            }
        }
        document.getElementById('sessionGrid').innerHTML = html || '<p style="color:#5a6e8a;padding:8px;font-size:10px;">Accumulating...</p>';
    }

    function drawCycleHistory(data) {
        if (!data.cycle_history || data.cycle_history.length === 0) return;

        let html = '<table><tr><th>#</th><th>Inst</th><th>TF</th><th>Strategy</th><th>IS Score</th><th>OOS</th><th>P&L</th><th>WR</th><th>Sharpe</th><th>AI</th><th>Fit</th><th>Time</th></tr>';
        const hist = [...data.cycle_history].reverse();
        for (const c of hist) {
            const pnlColor = c.pnl >= 0 ? 'positive' : 'negative';
            const overfitIcon = c.overfit_flag ? '<span class="negative">OVERFIT</span>' : '<span class="positive">OK</span>';
            html += `<tr>
                <td>${c.cycle}</td>
                <td>${(c.instrument || c.data_symbol || '').substring(0, 8)}</td>
                <td>${c.tf}m</td>
                <td>${c.best_strategy}</td>
                <td>${c.score.toFixed(0)}</td>
                <td>${(c.oos_score || 0).toFixed(0)}</td>
                <td class="${pnlColor}">${formatMoney(c.pnl)}</td>
                <td>${(c.win_rate*100).toFixed(1)}%</td>
                <td>${c.sharpe.toFixed(2)}</td>
                <td class="ai-color">${c.ai_insights || 0}</td>
                <td>${overfitIcon}</td>
                <td>${c.duration.toFixed(1)}s</td>
            </tr>`;
        }
        html += '</table>';
        document.getElementById('cycleHistory').innerHTML = html;
    }

    function drawParams(data) {
        if (!data.best_params || Object.keys(data.best_params).length === 0) return;

        const adj = (data.ai && data.ai.param_adjustments) || {};
        const adjCount = Object.keys(adj).length;
        document.getElementById('adjBadge').textContent = adjCount > 0 ? 'AI: ' + adjCount + ' adjustments active' : '';

        let html = '';
        for (const [key, val] of Object.entries(data.best_params)) {
            const displayVal = typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(3)) : val;
            const adjVal = adj[key];
            const adjStr = adjVal ? `<span class="padj">(${adjVal > 0 ? '+' : ''}${adjVal.toFixed(2)})</span>` : '';
            html += `<div class="param-item"><span class="pname">${key}:</span> <span class="pval">${displayVal}</span> ${adjStr}</div>`;
        }
        document.getElementById('paramsGrid').innerHTML = html;
    }

    function drawTradeLog(data) {
        if (!data.trades || data.trades.length === 0) {
            document.getElementById('tradeLog').innerHTML = '<p style="padding:12px;color:#5a6e8a;">No trades yet</p>';
            return;
        }

        let html = '<table><tr><th>#</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Pts</th><th>Exit</th><th>Bars</th><th>MFE</th><th>MAE</th><th>Reason</th></tr>';
        for (const t of data.trades) {
            const dirColor = t.direction === 'LONG' ? 'positive' : 'negative';
            const pnlColor = t.pnl_dollars >= 0 ? 'positive' : 'negative';
            html += `<tr>
                <td>${t.id}</td>
                <td class="${dirColor}">${t.direction}</td>
                <td>${t.entry_price.toFixed(2)}</td>
                <td>${t.exit_price.toFixed(2)}</td>
                <td class="${pnlColor}">${formatMoney(t.pnl_dollars)}</td>
                <td>${t.pnl_points.toFixed(2)}</td>
                <td>${t.exit_reason}</td>
                <td>${t.bars_held}</td>
                <td>${t.mfe}</td>
                <td>${t.mae}</td>
                <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${t.entry_reason}">${t.entry_reason}</td>
            </tr>`;
        }
        html += '</table>';
        document.getElementById('tradeLog').innerHTML = html;
    }

    refresh();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/state')
def api_state():
    global loop
    if loop is None:
        return jsonify({"status": "NOT STARTED", "cycle_count": 0})
    return jsonify(loop.get_state())


@app.route('/api/correlations')
def api_correlations():
    global loop
    if loop is None:
        return jsonify({})
    return jsonify(loop.correlation_engine.get_state())


@app.route('/api/workers')
def api_workers():
    global loop
    if loop is None:
        return jsonify([])
    return jsonify(loop.worker_stats())


@app.route('/api/lake')
def api_lake():
    global loop
    if loop is None:
        return jsonify({})
    return jsonify(loop.router.get_lake_stats())


def start_dashboard(backtest_loop, host='0.0.0.0', port=5000):
    """Start the Flask dashboard in a background thread."""
    global loop
    loop = backtest_loop

    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"\n  Dashboard running at http://localhost:{port}")
    return thread
