"""
Web Dashboard for NQ Fractal AI Trading Backtester

Real-time visualization of:
- Data source indicator (REAL vs synthetic)
- Candlestick chart with trade entries/exits
- Equity curve
- AI Brain: insights, conclusions, learning progress
- Regime analysis heatmap
- Strategy performance comparison
- Optimization progress across cycles
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
            min-width: 110px;
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

        /* AI Insights panel */
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
        .insight-card.validated {
            border-left-color: #00d4aa;
        }
        .insight-card.low-conf {
            border-left-color: #5a6e8a;
            opacity: 0.7;
        }
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
        <h1>NQ FRACTAL AI BACKTESTER</h1>
        <span id="dataBadge" class="data-badge data-real">REAL DATA</span>
        <div class="status" id="status">INITIALIZING...</div>
    </div>

    <div class="stats-bar" id="statsBar">
        <div class="stat-card">
            <div class="label">Cycle</div>
            <div class="value neutral" id="statCycle">0</div>
        </div>
        <div class="stat-card">
            <div class="label">Symbol</div>
            <div class="value neutral" id="statSymbol">--</div>
        </div>
        <div class="stat-card">
            <div class="label">Best Strategy</div>
            <div class="value neutral" id="statStrategy">&mdash;</div>
        </div>
        <div class="stat-card">
            <div class="label">Score</div>
            <div class="value neutral" id="statScore">0</div>
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
    </div>

    <div class="grid">
        <div class="chart-container grid-full">
            <h3>NQ Price Action & Trades <span id="chartDataLabel" class="ai-badge">REAL DATA</span></h3>
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

            // Data source indicator
            const dm = data.data_meta || {};
            const isReal = dm.is_real !== false;
            const badge = document.getElementById('dataBadge');
            badge.textContent = isReal ? 'REAL DATA' : 'SYNTHETIC';
            badge.className = 'data-badge ' + (isReal ? 'data-real' : 'data-synthetic');

            const sym = dm.symbol || data.data_stats && 'NQ' || '--';
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

            // Only redraw charts if cycle changed
            if (data.cycle_count !== lastCycle) {
                lastCycle = data.cycle_count;
                drawPriceChart(data);
                drawEquityChart(data);
                drawProgressChart(data);
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
            name: 'NQ',
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
            yaxis2: {domain: [0.18, 1], gridcolor: '#1a2744', title: 'NQ Price'},
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
            marker: {size: 4}, name: 'Best Score', yaxis: 'y2',
        }, {
            type: 'scatter', x: x, y: hist.map(h => h.pnl),
            mode: 'lines', line: {color: '#00d4aa', width: 1},
            name: 'P&L', yaxis: 'y',
        }, {
            type: 'scatter', x: x,
            y: hist.map(h => (h.ai_insights || 0)),
            mode: 'markers', marker: {color: '#a78bfa', size: 6, symbol: 'star'},
            name: 'AI Insights', yaxis: 'y2',
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
            const cardClass = ins.validated ? 'validated' : (ins.confidence < 0.5 ? 'low-conf' : '');
            html += `<div class="insight-card ${cardClass}">
                <div class="insight-category">${ins.category} ${ins.validated ? '&#10003; VALIDATED' : ''}</div>
                <div class="insight-text">${ins.conclusion}</div>
                <div class="insight-meta">
                    <span>Confidence: ${(ins.confidence*100).toFixed(0)}% <span class="conf-bar" style="width:${confPx}px;"></span></span>
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
        // Sessions
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
        // Volatility
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
        // Exit reasons
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

        let html = '<table><tr><th>#</th><th>TF</th><th>Data</th><th>Strategy</th><th>Score</th><th>P&L</th><th>WR</th><th>Sharpe</th><th>AI</th><th>Time</th></tr>';
        const hist = [...data.cycle_history].reverse();
        for (const c of hist) {
            const pnlColor = c.pnl >= 0 ? 'positive' : 'negative';
            const dataIcon = (c.data_source || '').includes('SYNTHETIC') ? '<span class="negative">SYN</span>' : '<span class="positive">REAL</span>';
            html += `<tr>
                <td>${c.cycle}</td>
                <td>${c.tf}m</td>
                <td>${dataIcon}</td>
                <td>${c.best_strategy}</td>
                <td>${c.score.toFixed(0)}</td>
                <td class="${pnlColor}">${formatMoney(c.pnl)}</td>
                <td>${(c.win_rate*100).toFixed(1)}%</td>
                <td>${c.sharpe.toFixed(2)}</td>
                <td class="ai-color">${c.ai_insights || 0}</td>
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
