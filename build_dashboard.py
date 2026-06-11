"""
Build a self-contained dashboard.html that backtests the QQQ 200-SMA strategy
on TQQQ vs. buy-and-hold QQQ / SPY.

Strategy (daily, signal evaluated at today's close):
  * Independent SMA windows for buy and sell signals (100 / 150 / 200 day).
  * Hysteresis with independent buy / sell buffers (each 0% disables that side):
      - If currently in cash AND qqq > buySMA*(1+B_buy)    -> buy TQQQ.
      - If currently in TQQQ AND qqq < sellSMA*(1-B_sell)  -> sell to cash.
      - Otherwise (inside the dead zone)                   -> hold current position.
  * Optional trailing-drawdown stop (Rule 1) — AND-combined with the SMA rule:
      - Tracks the rolling N-day high of QQQ close (N configurable, default 22).
      - "Stop tripped" when qqq <= peak_N * (1 - drop).
      - To be long: SMA rule says "in" AND stop is not tripped.
      - Re-entry from cash requires BOTH conditions to flip back to OK.
        (Option 1: no mitigation — fast exits, potential V-shape whipsaw.)
  * Optional vol-spike exit (Rule 2) — also AND-combined as a filter:
      - Computes W-day annualized realized volatility of QQQ log returns:
          sigma_year = stdev(log_returns[t-W+1..t]) * sqrt(252).
      - "Vol tripped" when sigma_year > threshold.
      - Same hold/exit/re-entry semantics as Rule 1: filter can force out or
        block re-entry, never force in. Pure threshold version (option 1) —
        no direction confirmation, so upward melt-up vol can also trigger.
  * On the first simulated day, the bare SMA test (qqq > sma) — combined with
    the stop check if enabled — sets the initial position.
  * Monthly contribution lands as cash and is then placed by the same
    signal-driven rebalance. A frequency slider splits it across the month:
    1x deposits everything on the first trading day; 2x deposits half on the
    first trading day and half on the trading day closest to the 15th
    (weekends/holidays roll to the nearest session).

All computation runs in the browser so sliders are interactive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def load(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def main() -> None:
    qqq = load(DATA_DIR / "synthetic-qqq.tsv").rename("qqq")
    tqqq = load(DATA_DIR / "synthetic-tqqq.tsv").rename("tqqq")
    spy = load(DATA_DIR / "spy.tsv").rename("spy")

    df = pd.concat([qqq, tqqq, spy], axis=1).dropna(how="all").sort_index()
    df = df.ffill().dropna()

    records = [
        {
            "d": idx.strftime("%Y-%m-%d"),
            "q": float(row.qqq),
            "t": float(row.tqqq),
            "s": float(row.spy),
        }
        for idx, row in df.iterrows()
    ]

    payload = json.dumps(records, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__DATA__", payload)
    out = ROOT / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(records):,} rows, {records[0]['d']} -> {records[-1]['d']})")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>QQQ 200-SMA Strategy Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/nouislider@15.7.1/dist/nouislider.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/nouislider@15.7.1/dist/nouislider.min.css" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:           #0a0c10;
    --bg-elev:      #11141b;
    --bg-elev-2:    #161a23;
    --bg-input:     #0d1017;
    --border:       #1c2230;
    --border-strong:#2a3243;
    --text:         #e9edf4;
    --text-dim:     #97a1b3;
    --text-faint:   #5e6878;
    --accent:       #6cc4ff;
    --accent-2:     #2196f3;
    --strat:        #6cc4ff;
    --tqqq:         #ff6b6b;
    --qqq:          #c7cdda;
    --spy:          #f5c451;
    --dip:          #a78bfa;
    --good:         #34d399;
    --bad:          #ef4444;
    --warn:         #fbbf24;
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(33,150,243,0.06), transparent 60%),
      radial-gradient(800px 400px at -10% 10%, rgba(108,196,255,0.04), transparent 60%),
      var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    font-size: 14px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .num { font-family: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  .container { max-width: 1520px; margin: 0 auto; padding: 28px 28px 56px; }

  /* ── Header ───────────────────────────────────────── */
  header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 26px; gap: 16px; flex-wrap: wrap; }
  header h1 { margin: 0 0 6px 0; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }
  header h1 .accent { background: linear-gradient(90deg, #6cc4ff, #2196f3); -webkit-background-clip: text; background-clip: text; color: transparent; }
  header .sub { color: var(--text-dim); font-size: 13px; max-width: 720px; }
  header .meta { color: var(--text-faint); font-size: 12px; text-align: right; }
  header .meta .badge { display: inline-block; padding: 4px 9px; border-radius: 999px; background: var(--bg-elev); border: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }

  /* ── KPI cards ────────────────────────────────────── */
  .kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 22px; }
  @media (max-width: 1400px) { .kpis { grid-template-columns: repeat(3, 1fr); } }
  .kpi {
    background: linear-gradient(180deg, var(--bg-elev) 0%, var(--bg-elev-2) 100%);
    border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px;
    position: relative; overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .kpi:hover { transform: translateY(-1px); border-color: var(--border-strong); }
  .kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--c, var(--accent)); }
  .kpi .kpi-name { display: flex; align-items: center; gap: 8px; color: var(--text-dim); font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
  .kpi .kpi-name .sw { width: 8px; height: 8px; border-radius: 2px; background: var(--c, var(--accent)); }
  .kpi .kpi-final { margin-top: 10px; font-size: 26px; font-weight: 600; letter-spacing: -0.02em; }
  .kpi .kpi-sub { margin-top: 6px; color: var(--text-faint); font-size: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .kpi .delta-pos { color: var(--good); }
  .kpi .delta-neg { color: var(--bad); }
  .kpi.is-strat { background: linear-gradient(180deg, rgba(108,196,255,0.07) 0%, var(--bg-elev-2) 100%); }

  /* ── Grid ─────────────────────────────────────────── */
  .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }
  @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } .kpis { grid-template-columns: repeat(2, 1fr); } }

  .panel { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }

  /* Section header inside a panel */
  .section { padding-bottom: 18px; margin-bottom: 18px; border-bottom: 1px solid var(--border); }
  .section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
  .section-title { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
  .section-title h2 { margin: 0; font-size: 10.5px; font-weight: 700; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.12em; }
  .section-title .info { font-size: 11.5px; color: var(--text-faint); }

  .ctrl { margin-bottom: 16px; }
  .ctrl:last-child { margin-bottom: 0; }
  .ctrl label.lbl { display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 10px; color: var(--text-dim); align-items: baseline; }
  .ctrl label.lbl .val { color: var(--text); font-weight: 600; font-size: 14px; }

  /* noUiSlider theme */
  .noUi-target { background: var(--border); border: none; box-shadow: none; height: 5px; }
  .noUi-connect { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
  .noUi-handle { background: #fff; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,0,0,0.5); border-radius: 50%; width: 16px !important; height: 16px !important; right: -8px !important; top: -6px !important; cursor: grab; }
  .noUi-handle::before, .noUi-handle::after { display: none; }
  .noUi-handle:focus, .noUi-handle.noUi-active { box-shadow: 0 0 0 5px rgba(108,196,255,0.18), 0 1px 3px rgba(0,0,0,0.6); cursor: grabbing; }

  .row { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
  .row input[type="number"] { background: var(--bg-input); color: var(--text); border: 1px solid var(--border-strong); border-radius: 8px; padding: 7px 10px; width: 100%; font: inherit; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
  .row input[type="number"]:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(108,196,255,0.15); }
  .row .prefix { color: var(--text-faint); font-size: 13px; padding: 0 2px; }

  /* Preset chips */
  .chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  .chip { background: var(--bg-input); border: 1px solid var(--border-strong); color: var(--text-dim); padding: 4px 10px; border-radius: 999px; font-size: 11.5px; font-family: 'JetBrains Mono', monospace; cursor: pointer; transition: all 0.12s ease; user-select: none; }
  .chip:hover { color: var(--text); border-color: var(--accent); }
  .chip.active { background: rgba(108,196,255,0.12); border-color: var(--accent); color: var(--accent); }

  /* Toggles */
  .toggle { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; font-size: 12.5px; }
  .toggle label { display: flex; align-items: center; gap: 7px; cursor: pointer; padding: 6px 8px; border-radius: 6px; transition: background 0.12s; }
  .toggle label:hover { background: var(--bg-elev-2); }
  .toggle .sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex-shrink: 0; }
  .toggle input[type="checkbox"] { accent-color: var(--accent); margin: 0; }

  /* Cash flow box */
  .stats-box { background: var(--bg-input); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; margin-top: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 10px; font-size: 12px; }
  .stats-box .k { color: var(--text-faint); }
  .stats-box .v { text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text); }

  /* Summary table */
  table.summary { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.summary th, table.summary td { padding: 9px 6px; border-bottom: 1px solid var(--border); text-align: right; font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
  table.summary th:first-child, table.summary td:first-child { text-align: left; font-family: 'Inter', sans-serif; }
  table.summary thead th { color: var(--text-faint); font-weight: 600; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; padding-bottom: 8px; }
  table.summary tbody td .sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block; margin-right: 8px; vertical-align: middle; }
  table.summary tbody tr:last-child td { border-bottom: none; }
  table.summary tbody tr.strat-row td { background: rgba(108,196,255,0.03); }
  table.summary tbody tr.strat-row td:first-child { font-weight: 600; }

  .footnote { color: var(--text-faint); font-size: 11.5px; margin-top: 14px; line-height: 1.6; }
  .footnote code { background: var(--bg-input); padding: 1px 5px; border-radius: 3px; font-size: 10.5px; }
  .footnote strong { color: var(--text-dim); }

  /* Charts */
  .chart-wrap { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 14px; padding: 14px 8px 8px 8px; }
  .chart-wrap + .chart-wrap { margin-top: 14px; }
  .chart-title { display: flex; justify-content: space-between; align-items: baseline; padding: 0 12px 8px 12px; gap: 10px; flex-wrap: wrap; }
  .chart-title h3 { margin: 0; font-size: 11.5px; font-weight: 700; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.1em; }
  .chart-title .hint { color: var(--text-dim); font-size: 11.5px; font-family: 'JetBrains Mono', monospace; }
  .chart-title .hint b { color: var(--text); font-weight: 600; }

  /* Tooltip / info dot */
  .info-dot { display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; background: var(--bg-input); border: 1px solid var(--border-strong); color: var(--text-faint); font-size: 9px; font-weight: 700; cursor: help; margin-left: 6px; }
  .info-dot:hover { color: var(--accent); border-color: var(--accent); }

  /* Section-level on/off toggle (rule toggle) */
  .rule-toggle { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }
  .rule-toggle input[type="checkbox"] { position: absolute; opacity: 0; pointer-events: none; }
  .rule-toggle-track { position: relative; width: 32px; height: 18px; background: var(--border-strong); border-radius: 999px; transition: background 0.15s ease; }
  .rule-toggle-thumb { position: absolute; left: 2px; top: 2px; width: 14px; height: 14px; background: var(--text-dim); border-radius: 50%; transition: transform 0.15s ease, background 0.15s ease; }
  .rule-toggle input:checked + .rule-toggle-track { background: rgba(108,196,255,0.35); }
  .rule-toggle input:checked + .rule-toggle-track .rule-toggle-thumb { transform: translateX(14px); background: var(--accent); }
  .rule-toggle-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-faint); }
  .rule-toggle input:checked ~ .rule-toggle-label { color: var(--accent); }

  /* When a rule's toggle is off, dim its controls */
  .section.rule-disabled .ctrl { opacity: 0.45; pointer-events: none; }
  .section.rule-disabled .ctrl .lbl,
  .section.rule-disabled .ctrl .chip,
  .section.rule-disabled .ctrl .val { filter: grayscale(0.5); }
</style>
</head>
<body>
<div class="container">

<header>
  <div>
    <h1><span class="accent">QQQ 200-SMA</span> Strategy Dashboard</h1>
    <div class="sub">TQQQ when QQQ trades above its 200-day SMA, cash otherwise — with optional buffer to ride out sideways chop. Daily evaluation, monthly contributions follow the same signal.</div>
  </div>
  <div class="meta">
    <div id="dataMeta" class="badge"></div>
    <div id="currentSignal" style="margin-top:8px"></div>
  </div>
</header>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="panel">

    <!-- Time period -->
    <div class="section">
      <div class="section-title"><h2>Time Period</h2><span class="info" id="rangeLabel"></span></div>
      <div class="ctrl">
        <div id="rangeSlider"></div>
        <div class="chips" id="rangePresets"></div>
      </div>
    </div>

    <!-- Capital -->
    <div class="section">
      <div class="section-title"><h2>Capital</h2></div>
      <div class="ctrl">
        <label class="lbl">Initial investment <span class="val num" id="initLabel"></span></label>
        <div id="initSlider"></div>
        <div class="row"><span class="prefix">$</span><input type="number" id="initInput" min="0" step="500" /></div>
      </div>
      <div class="ctrl">
        <label class="lbl">Monthly contribution <span class="val num" id="contribLabel"></span></label>
        <div id="contribSlider"></div>
        <div class="row"><span class="prefix">$</span><input type="number" id="contribInput" min="0" step="50" /></div>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Deposits per month
            <span class="info-dot" title="How the monthly contribution is spread across the month. 1× = all of it on the first trading day. 2× = half on the first trading day, half on the trading day closest to the 15th (rolls to the nearest session over weekends/holidays).">i</span>
          </span>
          <span class="val num" id="contribFreqLabel"></span>
        </label>
        <div id="contribFreqSlider"></div>
        <div class="chips" data-target="contribFreq">
          <span class="chip" data-freq="1">1× · 1st</span>
          <span class="chip" data-freq="2">2× · 1st &amp; 15th</span>
        </div>
      </div>
    </div>

    <!-- Strategy -->
    <div class="section">
      <div class="section-title"><h2>Strategy</h2></div>

      <!-- Buy side -->
      <div class="ctrl">
        <label class="lbl">
          <span>Buy signal SMA
            <span class="info-dot" title="Which SMA window to use for the re-entry signal. Shorter = more responsive (earlier re-entry).">i</span>
          </span>
          <span class="val num" id="buySMALabel"></span>
        </label>
        <div class="chips" data-target="buySMA">
          <span class="chip" data-sma="100">100D</span>
          <span class="chip" data-sma="150">150D</span>
          <span class="chip" data-sma="200">200D</span>
          <span class="chip" data-sma="250">250D</span>
        </div>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Buy buffer
            <span class="info-dot" title="When in cash: re-enter TQQQ only when QQQ &gt; buy-SMA × (1 + buy buffer). Higher = later re-entry, fewer false starts.">i</span>
          </span>
          <span class="val num" id="buyBufferLabel"></span>
        </label>
        <div id="buyBufferSlider"></div>
        <div class="chips" data-target="buy">
          <span class="chip" data-buffer="0">0%</span>
          <span class="chip" data-buffer="2">2%</span>
          <span class="chip" data-buffer="3">3%</span>
          <span class="chip" data-buffer="5">5%</span>
          <span class="chip" data-buffer="7">7%</span>
          <span class="chip" data-buffer="10">10%</span>
        </div>
      </div>

      <!-- Sell side -->
      <div class="ctrl">
        <label class="lbl">
          <span>Sell signal SMA
            <span class="info-dot" title="Which SMA window to use for the exit signal. Longer = more patient (later exit).">i</span>
          </span>
          <span class="val num" id="sellSMALabel"></span>
        </label>
        <div class="chips" data-target="sellSMA">
          <span class="chip" data-sma="100">100D</span>
          <span class="chip" data-sma="150">150D</span>
          <span class="chip" data-sma="200">200D</span>
          <span class="chip" data-sma="250">250D</span>
        </div>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Sell buffer
            <span class="info-dot" title="When in TQQQ: exit to cash only when QQQ &lt; sell-SMA × (1 − sell buffer). Higher = later exit, rides through small dips.">i</span>
          </span>
          <span class="val num" id="sellBufferLabel"></span>
        </label>
        <div id="sellBufferSlider"></div>
        <div class="chips" data-target="sell">
          <span class="chip" data-buffer="0">0%</span>
          <span class="chip" data-buffer="2">2%</span>
          <span class="chip" data-buffer="3">3%</span>
          <span class="chip" data-buffer="5">5%</span>
          <span class="chip" data-buffer="7">7%</span>
          <span class="chip" data-buffer="10">10%</span>
        </div>
      </div>
    </div>

    <!-- Rule 1: Trailing drawdown stop -->
    <div class="section">
      <div class="section-title">
        <h2>Rule 1 · Trailing Stop</h2>
        <label class="rule-toggle" for="dropEnabled">
          <input type="checkbox" id="dropEnabled" />
          <span class="rule-toggle-track"><span class="rule-toggle-thumb"></span></span>
          <span class="rule-toggle-label" id="dropEnabledLabel">Off</span>
        </label>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Drawdown threshold
            <span class="info-dot" title="Exit TQQQ when QQQ closes more than this % below its rolling high. Lower = more sensitive (more exits).">i</span>
          </span>
          <span class="val num" id="dropPctLabel"></span>
        </label>
        <div id="dropPctSlider"></div>
        <div class="chips" data-target="drop">
          <span class="chip" data-drop="5">5%</span>
          <span class="chip" data-drop="6">6%</span>
          <span class="chip" data-drop="8">8%</span>
          <span class="chip" data-drop="10">10%</span>
          <span class="chip" data-drop="12">12%</span>
          <span class="chip" data-drop="15">15%</span>
        </div>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Lookback window
            <span class="info-dot" title="Rolling N-day high used as the reference peak. Shorter = stop tracks more recent prices; longer = stop tracks an older peak so it's harder to clear after a drop.">i</span>
          </span>
          <span class="val num" id="dropWindowLabel"></span>
        </label>
        <div id="dropWindowSlider"></div>
        <div class="chips" data-target="dropWin">
          <span class="chip" data-window="10">10D</span>
          <span class="chip" data-window="22">22D</span>
          <span class="chip" data-window="44">44D</span>
          <span class="chip" data-window="66">66D</span>
        </div>
      </div>
    </div>

    <!-- Rule 2: Vol-spike exit -->
    <div class="section">
      <div class="section-title">
        <h2>Rule 2 · Vol Spike</h2>
        <label class="rule-toggle" for="volEnabled">
          <input type="checkbox" id="volEnabled" />
          <span class="rule-toggle-track"><span class="rule-toggle-thumb"></span></span>
          <span class="rule-toggle-label" id="volEnabledLabel">Off</span>
        </label>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Vol threshold
            <span class="info-dot" title="Exit TQQQ when QQQ's W-day annualized realized volatility exceeds this. Re-entry blocked until vol drops back below. Note: catches melt-ups too (option 1).">i</span>
          </span>
          <span class="val num" id="volPctLabel"></span>
        </label>
        <div id="volPctSlider"></div>
        <div class="chips" data-target="vol">
          <span class="chip" data-vol="20">20%</span>
          <span class="chip" data-vol="25">25%</span>
          <span class="chip" data-vol="30">30%</span>
          <span class="chip" data-vol="40">40%</span>
          <span class="chip" data-vol="50">50%</span>
        </div>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Lookback window
            <span class="info-dot" title="Number of trading days of returns used to estimate volatility. Shorter = faster reaction (noisier); longer = smoother (slower).">i</span>
          </span>
          <span class="val num" id="volWindowLabel"></span>
        </label>
        <div id="volWindowSlider"></div>
        <div class="chips" data-target="volWin">
          <span class="chip" data-window="5">5D</span>
          <span class="chip" data-window="10">10D</span>
          <span class="chip" data-window="15">15D</span>
          <span class="chip" data-window="20">20D</span>
        </div>
      </div>
      <div class="ctrl">
        <label class="lbl">
          <span>Direction confirm
            <span class="info-dot" title="Only trip the vol filter when QQQ is ALSO below this short SMA. Filters out melt-up false-positives (high vol with rising price). Off = pure threshold (Rule 2 trips on any high-vol day regardless of direction).">i</span>
          </span>
          <span class="val num" id="volDirLabel"></span>
        </label>
        <div class="chips" data-target="volDir">
          <span class="chip" data-vol-dir="0">Off</span>
          <span class="chip" data-vol-dir="15">15D</span>
          <span class="chip" data-vol-dir="20">20D</span>
          <span class="chip" data-vol-dir="25">25D</span>
          <span class="chip" data-vol-dir="50">50D</span>
        </div>
      </div>
    </div>

    <!-- View -->
    <div class="section">
      <div class="section-title"><h2>Display</h2></div>
      <div class="toggle">
        <label><input type="checkbox" id="showStrat" checked /> <span class="sw" style="background:var(--strat)"></span> Strategy</label>
        <label><input type="checkbox" id="showTqqq" checked /> <span class="sw" style="background:var(--tqqq)"></span> B&amp;H TQQQ</label>
        <label><input type="checkbox" id="showQqq" checked /> <span class="sw" style="background:var(--qqq)"></span> B&amp;H QQQ</label>
        <label><input type="checkbox" id="showSpy" checked /> <span class="sw" style="background:var(--spy)"></span> B&amp;H SPY</label>
        <label><input type="checkbox" id="showDip" checked /> <span class="sw" style="background:var(--dip)"></span> Dip-Buy TQQQ</label>
        <label><input type="checkbox" id="showShade" checked /> <span class="sw" style="background:rgba(239,68,68,0.4);border:1px solid var(--bad)"></span> Out-of-market</label>
        <label><input type="checkbox" id="logScale" checked /> <span class="sw" style="background:transparent;border:1px solid var(--text-dim)"></span> Log scale</label>
      </div>
    </div>

    <!-- Cash flow stats -->
    <div class="section">
      <div class="section-title"><h2>Run Summary</h2></div>
      <div class="stats-box">
        <div class="k">Total invested</div><div class="v" id="sInvested">—</div>
        <div class="k">Initial</div><div class="v" id="sInit">—</div>
        <div class="k">Contributions</div><div class="v" id="sContrib">—</div>
        <div class="k">Deposits</div><div class="v" id="sMonths">—</div>
        <div class="k">Strategy trades</div><div class="v" id="sTrades">—</div>
        <div class="k">Time in market</div><div class="v" id="sInMkt">—</div>
        <div class="k">Stop exits</div><div class="v" id="sStopExits">—</div>
        <div class="k">Stop hold-out days</div><div class="v" id="sStopHold">—</div>
        <div class="k">Vol exits</div><div class="v" id="sVolExits">—</div>
        <div class="k">Vol hold-out days</div><div class="v" id="sVolHold">—</div>
        <div class="k">Dip-Buy invested</div><div class="v" id="sDipInv">—</div>
        <div class="k">Dip-Buy events</div><div class="v" id="sDipEvents">—</div>
      </div>
      <table class="summary" id="summary" style="margin-top:14px">
        <thead><tr><th></th><th>Final</th><th>×</th><th>IRR</th><th>Max DD</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="footnote">
      Signal &amp; trades at the day's close (no lookahead — SMA at day <em>t</em> uses prices through day <em>t</em>). Pre-inception data is synthesized: QQQ pre-1999 from <code>^NDX</code>, TQQQ pre-2010 from 3× daily NDX-TR with ~1.84%/yr drag, SPY pre-1993 from <code>^SP500TR</code>/<code>^GSPC</code>. <strong>IRR</strong> is the money-weighted annualized return across all deposits.
    </div>
  </div>

  <div>
    <div class="chart-wrap">
      <div class="chart-title">
        <h3>Portfolio Value</h3>
        <span class="hint" id="portfolioHint"></span>
      </div>
      <div id="chart" style="height: 480px;"></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title"><h3>Drawdown from Peak</h3><span class="hint">linear · lower is worse</span></div>
      <div id="drawdown" style="height: 220px;"></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title"><h3>QQQ vs 200-Day SMA</h3><span class="hint" id="signalHint">red shading = out of market</span></div>
      <div id="signal" style="height: 260px;"></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">
        <h3>Realized Volatility (Rule 2)</h3>
        <span class="hint" id="volHint">annualized · close-to-close</span>
      </div>
      <div id="volChart" style="height: 180px;"></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title"><h3>TQQQ Price</h3><span class="hint">3× leveraged · same range · red shading = strategy out of market</span></div>
      <div id="tqqqChart" style="height: 260px;"></div>
    </div>
  </div>
</div>

</div>

<script>
const RAW = __DATA__;

const N = RAW.length;
const dates = new Array(N);
const qqq = new Float64Array(N);
const tqqq = new Float64Array(N);
const spy = new Float64Array(N);
for (let i = 0; i < N; i++) {
  dates[i] = RAW[i].d;
  qqq[i] = RAW[i].q;
  tqqq[i] = RAW[i].t;
  spy[i] = RAW[i].s;
}

// Pre-compute daily log returns of QQQ (used by Rule 2 — realized vol).
const logRet = new Float64Array(N);
logRet[0] = NaN;
for (let i = 1; i < N; i++) logRet[i] = Math.log(qqq[i] / qqq[i-1]);
const SQRT_252 = Math.sqrt(252);

// Pre-compute SMAs of QQQ for each supported window (today-inclusive rolling mean).
// Short windows (15/20/25/50) are used by Rule 2's direction-confirmation filter.
const SMA_WINDOWS = [15, 20, 25, 50, 100, 150, 200, 250];
const smas = {}; // smas[w] -> Float64Array(N)
for (const w of SMA_WINDOWS) {
  const arr = new Float64Array(N);
  let sum = 0;
  for (let i = 0; i < N; i++) {
    sum += qqq[i];
    if (i >= w) sum -= qqq[i - w];
    arr[i] = (i >= w - 1) ? (sum / w) : NaN;
  }
  smas[w] = arr;
}
const MAX_SMA = Math.max.apply(null, SMA_WINDOWS);

// First trading day of each month.
const firstOfMonth = new Uint8Array(N);
firstOfMonth[0] = 1;
for (let i = 1; i < N; i++) firstOfMonth[i] = dates[i].slice(0, 7) !== dates[i-1].slice(0, 7) ? 1 : 0;

// Mid-month deposit day: the trading day closest to the 15th of each month
// (a weekend/holiday 15th rolls to the nearest session, earlier on ties).
const midOfMonth = new Uint8Array(N);
{
  let mStart = 0;
  for (let i = 1; i <= N; i++) {
    if (i === N || firstOfMonth[i]) {
      let best = mStart, bestDist = Infinity;
      for (let j = mStart; j < i; j++) {
        const dist = Math.abs(+dates[j].slice(8, 10) - 15);
        if (dist < bestDist) { bestDist = dist; best = j; }
      }
      midOfMonth[best] = 1;
      mStart = i;
    }
  }
}

// Quarter index.
function dateToQuarter(d) {
  const y = +d.slice(0, 4), m = +d.slice(5, 7);
  const q = Math.floor((m - 1) / 3) + 1;
  return `${y}Q${q}`;
}
const quarters = [];
const quarterStartIdx = [];
{
  let prev = "";
  for (let i = 0; i < N; i++) {
    const q = dateToQuarter(dates[i]);
    if (q !== prev) { quarters.push(q); quarterStartIdx.push(i); prev = q; }
  }
}
// Require enough warmup for the longest SMA window so the user can freely switch.
const minStartQ = Math.max(0, quarters.findIndex((_, qi) => quarterStartIdx[qi] >= MAX_SMA - 1));

// ───────── Simulation with buffered hysteresis + optional trailing-stop ─────────
// Position rules:
//   Day startIdx: pos = (qqq > buySMA) AND (stop not tripped, if enabled) ? 1 : 0
//   Day i > start:
//     if pos==1: exit (target=0) when qqq < sellSMA[i]*(1-B_sell)  OR (stop tripped)
//     if pos==0: enter (target=1) when qqq > buySMA[i] *(1+B_buy) AND (stop not tripped)
//     else hold pos
//
// Trailing stop: stop_tripped when qqq[i] <= peak_N * (1 - dropPct),
//   where peak_N = max(qqq[i-N+1 .. i]) (today-inclusive rolling high).
// SMA undefined (early warmup) -> forced cash.
function simulate(startIdx, endIdx, initialCash, monthlyContrib, contribTimes,
                  buyBuffer, sellBuffer, buySMAWin, sellSMAWin,
                  dropEnabled, dropPct, dropWindow,
                  volEnabled, volPct, volWindow, volDirSMAWin) {
  const buySmaArr  = smas[buySMAWin];
  const sellSmaArr = smas[sellSMAWin];
  // Direction-confirmation SMA for Rule 2 (null when off / unused).
  const volDirSmaArr = (volDirSMAWin && smas[volDirSMAWin]) ? smas[volDirSMAWin] : null;
  const len = endIdx - startIdx + 1;
  const stratVal = new Float64Array(len);
  const tqqqBH = new Float64Array(len);
  const qqqBH = new Float64Array(len);
  const spyBH = new Float64Array(len);
  const posOut = new Uint8Array(len);   // for shading
  const stopLevel = new Float64Array(len); // trailing stop line for the chart (NaN when disabled)
  const volSeries = new Float64Array(len); // annualized realized vol per day (always computed, for chart)
  const dipBuyVal = new Float64Array(len); // "Dip-Buy TQQQ" strategy mark-to-market

  // Dip-Buy strategy state (independent of main strategy).
  // Rules: buy TQQQ only when QQQ < 200-SMA. Never sell. Skipped buys are forfeited
  // (cash is NOT held and NOT accumulated for future deployment).
  const sma200 = smas[200];
  let dipShares = 0;
  const dipCfIdx = [];
  const dipCfAmt = [];

  let cash = initialCash;
  let shares = 0;
  let pos = 0;
  let tShares = initialCash / tqqq[startIdx];
  let qShares = initialCash / qqq[startIdx];
  let sShares = initialCash / spy[startIdx];

  const cfIdx = [0];
  const cfAmt = [initialCash];
  let daysInMkt = 0;
  let trades = 0;
  let stopExits = 0;       // exits where the trailing stop was the binding constraint
  let stopHoldDays = 0;    // days where stop kept us out (would have been in by SMA alone)
  let volExits = 0;        // exits where the vol-spike was the binding constraint
  let volHoldDays = 0;     // days where vol kept us out (would have been in by SMA alone)

  for (let i = startIdx, k = 0; i <= endIdx; i++, k++) {
    // 1. Contribution lands as cash (skip startIdx — counted as initial deposit).
    //    1x/month: full amount on the first trading day.
    //    2x/month: half on the first trading day, half on the day nearest the 15th.
    const depositDay = firstOfMonth[i] || (contribTimes === 2 && midOfMonth[i]);
    if (depositDay && i !== startIdx && monthlyContrib > 0) {
      const amt = monthlyContrib / contribTimes;
      cash += amt;
      tShares += amt / tqqq[i];
      qShares += amt / qqq[i];
      sShares += amt / spy[i];
      cfIdx.push(i - startIdx);
      cfAmt.push(amt);
      // Dip-Buy strategy: contribution only deploys if QQQ < 200-SMA.
      // Otherwise skipped — money is NOT held as cash, just forfeited.
      const s200 = sma200[i];
      if (!isNaN(s200) && qqq[i] < s200) {
        dipShares += amt / tqqq[i];
        dipCfIdx.push(i - startIdx);
        dipCfAmt.push(amt);
      }
    }

    // Day-1 deploy for Dip-Buy: initial cash only invested if QQQ < 200-SMA on day 1.
    if (i === startIdx && initialCash > 0) {
      const s200 = sma200[i];
      if (!isNaN(s200) && qqq[i] < s200) {
        dipShares += initialCash / tqqq[i];
        dipCfIdx.push(0);
        dipCfAmt.push(initialCash);
      }
    }

    // 2a. Trailing-stop state (computed every day, used only when dropEnabled).
    let stopTripped = false;
    let curStopLvl = NaN;
    if (dropEnabled) {
      // Today-inclusive rolling N-day high. N is small (≤ ~120), naive scan is fine.
      const from = Math.max(0, i - dropWindow + 1);
      let peak = qqq[from];
      for (let j = from + 1; j <= i; j++) if (qqq[j] > peak) peak = qqq[j];
      curStopLvl = peak * (1 - dropPct);
      stopTripped = qqq[i] <= curStopLvl;
    }
    stopLevel[k] = curStopLvl;

    // 2b. Realized vol (computed every day for the chart; filter active only when volEnabled).
    //     Sample stdev (N-1 denominator) of log returns over [i-volWindow+1, i], annualized.
    let curVolAnn = NaN;
    let volTripped = false;
    {
      const from = Math.max(1, i - volWindow + 1);  // logRet[0] is NaN, skip
      const count = i - from + 1;
      if (count >= 2) {
        let sum = 0;
        for (let j = from; j <= i; j++) sum += logRet[j];
        const mean = sum / count;
        let sse = 0;
        for (let j = from; j <= i; j++) { const d = logRet[j] - mean; sse += d * d; }
        const sigDaily = Math.sqrt(sse / (count - 1));
        curVolAnn = sigDaily * SQRT_252;
        if (volEnabled && curVolAnn > volPct) {
          if (volDirSmaArr === null) {
            // No direction check — pure threshold.
            volTripped = true;
          } else {
            // Direction-confirmed: only trip when QQQ is below the short SMA too.
            const dirSma = volDirSmaArr[i];
            if (!isNaN(dirSma) && qqq[i] < dirSma) volTripped = true;
            // else: vol is high but price is at/above the short SMA → likely melt-up; don't trip.
          }
        }
      }
    }
    volSeries[k] = curVolAnn;

    // 3. Resolve today's target position. SMA hysteresis AND-combined with all
    //    active filters (Rule 1 stop, Rule 2 vol). Filters can force OUT or block
    //    re-entry; they never force IN.
    let target;
    let smaWouldBeIn;  // what the SMA-only rule would say for this position state
    const buySma  = buySmaArr[i];
    const sellSma = sellSmaArr[i];
    if (isNaN(buySma) || isNaN(sellSma)) {
      target = 0;
      smaWouldBeIn = false;
    } else if (i === startIdx) {
      smaWouldBeIn = (qqq[i] > buySma);             // bare buy-SMA test on day 1
      target = (smaWouldBeIn && !stopTripped && !volTripped) ? 1 : 0;
    } else {
      const upper = buySma  * (1 + buyBuffer);      // re-entry threshold
      const lower = sellSma * (1 - sellBuffer);     // exit threshold
      // First compute the SMA-only verdict (hysteresis as before).
      let smaTarget;
      if (pos === 0 && qqq[i] > upper) smaTarget = 1;
      else if (pos === 1 && qqq[i] < lower) smaTarget = 0;
      else smaTarget = pos;
      smaWouldBeIn = smaTarget === 1;
      // Overlay each active filter: any single trip forces cash.
      target = (smaWouldBeIn && !stopTripped && !volTripped) ? 1 : 0;
    }

    // Attribution counters — Rule 1 (stop).
    // "Stop exit": stop tripped, SMA still in, was in TQQQ -> stop was the binding reason.
    // (When multiple filters trip simultaneously, attribute to whichever is the relevant rule.)
    if (dropEnabled && pos === 1 && target === 0 && stopTripped && smaWouldBeIn) stopExits++;
    if (dropEnabled && pos === 0 && target === 0 && smaWouldBeIn && stopTripped) stopHoldDays++;
    // Attribution counters — Rule 2 (vol).
    if (volEnabled && pos === 1 && target === 0 && volTripped && smaWouldBeIn) volExits++;
    if (volEnabled && pos === 0 && target === 0 && smaWouldBeIn && volTripped) volHoldDays++;

    if (target !== pos) trades++;
    pos = target;

    // 3. Single rebalance to target.
    if (pos === 1) {
      if (cash > 0) { shares += cash / tqqq[i]; cash = 0; }
    } else {
      if (shares > 0) { cash += shares * tqqq[i]; shares = 0; }
    }
    if (pos === 1) daysInMkt++;

    stratVal[k] = cash + shares * tqqq[i];
    tqqqBH[k]   = tShares * tqqq[i];
    qqqBH[k]    = qShares * qqq[i];
    spyBH[k]    = sShares * spy[i];
    dipBuyVal[k]= dipShares * tqqq[i];
    posOut[k]   = pos;
  }

  return { stratVal, tqqqBH, qqqBH, spyBH, dipBuyVal,
           cfIdx, cfAmt, dipCfIdx, dipCfAmt,
           daysInMkt, trades, posOut,
           stopLevel, stopExits, stopHoldDays,
           volSeries, volExits, volHoldDays, len };
}

function maxDrawdown(arr) {
  let peak = -Infinity, mdd = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > peak) peak = arr[i];
    const dd = (arr[i] - peak) / peak;
    if (dd < mdd) mdd = dd;
  }
  return mdd;
}
function drawdownSeries(arr) {
  const out = new Float64Array(arr.length);
  let peak = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > peak) peak = arr[i];
    out[i] = (arr[i] - peak) / peak;
  }
  return out;
}

// Money-weighted return (IRR) by bisection.
function irr(cfIdx, cfAmt, finalIdx, finalAmt) {
  if (finalAmt <= 0) return NaN;
  const yrs = (t) => t / 252.0;
  function f(r) {
    if (r <= -0.999) return Infinity;
    let pv = 0;
    for (let i = 0; i < cfIdx.length; i++) pv += cfAmt[i] / Math.pow(1 + r, yrs(cfIdx[i]));
    return pv - finalAmt / Math.pow(1 + r, yrs(finalIdx));
  }
  let lo = -0.95, hi = 10.0;
  let flo = f(lo), fhi = f(hi);
  if (!isFinite(flo) || !isFinite(fhi) || flo * fhi > 0) return NaN;
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    const fm = f(mid);
    if (Math.abs(fm) < 1e-8 || (hi - lo) < 1e-7) return mid;
    if (flo * fm < 0) { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
  }
  return 0.5 * (lo + hi);
}

// ───────── Formatting ─────────
function fmtMoney(v) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return "$" + (v/1e9).toFixed(2) + "B";
  if (a >= 1e6) return "$" + (v/1e6).toFixed(2) + "M";
  if (a >= 1e3) return "$" + (v/1e3).toFixed(1) + "k";
  return "$" + v.toFixed(0);
}
function fmtMoneyFull(v) { return isFinite(v) ? "$" + Math.round(v).toLocaleString("en-US") : "—"; }
function fmtPct(v, d=2) { return isFinite(v) ? (v * 100).toFixed(d) + "%" : "—"; }
function fmtX(v) { return isFinite(v) ? v.toFixed(2) + "×" : "—"; }

// ───────── UI ─────────
const $ = (id) => document.getElementById(id);

const rangeSlider = $("rangeSlider");
const initSlider = $("initSlider");
const contribSlider = $("contribSlider");
const contribFreqSlider = $("contribFreqSlider");
const buyBufferSlider = $("buyBufferSlider");
const sellBufferSlider = $("sellBufferSlider");

document.getElementById("dataMeta").textContent =
  `${dates[0]} → ${dates[N-1]}  ·  ${N.toLocaleString()} days`;

const defaultStart = Math.max(minStartQ, quarters.indexOf("2010Q1"));
noUiSlider.create(rangeSlider, {
  start: [defaultStart >= 0 ? defaultStart : minStartQ, quarters.length - 1],
  connect: true, step: 1,
  range: { min: minStartQ, max: quarters.length - 1 },
});
noUiSlider.create(initSlider, { start: [10000], connect: "lower", range: { min: 0, max: 250000 }, step: 500 });
noUiSlider.create(contribSlider, { start: [500], connect: "lower", range: { min: 0, max: 10000 }, step: 50 });
noUiSlider.create(contribFreqSlider, { start: [1], connect: "lower", range: { min: 1, max: 2 }, step: 1 });
noUiSlider.create(buyBufferSlider,  { start: [0], connect: "lower", range: { min: 0, max: 15 }, step: 0.25 });
noUiSlider.create(sellBufferSlider, { start: [0], connect: "lower", range: { min: 0, max: 15 }, step: 0.25 });

// Rule 1 sliders
const dropPctSlider    = $("dropPctSlider");
const dropWindowSlider = $("dropWindowSlider");
noUiSlider.create(dropPctSlider,    { start: [8],  connect: "lower", range: { min: 2,  max: 25  }, step: 0.5 });
noUiSlider.create(dropWindowSlider, { start: [22], connect: "lower", range: { min: 5,  max: 120 }, step: 1 });

// Rule 2 sliders
const volPctSlider    = $("volPctSlider");
const volWindowSlider = $("volWindowSlider");
noUiSlider.create(volPctSlider,    { start: [30], connect: "lower", range: { min: 15, max: 60 }, step: 1 });
noUiSlider.create(volWindowSlider, { start: [10], connect: "lower", range: { min: 5,  max: 30 }, step: 1 });

let state = {
  startQ: defaultStart, endQ: quarters.length - 1,
  initial: 10000, contrib: 500, contribTimes: 1,
  buyBuffer: 0, sellBuffer: 0,
  buySMA: 200, sellSMA: 200,
  // Rule 1: trailing-drawdown stop
  dropEnabled: false, dropPct: 8, dropWindow: 22,
  // Rule 2: vol-spike exit
  // volDirSMA: 0 = pure threshold (no direction check); 15/20/25/50 = require qqq < that SMA to trip.
  volEnabled: false, volPct: 30, volWindow: 10, volDirSMA: 0,
};
$("initInput").value = state.initial;
$("contribInput").value = state.contrib;

// Range presets — populate after we know quarters.
const presetDefs = [
  ["5Y", -5*4],
  ["10Y", -10*4],
  ["20Y", -20*4],
  ["Since 2000", quarters.indexOf("2000Q1")],
  ["Since 2010", quarters.indexOf("2010Q1")],
  ["All", "all"],
];
{
  const wrap = $("rangePresets");
  for (const [label, target] of presetDefs) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = label;
    chip.addEventListener("click", () => {
      let s;
      if (target === "all") s = minStartQ;
      else if (target < 0) s = Math.max(minStartQ, quarters.length - 1 + target);
      else if (target < 0 || isNaN(target)) return;
      else s = Math.max(minStartQ, target);
      rangeSlider.noUiSlider.set([s, quarters.length - 1]);
    });
    if (target !== "all" && (isNaN(target) || target < -quarters.length)) chip.style.display = "none";
    wrap.appendChild(chip);
  }
}

// Buffer preset chips (buy / sell)
for (const chip of document.querySelectorAll('.chips[data-target="buy"] .chip')) {
  chip.addEventListener("click", () => buyBufferSlider.noUiSlider.set(+chip.dataset.buffer));
}
for (const chip of document.querySelectorAll('.chips[data-target="sell"] .chip')) {
  chip.addEventListener("click", () => sellBufferSlider.noUiSlider.set(+chip.dataset.buffer));
}
// Deposit-frequency chips
for (const chip of document.querySelectorAll('.chips[data-target="contribFreq"] .chip')) {
  chip.addEventListener("click", () => contribFreqSlider.noUiSlider.set(+chip.dataset.freq));
}
// SMA selector chips
for (const chip of document.querySelectorAll('.chips[data-target="buySMA"] .chip')) {
  chip.addEventListener("click", () => { state.buySMA = +chip.dataset.sma; update(); });
}
for (const chip of document.querySelectorAll('.chips[data-target="sellSMA"] .chip')) {
  chip.addEventListener("click", () => { state.sellSMA = +chip.dataset.sma; update(); });
}
// Rule 1 chips
for (const chip of document.querySelectorAll('.chips[data-target="drop"] .chip')) {
  chip.addEventListener("click", () => dropPctSlider.noUiSlider.set(+chip.dataset.drop));
}
for (const chip of document.querySelectorAll('.chips[data-target="dropWin"] .chip')) {
  chip.addEventListener("click", () => dropWindowSlider.noUiSlider.set(+chip.dataset.window));
}
// Rule 2 chips
for (const chip of document.querySelectorAll('.chips[data-target="vol"] .chip')) {
  chip.addEventListener("click", () => volPctSlider.noUiSlider.set(+chip.dataset.vol));
}
for (const chip of document.querySelectorAll('.chips[data-target="volWin"] .chip')) {
  chip.addEventListener("click", () => volWindowSlider.noUiSlider.set(+chip.dataset.window));
}
for (const chip of document.querySelectorAll('.chips[data-target="volDir"] .chip')) {
  chip.addEventListener("click", () => { state.volDirSMA = +chip.dataset.volDir; update(); });
}
function updateChips() {
  for (const chip of document.querySelectorAll('.chips[data-target="buy"] .chip')) {
    chip.classList.toggle("active", Math.abs(+chip.dataset.buffer - state.buyBuffer) < 1e-6);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="sell"] .chip')) {
    chip.classList.toggle("active", Math.abs(+chip.dataset.buffer - state.sellBuffer) < 1e-6);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="contribFreq"] .chip')) {
    chip.classList.toggle("active", +chip.dataset.freq === state.contribTimes);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="buySMA"] .chip')) {
    chip.classList.toggle("active", +chip.dataset.sma === state.buySMA);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="sellSMA"] .chip')) {
    chip.classList.toggle("active", +chip.dataset.sma === state.sellSMA);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="drop"] .chip')) {
    chip.classList.toggle("active", Math.abs(+chip.dataset.drop - state.dropPct) < 1e-6);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="dropWin"] .chip')) {
    chip.classList.toggle("active", +chip.dataset.window === state.dropWindow);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="vol"] .chip')) {
    chip.classList.toggle("active", Math.abs(+chip.dataset.vol - state.volPct) < 1e-6);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="volWin"] .chip')) {
    chip.classList.toggle("active", +chip.dataset.window === state.volWindow);
  }
  for (const chip of document.querySelectorAll('.chips[data-target="volDir"] .chip')) {
    chip.classList.toggle("active", +chip.dataset.volDir === state.volDirSMA);
  }
}

const SERIES = {
  strat: { name: "Strategy",      color: "#6cc4ff" },
  tqqq:  { name: "B&H TQQQ",      color: "#ff6b6b" },
  qqq:   { name: "B&H QQQ",       color: "#c7cdda" },
  spy:   { name: "B&H SPY",       color: "#f5c451" },
  dip:   { name: "Dip-Buy TQQQ",  color: "#a78bfa" },
};
const ORDER = ["strat","dip","tqqq","qqq","spy"];

// Build "out of market" shaded regions from the simulation's position track.
function outOfMarketShapes(startIdx, posOut) {
  if (!$("showShade").checked) return [];
  const shapes = [];
  let runStart = -1;
  const len = posOut.length;
  for (let k = 0; k <= len; k++) {
    const out = (k < len) && (posOut[k] === 0);
    if (out && runStart < 0) runStart = k;
    if (!out && runStart >= 0) {
      shapes.push({
        type: "rect", xref: "x", yref: "paper",
        x0: dates[startIdx + runStart], x1: dates[startIdx + k - 1],
        y0: 0, y1: 1,
        fillcolor: "rgba(239,68,68,0.10)",
        line: { width: 0 }, layer: "below",
      });
      runStart = -1;
    }
  }
  return shapes;
}

function update() {
  const startIdx = quarterStartIdx[state.startQ];
  const endQ = state.endQ;
  const endIdx = (endQ + 1 < quarters.length) ? quarterStartIdx[endQ + 1] - 1 : N - 1;
  if (endIdx <= startIdx) return;

  $("rangeLabel").textContent = `${quarters[state.startQ]} → ${quarters[endQ]}`;
  $("initLabel").textContent = fmtMoneyFull(state.initial);
  $("contribLabel").textContent = fmtMoneyFull(state.contrib) + "/mo";
  $("contribFreqLabel").textContent = state.contribTimes === 1
    ? "1× — all on the 1st"
    : `2× — ${fmtMoneyFull(state.contrib / 2)} on 1st & 15th`;
  $("buyBufferLabel").textContent = `+${state.buyBuffer.toFixed(2)}%`;
  $("sellBufferLabel").textContent = `−${state.sellBuffer.toFixed(2)}%`;
  $("buySMALabel").textContent = `${state.buySMA}-day`;
  $("sellSMALabel").textContent = `${state.sellSMA}-day`;
  $("dropPctLabel").textContent = `−${state.dropPct.toFixed(1)}%`;
  $("dropWindowLabel").textContent = `${state.dropWindow}-day`;
  $("dropEnabledLabel").textContent = state.dropEnabled ? "On" : "Off";
  $("volPctLabel").textContent = `${state.volPct}%`;
  $("volWindowLabel").textContent = `${state.volWindow}-day`;
  $("volDirLabel").textContent    = state.volDirSMA === 0 ? "Off" : `${state.volDirSMA}-day`;
  $("volEnabledLabel").textContent = state.volEnabled ? "On" : "Off";
  // Sync the checkbox visual state to state (covers initial render).
  $("dropEnabled").checked = state.dropEnabled;
  $("volEnabled").checked  = state.volEnabled;
  // Dim each rule's sub-controls when disabled.
  const r1Section = $("dropEnabled").closest(".section");
  if (r1Section) r1Section.classList.toggle("rule-disabled", !state.dropEnabled);
  const r2Section = $("volEnabled").closest(".section");
  if (r2Section) r2Section.classList.toggle("rule-disabled", !state.volEnabled);
  updateChips();

  const buyFrac  = state.buyBuffer  / 100;
  const sellFrac = state.sellBuffer / 100;
  const dropFrac = state.dropPct    / 100;
  const volFrac  = state.volPct     / 100;
  const sim = simulate(startIdx, endIdx, state.initial, state.contrib, state.contribTimes,
                       buyFrac, sellFrac, state.buySMA, state.sellSMA,
                       state.dropEnabled, dropFrac, state.dropWindow,
                       state.volEnabled,  volFrac,  state.volWindow, state.volDirSMA);
  const sliceDates = dates.slice(startIdx, endIdx + 1);
  const len = sim.len;

  // Cash flow + run stats.
  const totalContrib = sim.cfAmt.slice(1).reduce((a,b)=>a+b, 0);
  const totalInvested = state.initial + totalContrib;
  const deposits = sim.cfAmt.length - 1;
  $("sInit").textContent = fmtMoneyFull(state.initial);
  $("sContrib").textContent = fmtMoneyFull(totalContrib);
  $("sInvested").textContent = fmtMoneyFull(totalInvested);
  $("sMonths").textContent = deposits.toString();
  $("sTrades").textContent = sim.trades.toString();
  const inMktPct = (sim.daysInMkt / len * 100);
  $("sInMkt").textContent = inMktPct.toFixed(1) + "%";
  $("sStopExits").textContent = state.dropEnabled ? sim.stopExits.toString() : "—";
  $("sStopHold").textContent  = state.dropEnabled ? sim.stopHoldDays.toString() : "—";
  $("sVolExits").textContent  = state.volEnabled ? sim.volExits.toString() : "—";
  $("sVolHold").textContent   = state.volEnabled ? sim.volHoldDays.toString() : "—";
  const dipInvestedAmt = sim.dipCfAmt.reduce((a,b)=>a+b, 0);
  $("sDipInv").textContent    = fmtMoneyFull(dipInvestedAmt);
  $("sDipEvents").textContent = sim.dipCfIdx.length.toString();
  const sameSMA = state.buySMA === state.sellSMA;
  const smaTag  = sameSMA ? `<b>${state.buySMA}D</b> SMA` :
                            `buy <b>${state.buySMA}D</b> · sell <b>${state.sellSMA}D</b>`;
  const ruleParts = [];
  if (state.dropEnabled) ruleParts.push(`stop <b>−${state.dropPct.toFixed(1)}% / ${state.dropWindow}D</b>`);
  if (state.volEnabled) {
    const dirTag = state.volDirSMA === 0 ? "" : ` &amp; &lt;${state.volDirSMA}D-SMA`;
    ruleParts.push(`vol <b>&gt;${state.volPct}% / ${state.volWindow}D${dirTag}</b>`);
  }
  const ruleTag = ruleParts.length ? ` · ${ruleParts.join(" · ")}` : "";
  $("portfolioHint").innerHTML =
    `<b>${sim.trades}</b> trades · <b>${inMktPct.toFixed(0)}%</b> in market · ${smaTag} · buffers <b>+${state.buyBuffer.toFixed(2)}%</b> / <b>−${state.sellBuffer.toFixed(2)}%</b>${ruleTag}`;

  // Current signal badge (latest day in slice).
  const lastIdx = endIdx;
  const refSma = sim.posOut[len-1] === 1 ? smas[state.sellSMA][lastIdx]   // if in TQQQ, sell-SMA matters
                                         : smas[state.buySMA][lastIdx];   // if in cash, buy-SMA matters
  const refWin = sim.posOut[len-1] === 1 ? state.sellSMA : state.buySMA;
  let stateStr;
  if (isNaN(refSma)) stateStr = "warmup";
  else {
    const pct = ((qqq[lastIdx] / refSma) - 1) * 100;
    const inMktNow = sim.posOut[len-1] === 1;
    const tag = inMktNow ? "IN MARKET" : "OUT (CASH)";
    const cls = inMktNow ? "delta-pos" : "delta-neg";
    stateStr = `<span class="badge"><span class="${cls}">●</span> ${tag} · QQQ ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}% vs ${refWin}D-SMA</span>`;
  }
  $("currentSignal").innerHTML = stateStr;

  const useLog = $("logScale").checked;

  // Main chart.
  const series = { strat: sim.stratVal, tqqq: sim.tqqqBH, qqq: sim.qqqBH, spy: sim.spyBH, dip: sim.dipBuyVal };
  const show = {
    strat: $("showStrat").checked, tqqq: $("showTqqq").checked,
    qqq:   $("showQqq").checked,   spy:  $("showSpy").checked,
    dip:   $("showDip").checked,
  };

  const traces = [];
  for (const k of ORDER) {
    if (!show[k]) continue;
    traces.push({
      x: sliceDates, y: Array.from(series[k]),
      name: SERIES[k].name,
      line: { color: SERIES[k].color, width: k === "strat" ? 2.3 : 1.5 },
      hovertemplate: "%{x|%Y-%m-%d}<br>" + SERIES[k].name + ": <b>%{y:$,.0f}</b><extra></extra>",
    });
  }

  Plotly.react("chart", traces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e9edf4", family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { t: 10, r: 18, b: 36, l: 72 },
    yaxis: {
      type: useLog ? "log" : "linear",
      title: { text: useLog ? "Portfolio ($, log)" : "Portfolio ($)", font: { size: 11, color: "#97a1b3" } },
      gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, tickformat: "$,.0f",
    },
    xaxis: { gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: "rgba(231,236,244,0.65)", spikedash: "solid", spikesnap: "cursor" },
    legend: { orientation: "h", y: 1.07, font: { size: 11.5 }, bgcolor: "rgba(0,0,0,0)" },
    hovermode: "x unified", hoverlabel: { bgcolor: "#11141b", bordercolor: "#2a3243", font: { size: 12 } },
    shapes: outOfMarketShapes(startIdx, sim.posOut),
  }, { displaylogo: false, responsive: true });

  // Drawdown chart.
  const ddTraces = [];
  for (const k of ORDER) {
    if (!show[k]) continue;
    ddTraces.push({
      x: sliceDates, y: Array.from(drawdownSeries(series[k])),
      name: SERIES[k].name,
      line: { color: SERIES[k].color, width: k === "strat" ? 1.8 : 1.2 },
      hovertemplate: "%{x|%Y-%m-%d}<br>" + SERIES[k].name + ": <b>%{y:.1%}</b><extra></extra>",
    });
  }
  Plotly.react("drawdown", ddTraces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e9edf4", family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { t: 6, r: 18, b: 36, l: 72 },
    yaxis: { title: { text: "DD", font: { size: 11, color: "#97a1b3" } }, tickformat: ".0%", gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 } },
    xaxis: { gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: "rgba(231,236,244,0.65)", spikedash: "solid", spikesnap: "cursor" },
    showlegend: false, hovermode: "x unified",
    hoverlabel: { bgcolor: "#11141b", bordercolor: "#2a3243" },
  }, { displaylogo: false, responsive: true });

  // QQQ vs SMA chart — buy/sell trigger lines, with underlying SMAs as faint reference.
  const qSlice    = Array.from(qqq.slice(startIdx, endIdx + 1));
  const buySmaSl  = Array.from(smas[state.buySMA].slice(startIdx, endIdx + 1));
  const sellSmaSl = Array.from(smas[state.sellSMA].slice(startIdx, endIdx + 1));
  const buyLine   = buySmaSl.map(v => v * (1 + buyFrac));
  const sellLine  = sellSmaSl.map(v => v * (1 - sellFrac));

  const sigTraces = [];

  // Dead-zone shading between the two trigger lines (always — they may differ even with 0 buffers if SMAs differ).
  sigTraces.push({
    x: sliceDates, y: sellLine, line: { width: 0 }, hoverinfo: "skip", showlegend: false,
  });
  sigTraces.push({
    x: sliceDates, y: buyLine, fill: "tonexty", fillcolor: "rgba(108,196,255,0.06)",
    line: { width: 0 }, hoverinfo: "skip", showlegend: false,
  });

  // QQQ price (primary).
  sigTraces.push({
    x: sliceDates, y: qSlice, name: "QQQ",
    line: { color: "#c7cdda", width: 1.4 },
    hovertemplate: "%{x|%Y-%m-%d}<br>QQQ: <b>%{y:$,.2f}</b><extra></extra>",
  });

  // Underlying SMA(s) shown as faint reference when they're distinct from the trigger lines
  // (i.e. when buffer > 0) or when buy/sell SMAs differ.
  if (buyFrac > 0) {
    sigTraces.push({
      x: sliceDates, y: buySmaSl, name: `${state.buySMA}D SMA (buy)`,
      line: { color: "#6cc4ff", width: 1.1, dash: "dot" }, opacity: 0.6,
      hovertemplate: "%{x|%Y-%m-%d}<br>" + state.buySMA + "D SMA: <b>%{y:$,.2f}</b><extra></extra>",
    });
  }
  if (sellFrac > 0 && state.sellSMA !== state.buySMA) {
    sigTraces.push({
      x: sliceDates, y: sellSmaSl, name: `${state.sellSMA}D SMA (sell)`,
      line: { color: "#fbbf24", width: 1.1, dash: "dot" }, opacity: 0.6,
      hovertemplate: "%{x|%Y-%m-%d}<br>" + state.sellSMA + "D SMA: <b>%{y:$,.2f}</b><extra></extra>",
    });
  }

  // Trigger lines — the actual decision boundaries.
  const buyLabel  = `Buy line (${state.buySMA}D × ${(1+buyFrac).toFixed(3)})`;
  const sellLabel = `Sell line (${state.sellSMA}D × ${(1-sellFrac).toFixed(3)})`;
  sigTraces.push({
    x: sliceDates, y: buyLine, name: buyLabel,
    line: { color: "#34d399", width: 1.3, dash: buyFrac > 0 ? "dash" : "solid" },
    hovertemplate: "%{x|%Y-%m-%d}<br>Buy line: <b>%{y:$,.2f}</b><extra></extra>",
  });
  sigTraces.push({
    x: sliceDates, y: sellLine, name: sellLabel,
    line: { color: "#ef4444", width: 1.3, dash: sellFrac > 0 ? "dash" : "solid" },
    hovertemplate: "%{x|%Y-%m-%d}<br>Sell line: <b>%{y:$,.2f}</b><extra></extra>",
  });
  // Trailing-stop line (Rule 1) — only when enabled.
  if (state.dropEnabled) {
    sigTraces.push({
      x: sliceDates, y: Array.from(sim.stopLevel),
      name: `Trailing stop (peak${state.dropWindow}D × ${(1 - dropFrac).toFixed(3)})`,
      line: { color: "#fbbf24", width: 1.2, dash: "dot" },
      hovertemplate: "%{x|%Y-%m-%d}<br>Stop level: <b>%{y:$,.2f}</b><extra></extra>",
    });
  }
  // Rule 2 direction-confirm SMA — only when both Rule 2 and direction-confirm are on.
  if (state.volEnabled && state.volDirSMA !== 0 && smas[state.volDirSMA]) {
    const volDirSlice = Array.from(smas[state.volDirSMA].slice(startIdx, endIdx + 1));
    sigTraces.push({
      x: sliceDates, y: volDirSlice,
      name: `Vol-dir ${state.volDirSMA}D SMA`,
      line: { color: "#fb923c", width: 1.0, dash: "dot" }, opacity: 0.7,
      hovertemplate: "%{x|%Y-%m-%d}<br>" + state.volDirSMA + "D SMA: <b>%{y:$,.2f}</b><extra></extra>",
    });
  }

  const hintParts = ["red shading = out of market"];
  hintParts.push(`buy: ${state.buySMA}D` + (buyFrac > 0  ? ` +${state.buyBuffer.toFixed(2)}%`  : ""));
  hintParts.push(`sell: ${state.sellSMA}D` + (sellFrac > 0 ? ` −${state.sellBuffer.toFixed(2)}%` : ""));
  if (state.dropEnabled) hintParts.push(`stop: −${state.dropPct.toFixed(1)}% / ${state.dropWindow}D`);
  $("signalHint").textContent = hintParts.join(" · ");

  Plotly.react("signal", sigTraces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e9edf4", family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { t: 6, r: 18, b: 36, l: 72 },
    yaxis: { type: useLog ? "log" : "linear", title: { text: "QQQ ($)", font: { size: 11, color: "#97a1b3" } }, gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 } },
    xaxis: { gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: "rgba(231,236,244,0.65)", spikedash: "solid", spikesnap: "cursor" },
    legend: { orientation: "h", y: 1.12, font: { size: 11 }, bgcolor: "rgba(0,0,0,0)" },
    hovermode: "x unified", hoverlabel: { bgcolor: "#11141b", bordercolor: "#2a3243" },
    shapes: outOfMarketShapes(startIdx, sim.posOut),
  }, { displaylogo: false, responsive: true });

  // Realized-vol chart (Rule 2). Always plotted; shading + threshold line appear only when enabled.
  const volArr = Array.from(sim.volSeries);
  const volTraces = [{
    x: sliceDates, y: volArr, name: "Realized vol",
    line: { color: "#f5c451", width: 1.2 },
    hovertemplate: "%{x|%Y-%m-%d}<br>Vol: <b>%{y:.1%}</b><extra></extra>",
  }];
  const volShapes = [];
  if (state.volEnabled) {
    // Horizontal threshold line.
    volShapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: volFrac, y1: volFrac,
      line: { color: "#ef4444", width: 1.2, dash: "dash" },
      layer: "below",
    });
    // Red shading where vol > threshold.
    let runStart = -1;
    for (let k2 = 0; k2 <= volArr.length; k2++) {
      const tripped = (k2 < volArr.length) && (volArr[k2] > volFrac);
      if (tripped && runStart < 0) runStart = k2;
      if (!tripped && runStart >= 0) {
        volShapes.push({
          type: "rect", xref: "x", yref: "paper",
          x0: dates[startIdx + runStart], x1: dates[startIdx + k2 - 1],
          y0: 0, y1: 1,
          fillcolor: "rgba(239,68,68,0.10)",
          line: { width: 0 }, layer: "below",
        });
        runStart = -1;
      }
    }
  }
  const volHintParts = [`${state.volWindow}-day annualized`];
  if (state.volEnabled) {
    volHintParts.push(`threshold <b>${state.volPct}%</b> (red)`);
    if (state.volDirSMA !== 0) volHintParts.push(`dir-confirm: QQQ &lt; <b>${state.volDirSMA}D-SMA</b>`);
  } else {
    volHintParts.push("rule off");
  }
  $("volHint").innerHTML = volHintParts.join(" · ");

  Plotly.react("volChart", volTraces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e9edf4", family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { t: 6, r: 18, b: 36, l: 72 },
    yaxis: { title: { text: "σ (ann.)", font: { size: 11, color: "#97a1b3" } }, tickformat: ".0%", gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 } },
    xaxis: { gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: "rgba(231,236,244,0.65)", spikedash: "solid", spikesnap: "cursor" },
    showlegend: false, hovermode: "x unified",
    hoverlabel: { bgcolor: "#11141b", bordercolor: "#2a3243" },
    shapes: volShapes,
  }, { displaylogo: false, responsive: true });

  // TQQQ price chart (same slice, same out-of-market shading).
  const tSlice = Array.from(tqqq.slice(startIdx, endIdx + 1));
  Plotly.react("tqqqChart", [{
    x: sliceDates, y: tSlice, name: "TQQQ",
    line: { color: "#ff6b6b", width: 1.4 },
    hovertemplate: "%{x|%Y-%m-%d}<br>TQQQ: <b>%{y:$,.2f}</b><extra></extra>",
  }], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e9edf4", family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { t: 6, r: 18, b: 36, l: 72 },
    yaxis: { type: useLog ? "log" : "linear", title: { text: "TQQQ ($)", font: { size: 11, color: "#97a1b3" } }, gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, tickformat: "$,.2f" },
    xaxis: { gridcolor: "#1c2230", zerolinecolor: "#1c2230", tickfont: { size: 11 }, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: "rgba(231,236,244,0.65)", spikedash: "solid", spikesnap: "cursor" },
    showlegend: false, hovermode: "x unified",
    hoverlabel: { bgcolor: "#11141b", bordercolor: "#2a3243" },
    shapes: outOfMarketShapes(startIdx, sim.posOut),
  }, { displaylogo: false, responsive: true });

  // KPI cards + summary table.
  const finals = {
    strat: sim.stratVal[len-1], tqqq: sim.tqqqBH[len-1],
    qqq: sim.qqqBH[len-1], spy: sim.spyBH[len-1],
    dip: sim.dipBuyVal[len-1],
  };
  const dds = {
    strat: maxDrawdown(sim.stratVal), tqqq: maxDrawdown(sim.tqqqBH),
    qqq: maxDrawdown(sim.qqqBH), spy: maxDrawdown(sim.spyBH),
    dip: maxDrawdown(sim.dipBuyVal),
  };
  // Per-series cash flows. Strategy/B&H use the main cf (every dollar deposited).
  // Dip-Buy uses its own cf (only days where a buy actually fired).
  const dipInvested = sim.dipCfAmt.reduce((a,b)=>a+b, 0);
  const cfMap = {
    strat: { idx: sim.cfIdx, amt: sim.cfAmt, total: totalInvested },
    tqqq:  { idx: sim.cfIdx, amt: sim.cfAmt, total: totalInvested },
    qqq:   { idx: sim.cfIdx, amt: sim.cfAmt, total: totalInvested },
    spy:   { idx: sim.cfIdx, amt: sim.cfAmt, total: totalInvested },
    dip:   { idx: sim.dipCfIdx, amt: sim.dipCfAmt, total: dipInvested },
  };

  const kpisEl = $("kpis");
  kpisEl.innerHTML = "";
  for (const k of ORDER) {
    const fin = finals[k];
    const cf = cfMap[k];
    const mult = fin / Math.max(1, cf.total);
    const ir = irr(cf.idx, cf.amt, len - 1, fin);
    const card = document.createElement("div");
    card.className = "kpi" + (k === "strat" ? " is-strat" : "");
    card.style.setProperty("--c", SERIES[k].color);
    card.innerHTML = `
      <div class="kpi-name"><span class="sw"></span> ${SERIES[k].name}</div>
      <div class="kpi-final num">${fmtMoneyFull(fin)}</div>
      <div class="kpi-sub">
        <span class="num"><span class="${(fin >= cf.total) ? 'delta-pos' : 'delta-neg'}">${fmtX(mult)}</span></span>
        <span>·</span>
        <span class="num">IRR ${fmtPct(ir, 1)}</span>
        <span>·</span>
        <span class="num">DD ${fmtPct(dds[k], 0)}</span>
      </div>`;
    kpisEl.appendChild(card);
  }

  const tbody = document.querySelector("#summary tbody");
  tbody.innerHTML = "";
  for (const k of ORDER) {
    const fin = finals[k];
    const cf = cfMap[k];
    const mult = fin / Math.max(1, cf.total);
    const ir = irr(cf.idx, cf.amt, len - 1, fin);
    const tr = document.createElement("tr");
    if (k === "strat") tr.className = "strat-row";
    tr.innerHTML = `
      <td><span class="sw" style="background:${SERIES[k].color}"></span>${SERIES[k].name}</td>
      <td>${fmtMoney(fin)}</td>
      <td>${fmtX(mult)}</td>
      <td>${fmtPct(ir, 1)}</td>
      <td style="color:var(--bad)">${fmtPct(dds[k], 1)}</td>`;
    tbody.appendChild(tr);
  }
}

// Wiring
rangeSlider.noUiSlider.on("update", (vals) => {
  state.startQ = Math.round(+vals[0]);
  state.endQ = Math.round(+vals[1]);
  update();
});
initSlider.noUiSlider.on("update", (vals) => {
  state.initial = Math.round(+vals[0]);
  $("initInput").value = state.initial;
  update();
});
contribSlider.noUiSlider.on("update", (vals) => {
  state.contrib = Math.round(+vals[0]);
  $("contribInput").value = state.contrib;
  update();
});
contribFreqSlider.noUiSlider.on("update", (vals) => {
  state.contribTimes = Math.round(+vals[0]);
  update();
});
buyBufferSlider.noUiSlider.on("update", (vals) => {
  state.buyBuffer = Math.round((+vals[0]) * 100) / 100;
  update();
});
sellBufferSlider.noUiSlider.on("update", (vals) => {
  state.sellBuffer = Math.round((+vals[0]) * 100) / 100;
  update();
});
// Rule 1
dropPctSlider.noUiSlider.on("update", (vals) => {
  state.dropPct = Math.round((+vals[0]) * 10) / 10;
  update();
});
dropWindowSlider.noUiSlider.on("update", (vals) => {
  state.dropWindow = Math.round(+vals[0]);
  update();
});
$("dropEnabled").addEventListener("change", () => {
  state.dropEnabled = $("dropEnabled").checked;
  update();
});
// Rule 2 wiring
volPctSlider.noUiSlider.on("update", (vals) => {
  state.volPct = Math.round(+vals[0]);
  update();
});
volWindowSlider.noUiSlider.on("update", (vals) => {
  state.volWindow = Math.round(+vals[0]);
  update();
});
$("volEnabled").addEventListener("change", () => {
  state.volEnabled = $("volEnabled").checked;
  update();
});
$("initInput").addEventListener("change", () => {
  const v = Math.max(0, +$("initInput").value || 0);
  initSlider.noUiSlider.set(Math.min(v, 250000));
  if (v > 250000) { state.initial = v; update(); }
});
$("contribInput").addEventListener("change", () => {
  const v = Math.max(0, +$("contribInput").value || 0);
  contribSlider.noUiSlider.set(Math.min(v, 10000));
  if (v > 10000) { state.contrib = v; update(); }
});
for (const id of ["showStrat","showTqqq","showQqq","showSpy","showDip","showShade","logScale"]) {
  $(id).addEventListener("change", update);
}

update();

// ───────── Cross-chart hover sync ─────────
// Strategy:
//   - Hovered chart shows its native popup + spike line (via showspikes config).
//   - Other charts get a vertical cursor line drawn via Plotly.relayout shape.
// We can't reliably use Plotly.Fx.hover across charts because the xval format
// differs from points[0].x for date axes. Shapes accept the same x format that
// trace data uses (date strings here), so this is bulletproof.
const SYNC_CHARTS = ["chart", "drawdown", "signal", "volChart", "tqqqChart"];
const CURSOR_TAG = "__crosscursor";

function nonCursorShapes(gd) {
  const shapes = (gd && gd.layout && gd.layout.shapes) ? gd.layout.shapes : [];
  return shapes.filter(s => s && s.name !== CURSOR_TAG);
}

function drawCursor(gd, x) {
  if (!gd || !gd.layout) return;
  const shapes = nonCursorShapes(gd);
  shapes.push({
    type: "line", xref: "x", yref: "paper",
    x0: x, x1: x, y0: 0, y1: 1,
    line: { color: "rgba(231,236,244,0.65)", width: 1 },
    layer: "above",
    name: CURSOR_TAG,
  });
  Plotly.relayout(gd, { shapes: shapes });
}

function clearCursor(gd) {
  if (!gd || !gd.layout) return;
  const shapes = nonCursorShapes(gd);
  if (shapes.length === ((gd.layout.shapes || []).length)) return; // nothing to clear
  Plotly.relayout(gd, { shapes: shapes });
}

function attachHoverSync() {
  for (const id of SYNC_CHARTS) {
    const gd = $(id);
    if (!gd || gd._syncAttached || typeof gd.on !== "function") continue;
    gd._syncAttached = true;

    gd.on("plotly_hover", (evt) => {
      // points[0].x is in the same coordinate space as our trace x values
      // (date strings here). That's exactly what shape x0/x1 expects.
      if (!evt || !evt.points || !evt.points.length) return;
      const x = evt.points[0].x;
      for (const otherId of SYNC_CHARTS) {
        if (otherId === id) continue;
        drawCursor($(otherId), x);
      }
    });

    gd.on("plotly_unhover", () => {
      for (const otherId of SYNC_CHARTS) {
        if (otherId === id) continue;
        clearCursor($(otherId));
      }
    });
  }
}
attachHoverSync();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
