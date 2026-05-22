"""
Daily strategy signal check + Telegram notification.

Reads `config.json`, re-simulates the strategy over the last ~2 years to derive
today's position with proper hysteresis, compares to yesterday's position to
detect signal changes, and sends a Telegram message.

Env vars required:
  TELEGRAM_BOT_TOKEN  – your bot's API token (set as a GitHub repo secret)
  TELEGRAM_CHAT_ID    – chat / channel ID for the bot to message
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config.json"
STATE_PATH = ROOT / "state.json"

# Simulation backstop — re-derive position from this many trading days back so
# hysteresis is well-warmed. ~2 years is plenty for any SMA up to 250D.
BACKSTOP_DAYS = 504


# ---------- Data ----------

def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


# ---------- Simulation ----------

def simulate(qqq: pd.Series, cfg: dict) -> dict:
    """
    Re-derive position track over the last BACKSTOP_DAYS of QQQ history.
    Returns today's diagnostics + position(today, yesterday).
    """
    buy_sma_w  = int(cfg["buySMA"])
    sell_sma_w = int(cfg["sellSMA"])
    buy_buf    = float(cfg["buyBuffer"])  / 100.0
    sell_buf   = float(cfg["sellBuffer"]) / 100.0

    r1 = cfg.get("rule1", {})
    drop_on  = bool(r1.get("enabled", False))
    drop_pct = float(r1.get("dropPct", 8)) / 100.0
    drop_win = int(r1.get("dropWindow", 22))

    r2 = cfg.get("rule2", {})
    vol_on    = bool(r2.get("enabled", False))
    vol_pct   = float(r2.get("volPct", 30)) / 100.0
    vol_win   = int(r2.get("volWindow", 10))
    vol_dir_w = int(r2.get("volDirSMA", 0))  # 0 = off

    # All SMAs we may need.
    smas = {w: qqq.rolling(w).mean() for w in {buy_sma_w, sell_sma_w}.union({vol_dir_w} if vol_dir_w else set())}
    buy_sma  = smas[buy_sma_w]
    sell_sma = smas[sell_sma_w]
    vol_dir_sma = smas[vol_dir_w] if vol_dir_w else None

    log_ret = np.log(qqq / qqq.shift(1))

    # Restrict simulation to the backstop window. We still need full QQQ history
    # leading into this window for the SMA values to be defined.
    start_idx = max(0, len(qqq) - BACKSTOP_DAYS)

    pos = 0
    initial_set = False
    positions = []
    diag_today = {}

    for i in range(start_idx, len(qqq)):
        q  = float(qqq.iloc[i])
        bs = float(buy_sma.iloc[i])
        ss = float(sell_sma.iloc[i])
        date = qqq.index[i]

        if math.isnan(bs) or math.isnan(ss):
            positions.append(0)
            continue

        # --- Rule 1: trailing stop ---
        stop_tripped = False
        stop_level = None
        if drop_on:
            window_from = max(0, i - drop_win + 1)
            peak = float(qqq.iloc[window_from:i + 1].max())
            stop_level = peak * (1.0 - drop_pct)
            stop_tripped = q <= stop_level

        # --- Rule 2: realized vol ---
        vol_tripped = False
        vol_ann = float("nan")
        vol_from = max(1, i - vol_win + 1)
        recent = log_ret.iloc[vol_from:i + 1].dropna()
        if len(recent) >= 2:
            vol_ann = float(recent.std(ddof=1)) * math.sqrt(252)
            if vol_on and vol_ann > vol_pct:
                if vol_dir_sma is None:
                    vol_tripped = True
                else:
                    d = float(vol_dir_sma.iloc[i])
                    if not math.isnan(d) and q < d:
                        vol_tripped = True

        # --- SMA hysteresis verdict ---
        if not initial_set:
            sma_target = 1 if q > bs else 0
            initial_set = True
        else:
            upper = bs * (1.0 + buy_buf)
            lower = ss * (1.0 - sell_buf)
            if pos == 0 and q > upper:
                sma_target = 1
            elif pos == 1 and q < lower:
                sma_target = 0
            else:
                sma_target = pos

        # --- AND-combine with active filters ---
        target = 1 if (sma_target == 1 and not stop_tripped and not vol_tripped) else 0
        pos = target
        positions.append(pos)

        # Capture today's full snapshot on the last iteration.
        if i == len(qqq) - 1:
            diag_today = {
                "date": date,
                "qqq": q,
                "buy_sma": bs, "sell_sma": ss,
                "buy_line": bs * (1.0 + buy_buf),
                "sell_line": ss * (1.0 - sell_buf),
                "buy_buf": buy_buf, "sell_buf": sell_buf,
                "buy_sma_w": buy_sma_w, "sell_sma_w": sell_sma_w,
                "stop_on": drop_on, "stop_level": stop_level, "stop_tripped": stop_tripped,
                "stop_pct": drop_pct, "stop_win": drop_win,
                "vol_on": vol_on, "vol_ann": vol_ann, "vol_tripped": vol_tripped,
                "vol_pct": vol_pct, "vol_win": vol_win, "vol_dir_w": vol_dir_w,
                "vol_dir": float(vol_dir_sma.iloc[i]) if vol_dir_sma is not None else None,
                "sma_says_in": sma_target == 1,
            }

    if not diag_today:
        raise RuntimeError(
            f"No diagnostic snapshot for the last day "
            f"({qqq.index[-1].strftime('%Y-%m-%d')}); "
            f"SMA(s) undefined — check the QQQ TSV has at least "
            f"max(buySMA, sellSMA) days of history."
        )

    pos_today     = "TQQQ" if positions[-1] == 1 else "CASH"
    pos_yesterday = "TQQQ" if positions[-2] == 1 else "CASH"

    return {
        "position_today": pos_today,
        "position_yesterday": pos_yesterday,
        "diag": diag_today,
    }


# ---------- Message rendering ----------

def fmt_money(v):
    return f"${v:,.2f}"


def fmt_pct(v, d=2):
    return f"{v * 100:+.{d}f}%"


def build_message(result: dict, cfg: dict, changed: bool) -> str:
    d = result["diag"]
    pos = result["position_today"]
    prev = result["position_yesterday"]
    date_str = d["date"].strftime("%Y-%m-%d")

    pct_vs_sma = (d["qqq"] / d["sell_sma"] - 1) if pos == "TQQQ" else (d["qqq"] / d["buy_sma"] - 1)
    ref_sma_w = d["sell_sma_w"] if pos == "TQQQ" else d["buy_sma_w"]

    lines = []
    lines.append(f"📊 <b>TQQQ Strategy</b> — {date_str}")
    lines.append("")

    if changed:
        emoji = "🟢" if pos == "TQQQ" else "🔴"
        lines.append(f"🚨 <b>SIGNAL CHANGE: {prev} → {pos}</b> {emoji}")
        if pos == "CASH":
            # Reason
            reasons = []
            if not d["sma_says_in"]:
                reasons.append(f"QQQ below sell line ({fmt_money(d['sell_line'])})")
            if d["stop_on"] and d["stop_tripped"]:
                reasons.append(f"trailing stop tripped (level {fmt_money(d['stop_level'])})")
            if d["vol_on"] and d["vol_tripped"]:
                reasons.append(f"vol spike ({d['vol_ann'] * 100:.1f}% &gt; {d['vol_pct'] * 100:.0f}%)")
            lines.append(f"Trigger: {', '.join(reasons) or 'unknown'}")
            lines.append(f"<b>Action: sell all TQQQ at next open</b>")
        else:
            lines.append(f"Trigger: all filters cleared, QQQ &gt; {ref_sma_w}D-SMA buy line")
            lines.append(f"<b>Action: buy TQQQ at next open</b>")
        lines.append("")
    else:
        if pos == "TQQQ":
            lines.append(f"✅ Hold <b>TQQQ</b> (no change since previous run)")
        else:
            lines.append(f"⏸ Stay in <b>CASH</b> (no change since previous run)")
        lines.append("")

    # Detail block
    lines.append(f"<b>QQQ:</b> {fmt_money(d['qqq'])} ({fmt_pct(pct_vs_sma, 2)} vs {ref_sma_w}D-SMA)")
    lines.append(f"<b>Buy line:</b>  {fmt_money(d['buy_line'])}  ({d['buy_sma_w']}D × {1 + d['buy_buf']:.4f})")
    lines.append(f"<b>Sell line:</b> {fmt_money(d['sell_line'])}  ({d['sell_sma_w']}D × {1 - d['sell_buf']:.4f})")
    lines.append("")

    # Filters
    if d["stop_on"]:
        flag = "🔴 TRIPPED" if d["stop_tripped"] else "🟢 ok"
        lines.append(f"<b>Trailing stop</b> (−{d['stop_pct']*100:.1f}% / {d['stop_win']}D): "
                     f"{flag} (level {fmt_money(d['stop_level'])})")
    else:
        lines.append("<b>Trailing stop</b>: off")

    if d["vol_on"]:
        flag = "🔴 TRIPPED" if d["vol_tripped"] else "🟢 ok"
        dir_note = ""
        if d["vol_dir_w"]:
            dir_state = "below" if d["vol_dir"] is not None and d["qqq"] < d["vol_dir"] else "above"
            dir_note = f", dir-confirm {d['vol_dir_w']}D SMA: QQQ {dir_state}"
        lines.append(f"<b>Vol spike</b> (&gt;{d['vol_pct']*100:.0f}% / {d['vol_win']}D): "
                     f"{flag} (now {d['vol_ann']*100:.1f}%{dir_note})")
    else:
        lines.append("<b>Vol spike</b>: off")

    return "\n".join(lines)


# ---------- Telegram ----------

def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars not set.", file=sys.stderr)
        sys.exit(2)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        print(f"Telegram API error {r.status_code}: {r.text}", file=sys.stderr)
        r.raise_for_status()


# ---------- State persistence ----------

def load_previous_state(state_path: Path, expected_as_of: str) -> str | None:
    """Return the position the prior run recorded for `expected_as_of`, if any."""
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if state.get("as_of") != expected_as_of:
        return None
    pos = state.get("position")
    if pos not in ("TQQQ", "CASH"):
        return None
    return pos


# ---------- Main ----------

def main() -> int:
    cfg = json.loads(CONFIG_PATH.read_text())
    alert_mode = cfg.get("alertMode", "both")

    qqq = load_series(DATA_DIR / "synthetic-qqq.tsv")
    if len(qqq) < BACKSTOP_DAYS + 2:
        print("Not enough QQQ history for simulation.", file=sys.stderr)
        return 1

    result = simulate(qqq, cfg)
    pos_today = result["position_today"]
    sim_yesterday = result["position_yesterday"]

    # auto_adjust=True can retroactively shift historical prices on each
    # dividend, nudging the 200-SMA and flipping a borderline yesterday in
    # the re-simulation -> phantom SIGNAL CHANGE alerts. Prefer the prior
    # run's recorded position when its data date matches.
    yesterday_as_of = qqq.index[-2].strftime("%Y-%m-%d")
    stored_yesterday = load_previous_state(STATE_PATH, yesterday_as_of)
    if stored_yesterday is None:
        pos_yesterday = sim_yesterday
    else:
        pos_yesterday = stored_yesterday
        if stored_yesterday != sim_yesterday:
            print(
                f"Note: simulated yesterday is {sim_yesterday} but stored "
                f"state for {yesterday_as_of} recorded {stored_yesterday}; "
                f"trusting stored.",
                file=sys.stderr,
            )

    result["position_yesterday"] = pos_yesterday
    changed = (pos_today != pos_yesterday)

    print(f"As of {result['diag']['date'].date()}: {pos_today} (yesterday: {pos_yesterday}; changed: {changed})")

    # Decide whether to send.
    should_send = (
        alert_mode == "daily" or
        (alert_mode == "change" and changed) or
        alert_mode == "both"
    )

    if should_send:
        msg = build_message(result, cfg, changed)
        send_telegram(msg)
        print("Telegram message sent.")
    else:
        print("Skipping send per alertMode.")

    # Persist state for the next run.
    state = {
        "as_of": result["diag"]["date"].strftime("%Y-%m-%d"),
        "position": pos_today,
        "previous_position": pos_yesterday,
        "changed_today": changed,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Wrote {STATE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
