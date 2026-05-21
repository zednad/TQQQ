"""
Fetch and synthesize daily closing prices for QQQ, TQQQ, SPY.

Outputs (tab-separated, columns: date<TAB>close):
  data/synthetic-qqq.tsv   QQQ 1999-present; pre-1999 synthesized from ^NDX
  data/synthetic-tqqq.tsv  TQQQ 2010-present; pre-2010 synthesized via NDX-TR
                           (derived from QQQ where available) and ^NDX before that
  data/spy.tsv             SPY 1993-present; pre-1993 synthesized from
                           ^SP500TR (preferred) or ^GSPC

All series are dividend- and split-adjusted (auto_adjust=True) so comparisons
between them are on the same basis.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# TQQQ tracks the NASDAQ-100 with 3x daily leverage. We synthesize pre-2010
# TQQQ by levering the daily NDX-TR returns 3x with an annualized expense
# drag. Real TQQQ expense ratio is 0.84%; financing cost is roughly
# 2 * (fed funds + spread). For a coarse synthesis we apply a flat annual
# drag to capture both. Users who want a tighter fit can edit this constant.
TQQQ_ANNUAL_DRAG = 0.0084 + 0.01  # ~1.84%/yr, applied daily
TQQQ_DAILY_DRAG = TQQQ_ANNUAL_DRAG / 252.0


def fetch(ticker: str) -> pd.Series:
    """Return adjusted close as a pandas Series indexed by date."""
    print(f"  fetching {ticker} ...", flush=True)
    df = yf.download(
        ticker,
        period="max",
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    # yfinance may return a MultiIndex column frame for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = ticker
    return s


def splice(early: pd.Series, late: pd.Series) -> pd.Series:
    """Splice `early` (synthetic) onto the start of `late` (real).

    The early series is rescaled so its last overlapping (or last available
    pre-`late`) value matches the first `late` value, preserving daily returns.
    """
    if late.empty:
        return early.copy()
    if early.empty:
        return late.copy()

    first_late = late.index.min()
    first_late_val = late.loc[first_late]

    early_before = early.loc[early.index < first_late]
    if early_before.empty:
        return late.copy()

    # Scale early so its last value equals the first real value.
    scale = first_late_val / early_before.iloc[-1]
    early_scaled = early_before * scale

    out = pd.concat([early_scaled, late])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def synth_qqq(qqq: pd.Series, ndx: pd.Series) -> pd.Series:
    return splice(ndx, qqq)


def synth_tqqq(tqqq: pd.Series, qqq: pd.Series, ndx: pd.Series) -> pd.Series:
    """3x daily NDX-TR (proxied by QQQ where available, else ^NDX) with drag."""
    # Build the longest NDX-TR-style proxy: QQQ (TR) early-spliced onto ^NDX.
    ndx_tr = splice(ndx, qqq)
    daily_ret = ndx_tr.pct_change().dropna()
    lev_ret = 3.0 * daily_ret - TQQQ_DAILY_DRAG
    synth = (1.0 + lev_ret).cumprod()
    # Start the synthetic series at 1.0 on the first available date.
    synth = pd.concat([pd.Series([1.0], index=[daily_ret.index[0] - pd.Timedelta(days=1)]), synth])
    synth = synth.sort_index()
    return splice(synth, tqqq)


def synth_spy(spy: pd.Series, sp500tr: pd.Series, gspc: pd.Series) -> pd.Series:
    base = sp500tr if not sp500tr.empty else gspc
    return splice(base, spy)


def write_tsv(series: pd.Series, path: Path) -> None:
    df = series.rename("close").to_frame()
    df.index.name = "date"
    df.index = df.index.strftime("%Y-%m-%d")
    df.to_csv(path, sep="\t", float_format="%.6f")
    print(f"  wrote {path}  ({len(df):,} rows, {df.index[0]} -> {df.index[-1]})")


def main() -> int:
    print("Fetching source series...")
    qqq = fetch("QQQ")
    tqqq = fetch("TQQQ")
    spy = fetch("SPY")
    ndx = fetch("^NDX")
    try:
        sp500tr = fetch("^SP500TR")
    except Exception as e:
        print(f"  (^SP500TR unavailable: {e})")
        sp500tr = pd.Series(dtype=float)
    gspc = fetch("^GSPC")

    print("Building synthetic series...")
    s_qqq = synth_qqq(qqq, ndx)
    s_tqqq = synth_tqqq(tqqq, qqq, ndx)
    s_spy = synth_spy(spy, sp500tr, gspc)

    print("Writing outputs...")
    write_tsv(s_qqq, DATA_DIR / "synthetic-qqq.tsv")
    write_tsv(s_tqqq, DATA_DIR / "synthetic-tqqq.tsv")
    write_tsv(s_spy, DATA_DIR / "spy.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
