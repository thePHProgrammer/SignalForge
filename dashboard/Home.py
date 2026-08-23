"""SignalForge dashboard entry point.

Run with:  streamlit run dashboard/Home.py

Full CLI parity: every page here calls the same library functions
scripts/*.py already calls (adapters, signals, backtest, ML) directly --
this is a UI on the library, not a wrapper around the CLI scripts. Use the
sidebar to navigate to Fetch Candles, Signals, Backtest, or ML Comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

# streamlit run doesn't put the repo root on sys.path (unlike python -m,
# which every scripts/*.py CLI relies on) -- find it by walking up from this
# file until we find the signalforge package, so `dashboard` and
# `signalforge` are both importable regardless of how/where streamlit was
# launched from. Depth-independent so the identical snippet also works
# unchanged in dashboard/pages/*.py, which sits one level deeper.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "signalforge").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import streamlit as st

from dashboard._shared import get_settings

st.set_page_config(page_title="SignalForge", layout="wide")

st.title("SignalForge")
st.caption("Personal-use quant trading signal generator across crypto and forex. Not investment advice.")

st.markdown(
    """
Use the sidebar to:

- **Fetch Candles** — pull real OHLCV data from Kraken (crypto) or Twelve Data / OANDA (forex) into the local database.
- **Signals** — view the rule-based RSI/MACD/Bollinger signal (with optional multi-timeframe confirmation) over stored candles.
- **Backtest** — run a walk-forward backtest of the rule-based signal.
- **ML Comparison** — walk-forward compare a LightGBM model against the rule-based baseline.

Fetching happens right here in the dashboard, so `scripts/fetch_candles.py` isn't required first -- but candles must
exist in the database (fetched here or via the CLI) before Signals/Backtest/ML Comparison have anything to show.
"""
)

settings = get_settings()
st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Database path", str(settings.db_path))
col2.metric("Twelve Data key configured", "yes" if settings.twelve_data_api_key else "no")
col3.metric("OANDA token configured", "yes" if settings.oanda_api_token else "no")
