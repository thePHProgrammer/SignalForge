"""Walk-forward backtest of the rule-based signal generator -- the dashboard
equivalent of scripts/run_backtest.py, calling run_walk_forward() directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# See dashboard/Home.py for why this bootstrap is needed on every page.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "signalforge").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import datetime as dt

import pandas as pd
import streamlit as st

from dashboard._shared import (
    CRYPTO_SYMBOLS,
    FOREX_SYMBOLS,
    TIMEFRAMES,
    build_equity_curve_figure,
    cost_note_text,
    get_settings,
    open_connection,
    to_utc,
)
from signalforge.backtest.costs import default_cost_model
from signalforge.backtest.runner import run_walk_forward
from signalforge.signals import REQUIRED_WARMUP_BARS

st.set_page_config(page_title="Backtest - SignalForge", layout="wide")
st.title("Backtest")

settings = get_settings()

asset_class = st.radio("Asset class", ["crypto", "forex"], horizontal=True)
default_symbol = CRYPTO_SYMBOLS[0] if asset_class == "crypto" else FOREX_SYMBOLS[0]
symbol = st.text_input('Symbol ("BASE/QUOTE")', value=default_symbol)

col1, col2 = st.columns(2)
timeframe = col1.selectbox("Primary timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1h"))
confirm_timeframe = col2.selectbox("Confirmation timeframe (optional)", ["None"] + TIMEFRAMES)
confirm_timeframe = None if confirm_timeframe == "None" else confirm_timeframe

col3, col4 = st.columns(2)
start_date = col3.date_input("Start", value=dt.date.today() - dt.timedelta(days=180))
end_date = col4.date_input("End", value=dt.date.today())

col5, col6, col7 = st.columns(3)
train_days = col5.number_input("Train days", min_value=1.0, value=90.0)
test_days = col6.number_input("Test days", min_value=1.0, value=30.0)
step_days = col7.number_input("Step days (0 = same as test days)", min_value=0.0, value=0.0)

col8, col9 = st.columns(2)
initial_capital = col8.number_input("Initial capital", min_value=1.0, value=10_000.0)
position_mode = col9.selectbox("Position mode", ["long_only", "long_short"])

with st.expander("Advanced"):
    warmup_buffer_bars = st.number_input("Warm-up buffer bars", min_value=0, value=REQUIRED_WARMUP_BARS)
    risk_free_rate = st.number_input("Risk-free rate (annual)", value=0.0, format="%.4f")

if position_mode == "long_short" and asset_class == "crypto":
    st.warning("long_short is not supported for crypto (Kraken is spot-only) -- switch to long_only.")
    st.stop()

if st.button("Run backtest", type="primary"):
    resolved_cost = default_cost_model(asset_class, symbol)
    note = cost_note_text(resolved_cost)
    if note:
        st.info(f"Cost model note: {note}")

    with open_connection(settings) as conn, st.spinner("Running walk-forward backtest..."):
        report = run_walk_forward(
            conn, symbol, asset_class, timeframe, to_utc(start_date), to_utc(end_date),
            train_period=pd.Timedelta(days=train_days),
            test_period=pd.Timedelta(days=test_days),
            step=pd.Timedelta(days=step_days) if step_days else None,
            confirm_timeframe=confirm_timeframe,
            warmup_buffer_bars=int(warmup_buffer_bars),
            initial_capital=initial_capital,
            position_mode=position_mode,
            cost_model=resolved_cost.cost_model,
            risk_free_rate=risk_free_rate,
        )

    if not report.windows:
        st.warning(f"No test windows produced results for {symbol} ({timeframe}) in this range. "
                   f"Check that candles are stored (Fetch Candles page) covering the range plus warm-up.")
        st.stop()

    rows = [
        {
            "test_start": wr.window.test_start, "test_end": wr.window.test_end,
            "trades": wr.metrics.total_trades,
            "forced_closes": sum(1 for t in wr.run.trades if t.is_forced_close),
            "sharpe": wr.metrics.sharpe_ratio, "max_dd": wr.metrics.max_drawdown,
            "win_rate": wr.metrics.win_rate, "profit_factor": wr.metrics.profit_factor,
            "final_equity": wr.metrics.final_equity,
        }
        for wr in report.windows
    ]
    st.subheader(f"{len(report.windows)} test window(s)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Combined (stitched out-of-sample)")
    if report.combined_metrics is not None:
        m = report.combined_metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", m.total_trades)
        c2.metric("Sharpe", f"{m.sharpe_ratio:.4f}")
        c3.metric("Max DD", f"{m.max_drawdown:.4f}")
        c4.metric("Win rate", f"{m.win_rate:.4f}")
        c5.metric("Final equity", f"{m.final_equity:.2f}")
        st.plotly_chart(
            build_equity_curve_figure({"Rule-based": report.combined_run.equity_curve}, title="Equity curve"),
            use_container_width=True,
        )
    else:
        st.warning("Combined metrics unavailable -- test windows are not contiguous (step != test_days). "
                   "Per-window results above are still meaningful on their own.")
