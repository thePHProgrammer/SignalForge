"""Walk-forward ML (LightGBM) vs. rule-based comparison -- the dashboard
equivalent of scripts/run_ml_backtest.py, calling run_ml_vs_rule_comparison()
directly.
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
from signalforge.ml.runner import run_ml_vs_rule_comparison
from signalforge.signals import REQUIRED_WARMUP_BARS

st.set_page_config(page_title="ML Comparison - SignalForge", layout="wide")
st.title("ML Comparison")

st.info("The ML side is always **long-only** with a fixed holding period (`horizon` bars) -- a binary up/down "
        "label can't support principled short decisions. `rule_position_mode` governs only the rule-based "
        "baseline, which can still run long_short for forex; this is a deliberate, documented asymmetry, "
        "not a bug.")

settings = get_settings()

asset_class = st.radio("Asset class", ["crypto", "forex"], horizontal=True)
default_symbol = CRYPTO_SYMBOLS[0] if asset_class == "crypto" else FOREX_SYMBOLS[0]
symbol = st.text_input('Symbol ("BASE/QUOTE")', value=default_symbol)
timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1h"))

col1, col2 = st.columns(2)
start_date = col1.date_input("Start", value=dt.date.today() - dt.timedelta(days=180))
end_date = col2.date_input("End", value=dt.date.today())

col3, col4, col5 = st.columns(3)
train_days = col3.number_input("Train days", min_value=1.0, value=60.0)
test_days = col4.number_input("Test days", min_value=1.0, value=14.0)
step_days = col5.number_input("Step days (0 = same as test days)", min_value=0.0, value=0.0)

col6, col7, col8 = st.columns(3)
horizon = col6.number_input("Horizon (bars)", min_value=1, value=1)
buy_threshold = col7.number_input("Buy threshold", min_value=0.0, max_value=1.0, value=0.60)
min_train_rows = col8.number_input("Min train rows", min_value=1, value=200)

include_rule_based_features = st.checkbox("Include rule-based features (vote_sum/confidence)", value=False)

col9, col10 = st.columns(2)
rule_position_mode = col9.selectbox("Rule-based position mode", ["long_only", "long_short"])
initial_capital = col10.number_input("Initial capital", min_value=1.0, value=10_000.0)

with st.expander("Advanced"):
    warmup_buffer_bars = st.number_input("Warm-up buffer bars", min_value=0, value=REQUIRED_WARMUP_BARS)
    risk_free_rate = st.number_input("Risk-free rate (annual)", value=0.0, format="%.4f")

if rule_position_mode == "long_short" and asset_class == "crypto":
    st.warning("long_short is not supported for crypto (Kraken is spot-only) -- switch to long_only.")
    st.stop()

feature_set = "rule-aware (price/indicator features + vote_sum/confidence)" if include_rule_based_features \
    else "independent (price/indicator features only)"
st.caption(f"ML feature set: {feature_set}")

if st.button("Run comparison", type="primary"):
    resolved_cost = default_cost_model(asset_class, symbol)
    note = cost_note_text(resolved_cost)
    if note:
        st.info(f"Cost model note: {note}")

    with open_connection(settings) as conn, st.spinner("Running walk-forward ML vs. rule-based comparison..."):
        report = run_ml_vs_rule_comparison(
            conn, symbol, asset_class, timeframe, to_utc(start_date), to_utc(end_date),
            train_period=pd.Timedelta(days=train_days),
            test_period=pd.Timedelta(days=test_days),
            step=pd.Timedelta(days=step_days) if step_days else None,
            horizon=int(horizon),
            include_rule_based_features=include_rule_based_features,
            buy_threshold=buy_threshold,
            min_train_rows=int(min_train_rows),
            warmup_buffer_bars=int(warmup_buffer_bars),
            initial_capital=initial_capital,
            rule_position_mode=rule_position_mode,
            cost_model=resolved_cost.cost_model,
            risk_free_rate=risk_free_rate,
        )

    if not report.windows:
        st.warning(f"No test windows produced results for {symbol} ({timeframe}) in this range. "
                   f"Check that candles are stored (Fetch Candles page) covering the range plus warm-up and train period.")
        st.stop()

    rows = []
    for wr in report.windows:
        row = {
            "test_start": wr.window.test_start, "test_end": wr.window.test_end,
            "train_rows": wr.train_rows_used, "ml_skip": wr.ml_skip_reason or "",
        }
        if wr.ml_metrics is not None:
            row.update({"ml_trades": wr.ml_metrics.total_trades, "ml_sharpe": wr.ml_metrics.sharpe_ratio,
                        "ml_win_rate": wr.ml_metrics.win_rate, "ml_profit_factor": wr.ml_metrics.profit_factor})
        else:
            row.update({"ml_trades": None, "ml_sharpe": None, "ml_win_rate": None, "ml_profit_factor": None})
        if wr.ml_probability_metrics is not None:
            row.update({"ml_roc_auc": wr.ml_probability_metrics.roc_auc,
                        "ml_brier": wr.ml_probability_metrics.brier_score,
                        "ml_log_loss": wr.ml_probability_metrics.log_loss})
        else:
            row.update({"ml_roc_auc": None, "ml_brier": None, "ml_log_loss": None})
        row.update({"rule_trades": wr.rule_metrics.total_trades, "rule_sharpe": wr.rule_metrics.sharpe_ratio,
                    "rule_win_rate": wr.rule_metrics.win_rate, "rule_profit_factor": wr.rule_metrics.profit_factor})
        rows.append(row)

    st.subheader(f"{len(report.windows)} test window(s)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Combined (stitched out-of-sample)")
    ml_col, rule_col = st.columns(2)

    with ml_col:
        st.markdown("**ML**")
        if report.ml_combined_metrics is not None:
            m = report.ml_combined_metrics
            st.metric("Trades", m.total_trades)
            st.metric("Sharpe", f"{m.sharpe_ratio:.4f}")
            st.metric("Win rate", f"{m.win_rate:.4f}")
            st.metric("Final equity", f"{m.final_equity:.2f}")
        else:
            st.warning("Unavailable -- windows not contiguous, or at least one window skipped model fitting "
                       "(see ml_skip column above).")

    with rule_col:
        st.markdown("**Rule-based**")
        if report.rule_combined_metrics is not None:
            m = report.rule_combined_metrics
            st.metric("Trades", m.total_trades)
            st.metric("Sharpe", f"{m.sharpe_ratio:.4f}")
            st.metric("Win rate", f"{m.win_rate:.4f}")
            st.metric("Final equity", f"{m.final_equity:.2f}")
        else:
            st.warning("Unavailable -- test windows are not contiguous (step != test_days).")

    if report.ml_combined_run is not None or report.rule_combined_run is not None:
        curves = {
            "ML": report.ml_combined_run.equity_curve if report.ml_combined_run is not None else None,
            "Rule-based": report.rule_combined_run.equity_curve if report.rule_combined_run is not None else None,
        }
        st.plotly_chart(build_equity_curve_figure(curves, title="ML vs. rule-based equity"), use_container_width=True)
