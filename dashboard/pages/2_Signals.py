"""Rule-based RSI/MACD/Bollinger signal, with optional multi-timeframe
confirmation, over stored candles -- the dashboard equivalent of
scripts/generate_signals.py. Read-only, no network call.
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

import streamlit as st

from dashboard._shared import (
    CRYPTO_SYMBOLS,
    FOREX_SYMBOLS,
    TIMEFRAMES,
    build_candlestick_figure,
    get_settings,
    load_price_frame,
    open_connection,
    to_utc,
)
from signalforge.signals import confirm_signals, generate_signals

st.set_page_config(page_title="Signals - SignalForge", layout="wide")
st.title("Signals")

settings = get_settings()

asset_class = st.radio("Asset class", ["crypto", "forex"], horizontal=True)
default_symbol = CRYPTO_SYMBOLS[0] if asset_class == "crypto" else FOREX_SYMBOLS[0]
symbol = st.text_input('Symbol ("BASE/QUOTE")', value=default_symbol)

col1, col2 = st.columns(2)
timeframe = col1.selectbox("Primary timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1h"))
confirm_timeframe = col2.selectbox("Confirmation timeframe (optional)", ["None"] + TIMEFRAMES)
confirm_timeframe = None if confirm_timeframe == "None" else confirm_timeframe

col3, col4 = st.columns(2)
start_date = col3.date_input("Start", value=dt.date.today() - dt.timedelta(days=25))
end_date = col4.date_input("End", value=dt.date.today())
history_rows = st.number_input("Rows to show in the history table", min_value=1, max_value=200, value=10)

start, end = to_utc(start_date), to_utc(end_date)

with open_connection(settings) as conn:
    primary_price = load_price_frame(conn, symbol, asset_class, timeframe, start, end)
    confirm_price = None
    if confirm_timeframe:
        confirm_price = load_price_frame(conn, symbol, asset_class, confirm_timeframe, start, end)

if primary_price.empty:
    st.warning(f"No stored candles for {symbol} ({timeframe}) in this range. Fetch candles first.")
    st.stop()

primary_signals = generate_signals(primary_price)

if confirm_timeframe:
    if confirm_price is None or confirm_price.empty:
        st.warning(f"No stored candles for {symbol} ({confirm_timeframe}) in this range. "
                    f"Fetch the confirmation timeframe too.")
        st.stop()
    confirmation_signals = generate_signals(confirm_price)
    combined = confirm_signals(primary_signals, confirmation_signals)
    display_signal_col = "signal"
else:
    combined = primary_price.join(primary_signals)
    display_signal_col = "signal"

latest = combined.iloc[-1]

st.subheader(f"{symbol} ({timeframe}" + (f", confirmed by {confirm_timeframe})" if confirm_timeframe else ")"))

signal_color = {"buy": "green", "sell": "red", "hold": "gray"}[latest[display_signal_col]]
badge_col, meta_col = st.columns([1, 3])
badge_col.markdown(f"### :{signal_color}[{latest[display_signal_col].upper()}]")
meta_col.metric("Confidence", f"{latest['confidence']:.2f}")
if latest["is_warmup"]:
    st.caption("WARM-UP -- not enough history yet for a reliable reading.")

st.plotly_chart(
    build_candlestick_figure(primary_price, title=f"{symbol} ({timeframe})", signal=combined[display_signal_col]),
    use_container_width=True,
)

st.subheader("Recent history")
if confirm_timeframe:
    cols = ["primary_signal", "primary_confidence", "confirmation_signal", "confirmation_confidence",
            "signal", "confidence", "is_warmup"]
else:
    cols = ["close", "rsi", "macd", "macd_hist", "bb_upper", "bb_lower", "atr",
            "vote_sum", "signal", "confidence", "is_warmup"]
st.dataframe(combined[cols].tail(int(history_rows)), use_container_width=True)
