"""Fetch OHLCV candles from Kraken (crypto) or Twelve Data / OANDA (forex)
and store them in the local database -- the dashboard equivalent of
scripts/fetch_candles.py, calling the same adapters directly.
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
from signalforge.data.adapters.kraken import KrakenAdapter
from signalforge.data.adapters.oanda import OANDAAdapter
from signalforge.data.adapters.twelvedata import TwelveDataAdapter
from signalforge.data.exceptions import DataAdapterError
from signalforge.data.storage.db import upsert_candles

st.set_page_config(page_title="Fetch Candles - SignalForge", layout="wide")
st.title("Fetch Candles")

settings = get_settings()

asset_class = st.radio("Asset class", ["crypto", "forex"], horizontal=True)
default_symbol = CRYPTO_SYMBOLS[0] if asset_class == "crypto" else FOREX_SYMBOLS[0]
known_symbols = CRYPTO_SYMBOLS if asset_class == "crypto" else FOREX_SYMBOLS
symbol = st.text_input('Symbol ("BASE/QUOTE")', value=default_symbol)
st.caption(f"Known-good symbols for {asset_class}: {', '.join(known_symbols)}")

timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1h"))

col1, col2 = st.columns(2)
start_date = col1.date_input("Start", value=dt.date.today() - dt.timedelta(days=25))
end_date = col2.date_input("End", value=dt.date.today())

forex_provider = "twelvedata"
if asset_class == "forex":
    forex_provider = st.radio("Forex data provider", ["twelvedata", "oanda"], horizontal=True,
                               help="Twelve Data is the default (free signup, no account approval or "
                                    "country restriction). OANDA requires an OANDA account.")

if asset_class == "crypto":
    st.info("Kraken's free endpoint returns at most ~720 candles per fetch, regardless of the date range "
            "requested (~30 days at 1h). Fetch in ~29-day chunks for a longer backfill.")

if st.button("Fetch", type="primary"):
    start = to_utc(start_date)
    end = to_utc(end_date)
    try:
        if asset_class == "crypto":
            adapter = KrakenAdapter()
        elif forex_provider == "oanda":
            adapter = OANDAAdapter(api_token=settings.oanda_api_token, environment=settings.oanda_environment)
        else:
            adapter = TwelveDataAdapter(api_key=settings.twelve_data_api_key)

        with st.spinner(f"Fetching {symbol} ({timeframe}) from {start.date()} to {end.date()}..."):
            candles = adapter.fetch_ohlcv(symbol, timeframe, start, end)
            with open_connection(settings) as conn:
                written = upsert_candles(conn, candles)

        if candles:
            st.success(f"Fetched {len(candles)} candles, "
                       f"{dt.datetime.fromtimestamp(candles[0].timestamp, tz=dt.timezone.utc)} -> "
                       f"{dt.datetime.fromtimestamp(candles[-1].timestamp, tz=dt.timezone.utc)}. "
                       f"Wrote {written} rows to {settings.db_path}")
        else:
            st.warning("Fetched 0 candles for the requested range.")
    except DataAdapterError as exc:
        st.error(str(exc))

st.divider()
st.subheader("Currently stored candles")
with open_connection(settings) as conn:
    price = load_price_frame(conn, symbol, asset_class, timeframe, to_utc(start_date), to_utc(end_date))

if price.empty:
    st.caption("No candles stored yet for this symbol/timeframe/range.")
else:
    st.plotly_chart(build_candlestick_figure(price, title=f"{symbol} ({timeframe})"), use_container_width=True)
