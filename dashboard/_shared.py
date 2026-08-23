"""Shared helpers for every dashboard page: settings/DB access, symbol and
timeframe choices, and pure chart-building functions kept separate from any
Streamlit call so they're directly unit-testable (see tests/dashboard/).

Every page opens its own DB connection per action rather than holding one
across Streamlit reruns -- sqlite3 connections aren't safe to share across
Streamlit's rerun/thread model without extra care, and a fresh connection
per action is cheap and matches how every scripts/*.py CLI already works
(open, use, close in a try/finally).
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

import pandas as pd
import plotly.graph_objects as go

from signalforge.config import Settings, load_settings
from signalforge.data.frames import candles_to_frame
from signalforge.data.storage.db import get_connection, init_db, query_candles

CRYPTO_SYMBOLS = ["BTC/USD", "BTC/USDT", "ETH/USD", "ETH/USDT"]
FOREX_SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def get_settings() -> Settings:
    return load_settings()


@contextmanager
def open_connection(settings: Settings | None = None):
    """Opens a fresh connection for one action and closes it on exit -- see
    module docstring for why this isn't cached across reruns instead."""
    settings = settings or get_settings()
    conn = get_connection(settings.db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def to_utc(value: dt.date | dt.datetime) -> dt.datetime:
    """Normalizes a date/datetime (st.date_input returns a naive
    datetime.date) to a UTC-aware datetime at midnight -- query_candles and
    every adapter require timezone-aware datetimes."""
    if isinstance(value, dt.datetime):
        result = value
    else:
        result = dt.datetime(value.year, value.month, value.day)
    return result.replace(tzinfo=dt.timezone.utc) if result.tzinfo is None else result.astimezone(dt.timezone.utc)


def load_price_frame(conn, symbol: str, asset_class: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    candles = query_candles(conn, symbol=symbol, asset_class=asset_class, timeframe=timeframe, start=start, end=end)
    return candles_to_frame(candles)


def cost_note_text(resolved) -> str | None:
    """None when the cost model is a researched/verified figure; otherwise
    the note explaining what's estimated, for an st.info/st.warning caller."""
    return None if resolved.is_verified else resolved.note


def build_candlestick_figure(price: pd.DataFrame, *, title: str, signal: pd.Series | None = None) -> go.Figure:
    """Pure function: OHLC candlesticks, optionally with buy/sell markers
    overlaid from a `signal` Series (values in {"buy","sell","hold"}) aligned
    to price.index. No Streamlit calls here -- kept directly unit-testable."""
    fig = go.Figure()
    if not price.empty:
        fig.add_trace(go.Candlestick(
            x=price.index, open=price["open"], high=price["high"], low=price["low"], close=price["close"],
            name="price",
        ))
    if signal is not None and not price.empty:
        aligned = signal.reindex(price.index)
        buys = aligned[aligned == "buy"]
        sells = aligned[aligned == "sell"]
        if len(buys):
            fig.add_trace(go.Scatter(
                x=buys.index, y=price.loc[buys.index, "low"] * 0.999, mode="markers", name="buy",
                marker=dict(symbol="triangle-up", size=11, color="green"),
            ))
        if len(sells):
            fig.add_trace(go.Scatter(
                x=sells.index, y=price.loc[sells.index, "high"] * 1.001, mode="markers", name="sell",
                marker=dict(symbol="triangle-down", size=11, color="red"),
            ))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=450, margin=dict(t=40))
    return fig


def build_equity_curve_figure(curves: dict[str, pd.Series | None], *, title: str) -> go.Figure:
    """Pure function: one or more named equity curves on a single line chart
    (e.g. {"ML": ml_run.equity_curve, "Rule-based": rule_run.equity_curve})."""
    fig = go.Figure()
    for name, curve in curves.items():
        if curve is not None and len(curve):
            fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=name))
    fig.update_layout(title=title, height=350, yaxis_title="Equity", margin=dict(t=40))
    return fig
