"""Candle -> pandas DataFrame conversion.

Keeps signalforge/data/ as the sole owner of Candle-shape concerns, so
indicators.py/signals.py never import Candle at all -- they only ever see
plain OHLCV DataFrames, which is what keeps them asset-class-agnostic.
"""

from __future__ import annotations

import pandas as pd

from signalforge.data.models import Candle

_COLUMNS = ["open", "high", "low", "close", "volume"]


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    if not candles:
        index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return pd.DataFrame(columns=_COLUMNS, index=index, dtype="float64")

    keys = {(c.symbol, c.asset_class, c.timeframe) for c in candles}
    if len(keys) > 1:
        raise ValueError(f"candles_to_frame requires a single symbol/asset_class/timeframe series, got {keys}")

    # Sorted defensively rather than trusting the caller (e.g. query_candles
    # already returns ascending order, but this shouldn't assume it's the
    # only caller) -- every downstream indicator depends on time order.
    ordered = sorted(candles, key=lambda c: c.timestamp)
    index = pd.DatetimeIndex(
        pd.to_datetime([c.timestamp for c in ordered], unit="s", utc=True), name="timestamp"
    )
    return pd.DataFrame(
        {
            "open": [c.open for c in ordered],
            "high": [c.high for c in ordered],
            "low": [c.low for c in ordered],
            "close": [c.close for c in ordered],
            "volume": [c.volume for c in ordered],
        },
        index=index,
    ).astype("float64")
