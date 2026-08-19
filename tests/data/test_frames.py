import pandas as pd
import pytest

from signalforge.data.frames import candles_to_frame
from signalforge.data.models import Candle


def make_candle(timestamp: int, close: float = 100.0, volume: float | None = 10.0) -> Candle:
    return Candle(
        symbol="BTC/USD",
        asset_class="crypto",
        timeframe="1h",
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        source="kraken",
    )


def test_round_trip_shape_and_values():
    candles = [make_candle(1_700_000_000, close=100.0), make_candle(1_700_003_600, close=105.0)]

    df = candles_to_frame(candles)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df["close"].tolist() == [100.0, 105.0]
    assert df.dtypes.unique().tolist() == [pd.Series(dtype="float64").dtype]


def test_out_of_order_input_is_sorted():
    candles = [make_candle(1_700_003_600, close=105.0), make_candle(1_700_000_000, close=100.0)]

    df = candles_to_frame(candles)

    assert df["close"].tolist() == [100.0, 105.0]
    assert df.index.is_monotonic_increasing


def test_none_volume_becomes_nan():
    candles = [make_candle(1_700_000_000, volume=None)]

    df = candles_to_frame(candles)

    assert pd.isna(df["volume"].iloc[0])
    assert df["volume"].dtype == "float64"


def test_empty_list_returns_empty_frame():
    df = candles_to_frame([])

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_mixed_symbol_raises():
    candles = [
        make_candle(1_700_000_000),
        Candle(
            symbol="ETH/USD", asset_class="crypto", timeframe="1h", timestamp=1_700_003_600,
            open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, source="kraken",
        ),
    ]

    with pytest.raises(ValueError):
        candles_to_frame(candles)


def test_mixed_timeframe_raises():
    candles = [
        make_candle(1_700_000_000),
        Candle(
            symbol="BTC/USD", asset_class="crypto", timeframe="4h", timestamp=1_700_003_600,
            open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, source="kraken",
        ),
    ]

    with pytest.raises(ValueError):
        candles_to_frame(candles)
