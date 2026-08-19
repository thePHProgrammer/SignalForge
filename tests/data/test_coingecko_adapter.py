from datetime import datetime, timedelta, timezone

import pytest
import responses

from signalforge.data.adapters import coingecko as coingecko_module
from signalforge.data.adapters.coingecko import CoinGeckoAdapter
from signalforge.data.exceptions import ApiRequestError, SymbolNotFoundError, UnsupportedTimeframeError

FIXED_NOW = datetime(2023, 11, 15, 6, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch):
    monkeypatch.setattr(coingecko_module, "_utcnow", lambda: FIXED_NOW)


@responses.activate
def test_fetch_ohlcv_happy_path(load_fixture):
    fixture = load_fixture("coingecko_ohlc_response.json")
    responses.add(
        responses.GET,
        "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",
        json=fixture,
        status=200,
    )

    adapter = CoinGeckoAdapter()
    start = FIXED_NOW - timedelta(hours=20)
    end = FIXED_NOW

    candles = adapter.fetch_ohlcv("BTC/USD", "30m", start, end)

    assert len(candles) == 4
    assert candles[0].timestamp == 1700000000
    assert candles[0].symbol == "BTC/USD"
    assert candles[0].asset_class == "crypto"
    assert candles[0].timeframe == "30m"
    assert candles[0].source == "coingecko"
    assert candles[0].volume is None
    assert candles == sorted(candles, key=lambda c: c.timestamp)

    request_params = responses.calls[0].request.params
    assert request_params["vs_currency"] == "usd"
    assert request_params["days"] == "1"


@responses.activate
def test_fetch_ohlcv_filters_rows_outside_requested_range(load_fixture):
    fixture = load_fixture("coingecko_ohlc_response.json")
    responses.add(
        responses.GET,
        "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",
        json=fixture,
        status=200,
    )

    adapter = CoinGeckoAdapter()
    start = datetime.fromtimestamp(1700001800, tz=timezone.utc)
    end = datetime.fromtimestamp(1700003600, tz=timezone.utc)

    candles = adapter.fetch_ohlcv("BTC/USD", "30m", start, end)

    assert [c.timestamp for c in candles] == [1700001800, 1700003600]


def test_fetch_ohlcv_unknown_symbol_raises():
    adapter = CoinGeckoAdapter()
    with pytest.raises(SymbolNotFoundError):
        adapter.fetch_ohlcv("DOGE/USD", "30m", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_fetch_ohlcv_mismatched_timeframe_raises():
    adapter = CoinGeckoAdapter()
    # A 1-day-back window only ever returns 30m candles from CoinGecko's free tier.
    with pytest.raises(UnsupportedTimeframeError):
        adapter.fetch_ohlcv("BTC/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_fetch_ohlcv_window_beyond_30_days_raises():
    adapter = CoinGeckoAdapter()
    with pytest.raises(UnsupportedTimeframeError):
        adapter.fetch_ohlcv("BTC/USD", "4h", FIXED_NOW - timedelta(days=45), FIXED_NOW)


def test_fetch_ohlcv_naive_datetime_raises():
    adapter = CoinGeckoAdapter()
    with pytest.raises(ValueError):
        adapter.fetch_ohlcv("BTC/USD", "30m", datetime(2023, 11, 14), FIXED_NOW)


@responses.activate
def test_fetch_ohlcv_non_200_raises_api_request_error():
    responses.add(
        responses.GET,
        "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",
        json={"error": "rate limited"},
        status=429,
    )

    adapter = CoinGeckoAdapter()
    with pytest.raises(ApiRequestError):
        adapter.fetch_ohlcv("BTC/USD", "30m", FIXED_NOW - timedelta(hours=1), FIXED_NOW)
