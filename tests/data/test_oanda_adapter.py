from datetime import datetime, timedelta, timezone

import pytest
import responses

from signalforge.data.adapters.oanda import OANDAAdapter, _parse_oanda_time
from signalforge.data.exceptions import ApiRequestError, MissingCredentialsError, UnsupportedTimeframeError

NOW = datetime(2023, 11, 15, 6, 30, 0, tzinfo=timezone.utc)


@responses.activate
def test_fetch_ohlcv_happy_path_drops_incomplete_candle(load_fixture):
    fixture = load_fixture("oanda_candles_response.json")
    responses.add(
        responses.GET,
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles",
        json=fixture,
        status=200,
    )

    adapter = OANDAAdapter(api_token="fake-token")
    candles = adapter.fetch_ohlcv("EUR/USD", "1h", NOW - timedelta(hours=3), NOW)

    # third fixture candle has "complete": false and must be dropped
    assert len(candles) == 2
    assert candles[0].timestamp == 1700020800  # 2023-11-15T04:00:00Z
    assert candles[0].open == 1.08421
    assert candles[0].volume == 1204
    assert candles[0].symbol == "EUR/USD"
    assert candles[0].asset_class == "forex"
    assert candles[0].source == "oanda"
    assert candles == sorted(candles, key=lambda c: c.timestamp)

    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer fake-token"
    assert "granularity=H1" in request.url


def test_missing_token_raises_missing_credentials_error():
    with pytest.raises(MissingCredentialsError):
        OANDAAdapter(api_token=None)


def test_unsupported_timeframe_raises():
    adapter = OANDAAdapter(api_token="fake-token")
    with pytest.raises(UnsupportedTimeframeError):
        adapter.fetch_ohlcv("EUR/USD", "2h", NOW - timedelta(hours=1), NOW)


@responses.activate
def test_non_200_raises_api_request_error():
    responses.add(
        responses.GET,
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles",
        json={"errorMessage": "Invalid token"},
        status=401,
    )
    adapter = OANDAAdapter(api_token="fake-token")
    with pytest.raises(ApiRequestError):
        adapter.fetch_ohlcv("EUR/USD", "1h", NOW - timedelta(hours=1), NOW)


def test_parse_oanda_time_handles_nanosecond_precision():
    ts = _parse_oanda_time("2023-11-15T04:00:00.000000000Z")
    assert ts == 1700020800


def test_fetch_ohlcv_naive_datetime_raises():
    adapter = OANDAAdapter(api_token="fake-token")
    with pytest.raises(ValueError):
        adapter.fetch_ohlcv("EUR/USD", "1h", datetime(2023, 11, 15), NOW)
