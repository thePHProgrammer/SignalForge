from datetime import datetime, timedelta, timezone

import pytest
import responses

from signalforge.data.adapters import twelvedata as twelvedata_module
from signalforge.data.adapters.twelvedata import TwelveDataAdapter, _parse_twelvedata_datetime
from signalforge.data.exceptions import ApiRequestError, MissingCredentialsError, SymbolNotFoundError, UnsupportedTimeframeError

# One hour past the last fixture candle's period end (1700028000 + 3600),
# so all three fixture rows count as closed.
FIXED_NOW = datetime.fromtimestamp(1700028000 + 3600 + 3600, tz=timezone.utc)


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch):
    monkeypatch.setattr(twelvedata_module, "_utcnow", lambda: FIXED_NOW)


@responses.activate
def test_fetch_ohlcv_happy_path(load_fixture):
    fixture = load_fixture("twelvedata_time_series_response.json")
    responses.add(responses.GET, "https://api.twelvedata.com/time_series", json=fixture, status=200)

    adapter = TwelveDataAdapter(api_key="fake-key")
    start = datetime.fromtimestamp(1700020800, tz=timezone.utc)
    end = datetime.fromtimestamp(1700028000, tz=timezone.utc)

    candles = adapter.fetch_ohlcv("EUR/USD", "1h", start, end)

    assert len(candles) == 3
    assert candles[0].timestamp == 1700020800
    assert candles[0].open == 1.08421
    assert candles[0].close == 1.08477
    assert candles[0].volume is None  # forex responses carry no volume field
    assert candles[0].symbol == "EUR/USD"
    assert candles[0].asset_class == "forex"
    assert candles[0].timeframe == "1h"
    assert candles[0].source == "twelvedata"
    assert candles == sorted(candles, key=lambda c: c.timestamp)

    request_params = responses.calls[0].request.params
    assert request_params["symbol"] == "EUR/USD"
    assert request_params["interval"] == "1h"
    assert request_params["order"] == "ASC"
    assert request_params["timezone"] == "UTC"
    assert request_params["apikey"] == "fake-key"


@responses.activate
def test_fetch_ohlcv_drops_unclosed_final_candle(load_fixture):
    fixture = load_fixture("twelvedata_time_series_response.json")
    responses.add(responses.GET, "https://api.twelvedata.com/time_series", json=fixture, status=200)

    adapter = TwelveDataAdapter(api_key="fake-key")
    start = datetime.fromtimestamp(1700020800, tz=timezone.utc)
    end = datetime.fromtimestamp(1700028000, tz=timezone.utc)

    # "now" only just past the second candle's close (1700024400 + 3600), so
    # the third (last) row is still forming and must be dropped even though
    # it's within [start, end].
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(twelvedata_module, "_utcnow", lambda: datetime.fromtimestamp(1700028000 + 1800, tz=timezone.utc))
        candles = adapter.fetch_ohlcv("EUR/USD", "1h", start, end)

    assert [c.timestamp for c in candles] == [1700020800, 1700024400]


def test_missing_api_key_raises_missing_credentials_error():
    with pytest.raises(MissingCredentialsError):
        TwelveDataAdapter(api_key=None)


def test_unsupported_timeframe_raises():
    adapter = TwelveDataAdapter(api_key="fake-key")
    with pytest.raises(UnsupportedTimeframeError):
        adapter.fetch_ohlcv("EUR/USD", "2h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_malformed_symbol_raises_symbol_not_found_error():
    adapter = TwelveDataAdapter(api_key="fake-key")
    with pytest.raises(SymbolNotFoundError):
        adapter.fetch_ohlcv("EURUSD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_fetch_ohlcv_naive_datetime_raises():
    adapter = TwelveDataAdapter(api_key="fake-key")
    with pytest.raises(ValueError):
        adapter.fetch_ohlcv("EUR/USD", "1h", datetime(2023, 11, 15), FIXED_NOW)


@responses.activate
def test_non_200_raises_api_request_error():
    responses.add(
        responses.GET,
        "https://api.twelvedata.com/time_series",
        json={"code": 401, "message": "Invalid apikey", "status": "error"},
        status=401,
    )
    adapter = TwelveDataAdapter(api_key="fake-key")
    with pytest.raises(ApiRequestError):
        adapter.fetch_ohlcv("EUR/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


@responses.activate
def test_status_error_with_http_200_raises_api_request_error():
    # Documented Twelve Data quirk: some errors (e.g. an invalid interval)
    # come back with HTTP 200 and "status": "error" in the body.
    responses.add(
        responses.GET,
        "https://api.twelvedata.com/time_series",
        json={"code": 400, "message": "Invalid interval provided", "status": "error"},
        status=200,
    )
    adapter = TwelveDataAdapter(api_key="fake-key")
    with pytest.raises(ApiRequestError):
        adapter.fetch_ohlcv("EUR/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_parse_twelvedata_datetime_handles_date_only_format():
    ts = _parse_twelvedata_datetime("2023-11-15")
    assert ts == 1700006400


def test_parse_twelvedata_datetime_handles_datetime_format():
    ts = _parse_twelvedata_datetime("2023-11-15 04:00:00")
    assert ts == 1700020800


@responses.activate
def test_volume_is_parsed_when_present():
    payload = {
        "values": [{"datetime": "2023-11-15 04:00:00", "open": "1.0", "high": "1.1", "low": "0.9", "close": "1.05", "volume": "1234"}],
        "status": "ok",
    }
    responses.add(responses.GET, "https://api.twelvedata.com/time_series", json=payload, status=200)

    adapter = TwelveDataAdapter(api_key="fake-key")
    candles = adapter.fetch_ohlcv("EUR/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)

    assert candles[0].volume == 1234.0
