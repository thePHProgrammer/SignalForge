from datetime import datetime, timedelta, timezone

import pytest
import responses

from signalforge.data.adapters import kraken as kraken_module
from signalforge.data.adapters.kraken import KrakenAdapter
from signalforge.data.exceptions import ApiRequestError, SymbolNotFoundError, UnsupportedTimeframeError

# One hour past the last (closed) fixture candle's period end (1700031600 + 3600),
# so all four fixture rows count as closed.
FIXED_NOW = datetime.fromtimestamp(1700031600 + 3600 + 3600, tz=timezone.utc)


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch):
    monkeypatch.setattr(kraken_module, "_utcnow", lambda: FIXED_NOW)


@responses.activate
def test_fetch_ohlcv_happy_path(load_fixture):
    fixture = load_fixture("kraken_ohlc_response.json")
    responses.add(responses.GET, "https://api.kraken.com/0/public/OHLC", json=fixture, status=200)

    adapter = KrakenAdapter()
    start = datetime.fromtimestamp(1700020800, tz=timezone.utc)
    end = datetime.fromtimestamp(1700031600, tz=timezone.utc)

    candles = adapter.fetch_ohlcv("BTC/USD", "1h", start, end)

    assert len(candles) == 4
    assert candles[0].timestamp == 1700020800
    assert candles[0].open == 34500.1
    assert candles[0].volume == 12.4521
    assert candles[0].symbol == "BTC/USD"
    assert candles[0].asset_class == "crypto"
    assert candles[0].timeframe == "1h"
    assert candles[0].source == "kraken"
    assert candles == sorted(candles, key=lambda c: c.timestamp)

    request_params = responses.calls[0].request.params
    assert request_params["pair"] == "XBTUSD"
    assert request_params["interval"] == "60"
    assert request_params["since"] == "1700020800"


@responses.activate
def test_fetch_ohlcv_drops_unclosed_final_candle(load_fixture):
    fixture = load_fixture("kraken_ohlc_response.json")
    responses.add(responses.GET, "https://api.kraken.com/0/public/OHLC", json=fixture, status=200)

    adapter = KrakenAdapter()
    start = datetime.fromtimestamp(1700020800, tz=timezone.utc)
    end = datetime.fromtimestamp(1700031600, tz=timezone.utc)

    # "now" only just past the third candle's close, so the fourth (last) row
    # is still forming and must be dropped even though it's within [start, end].
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kraken_module, "_utcnow", lambda: datetime.fromtimestamp(1700028000 + 3600, tz=timezone.utc))
        candles = adapter.fetch_ohlcv("BTC/USD", "1h", start, end)

    assert [c.timestamp for c in candles] == [1700020800, 1700024400, 1700028000]


@responses.activate
def test_fetch_ohlcv_filters_rows_outside_requested_range(load_fixture):
    fixture = load_fixture("kraken_ohlc_response.json")
    responses.add(responses.GET, "https://api.kraken.com/0/public/OHLC", json=fixture, status=200)

    adapter = KrakenAdapter()
    start = datetime.fromtimestamp(1700024400, tz=timezone.utc)
    end = datetime.fromtimestamp(1700028000, tz=timezone.utc)

    candles = adapter.fetch_ohlcv("BTC/USD", "1h", start, end)

    assert [c.timestamp for c in candles] == [1700024400, 1700028000]


def test_fetch_ohlcv_unknown_symbol_raises():
    adapter = KrakenAdapter()
    with pytest.raises(SymbolNotFoundError):
        adapter.fetch_ohlcv("DOGE/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_fetch_ohlcv_unsupported_timeframe_raises():
    adapter = KrakenAdapter()
    with pytest.raises(UnsupportedTimeframeError):
        adapter.fetch_ohlcv("BTC/USD", "2h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


def test_fetch_ohlcv_naive_datetime_raises():
    adapter = KrakenAdapter()
    with pytest.raises(ValueError):
        adapter.fetch_ohlcv("BTC/USD", "1h", datetime(2023, 11, 15), FIXED_NOW)


@responses.activate
def test_fetch_ohlcv_non_200_raises_api_request_error():
    responses.add(
        responses.GET,
        "https://api.kraken.com/0/public/OHLC",
        json={"error": ["EGeneral:Internal error"]},
        status=520,
    )
    adapter = KrakenAdapter()
    with pytest.raises(ApiRequestError):
        adapter.fetch_ohlcv("BTC/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)


@responses.activate
def test_fetch_ohlcv_kraken_error_array_raises():
    responses.add(
        responses.GET,
        "https://api.kraken.com/0/public/OHLC",
        json={"error": ["EQuery:Unknown asset pair"], "result": {}},
        status=200,
    )
    adapter = KrakenAdapter()
    with pytest.raises(ApiRequestError):
        adapter.fetch_ohlcv("BTC/USD", "1h", FIXED_NOW - timedelta(hours=1), FIXED_NOW)
