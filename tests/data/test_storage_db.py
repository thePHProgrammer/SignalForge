from datetime import datetime, timezone

from signalforge.data.models import Candle
from signalforge.data.storage.db import get_connection, init_db, query_candles, upsert_candles


def make_candle(timestamp: int, close: float = 100.0) -> Candle:
    return Candle(
        symbol="BTC/USD",
        asset_class="crypto",
        timeframe="1h",
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=None,
        source="coingecko",
    )


def test_init_db_is_idempotent(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    init_db(conn)  # must not raise
    conn.close()


def test_upsert_then_query_round_trips(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    candle = make_candle(1_700_000_000)
    written = upsert_candles(conn, [candle])
    assert written == 1

    results = query_candles(
        conn,
        symbol="BTC/USD",
        asset_class="crypto",
        timeframe="1h",
        start=datetime.fromtimestamp(1_699_999_000, tz=timezone.utc),
        end=datetime.fromtimestamp(1_700_001_000, tz=timezone.utc),
    )
    assert len(results) == 1
    assert results[0].timestamp == 1_700_000_000
    assert results[0].close == 100.0


def test_upsert_on_conflict_updates_in_place(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    upsert_candles(conn, [make_candle(1_700_000_000, close=100.0)])
    upsert_candles(conn, [make_candle(1_700_000_000, close=105.0)])

    row_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    assert row_count == 1

    results = query_candles(
        conn,
        symbol="BTC/USD",
        asset_class="crypto",
        timeframe="1h",
        start=datetime.fromtimestamp(1_699_999_000, tz=timezone.utc),
        end=datetime.fromtimestamp(1_700_001_000, tz=timezone.utc),
    )
    assert results[0].close == 105.0


def test_query_candles_filters_by_range(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    upsert_candles(
        conn,
        [make_candle(1_700_000_000), make_candle(1_700_003_600), make_candle(1_700_007_200)],
    )

    results = query_candles(
        conn,
        symbol="BTC/USD",
        asset_class="crypto",
        timeframe="1h",
        start=datetime.fromtimestamp(1_700_000_000, tz=timezone.utc),
        end=datetime.fromtimestamp(1_700_003_600, tz=timezone.utc),
    )
    assert [c.timestamp for c in results] == [1_700_000_000, 1_700_003_600]
