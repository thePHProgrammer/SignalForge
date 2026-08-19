import logging
import math

import pandas as pd
import pytest

import signalforge.backtest.runner as runner_module
from signalforge.backtest.costs import CostModel
from signalforge.backtest.runner import run_walk_forward
from signalforge.data.models import Candle
from signalforge.data.storage.db import get_connection, init_db, upsert_candles

START = pd.Timestamp("2024-01-01", tz="UTC")


def _seed_oscillating_candles(conn, symbol, asset_class, timeframe, start, n, bar):
    candles = []
    for i in range(n):
        price = 100.0 + 5.0 * math.sin(i / 8.0)
        ts = int((start + i * bar).timestamp())
        candles.append(
            Candle(symbol=symbol, asset_class=asset_class, timeframe=timeframe, timestamp=ts,
                   open=price, high=price + 0.5, low=price - 0.5, close=price, volume=1.0, source="test")
        )
    upsert_candles(conn, candles)


@pytest.fixture
def conn(tmp_path):
    c = get_connection(tmp_path / "test.db")
    init_db(c)
    return c


def test_window_count_and_internally_consistent_metrics(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 500, pd.Timedelta(hours=1))
    report = run_walk_forward(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=18),
        train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2),
    )
    assert len(report.windows) == 6
    for wr in report.windows:
        assert wr.metrics.total_trades == len(wr.run.trades)


def test_combined_metrics_populated_when_contiguous(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 500, pd.Timedelta(hours=1))
    report = run_walk_forward(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=18),
        train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2),
    )
    assert report.combined_metrics is not None
    assert report.combined_run is not None


def test_combined_metrics_none_when_overlapping(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 500, pd.Timedelta(hours=1))
    report = run_walk_forward(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=18),
        train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2), step=pd.Timedelta(days=1),
    )
    assert report.combined_metrics is None
    assert report.combined_run is None
    assert len(report.windows) > 0  # per-window results still reported


def test_combined_metrics_none_when_gapped(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 500, pd.Timedelta(hours=1))
    report = run_walk_forward(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=18),
        train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2), step=pd.Timedelta(days=3),
    )
    assert report.combined_metrics is None
    assert report.combined_run is None


def test_warmup_buffer_is_excluded_from_reported_window(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 500, pd.Timedelta(hours=1))
    report = run_walk_forward(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=9),
        train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2),
    )
    first_window = report.windows[0]
    assert first_window.run.equity_curve.index[0] == first_window.window.test_start
    assert not first_window.run.equity_curve.index.min() < first_window.window.test_start


def test_zero_candle_window_is_skipped_with_warning(conn, caplog):
    caplog.set_level(logging.WARNING)
    # Only seed enough data for the first window; later windows have nothing.
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 24 * 8, pd.Timedelta(hours=1))
    report = run_walk_forward(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=18),
        train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2),
    )
    assert len(report.windows) < 6
    assert any("skipping" in record.message for record in caplog.records)


def test_mtf_uses_per_timeframe_warmup_buffer(conn, monkeypatch):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "5m", START, 3000, pd.Timedelta(minutes=5))
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1m", START, 15000, pd.Timedelta(minutes=1))

    calls = []
    original_query_candles = runner_module.query_candles

    def spy(conn, *, symbol, asset_class, timeframe, start, end):
        calls.append((timeframe, start))
        return original_query_candles(conn, symbol=symbol, asset_class=asset_class, timeframe=timeframe, start=start, end=end)

    monkeypatch.setattr(runner_module, "query_candles", spy)

    run_walk_forward(
        conn, "BTC/USD", "crypto", "5m", START, START + pd.Timedelta(days=5),
        train_period=pd.Timedelta(days=1), test_period=pd.Timedelta(hours=6),
        confirm_timeframe="1m", warmup_buffer_bars=50,
    )

    primary_starts = [c[1] for c in calls if c[0] == "5m"]
    confirm_starts = [c[1] for c in calls if c[0] == "1m"]
    assert primary_starts and confirm_starts

    window_test_start = START + pd.Timedelta(days=1)  # first window's test_start
    primary_buffer = window_test_start - primary_starts[0]
    confirm_buffer = window_test_start - confirm_starts[0]
    assert primary_buffer == pd.Timedelta(minutes=50 * 5)
    assert confirm_buffer == pd.Timedelta(minutes=50 * 1)
    assert primary_buffer != confirm_buffer


def test_long_short_crypto_raises_before_any_window(conn):
    with pytest.raises(ValueError):
        run_walk_forward(
            conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=18),
            train_period=pd.Timedelta(days=5), test_period=pd.Timedelta(days=2),
            position_mode="long_short",
        )
