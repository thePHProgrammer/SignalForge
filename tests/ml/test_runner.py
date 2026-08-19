import logging
import math

import pandas as pd
import pytest

from signalforge.data.models import Candle
from signalforge.data.storage.db import get_connection, init_db, upsert_candles
from signalforge.ml.runner import run_ml_vs_rule_comparison

START = pd.Timestamp("2024-01-01", tz="UTC")


def _seed_oscillating_candles(conn, symbol, asset_class, timeframe, start, n, bar, amplitude=5.0, base=100.0):
    candles = []
    for i in range(n):
        price = base + amplitude * math.sin(i / 8.0)
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


def test_ml_and_rule_metrics_computed_over_same_test_window(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 550, pd.Timedelta(hours=1))
    report = run_ml_vs_rule_comparison(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=21),
        train_period=pd.Timedelta(days=10), test_period=pd.Timedelta(days=3),
        min_train_rows=100,
    )
    assert len(report.windows) == 3
    for wr in report.windows:
        assert wr.rule_run.equity_curve.index[0] == wr.window.test_start
        if wr.ml_run is not None:
            assert wr.ml_run.equity_curve.index[0] == wr.window.test_start
            assert wr.ml_probability_metrics is not None
        else:
            assert wr.ml_probability_metrics is None
    assert any(wr.ml_metrics is not None for wr in report.windows)


def test_forced_skip_still_reports_rule_metrics(conn, caplog):
    caplog.set_level(logging.WARNING)
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 550, pd.Timedelta(hours=1))
    report = run_ml_vs_rule_comparison(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=21),
        train_period=pd.Timedelta(days=10), test_period=pd.Timedelta(days=3),
        min_train_rows=100_000,
    )
    assert len(report.windows) == 3
    for wr in report.windows:
        assert wr.ml_metrics is None
        assert wr.ml_run is None
        assert wr.ml_probability_metrics is None
        assert wr.ml_skip_reason == "insufficient_train_rows"
        assert wr.rule_metrics is not None
    assert any("insufficient_train_rows" in record.message for record in caplog.records)


def test_combined_metrics_require_no_skipped_ml_windows(conn):
    # Truncate seeded history so the first window's train range is missing
    # most of its data (forcing an ML skip there) while later windows' train
    # ranges -- which extend further right as the walk-forward cursor
    # advances -- are fully covered.
    seed_start = START + pd.Timedelta(days=7)
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", seed_start, 24 * 17, pd.Timedelta(hours=1))

    report = run_ml_vs_rule_comparison(
        conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=19),
        train_period=pd.Timedelta(days=10), test_period=pd.Timedelta(days=3),
        min_train_rows=80,
    )
    assert len(report.windows) == 3
    assert report.windows[0].ml_metrics is None
    assert report.windows[0].ml_skip_reason == "insufficient_train_rows"
    assert report.windows[1].ml_metrics is not None
    assert report.windows[2].ml_metrics is not None

    assert report.rule_combined_metrics is not None
    assert report.rule_combined_run is not None
    assert report.ml_combined_metrics is None
    assert report.ml_combined_run is None


def test_include_rule_based_features_toggle_runs_end_to_end(conn):
    _seed_oscillating_candles(conn, "BTC/USD", "crypto", "1h", START, 550, pd.Timedelta(hours=1))
    for toggle in (False, True):
        report = run_ml_vs_rule_comparison(
            conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=21),
            train_period=pd.Timedelta(days=10), test_period=pd.Timedelta(days=3),
            min_train_rows=100, include_rule_based_features=toggle,
        )
        assert report.include_rule_based_features is toggle
        assert len(report.windows) == 3


def test_long_short_crypto_raises_before_any_window(conn):
    with pytest.raises(ValueError):
        run_ml_vs_rule_comparison(
            conn, "BTC/USD", "crypto", "1h", START, START + pd.Timedelta(days=21),
            train_period=pd.Timedelta(days=10), test_period=pd.Timedelta(days=3),
            rule_position_mode="long_short",
        )


def test_rule_position_mode_asymmetry_forex_long_short(conn):
    _seed_oscillating_candles(conn, "EUR/USD", "forex", "1h", START, 550, pd.Timedelta(hours=1), amplitude=0.02, base=1.10)
    report = run_ml_vs_rule_comparison(
        conn, "EUR/USD", "forex", "1h", START, START + pd.Timedelta(days=21),
        train_period=pd.Timedelta(days=10), test_period=pd.Timedelta(days=3),
        min_train_rows=100, rule_position_mode="long_short",
    )
    assert report.rule_position_mode == "long_short"

    rule_sides = {t.side for wr in report.windows for t in wr.rule_run.trades}
    ml_sides = {t.side for wr in report.windows if wr.ml_run is not None for t in wr.ml_run.trades}
    assert "short" in rule_sides
    assert ml_sides <= {"long"}
