import datetime as dt

import pandas as pd
import pytest

from dashboard._shared import build_candlestick_figure, build_equity_curve_figure, cost_note_text, to_utc


def _price_frame(n=5):
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"), name="timestamp")
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.5] * n, "volume": [1.0] * n},
        index=index,
    )


def test_to_utc_naive_date_becomes_utc_midnight():
    result = to_utc(dt.date(2024, 3, 1))
    assert result == dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)


def test_to_utc_naive_datetime_gets_utc_attached():
    result = to_utc(dt.datetime(2024, 3, 1, 14, 30))
    assert result == dt.datetime(2024, 3, 1, 14, 30, tzinfo=dt.timezone.utc)


def test_to_utc_aware_datetime_converted_not_reinterpreted():
    tz = dt.timezone(dt.timedelta(hours=-5))
    result = to_utc(dt.datetime(2024, 3, 1, 9, 0, tzinfo=tz))
    assert result == dt.datetime(2024, 3, 1, 14, 0, tzinfo=dt.timezone.utc)


class _FakeResolvedCost:
    def __init__(self, is_verified, note):
        self.is_verified = is_verified
        self.note = note


def test_cost_note_text_none_when_verified():
    assert cost_note_text(_FakeResolvedCost(is_verified=True, note="researched")) is None


def test_cost_note_text_returns_note_when_unverified():
    assert cost_note_text(_FakeResolvedCost(is_verified=False, note="an estimate")) == "an estimate"


def test_build_candlestick_figure_empty_price_produces_no_candlestick_trace():
    fig = build_candlestick_figure(pd.DataFrame(columns=["open", "high", "low", "close"]), title="empty")
    assert not any(trace.type == "candlestick" for trace in fig.data)


def test_build_candlestick_figure_adds_one_candlestick_trace():
    fig = build_candlestick_figure(_price_frame(), title="BTC/USD")
    candlestick_traces = [t for t in fig.data if t.type == "candlestick"]
    assert len(candlestick_traces) == 1
    assert len(candlestick_traces[0].close) == 5


def test_build_candlestick_figure_adds_buy_and_sell_marker_traces():
    price = _price_frame()
    signal = pd.Series(["hold", "buy", "hold", "sell", "hold"], index=price.index)
    fig = build_candlestick_figure(price, title="BTC/USD", signal=signal)
    scatter_traces = {t.name: t for t in fig.data if t.type == "scatter"}
    assert set(scatter_traces) == {"buy", "sell"}
    assert len(scatter_traces["buy"].x) == 1
    assert len(scatter_traces["sell"].x) == 1


def test_build_candlestick_figure_no_signal_means_no_marker_traces():
    fig = build_candlestick_figure(_price_frame(), title="BTC/USD")
    assert not any(t.type == "scatter" for t in fig.data)


def test_build_equity_curve_figure_skips_none_and_empty_curves():
    price = _price_frame()
    curve = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0], index=price.index)
    fig = build_equity_curve_figure({"ML": curve, "Rule": None, "Empty": pd.Series(dtype=float)}, title="Equity")
    assert len(fig.data) == 1
    assert fig.data[0].name == "ML"


def test_build_equity_curve_figure_multiple_curves_both_present():
    price = _price_frame()
    curve_a = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0], index=price.index)
    curve_b = pd.Series([100.0, 100.5, 100.2, 100.9, 101.5], index=price.index)
    fig = build_equity_curve_figure({"ML": curve_a, "Rule-based": curve_b}, title="Equity")
    names = {t.name for t in fig.data}
    assert names == {"ML", "Rule-based"}
