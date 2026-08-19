import math
import random

import pandas as pd
import pytest

from signalforge.indicators import atr, bollinger_bands, macd, rsi


def _oracle_ewm_adjust_false(values: list[float], alpha: float, min_periods: int) -> list[float]:
    """Independent reference for pandas' ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean(),
    for the leading-NaN-then-clean-data shape every indicator here produces: skips leading NaN, seeds
    from the first valid value, then recurses. Not imported from production code."""
    result = [float("nan")] * len(values)
    ema = None
    count = 0
    for i, v in enumerate(values):
        if v != v:  # NaN
            continue
        ema = v if ema is None else (1 - alpha) * ema + alpha * v
        count += 1
        if count >= min_periods:
            result[i] = ema
    return result


def _oracle_rsi(closes: list[float], period: int = 14) -> list[float]:
    n = len(closes)
    gains = [float("nan")] + [max(closes[i] - closes[i - 1], 0.0) for i in range(1, n)]
    losses = [float("nan")] + [max(closes[i - 1] - closes[i], 0.0) for i in range(1, n)]
    avg_gain = _oracle_ewm_adjust_false(gains, 1 / period, period)
    avg_loss = _oracle_ewm_adjust_false(losses, 1 / period, period)
    result = []
    for g, l in zip(avg_gain, avg_loss):
        if g != g:
            result.append(float("nan"))
        else:
            denom = g + l
            result.append(50.0 if denom == 0 else 100 * g / denom)
    return result


def _oracle_macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _oracle_ewm_adjust_false(closes, 2 / (fast + 1), fast)
    ema_slow = _oracle_ewm_adjust_false(closes, 2 / (slow + 1), slow)
    macd_line = [f - s if (f == f and s == s) else float("nan") for f, s in zip(ema_fast, ema_slow)]
    signal_line = _oracle_ewm_adjust_false(macd_line, 2 / (signal + 1), signal)
    histogram = [m - s if (m == m and s == s) else float("nan") for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def _oracle_atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    n = len(high)
    true_range = []
    for i in range(n):
        candidates = [high[i] - low[i]]
        if i > 0:
            candidates.append(abs(high[i] - close[i - 1]))
            candidates.append(abs(low[i] - close[i - 1]))
        true_range.append(max(candidates))
    return _oracle_ewm_adjust_false(true_range, 1 / period, period)


def _random_walk_closes(n: int, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    price = 100.0
    closes = []
    for _ in range(n):
        price += rng.uniform(-2, 2)
        closes.append(price)
    return closes


# --- oracle-loop reference tests ---

def test_rsi_matches_oracle():
    closes = _random_walk_closes(60)
    actual = rsi(pd.Series(closes))
    expected = pd.Series(_oracle_rsi(closes))
    pd.testing.assert_series_equal(actual.reset_index(drop=True), expected, check_exact=False, rtol=1e-6, check_names=False)


def test_macd_matches_oracle():
    closes = _random_walk_closes(60)
    actual = macd(pd.Series(closes))
    exp_macd, exp_signal, exp_hist = _oracle_macd(closes)
    pd.testing.assert_series_equal(actual["macd"].reset_index(drop=True), pd.Series(exp_macd), check_exact=False, rtol=1e-6, check_names=False)
    pd.testing.assert_series_equal(actual["signal"].reset_index(drop=True), pd.Series(exp_signal), check_exact=False, rtol=1e-6, check_names=False)
    pd.testing.assert_series_equal(actual["histogram"].reset_index(drop=True), pd.Series(exp_hist), check_exact=False, rtol=1e-6, check_names=False)


def test_atr_matches_oracle():
    rng = random.Random(7)
    closes = _random_walk_closes(60, seed=7)
    highs = [c + rng.uniform(0.5, 2.0) for c in closes]
    lows = [c - rng.uniform(0.5, 2.0) for c in closes]
    actual = atr(pd.Series(highs), pd.Series(lows), pd.Series(closes))
    expected = pd.Series(_oracle_atr(highs, lows, closes))
    pd.testing.assert_series_equal(actual.reset_index(drop=True), expected, check_exact=False, rtol=1e-6, check_names=False)


# --- hand-verifiable boundary cases ---

def test_rsi_strictly_increasing_is_exactly_100():
    closes = pd.Series(range(1, 30), dtype=float)
    result = rsi(closes, period=14)
    assert (result.iloc[14:] == 100.0).all()


def test_rsi_strictly_decreasing_is_exactly_0():
    closes = pd.Series(range(30, 1, -1), dtype=float)
    result = rsi(closes, period=14)
    assert (result.iloc[14:] == 0.0).all()


def test_rsi_flat_is_exactly_50():
    closes = pd.Series([100.0] * 20)
    result = rsi(closes, period=14)
    assert (result.iloc[14:] == 50.0).all()


def test_bollinger_bands_hand_computed():
    closes = pd.Series([10.0, 12.0, 14.0, 12.0, 10.0])
    result = bollinger_bands(closes, period=3, std_dev=2.0)

    assert result["middle"].iloc[2] == pytest.approx(12.0, rel=1e-4)
    assert result["upper"].iloc[2] == pytest.approx(15.26599, rel=1e-4)
    assert result["lower"].iloc[2] == pytest.approx(8.73401, rel=1e-4)

    assert result["middle"].iloc[3] == pytest.approx(12.6667, rel=1e-4)
    assert result["upper"].iloc[3] == pytest.approx(14.5523, rel=1e-4)
    assert result["lower"].iloc[3] == pytest.approx(10.781, rel=1e-3)

    assert result["upper"].iloc[4] == pytest.approx(15.26599, rel=1e-4)
    assert result["lower"].iloc[4] == pytest.approx(8.73401, rel=1e-4)


def test_atr_hand_computed_with_spike():
    high = pd.Series([101.0, 101.0, 101.0, 104.0, 103.0])
    low = pd.Series([99.0, 99.0, 99.0, 100.0, 101.0])
    close = pd.Series([100.0, 100.0, 100.0, 102.0, 102.0])

    result = atr(high, low, close, period=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0, rel=1e-6)
    assert result.iloc[3] == pytest.approx(8 / 3, rel=1e-6)
    assert result.iloc[4] == pytest.approx(22 / 9, rel=1e-6)


# --- warm-up NaN-position assertions ---

def test_warmup_positions_match_expected_row_table():
    closes = pd.Series(_random_walk_closes(60))
    highs = closes + 1.0
    lows = closes - 1.0

    rsi_values = rsi(closes, period=14)
    assert rsi_values.iloc[:14].isna().all()
    assert rsi_values.iloc[14:].notna().all()

    atr_values = atr(highs, lows, closes, period=14)
    assert atr_values.iloc[:13].isna().all()
    assert atr_values.iloc[13:].notna().all()

    bb = bollinger_bands(closes, period=20)
    assert bb["middle"].iloc[:19].isna().all()
    assert bb["middle"].iloc[19:].notna().all()

    macd_df = macd(closes)
    assert macd_df["signal"].iloc[:33].isna().all()
    assert macd_df["signal"].iloc[33:].notna().all()
    # explicit MACD warm-up row assertion, verified by test rather than just documented
    assert math.isnan(macd_df["signal"].iloc[32])
    assert not math.isnan(macd_df["signal"].iloc[33])
