import random

import pandas as pd
import pytest

from signalforge.data.frames import candles_to_frame
from signalforge.data.models import Candle
from signalforge.signals import _threshold_vote, confirm_signals, generate_signals


def _random_walk_ohlc(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    price = 100.0
    closes = []
    for _ in range(n):
        price += rng.uniform(-2, 2)
        closes.append(price)
    close = pd.Series(closes)
    return pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0})


def _signal_frame(rows: list[tuple[int, str, float, bool]]) -> pd.DataFrame:
    """rows: (timestamp_seconds, signal, confidence, is_warmup)."""
    index = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows], unit="s", utc=True), name="timestamp")
    return pd.DataFrame(
        {"signal": [r[1] for r in rows], "confidence": [r[2] for r in rows], "is_warmup": [r[3] for r in rows]},
        index=index,
    )


# --- generate_signals: individual vote conditions ---

def test_rsi_vote_bullish_on_strictly_decreasing_series():
    close = pd.Series(range(60, 0, -1), dtype=float)
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    assert (result["rsi_vote"].iloc[14:] == 1.0).all()


def test_rsi_vote_bearish_on_strictly_increasing_series():
    close = pd.Series(range(1, 61), dtype=float)
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    assert (result["rsi_vote"].iloc[14:] == -1.0).all()


def test_macd_vote_bullish_on_sustained_uptrend():
    close = pd.Series(range(1, 81), dtype=float)  # steady linear ramp
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    assert (result["macd_vote"].iloc[40:] == 1.0).all()


def test_macd_vote_bearish_on_sustained_downtrend():
    close = pd.Series(range(80, 0, -1), dtype=float)
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    assert (result["macd_vote"].iloc[40:] == -1.0).all()


def test_bb_vote_bullish_on_downward_spike():
    close = pd.Series([100.0] * 25 + [90.0] + [100.0] * 5)
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    assert result["bb_vote"].iloc[25] == 1.0


def test_bb_vote_bearish_on_upward_spike():
    close = pd.Series([100.0] * 25 + [110.0] + [100.0] * 5)
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    assert result["bb_vote"].iloc[25] == -1.0


# --- boundary/equality cases ---

def test_threshold_vote_exact_boundary_is_neutral():
    values = pd.Series([25.0, 30.0, 50.0, 70.0, 75.0])
    vote = _threshold_vote(values < 30, values > 70, ready=pd.Series([True] * 5))
    assert vote.tolist() == [1.0, 0.0, 0.0, 0.0, -1.0]


def test_flat_series_produces_exact_equality_neutral_votes():
    # A flat close series makes macd_line == signal_line == 0 exactly, and
    # bb upper == middle == lower == close exactly -- genuine, naturally
    # occurring equality-boundary cases (not contrived), on top of the
    # explicit _threshold_vote boundary test above.
    close = pd.Series([100.0] * 50)
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)
    tail = result.iloc[34:]
    assert (tail["macd_vote"] == 0.0).all()
    assert (tail["bb_vote"] == 0.0).all()
    assert (tail["rsi_vote"] == 0.0).all()


# --- warm-up / NaN propagation ---

def test_warmup_columns_and_partial_readiness():
    close = pd.Series(range(1, 31), dtype=float)  # 30 rows: rsi/bb warm up, macd never does
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    result = generate_signals(df)

    assert result["is_warmup"].all()  # macd never warms up in 30 rows
    assert (result["signal"] == "hold").all()
    assert (result["confidence"] == 0.0).all()
    # but rsi/bb, which warm up earlier, still expose real (non-NaN) votes
    assert result["rsi_vote"].iloc[14:].notna().all()
    assert result["bb_vote"].iloc[19:].notna().all()
    assert result["macd_vote"].isna().all()


def test_confidence_is_vote_agreement_strength_property():
    df = _random_walk_ohlc(80)
    result = generate_signals(df)
    ready = ~result["is_warmup"]
    expected = result.loc[ready, "vote_sum"].abs() / 3.0
    pd.testing.assert_series_equal(result.loc[ready, "confidence"], expected, check_names=False)
    assert (result.loc[~ready, "confidence"] == 0.0).all()


def test_signal_always_one_of_three_labels():
    df = _random_walk_ohlc(80)
    result = generate_signals(df)
    assert result["signal"].isin(["buy", "sell", "hold"]).all()
    assert (result.loc[result["is_warmup"], "signal"] == "hold").all()


def test_end_to_end_from_candles():
    rng = random.Random(42)
    price = 100.0
    candles = []
    for i in range(80):
        price += rng.uniform(-2, 2)
        candles.append(
            Candle(
                symbol="BTC/USD", asset_class="crypto", timeframe="1h", timestamp=1_700_000_000 + i * 3600,
                open=price, high=price + 1, low=price - 1, close=price, volume=10.0, source="kraken",
            )
        )

    df = candles_to_frame(candles)
    result = generate_signals(df)

    assert len(result) == len(candles)
    assert result.index.equals(df.index)
    assert result["is_warmup"].iloc[:33].all()
    assert (result.loc[result["is_warmup"], "signal"] == "hold").all()
    assert (result.loc[result["is_warmup"], "confidence"] == 0.0).all()
    assert (result.loc[~result["is_warmup"], "signal"] != "hold").any()


# --- confirm_signals ---

def test_confirm_signals_agreement_confirms():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    confirmation = _signal_frame([(900, "buy", 1.0, False)])
    result = confirm_signals(primary, confirmation)
    assert result["signal"].iloc[0] == "buy"
    assert result["confidence"].iloc[0] == pytest.approx(0.667)


def test_confirm_signals_disagreement_holds():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    confirmation = _signal_frame([(900, "sell", 1.0, False)])
    result = confirm_signals(primary, confirmation)
    assert result["signal"].iloc[0] == "hold"
    assert result["confidence"].iloc[0] == 0.0


def test_confirm_signals_confirmation_hold_vetoes():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    confirmation = _signal_frame([(900, "hold", 0.0, False)])
    result = confirm_signals(primary, confirmation)
    assert result["signal"].iloc[0] == "hold"


def test_confirm_signals_never_uses_future_confirmation_bar():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    confirmation = _signal_frame([(990, "buy", 1.0, False), (1010, "sell", 1.0, False)])
    result = confirm_signals(primary, confirmation)
    assert result["confirmation_signal"].iloc[0] == "buy"  # the 990 bar, never the 1010 one
    assert result["signal"].iloc[0] == "buy"


def test_confirm_signals_uses_latest_eligible_confirmation_bar():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    confirmation = _signal_frame([(900, "sell", 1.0, False), (950, "hold", 0.0, False), (999, "buy", 1.0, False)])
    result = confirm_signals(primary, confirmation)
    assert result["confirmation_signal"].iloc[0] == "buy"
    assert result["signal"].iloc[0] == "buy"


def test_confirm_signals_no_eligible_confirmation_yet_falls_through_to_hold():
    primary = _signal_frame([(100, "buy", 0.667, False), (200, "buy", 0.667, False), (300, "buy", 0.667, False)])
    confirmation = _signal_frame([(250, "buy", 1.0, False)])
    result = confirm_signals(primary, confirmation)
    assert result["signal"].tolist() == ["hold", "hold", "buy"]


def test_confirm_signals_warmup_propagates_via_or():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    confirmation = _signal_frame([(900, "buy", 1.0, True)])  # confirmation still warming up
    result = confirm_signals(primary, confirmation)
    assert result["is_warmup"].iloc[0] == True  # noqa: E712


def test_confirm_signals_rejects_naive_index():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    naive = primary.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError):
        confirm_signals(naive, primary)


def test_confirm_signals_rejects_non_utc_index():
    primary = _signal_frame([(1000, "buy", 0.667, False)])
    non_utc = primary.copy()
    non_utc.index = non_utc.index.tz_convert("US/Eastern")
    with pytest.raises(ValueError):
        confirm_signals(non_utc, primary)


def test_confirm_signals_sorts_unsorted_inputs():
    primary_sorted = _signal_frame([(900, "buy", 0.667, False), (1000, "buy", 0.667, False)])
    primary_shuffled = _signal_frame([(1000, "buy", 0.667, False), (900, "buy", 0.667, False)])
    confirmation = _signal_frame([(890, "buy", 1.0, False), (990, "buy", 1.0, False)])

    expected = confirm_signals(primary_sorted, confirmation)
    actual = confirm_signals(primary_shuffled, confirmation)

    pd.testing.assert_frame_equal(actual.sort_index(), expected)
