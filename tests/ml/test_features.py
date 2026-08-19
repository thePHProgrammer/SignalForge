import random

import numpy as np
import pandas as pd

from signalforge.indicators import atr, bollinger_bands, macd
from signalforge.ml.features import (
    FEATURE_COLUMNS_BASE,
    MOMENTUM_WINDOWS,
    engineer_features,
    feature_columns,
)
from signalforge.signals import generate_signals


def _random_walk_price(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    price = 100.0
    closes = []
    for _ in range(n):
        price += rng.uniform(-2, 2)
        closes.append(price)
    close = pd.Series(closes)
    return pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0})


# --- no look-ahead ---

def test_no_lookahead_truncated_matches_full():
    price = _random_walk_price(100)
    full = engineer_features(price)
    truncated = engineer_features(price.iloc[:80])
    pd.testing.assert_frame_equal(full.iloc[:80], truncated)


# --- formulas match the underlying (already-tested) indicators, as ratios ---

def test_macd_norm_matches_indicator_ratio():
    price = _random_walk_price(80)
    features = engineer_features(price)
    macd_df = macd(price["close"])
    pd.testing.assert_series_equal(features["macd_norm"], macd_df["macd"] / price["close"], check_names=False)
    pd.testing.assert_series_equal(features["macd_signal_norm"], macd_df["signal"] / price["close"], check_names=False)
    pd.testing.assert_series_equal(features["macd_hist_norm"], macd_df["histogram"] / price["close"], check_names=False)


def test_bb_pct_b_and_bandwidth_match_indicator_ratio():
    price = _random_walk_price(80)
    features = engineer_features(price)
    bb = bollinger_bands(price["close"])
    expected_pct_b = (price["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
    expected_bandwidth = (bb["upper"] - bb["lower"]) / bb["middle"]
    pd.testing.assert_series_equal(features["bb_pct_b"], expected_pct_b, check_names=False)
    pd.testing.assert_series_equal(features["bb_bandwidth"], expected_bandwidth, check_names=False)


def test_atr_norm_matches_indicator_ratio():
    price = _random_walk_price(80)
    features = engineer_features(price)
    expected = atr(price["high"], price["low"], price["close"]) / price["close"]
    pd.testing.assert_series_equal(features["atr_norm"], expected, check_names=False)


def test_return_windows_match_pct_change():
    price = _random_walk_price(80)
    features = engineer_features(price)
    for w in MOMENTUM_WINDOWS:
        pd.testing.assert_series_equal(features[f"return_{w}"], price["close"].pct_change(w), check_names=False)


# --- degenerate denominators ---

def test_bb_pct_b_is_nan_on_flat_price_not_warmup():
    close = pd.Series([100.0] * 60)
    price = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    features = engineer_features(price)
    # past BB's own warm-up (row 19) but flat -> upper==lower==close -> 0/0 -> NaN,
    # not 0 or 1 -- mathematical degeneracy, not a warm-up artifact
    assert features["bb_pct_b"].iloc[19:].isna().all()
    # bb_bandwidth's denominator is `middle`, never zero for real price data, so it
    # stays well-defined (0.0) at the same flat rows -- a useful contrast
    assert (features["bb_bandwidth"].iloc[19:] == 0.0).all()


def test_no_feature_is_ever_inf():
    price = _random_walk_price(80)
    features = engineer_features(price)
    numeric = features.to_numpy(dtype="float64")
    assert not np.isinf(numeric).any()


# --- volatility regime ---

def test_vol_regime_near_one_under_constant_volatility():
    rng = random.Random(7)
    price_val = 100.0
    closes = []
    for _ in range(100):
        price_val += rng.uniform(-0.5, 0.5)
        closes.append(price_val)
    close = pd.Series(closes)
    price = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0})
    features = engineer_features(price)
    tail = features["vol_regime"].iloc[-30:]
    assert ((tail - 1.0).abs() < 0.4).all()


def test_vol_regime_detects_expanding_volatility():
    rng = random.Random(1)
    price_val = 100.0
    closes = []
    for _ in range(60):
        price_val += rng.uniform(-0.2, 0.2)
        closes.append(price_val)
    for _ in range(40):
        price_val += rng.uniform(-3.0, 3.0)
        closes.append(price_val)
    close = pd.Series(closes)
    price = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0})
    features = engineer_features(price)

    low_vol_regime = features["vol_regime"].iloc[40:59].mean()
    transition_vol_regime = features["vol_regime"].iloc[60:70].mean()
    assert transition_vol_regime > low_vol_regime


# --- warm-up positions ---

def test_warmup_positions_match_derivation_table():
    price = _random_walk_price(80)
    features = engineer_features(price)
    assert features["atr_norm"].iloc[:13].isna().all() and features["atr_norm"].iloc[13:].notna().all()
    assert features["rsi"].iloc[:14].isna().all() and features["rsi"].iloc[14:].notna().all()
    assert features["bb_pct_b"].iloc[:19].isna().all() and features["bb_pct_b"].iloc[19:].notna().all()
    assert features["vol_regime"].iloc[:32].isna().all() and features["vol_regime"].iloc[32:].notna().all()
    assert features["macd_signal_norm"].iloc[:33].isna().all() and features["macd_signal_norm"].iloc[33:].notna().all()


def test_feature_warmup_binding_constraint_with_and_without_rule_features():
    price = _random_walk_price(80)

    features_indep = engineer_features(price)
    mask_indep = features_indep[feature_columns(False)].isna().any(axis=1)
    assert mask_indep.iloc[:33].all() and not mask_indep.iloc[33:].any()

    rule_signals = generate_signals(price)
    features_rule = engineer_features(price, rule_signals=rule_signals)
    mask_rule = features_rule[feature_columns(True)].isna().any(axis=1)
    assert mask_rule.iloc[:33].all() and not mask_rule.iloc[33:].any()


def test_rule_based_confidence_warmup_still_flagged_via_paired_vote_sum_nan():
    price = _random_walk_price(80)
    rule_signals = generate_signals(price)
    features = engineer_features(price, rule_signals=rule_signals)

    # confidence alone is 0.0-filled (never NaN) during warm-up -- would NOT
    # flag warm-up if used by itself
    assert not features["confidence"].iloc[:33].isna().any()
    assert (features["confidence"].iloc[:33] == 0.0).all()
    # but vote_sum IS NaN during warm-up, so the paired mask still catches it
    assert features["vote_sum"].iloc[:33].isna().all()
    combined_mask = features[feature_columns(True)].isna().any(axis=1)
    assert combined_mask.iloc[:33].all()


# --- toggle ---

def test_include_rule_based_features_toggle():
    price = _random_walk_price(80)

    features_without = engineer_features(price)
    assert "vote_sum" not in features_without.columns
    assert "confidence" not in features_without.columns

    rule_signals = generate_signals(price)
    features_with = engineer_features(price, rule_signals=rule_signals)
    assert "vote_sum" in features_with.columns
    assert "confidence" in features_with.columns
    pd.testing.assert_series_equal(features_with["vote_sum"], rule_signals["vote_sum"], check_names=False)
    pd.testing.assert_series_equal(features_with["confidence"], rule_signals["confidence"], check_names=False)


def test_feature_columns_matches_dataframe_columns():
    price = _random_walk_price(80)
    rule_signals = generate_signals(price)
    for include in (False, True):
        features = engineer_features(price, rule_signals=rule_signals if include else None)
        assert list(features.columns) == feature_columns(include)
    assert set(FEATURE_COLUMNS_BASE) <= set(feature_columns(False))
