"""Engineered ML features: scale-normalized indicator values, multi-window
momentum, and a self-relative volatility regime measure -- causal-only
(every value at row T uses only data at/before T), same discipline as
indicators.py/signals.py.

Raw MACD and Bollinger Band values are absolute price-unit quantities, so a
model trained on BTC at $30k doesn't transfer to BTC at $90k, or to a forex
pair near 1.0 -- exactly the problem generate_signals() never had to solve
because it only ever compares an indicator to itself (macd > signal, close <
bb_lower), never feeds a raw magnitude into a model that has to generalize
across price regimes. Every indicator-derived feature here is therefore
expressed as a ratio (dimensionless), except RSI, which is already 0-100
bounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signalforge.indicators import atr, bollinger_bands, macd, rsi
from signalforge.signals import REQUIRED_WARMUP_BARS  # noqa: F401 -- re-exported; single source of truth

MOMENTUM_WINDOWS: tuple[int, ...] = (1, 5, 10)
VOL_REGIME_WINDOW = 20  # matches bollinger_bands()'s own period=20 convention

FEATURE_COLUMNS_BASE: list[str] = [
    "rsi", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_pct_b", "bb_bandwidth", "atr_norm", "vol_regime",
    *(f"return_{w}" for w in MOMENTUM_WINDOWS),
]
# Always included as a pair, never confidence alone: confidence is 0.0-filled
# (never NaN) during warm-up in generate_signals(), while vote_sum is
# genuinely NaN then. The generic X.notna().all(axis=1) mask used throughout
# this codebase (feature warm-up flag, fit_model's row filter) only catches
# warm-up correctly because vote_sum's NaN is present -- pairing them keeps
# that true even if confidence alone would otherwise slip past it.
FEATURE_COLUMNS_RULE_BASED: list[str] = ["vote_sum", "confidence"]


def feature_columns(include_rule_based_features: bool) -> list[str]:
    return FEATURE_COLUMNS_BASE + (FEATURE_COLUMNS_RULE_BASED if include_rule_based_features else [])


def engineer_features(price: pd.DataFrame, *, rule_signals: pd.DataFrame | None = None) -> pd.DataFrame:
    """price: an OHLCV frame as candles_to_frame() produces. rule_signals:
    generate_signals(price) output, passed in (not recomputed here) when
    the caller wants vote_sum/confidence as extra features -- omit for the
    default fully-independent feature set. Always returns a frame with the
    same index/length as `price` (NaN, not dropped rows, during warm-up --
    same invariant as generate_signals()).
    """
    close, high, low = price["close"], price["high"], price["low"]

    rsi_values = rsi(close)
    macd_df = macd(close)
    bb_df = bollinger_bands(close)
    atr_values = atr(high, low, close)

    macd_norm = macd_df["macd"] / close
    macd_signal_norm = macd_df["signal"] / close
    macd_hist_norm = macd_df["histogram"] / close
    bb_pct_b = (close - bb_df["lower"]) / (bb_df["upper"] - bb_df["lower"])
    bb_bandwidth = (bb_df["upper"] - bb_df["lower"]) / bb_df["middle"]
    atr_norm = atr_values / close
    vol_regime = atr_norm / atr_norm.rolling(window=VOL_REGIME_WINDOW, min_periods=VOL_REGIME_WINDOW).mean()

    features = {
        "rsi": rsi_values,
        "macd_norm": macd_norm, "macd_signal_norm": macd_signal_norm, "macd_hist_norm": macd_hist_norm,
        "bb_pct_b": bb_pct_b, "bb_bandwidth": bb_bandwidth,
        "atr_norm": atr_norm, "vol_regime": vol_regime,
        **{f"return_{w}": close.pct_change(w) for w in MOMENTUM_WINDOWS},
    }
    if rule_signals is not None:
        features["vote_sum"] = rule_signals["vote_sum"]
        features["confidence"] = rule_signals["confidence"]

    result = pd.DataFrame(features, index=price.index)
    # General safety net rather than case-by-case zero-denominator reasoning:
    # real market prices never hit most theoretical zero-denominator cases
    # (close/middle are never exactly zero), but this closes the class of
    # bug uniformly and guards against inf slipping past the notna() mask
    # fit_model relies on -- unlike NaN, inf is NOT caught by .notna().
    return result.replace([np.inf, -np.inf], np.nan)
