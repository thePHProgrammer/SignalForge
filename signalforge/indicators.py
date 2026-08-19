"""Reusable technical indicators: RSI, MACD, Bollinger Bands, ATR.

Hand-rolled rather than via pandas-ta: pandas-ta's maintainer has said the
project gets archived July 2026 without more sponsorship funding, and it has
a history of breaking on numpy 2.x. Each of these four is well-known and
short enough (~15 lines) to hand-roll and test directly instead of taking on
that dependency risk.

rsi/atr use `ewm(alpha=1/period, adjust=False)` as a Wilder-smoothing
*approximation* -- it seeds from the first observation rather than TA-Lib's
SMA-seeded method. The two converge but aren't byte-identical for the first
few multiples of `period` bars after warm-up. Accepted tradeoff: avoids
hand-rolling a stateful recursive loop for a difference that washes out.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    denom = avg_gain + avg_loss
    # `denom != 0` is True even when denom is NaN (IEEE-754), so this
    # preserves NaN during warm-up without extra masking. A perfectly flat
    # price run (denom == 0) maps to a neutral 50 rather than 0/0 -> NaN.
    return (100 * avg_gain / denom).where(denom != 0, 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line})


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)  # population std, matches Bollinger's/TA-Lib's definition
    return pd.DataFrame({"upper": middle + std_dev * std, "middle": middle, "lower": middle - std_dev * std})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    # Row 0 has no prev_close (NaN), but .max(skipna=True) falls back to
    # high-low rather than propagating NaN -- so true_range has no leading
    # NaN, and atr warms up one row *earlier* than rsi despite using the
    # same smoothing. Real, correct asymmetry, not a bug.
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
