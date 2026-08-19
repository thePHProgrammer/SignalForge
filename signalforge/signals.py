"""Signal generation: discrete-vote buy/sell/hold + confidence, and optional
multi-timeframe confirmation.

`confidence` is vote-agreement strength (how many of the three directional
indicators agree, 0 to 1), not a calibrated probability of future price
movement -- don't conflate the two, especially once Phase 4's ML layer
introduces an actual predicted probability.

The three directional votes don't all measure the same hypothesis: MACD is
trend-following (line-above-signal = bullish momentum), while RSI and
Bollinger Bands are mean-reversion (an extreme reading = expect a reversal).
A "sell" can mean either "trend turned bearish" or "price is due to revert
from overbought," and they can and will disagree during strong trends. This
isn't a defect to fix here -- it's a real property of the strategy that
matters once backtest results get evaluated. The per-indicator vote columns
are kept in the output specifically so this is inspectable per-row, not
just the final signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signalforge.indicators import atr, bollinger_bands, macd, rsi

# MACD (default 12/26/9) is the binding warm-up constraint among the three
# directional indicators -- signal/histogram first turn valid at row
# slow + signal - 1 = 34 (0-indexed row 33). Declared here, once, so Phase 3's
# backtest warm-up buffering has a single source of truth instead of a number
# invented independently downstream.
REQUIRED_WARMUP_BARS = 34


def _threshold_vote(bullish: pd.Series, bearish: pd.Series, ready: pd.Series) -> pd.Series:
    """+1/-1/0 vote from boolean masks, NaN where the underlying indicator isn't warmed up.

    `bullish`/`bearish` come from comparisons against a possibly-NaN indicator
    value, and NaN comparisons silently evaluate to False rather than
    propagating NaN -- so `ready` must be passed explicitly, or a warm-up row
    would default to a neutral 0 vote instead of NaN.
    """
    vote = pd.Series(0.0, index=bullish.index)
    vote[bullish] = 1.0
    vote[bearish] = -1.0
    vote[~ready] = np.nan
    return vote


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]

    rsi_values = rsi(close)
    macd_df = macd(close)
    bb_df = bollinger_bands(close)
    atr_values = atr(high, low, close)

    rsi_vote = _threshold_vote(rsi_values < 30, rsi_values > 70, ready=rsi_values.notna())
    macd_vote = _threshold_vote(
        macd_df["macd"] > macd_df["signal"], macd_df["macd"] < macd_df["signal"],
        ready=macd_df["signal"].notna(),
    )
    bb_vote = _threshold_vote(close < bb_df["lower"], close > bb_df["upper"], ready=bb_df["lower"].notna())

    vote_sum = rsi_vote + macd_vote + bb_vote
    is_warmup = vote_sum.isna()
    # vote_sum is NaN during warm-up, and NaN comparisons are always False, so
    # np.select's default already falls through to "hold" here -- this is the
    # warm-up safety net, not just the tie-break for a genuine zero-sum vote.
    signal = pd.Series(np.select([vote_sum > 0, vote_sum < 0], ["buy", "sell"], default="hold"), index=df.index)
    confidence = (vote_sum.abs() / 3.0).fillna(0.0)

    return pd.DataFrame({
        "rsi": rsi_values,
        "macd": macd_df["macd"], "macd_signal": macd_df["signal"], "macd_hist": macd_df["histogram"],
        "bb_upper": bb_df["upper"], "bb_middle": bb_df["middle"], "bb_lower": bb_df["lower"],
        "atr": atr_values,
        "rsi_vote": rsi_vote, "macd_vote": macd_vote, "bb_vote": bb_vote,
        "vote_sum": vote_sum, "is_warmup": is_warmup,
        "signal": signal, "confidence": confidence,
    }, index=df.index)


def confirm_signals(primary: pd.DataFrame, confirmation: pd.DataFrame) -> pd.DataFrame:
    """Combine two generate_signals() outputs at different timeframes.

    A primary bar closing at time T must only be confirmed using
    confirmation-timeframe data that had actually closed by T -- never a
    confirmation candle still in the future relative to the primary bar.
    Since confirmation is the faster timeframe, there are multiple
    confirmation bars per primary bar; the correct one is the last
    confirmation bar at or before the primary bar's close, which is exactly
    what merge_asof(direction="backward") computes. A plain join/merge would
    either misalign or need manual forward-filling that's easy to get
    subtly wrong in the leaking direction.

    generate_signals()'s only caller sources data through candles_to_frame
    (sorted, UTC-aware by construction), so it leans on that contract. This
    function takes two already-built signal frames that could originate from
    anywhere, so its precondition is enforced explicitly rather than assumed.
    """
    for name, frame in (("primary", primary), ("confirmation", confirmation)):
        if not (isinstance(frame.index, pd.DatetimeIndex) and str(frame.index.tz) == "UTC"):
            raise ValueError(f"{name} must have a UTC-aware DatetimeIndex")
    primary = primary.sort_index()
    confirmation = confirmation.sort_index()

    merged = pd.merge_asof(
        primary.reset_index(), confirmation.reset_index(),
        on="timestamp", direction="backward", suffixes=("", "_confirm"),
    ).set_index("timestamp")

    # NaN comparisons are False, so a primary row with no eligible confirmation
    # data yet (confirmation history starts later than primary's) correctly
    # falls through to "hold"/0.0 below rather than crashing or confirming.
    confirmed = (
        (merged["signal"] == "buy") & (merged["signal_confirm"] == "buy")
    ) | (
        (merged["signal"] == "sell") & (merged["signal_confirm"] == "sell")
    )

    return pd.DataFrame({
        "primary_signal": merged["signal"], "primary_confidence": merged["confidence"],
        "confirmation_signal": merged["signal_confirm"], "confirmation_confidence": merged["confidence_confirm"],
        "is_warmup": merged["is_warmup"] | merged["is_warmup_confirm"],
        "signal": merged["signal"].where(confirmed, "hold"),
        "confidence": merged["confidence"].where(confirmed, 0.0),
    }, index=merged.index)
