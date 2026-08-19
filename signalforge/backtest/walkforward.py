"""Rolling train/test window splitter.

Zero dependencies on the rest of backtest/ -- works in abstract half-open
time, timeframe-agnostic, independently testable with plain timestamps.
query_candles is inclusive-both-ends, so translating test_end (exclusive)
into an inclusive query bound is the runner's job (it already knows the
timeframe); kept out of this module so it stays reusable for Phase 4
without dragging in timeframe concerns.

Phase 3 has nothing to fit on train_start/train_end yet -- generate_signals()
has zero fittable parameters. Those fields exist as reusable scaffolding for
Phase 4's ML layer; this module just produces them, it doesn't use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive; == test_start (contiguous, no gap)
    test_start: pd.Timestamp  # inclusive
    test_end: pd.Timestamp  # exclusive


def walk_forward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    step: pd.Timedelta | None = None,
) -> Iterator[WalkForwardWindow]:
    # These raise rather than following the project's usual lazy-validation
    # convention (that's specifically about business config like missing
    # credentials) -- this is an infinite-loop guard, a different tier.
    step = step if step is not None else test_period
    if step <= pd.Timedelta(0):
        raise ValueError(f"step must be positive, got {step}")
    if train_period <= pd.Timedelta(0) or test_period <= pd.Timedelta(0):
        raise ValueError("train_period and test_period must be positive")

    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train_period
        test_start = train_end
        test_end = test_start + test_period
        if test_end > end:
            return  # never yields a truncated trailing partial window
        yield WalkForwardWindow(train_start, train_end, test_start, test_end)
        cursor = cursor + step
