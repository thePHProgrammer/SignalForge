"""Open-anchored, fixed-horizon forward-direction label, with purge-by-
construction for walk-forward train windows.

The label must match what simulate_trades() (Phase 3's engine, reused
unchanged) actually realizes: a signal at row T fills at row T+1's open, and
this codebase's fixed-horizon ML strategy (see ml/model.py's
to_fixed_horizon_signal) force-exits exactly `horizon` bars later via a
sell at row T+horizon, which itself fills at row T+1+horizon's open. A
label anchored to any close price -- including close[T] as a reference or
close[T+horizon] as an exit -- would score a price move the strategy either
never captured (entry gapped past it) or can't actually realize
(simulate_trades() has no mechanism to fill a signal-driven trade at a
close price at all, only at opens). This label is therefore expressed
purely in opens, matching the engine exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_forward_direction(open_: pd.Series, horizon: int) -> pd.Series:
    """1.0 if open[T+1+horizon] > open[T+1] else 0.0 -- exactly the entry
    and exit prices to_fixed_horizon_signal() realizes (entry at T+1's open
    from a buy at T, forced exit at T+1+horizon's open from a forced sell
    at T+horizon).

    NaN for the trailing horizon+1 rows where either shift reads past the
    end of THIS series. Calling this only on a train-window-local `open_`
    slice (never extending past train_end) makes that trailing-NaN
    behavior the leakage purge, by construction: there is no bar at/after
    train_end in that local slice for the shift to read, so a training
    row's label can never encode information from the test window.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    entry = open_.shift(-1)
    exit_price = open_.shift(-(horizon + 1))
    future_return = exit_price / entry - 1
    label = pd.Series(np.where(future_return > 0, 1.0, 0.0), index=open_.index)
    label[future_return.isna()] = np.nan
    return label
