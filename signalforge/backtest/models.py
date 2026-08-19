"""Shared dataclasses for the backtest package, mirroring data/models.py's role
as the common type home so engine.py, metrics.py, and runner.py don't need to
depend on each other just for types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Trade:
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float  # post-slippage/spread fill
    exit_time: pd.Timestamp
    exit_price: float  # post-slippage/spread fill
    quantity: float
    entry_fee: float
    exit_fee: float
    gross_pnl: float  # pre-fee
    net_pnl: float  # gross_pnl - entry_fee - exit_fee
    is_forced_close: bool = False  # True for a window-boundary liquidation, not a signal-driven exit


@dataclass(frozen=True)
class OpenPosition:
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    entry_fee: float
    unrealized_pnl: float  # mark-to-market at the final bar's close; never realized


@dataclass(frozen=True)
class BacktestRun:
    equity_curve: pd.Series  # UTC index matching input price index
    trades: list[Trade]  # closed trades only
    open_position: OpenPosition | None  # always None when force_close_at_end=True
    initial_capital: float
