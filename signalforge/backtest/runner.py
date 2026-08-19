"""Ties the backtest package together: fetches stored candles, computes
signals, simulates trades per walk-forward test window, and reports metrics.

Phase 3 is rolling out-of-sample evaluation of a fixed-parameter strategy,
not walk-forward optimization -- generate_signals() has nothing to fit, so
train_start/train_end are generated (reusable scaffolding for Phase 4) but
never queried or used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from signalforge.backtest.costs import CostModel, default_cost_model
from signalforge.backtest.engine import simulate_trades
from signalforge.backtest.metrics import BacktestResult, compute_metrics
from signalforge.backtest.models import BacktestRun
from signalforge.backtest.walkforward import WalkForwardWindow, walk_forward_windows
from signalforge.data.frames import candles_to_frame
from signalforge.data.storage.db import query_candles
from signalforge.signals import REQUIRED_WARMUP_BARS, confirm_signals, generate_signals

logger = logging.getLogger(__name__)

_TIMEFRAME_TIMEDELTAS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


@dataclass(frozen=True)
class WalkForwardWindowResult:
    window: WalkForwardWindow
    metrics: BacktestResult
    run: BacktestRun  # test-slice-only equity curve + trade log


@dataclass(frozen=True)
class WalkForwardReport:
    windows: list[WalkForwardWindowResult]
    # "Stitched" out-of-sample performance built by compounding each window's own
    # returns -- NOT the equity history of one continuous real account (each
    # window independently restarts at initial_capital before compounding).
    # Only populated when test windows are contiguous and non-overlapping
    # (step == test_period); otherwise both fields are None, since a naive
    # concatenation would double-count overlapping periods or misrepresent
    # gapped periods as temporally consecutive.
    combined_metrics: BacktestResult | None
    combined_run: BacktestRun | None


def load_buffered(conn, symbol, asset_class, timeframe, range_start, range_end, warmup_buffer_bars):
    """Fetches candles covering [range_start - warmup_buffer_bars*bar, range_end),
    with the buffer sized in that timeframe's own calendar time. `range_start`/
    `range_end` are generic -- this is used for both train and test ranges
    (Phase 4 onward), not just test windows."""
    bar = _TIMEFRAME_TIMEDELTAS[timeframe]
    query_start = range_start - warmup_buffer_bars * bar
    query_end = range_end - bar  # range_end is exclusive; the last real candle is one bar before it
    candles = query_candles(conn, symbol=symbol, asset_class=asset_class, timeframe=timeframe,
                             start=query_start, end=query_end)
    if not candles:
        return None, None
    price_df = candles_to_frame(candles)
    return price_df, generate_signals(price_df)


def run_single_window(conn, symbol, asset_class, timeframe, confirm_timeframe, window,
                       warmup_buffer_bars, initial_capital, position_mode, cost_model) -> BacktestRun | None:
    primary_price, primary_signals = load_buffered(
        conn, symbol, asset_class, timeframe, window.test_start, window.test_end, warmup_buffer_bars
    )
    if primary_price is None:
        logger.warning("No candles for %s %s in window %s - %s; skipping.",
                        symbol, timeframe, window.test_start, window.test_end)
        return None

    if confirm_timeframe:
        _, confirmation_signals = load_buffered(
            conn, symbol, asset_class, confirm_timeframe, window.test_start, window.test_end, warmup_buffer_bars
        )
        if confirmation_signals is None:
            logger.warning("No candles for %s %s (confirmation) in window %s - %s; skipping.",
                            symbol, confirm_timeframe, window.test_start, window.test_end)
            return None
        combined_signals = confirm_signals(primary_signals, confirmation_signals)
    else:
        combined_signals = primary_signals

    price = primary_price.loc[window.test_start:]
    signals = combined_signals.loc[window.test_start:]
    if price.empty:
        return None

    return simulate_trades(price, signals, cost_model, initial_capital=initial_capital,
                            position_mode=position_mode, force_close_at_end=True)


def stitch_runs(runs: list[BacktestRun], initial_capital: float) -> BacktestRun:
    returns = pd.concat([r.equity_curve.pct_change().dropna() for r in runs])
    stitched_equity = initial_capital * (1 + returns).cumprod()
    all_trades = [t for r in runs for t in r.trades]
    return BacktestRun(equity_curve=stitched_equity, trades=all_trades, open_position=None, initial_capital=initial_capital)


def run_walk_forward(
    conn,
    symbol: str,
    asset_class: str,
    timeframe: str,
    start,
    end,
    train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    *,
    step: pd.Timedelta | None = None,
    confirm_timeframe: str | None = None,
    warmup_buffer_bars: int = REQUIRED_WARMUP_BARS,
    initial_capital: float = 10_000.0,
    position_mode: Literal["long_only", "long_short"] = "long_only",
    cost_model: CostModel | None = None,
    risk_free_rate: float = 0.0,
) -> WalkForwardReport:
    if position_mode == "long_short" and asset_class == "crypto":
        raise ValueError("long_short position_mode is not supported for crypto (Kraken is spot-only in this codebase)")

    resolved_cost_model = cost_model if cost_model is not None else default_cost_model(asset_class, symbol).cost_model
    step = step if step is not None else test_period
    is_contiguous = step == test_period

    window_results: list[WalkForwardWindowResult] = []
    for window in walk_forward_windows(pd.Timestamp(start), pd.Timestamp(end), train_period, test_period, step):
        run = run_single_window(conn, symbol, asset_class, timeframe, confirm_timeframe, window,
                                 warmup_buffer_bars, initial_capital, position_mode, resolved_cost_model)
        if run is None:
            continue
        metrics = compute_metrics(run, timeframe=timeframe, asset_class=asset_class, risk_free_rate=risk_free_rate)
        window_results.append(WalkForwardWindowResult(window=window, metrics=metrics, run=run))

    combined_run = combined_metrics = None
    if is_contiguous and window_results:
        combined_run = stitch_runs([r.run for r in window_results], initial_capital)
        combined_metrics = compute_metrics(combined_run, timeframe=timeframe, asset_class=asset_class, risk_free_rate=risk_free_rate)

    return WalkForwardReport(windows=window_results, combined_metrics=combined_metrics, combined_run=combined_run)
