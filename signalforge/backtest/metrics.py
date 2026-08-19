"""Performance metrics from a BacktestRun's equity curve and trade log.

Risk-free rate defaults to 0.0: idle-cash (flat) periods are modeled as
literally zero return per bar, so a nonzero risk-free rate would penalize
those as negative excess return relative to a benchmark this engine doesn't
actually invest idle cash into -- that mismatch would be misleading.

win_rate/profit_factor use net_pnl (post-fee), not gross_pnl -- using
pre-fee numbers would overstate a marginal strategy's real viability.

NaN vs. 0.0 is deliberate throughout, mirroring how indicators.py/signals.py
already distinguish "genuinely computed and neutral" from "not computable":
no trades or zero-variance returns are the second kind (NaN), a real
meaningful zero (equity that never dipped below its running peak) is the
first kind and stays 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from signalforge.backtest.models import BacktestRun, Trade

_BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
_TRADING_DAYS_PER_YEAR = {"crypto": 365, "forex": 260}  # forex ~5 days/week * 52 -- ties to "weekend gap risk"


@dataclass(frozen=True)
class BacktestResult:
    sharpe_ratio: float
    max_drawdown: float  # negative fraction, e.g. -0.42; 0.0 if equity never dipped below its running peak
    win_rate: float  # fraction of closed trades with net_pnl > 0; NaN if no closed trades
    profit_factor: float  # sum(net_pnl>0) / abs(sum(net_pnl<0)); NaN if no trades; +inf if no losers
    total_trades: int
    total_return: float
    final_equity: float


def annualization_factor(timeframe: str, asset_class: str) -> float:
    if timeframe not in _BARS_PER_DAY or asset_class not in _TRADING_DAYS_PER_YEAR:
        raise ValueError(f"unknown timeframe/asset_class combination: {timeframe!r}/{asset_class!r}")
    return _BARS_PER_DAY[timeframe] * _TRADING_DAYS_PER_YEAR[asset_class]


def sharpe_ratio(equity_curve: pd.Series, *, timeframe: str, asset_class: str, risk_free_rate: float = 0.0) -> float:
    # This Sharpe is only comparable across SignalForge's own backtests using this
    # same bar-return sampling and annualization convention -- not necessarily to
    # a Sharpe reported by another platform using a different trading calendar.
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")  # undefined, not a defensible 0.0
    factor = annualization_factor(timeframe, asset_class)
    excess = returns - (risk_free_rate / factor)
    return (excess.mean() / returns.std(ddof=1)) * math.sqrt(factor)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    return ((equity_curve - running_max) / running_max).min()


def win_rate(trades: list[Trade]) -> float:
    if not trades:
        return float("nan")
    return sum(1 for t in trades if t.net_pnl > 0) / len(trades)  # breakeven does NOT count as a win


def profit_factor(trades: list[Trade]) -> float:
    if not trades:
        return float("nan")
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = sum(t.net_pnl for t in trades if t.net_pnl < 0)
    if gross_profit == 0 and gross_loss == 0:
        return float("nan")  # all-breakeven, genuinely undefined
    if gross_loss == 0:
        return float("inf")  # winners with zero losers -- a real, unbounded PF, not an error
    return gross_profit / abs(gross_loss)


def compute_metrics(run: BacktestRun, *, timeframe: str, asset_class: str, risk_free_rate: float = 0.0) -> BacktestResult:
    final_equity = float(run.equity_curve.iloc[-1])
    return BacktestResult(
        sharpe_ratio=sharpe_ratio(run.equity_curve, timeframe=timeframe, asset_class=asset_class, risk_free_rate=risk_free_rate),
        max_drawdown=max_drawdown(run.equity_curve),
        win_rate=win_rate(run.trades),
        profit_factor=profit_factor(run.trades),
        total_trades=len(run.trades),
        total_return=(final_equity - run.initial_capital) / run.initial_capital,
        final_equity=final_equity,
    )
