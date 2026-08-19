import math

import pandas as pd
import pytest

from signalforge.backtest.metrics import (
    annualization_factor,
    compute_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from signalforge.backtest.models import BacktestRun, Trade


def _trade(net_pnl: float) -> Trade:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return Trade(
        side="long", entry_time=ts, entry_price=100.0, exit_time=ts, exit_price=100.0,
        quantity=1.0, entry_fee=0.0, exit_fee=0.0, gross_pnl=net_pnl, net_pnl=net_pnl,
    )


# --- annualization_factor ---

def test_annualization_factor_crypto_daily():
    assert annualization_factor("1d", "crypto") == 365


def test_annualization_factor_forex_hourly():
    assert annualization_factor("1h", "forex") == 24 * 260


def test_annualization_factor_unknown_timeframe_raises():
    with pytest.raises(ValueError):
        annualization_factor("2h", "crypto")


def test_annualization_factor_unknown_asset_class_raises():
    with pytest.raises(ValueError):
        annualization_factor("1d", "stocks")


# --- sharpe_ratio ---

def test_sharpe_ratio_hand_computed():
    equity = pd.Series([100.0, 102.0, 101.0, 105.0])
    returns = equity.pct_change().dropna()
    expected = (returns.mean() / returns.std(ddof=1)) * math.sqrt(annualization_factor("1d", "crypto"))
    assert sharpe_ratio(equity, timeframe="1d", asset_class="crypto") == pytest.approx(expected)


def test_sharpe_ratio_single_point_is_nan():
    assert math.isnan(sharpe_ratio(pd.Series([100.0]), timeframe="1d", asset_class="crypto"))


def test_sharpe_ratio_zero_variance_is_nan():
    assert math.isnan(sharpe_ratio(pd.Series([100.0, 100.0, 100.0]), timeframe="1d", asset_class="crypto"))


def test_sharpe_ratio_risk_free_rate_reduces_result():
    equity = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0])
    zero_rf = sharpe_ratio(equity, timeframe="1d", asset_class="crypto", risk_free_rate=0.0)
    with_rf = sharpe_ratio(equity, timeframe="1d", asset_class="crypto", risk_free_rate=0.10)
    assert with_rf < zero_rf


# --- max_drawdown ---

def test_max_drawdown_hand_computed():
    equity = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_never_dipped_is_zero():
    equity = pd.Series([100.0, 105.0, 110.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


# --- win_rate / profit_factor ---

def test_win_rate_excludes_breakeven():
    trades = [_trade(10), _trade(-5), _trade(0)]
    assert win_rate(trades) == pytest.approx(1 / 3)


def test_win_rate_empty_is_nan():
    assert math.isnan(win_rate([]))


def test_profit_factor_all_winners_is_inf():
    assert profit_factor([_trade(10), _trade(5)]) == float("inf")


def test_profit_factor_empty_is_nan():
    assert math.isnan(profit_factor([]))


def test_profit_factor_all_breakeven_is_nan():
    assert math.isnan(profit_factor([_trade(0), _trade(0)]))


def test_profit_factor_mixed():
    trades = [_trade(30), _trade(-10), _trade(-5)]
    assert profit_factor(trades) == pytest.approx(30 / 15)


# --- compute_metrics integration ---

def test_compute_metrics_integration():
    equity = pd.Series(
        [10_000.0, 10_100.0, 9_900.0, 10_500.0],
        index=pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC"),
    )
    run = BacktestRun(equity_curve=equity, trades=[_trade(100), _trade(-50)], open_position=None, initial_capital=10_000.0)
    result = compute_metrics(run, timeframe="1d", asset_class="crypto")

    assert result.total_trades == 2
    assert result.final_equity == pytest.approx(10_500.0)
    assert result.total_return == pytest.approx(0.05)
    assert result.win_rate == pytest.approx(0.5)
    assert result.profit_factor == pytest.approx(2.0)
