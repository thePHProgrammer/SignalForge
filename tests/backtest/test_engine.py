import pandas as pd
import pytest

from signalforge.backtest.costs import CostModel
from signalforge.backtest.engine import simulate_trades


def _make_price_signals(bars: list[tuple[float, float, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bars: list of (open, close, signal) tuples, one per bar."""
    n = len(bars)
    index = pd.DatetimeIndex(
        pd.to_datetime([1_700_000_000 + i * 3600 for i in range(n)], unit="s", utc=True), name="timestamp"
    )
    price = pd.DataFrame({"open": [b[0] for b in bars], "close": [b[1] for b in bars]}, index=index)
    signals = pd.DataFrame({"signal": [b[2] for b in bars]}, index=index)
    return price, signals


# --- hand-verified core trade, no costs ---

def test_hand_verified_buy_then_sell_no_costs():
    price, signals = _make_price_signals(
        [(100, 100, "hold"), (100, 100, "buy"), (102, 102.5, "hold"), (103, 103.5, "sell"), (104, 104.5, "hold")]
    )
    run = simulate_trades(price, signals, CostModel(), initial_capital=10_000.0, position_mode="long_only")

    expected_quantity = 10_000.0 / 102  # buy fills at bar 2's open (102), no cost
    assert len(run.trades) == 1
    trade = run.trades[0]
    assert trade.entry_price == pytest.approx(102)
    assert trade.exit_price == pytest.approx(104)  # sell fills at bar 4's open
    assert trade.quantity == pytest.approx(expected_quantity)
    assert trade.entry_fee == 0.0
    assert trade.exit_fee == 0.0
    expected_gross_pnl = (104 - 102) * expected_quantity
    assert trade.gross_pnl == pytest.approx(expected_gross_pnl)
    assert trade.net_pnl == pytest.approx(expected_gross_pnl)
    assert run.open_position is None

    assert run.equity_curve.iloc[0] == pytest.approx(10_000.0)
    assert run.equity_curve.iloc[1] == pytest.approx(10_000.0)  # still flat when marked (action fills next bar)
    assert run.equity_curve.iloc[2] == pytest.approx(expected_quantity * 102.5)
    assert run.equity_curve.iloc[3] == pytest.approx(expected_quantity * 103.5)
    assert run.equity_curve.iloc[4] == pytest.approx(10_000.0 + expected_gross_pnl)


# --- entry-fee sizing correctness (regression test for the bug caught in review) ---

def test_entry_fee_is_consistent_with_executed_notional():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 100, "hold")])
    cost = CostModel(fee_pct=0.004)  # 0.40%, no slippage -> fill is exactly the bar's open (100)
    run = simulate_trades(price, signals, cost, initial_capital=10_000.0, position_mode="long_only")

    position = run.open_position
    expected_quantity = 10_000.0 / (100 * 1.004)
    expected_fee = expected_quantity * 100 * 0.004

    assert position.quantity == pytest.approx(expected_quantity)
    assert position.entry_fee == pytest.approx(expected_fee)
    assert position.entry_fee == pytest.approx(39.84063745019920, rel=1e-6)
    # the bug this regresses: the old (incorrect) formula would have charged
    # exactly $40 (0.4% of pre-fee cash) instead of 0.4% of executed notional
    assert position.entry_fee != pytest.approx(40.0, rel=1e-6)
    # no leverage: executed notional + fee == cash, exactly
    assert position.quantity * 100 + position.entry_fee == pytest.approx(10_000.0)


# --- long_only state transitions ---

def test_flat_sell_is_noop_in_long_only():
    price, signals = _make_price_signals([(100, 100, "sell"), (100, 100, "hold"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only")
    assert run.trades == []
    assert run.open_position is None


def test_flat_hold_is_noop():
    price, signals = _make_price_signals([(100, 100, "hold"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only")
    assert run.trades == []
    assert run.open_position is None


def test_long_buy_is_noop_already_long():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 100, "buy"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only")
    assert run.trades == []
    assert run.open_position.side == "long"
    assert run.open_position.entry_price == pytest.approx(100)


def test_long_hold_is_noop():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 100, "hold"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only")
    assert run.trades == []
    assert run.open_position.side == "long"


def test_long_sell_closes_to_flat_in_long_only():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 100, "sell"), (110, 110, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only")
    assert len(run.trades) == 1
    assert run.trades[0].side == "long"
    assert run.trades[0].exit_price == pytest.approx(110)
    assert run.open_position is None


# --- long_short state transitions, including both reversal directions ---

def test_flat_sell_opens_short_in_long_short():
    price, signals = _make_price_signals([(100, 100, "sell"), (100, 100, "hold"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_short")
    assert run.trades == []
    assert run.open_position.side == "short"
    assert run.open_position.entry_price == pytest.approx(100)


def test_short_sell_is_noop_already_short():
    price, signals = _make_price_signals([(100, 100, "sell"), (100, 100, "sell"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_short")
    assert run.trades == []
    assert run.open_position.side == "short"
    assert run.open_position.entry_price == pytest.approx(100)


def test_short_hold_is_noop():
    price, signals = _make_price_signals([(100, 100, "sell"), (100, 100, "hold"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_short")
    assert run.trades == []
    assert run.open_position.side == "short"


def test_long_sell_reverses_to_short_in_long_short():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 100, "sell"), (110, 110, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_short")
    assert len(run.trades) == 1
    assert run.trades[0].side == "long"
    assert run.trades[0].exit_price == pytest.approx(110)
    assert run.open_position is not None
    assert run.open_position.side == "short"
    assert run.open_position.entry_price == pytest.approx(110)


def test_short_buy_reverses_to_long_in_long_short():
    price, signals = _make_price_signals([(100, 100, "sell"), (100, 100, "buy"), (90, 90, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_short")
    assert len(run.trades) == 1
    assert run.trades[0].side == "short"
    assert run.trades[0].exit_price == pytest.approx(90)
    # short profits when price falls: entry 100, exit 90 -> positive gross_pnl
    assert run.trades[0].gross_pnl > 0
    assert run.open_position is not None
    assert run.open_position.side == "long"
    assert run.open_position.entry_price == pytest.approx(90)


def test_reversal_charges_fees_on_both_legs():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 100, "sell"), (100, 100, "hold")])
    cost = CostModel(fee_pct=0.01)
    run = simulate_trades(price, signals, cost, position_mode="long_short")
    assert len(run.trades) == 1
    closed_trade = run.trades[0]
    assert closed_trade.entry_fee > 0  # fee from the original long entry
    assert closed_trade.exit_fee > 0   # fee from closing the long
    assert run.open_position.entry_fee > 0  # fee from opening the new short


# --- last-bar / force-close behavior ---

def test_last_bar_signal_never_executes():
    price, signals = _make_price_signals([(100, 100, "hold"), (100, 100, "buy")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only")
    assert run.trades == []
    assert run.open_position is None


def test_force_close_at_end_true_liquidates_open_position():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 105, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only", force_close_at_end=True)
    assert run.open_position is None
    assert len(run.trades) == 1
    assert run.trades[0].is_forced_close is True
    assert run.trades[0].exit_price == pytest.approx(105)  # last bar's close, no next-bar-open available
    assert run.equity_curve.iloc[-1] == pytest.approx(run.trades[0].exit_price / 100 * 10_000.0)


def test_force_close_at_end_false_leaves_position_open():
    price, signals = _make_price_signals([(100, 100, "buy"), (100, 105, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only", force_close_at_end=False)
    assert run.open_position is not None
    assert run.trades == []


def test_force_close_no_open_position_is_noop():
    price, signals = _make_price_signals([(100, 100, "hold"), (100, 100, "hold")])
    run = simulate_trades(price, signals, CostModel(), position_mode="long_only", force_close_at_end=True)
    assert run.trades == []
    assert run.open_position is None


# --- precondition ---

def test_price_signals_index_mismatch_raises():
    price, signals = _make_price_signals([(100, 100, "hold"), (100, 100, "hold")])
    signals_bad = signals.copy()
    signals_bad.index = signals_bad.index + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        simulate_trades(price, signals_bad, CostModel())
