"""Single-asset position simulation: signals in, trades + equity curve out.

Execution is next-bar-open: a signal computed from bar T's fully-closed data
fills at bar T+1's open -- never at bar T's own close, which would be the
trade-execution analog of the look-ahead leak confirm_signals() already
guards against elsewhere in this codebase. The consequence: the final bar's
signal in any given range can never execute as a signal-driven trade (there
is no bar T+1 to fill it at) -- it's skipped silently, not an error.

Position state is inherently path-dependent (whether "sell" does anything
depends on whether you're currently long), so this is a single explicit
sequential pass over pre-extracted numpy arrays rather than forced into
branch-free vectorized ops -- O(n) either way, correct and testable. This is
also why even "vectorized" backtest libraries run compiled loops under the
hood for exactly this kind of state.

long_short mode has no signal-driven path back to flat once a position is
opened: "hold" never exits (consistent with how "hold" already behaves
elsewhere -- it means "not enough agreement," not "close"), and a signal
flip always reverses straight into the opposite side rather than passing
through flat. The position only returns to flat via force_close_at_end or
the series simply ending. This is a deliberate "always positioned"
convention, not an oversight.

Short positions are modeled as cash-collateralized synthetic shorts (no real
margin/borrow mechanics): opening a short commits quantity*entry_price as
collateral; mark-to-market is that collateral plus unrealized price-move
PnL; closing crystallizes it minus the exit fee. See _close()/_mark_to_market.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from signalforge.backtest.costs import CostModel, _fill_price
from signalforge.backtest.models import BacktestRun, OpenPosition, Trade


@dataclass(frozen=True)
class _OpenState:
    side: Literal["long", "short"]
    quantity: float
    entry_price: float
    entry_fee: float
    entry_time: pd.Timestamp


def _open(cash: float, side: Literal["long", "short"], fill: float, fill_time: pd.Timestamp, cost: CostModel) -> _OpenState:
    # Solves size and fee simultaneously: entry_fee = cash * fee_pct then
    # quantity = (cash - entry_fee) / fill would be internally inconsistent --
    # the actual executed notional is quantity*fill, and fee_pct of *that*
    # doesn't equal the fee already subtracted to compute it. This formula
    # guarantees quantity*fill + entry_fee == cash exactly: no leverage, and
    # the fee is an exact percentage of the executed notional.
    quantity = cash / (fill * (1 + cost.fee_pct))
    entry_fee = quantity * fill * cost.fee_pct
    return _OpenState(side, quantity, fill, entry_fee, fill_time)


def _close(position: _OpenState, fill: float, fill_time: pd.Timestamp, cost: CostModel) -> tuple[Trade, float]:
    if position.side == "long":
        gross_proceeds = position.quantity * fill
        exit_fee = gross_proceeds * cost.fee_pct
        cash = gross_proceeds - exit_fee
        gross_pnl = (fill - position.entry_price) * position.quantity
    else:
        # Cash-collateralized synthetic short: closing crystallizes the
        # committed collateral (quantity*entry_price) plus the price-move
        # PnL, minus the exit fee (charged on the buy-back notional, same
        # notional-based convention as the long side).
        exit_notional = position.quantity * fill
        exit_fee = exit_notional * cost.fee_pct
        gross_pnl = (position.entry_price - fill) * position.quantity
        cash = position.quantity * position.entry_price + gross_pnl - exit_fee
    net_pnl = gross_pnl - position.entry_fee - exit_fee
    trade = Trade(
        position.side, position.entry_time, position.entry_price, fill_time, fill,
        position.quantity, position.entry_fee, exit_fee, gross_pnl, net_pnl,
    )
    return trade, cash


def _mark_to_market(position: _OpenState, close_price: float) -> float:
    if position.side == "long":
        return position.quantity * close_price
    return position.quantity * (2 * position.entry_price - close_price)


def _unrealized_pnl(position: _OpenState, close_price: float) -> float:
    if position.side == "long":
        return (close_price - position.entry_price) * position.quantity
    return (position.entry_price - close_price) * position.quantity


def _apply_transition(
    position: _OpenState | None, sig: str, position_mode: str, cash: float,
    fill_at: float, fill_time: pd.Timestamp, cost_model: CostModel,
) -> tuple[_OpenState | None, float, Trade | None]:
    if position is None:
        if sig == "buy":
            return _open(cash, "long", _fill_price(fill_at, "buy", cost_model), fill_time, cost_model), 0.0, None
        if sig == "sell" and position_mode == "long_short":
            return _open(cash, "short", _fill_price(fill_at, "sell", cost_model), fill_time, cost_model), 0.0, None
        return None, cash, None  # hold, or sell-while-flat in long_only -> no-op

    if position.side == "long":
        if sig == "sell":
            fill = _fill_price(fill_at, "sell", cost_model)
            trade, cash_after = _close(position, fill, fill_time, cost_model)
            if position_mode == "long_short":
                return _open(cash_after, "short", fill, fill_time, cost_model), 0.0, trade
            return None, cash_after, trade
        return position, cash, None  # buy-while-long or hold -> no-op

    # position.side == "short" (only reachable in long_short mode)
    if sig == "buy":
        fill = _fill_price(fill_at, "buy", cost_model)
        trade, cash_after = _close(position, fill, fill_time, cost_model)
        return _open(cash_after, "long", fill, fill_time, cost_model), 0.0, trade
    return position, cash, None  # sell-while-short or hold -> no-op


def simulate_trades(
    price: pd.DataFrame,
    signals: pd.DataFrame,
    cost_model: CostModel,
    *,
    initial_capital: float = 10_000.0,
    position_mode: Literal["long_only", "long_short"] = "long_only",
    force_close_at_end: bool = False,
) -> BacktestRun:
    if not price.index.equals(signals.index):
        raise ValueError("price and signals must share the same index")

    opens = price["open"].to_numpy()
    closes = price["close"].to_numpy()
    signal_arr = signals["signal"].to_numpy()
    idx = price.index
    n = len(price)

    cash = initial_capital
    position: _OpenState | None = None
    trades: list[Trade] = []
    equity = np.empty(n)

    for i in range(n):
        equity[i] = cash if position is None else _mark_to_market(position, closes[i])
        is_last = i == n - 1
        if is_last:
            if force_close_at_end and position is not None:
                fill = _fill_price(closes[i], "sell" if position.side == "long" else "buy", cost_model)
                trade, cash = _close(position, fill, idx[i], cost_model)
                trades.append(dataclasses.replace(trade, is_forced_close=True))
                position = None
                equity[i] = cash
            continue  # no bar i+1 to fill a signal-driven action at

        sig, fill_at, fill_time = signal_arr[i], opens[i + 1], idx[i + 1]
        position, cash, trade = _apply_transition(position, sig, position_mode, cash, fill_at, fill_time, cost_model)
        if trade is not None:
            trades.append(trade)

    open_position = None
    if position is not None:
        open_position = OpenPosition(
            side=position.side, entry_time=position.entry_time, entry_price=position.entry_price,
            quantity=position.quantity, entry_fee=position.entry_fee,
            unrealized_pnl=_unrealized_pnl(position, closes[-1]),
        )

    return BacktestRun(
        equity_curve=pd.Series(equity, index=idx, name="equity"),
        trades=trades,
        open_position=open_position,
        initial_capital=initial_capital,
    )
