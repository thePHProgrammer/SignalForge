"""Fee/slippage model, asset-class-aware.

Kraken (crypto): a percentage taker fee (market-order fill, not a resting
limit order -- next-bar-open execution means every fill is effectively a
market order) plus a percentage slippage placeholder.

OANDA (forex): spread-only pricing on the default retail account type (no
separate commission), modeled as half the spread against the trader on each
leg. Spreads are absolute price-unit values, NOT pips -- not every OANDA
instrument quotes in currency-pair pips (XAU/USD trades in direct USD price
terms; forcing it through a "pips * 0.0001" assumption would silently
produce a nonsensically tiny spread).

Only EUR/USD's spread below is independently researched. Everything else is
an explicit estimate -- ResolvedCost.is_verified tells the caller which is
which, so an unverified number is never silently trusted as fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KRAKEN_TAKER_FEE_PCT = 0.0040  # 0.40% base tier -- researched, override if your account tier differs
DEFAULT_CRYPTO_SLIPPAGE_PCT = 0.0005  # 0.05% -- engineering placeholder, not a published number

DEFAULT_FOREX_SPREADS: dict[str, float] = {
    "EUR/USD": 0.00012,  # ~1.2 pips -- researched (OANDA-published average ~1.06 pips, standard-account ~1.4)
    "GBP/USD": 0.00016,  # ~1.6 pips -- ESTIMATE, not independently verified
    "USD/JPY": 0.012,  # ~1.2 pips (JPY pairs: 1 pip = 0.01) -- ESTIMATE
    "XAU/USD": 0.30,  # spot gold spread in USD -- ESTIMATE
}
VERIFIED_FOREX_SPREADS = {"EUR/USD"}  # everything else above, plus the fallback, is an estimate
_FALLBACK_FOREX_SPREAD = 0.0002


@dataclass(frozen=True)
class CostModel:
    fee_pct: float = 0.0  # commission, fraction of notional, charged each leg (entry AND exit)
    slippage_pct: float = 0.0  # crypto: adverse price-fraction offset, each leg
    half_spread: float = 0.0  # forex: half the absolute spread, each leg


@dataclass(frozen=True)
class ResolvedCost:
    """What default_cost_model() actually used, plus whether it's a verified figure --
    so a caller (the CLI) can warn the user rather than silently trusting an estimate."""

    cost_model: CostModel
    is_verified: bool
    note: str


def _fill_price(mid_price: float, side: Literal["buy", "sell"], cost: CostModel) -> float:
    """`side` is this leg's own order direction (buy pays the ask-side cost, sell
    receives the bid-side cost) -- not the resulting position's side. A short entry
    is a "sell" order and a short exit is a "buy" order, so this handles long and
    short legs uniformly with no asset-class or position-side branch here.
    """
    adverse = mid_price * cost.slippage_pct + cost.half_spread
    return mid_price + adverse if side == "buy" else mid_price - adverse


def kraken_cost_model(
    fee_pct: float = KRAKEN_TAKER_FEE_PCT, slippage_pct: float = DEFAULT_CRYPTO_SLIPPAGE_PCT
) -> ResolvedCost:
    return ResolvedCost(
        CostModel(fee_pct=fee_pct, slippage_pct=slippage_pct),
        is_verified=False,
        note="Kraken fee is researched; crypto slippage is an engineering placeholder, not verified.",
    )


def oanda_cost_model(symbol: str, spread: float | None = None) -> ResolvedCost:
    resolved_spread = spread if spread is not None else DEFAULT_FOREX_SPREADS.get(symbol, _FALLBACK_FOREX_SPREAD)
    verified = spread is None and symbol in VERIFIED_FOREX_SPREADS
    note = (
        "Researched spread."
        if verified
        else f"Estimated spread for {symbol} -- override with a real value before trusting results."
    )
    return ResolvedCost(CostModel(half_spread=resolved_spread / 2), is_verified=verified, note=note)


def default_cost_model(asset_class: str, symbol: str) -> ResolvedCost:
    return kraken_cost_model() if asset_class == "crypto" else oanda_cost_model(symbol)
