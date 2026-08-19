import pytest

from signalforge.backtest.costs import (
    CostModel,
    _fill_price,
    default_cost_model,
    kraken_cost_model,
    oanda_cost_model,
)


def test_fill_price_buy_slippage_is_above_mid():
    cost = CostModel(slippage_pct=0.001)
    assert _fill_price(100.0, "buy", cost) == pytest.approx(100.1)


def test_fill_price_sell_slippage_is_below_mid():
    cost = CostModel(slippage_pct=0.001)
    assert _fill_price(100.0, "sell", cost) == pytest.approx(99.9)


def test_fill_price_buy_half_spread_is_above_mid():
    cost = CostModel(half_spread=0.05)
    assert _fill_price(100.0, "buy", cost) == pytest.approx(100.05)


def test_fill_price_sell_half_spread_is_below_mid():
    cost = CostModel(half_spread=0.05)
    assert _fill_price(100.0, "sell", cost) == pytest.approx(99.95)


def test_default_cost_model_dispatches_by_asset_class():
    crypto = default_cost_model("crypto", "BTC/USD")
    forex = default_cost_model("forex", "EUR/USD")
    assert crypto.cost_model.fee_pct > 0
    assert crypto.cost_model.half_spread == 0.0
    assert forex.cost_model.fee_pct == 0.0
    assert forex.cost_model.half_spread > 0.0


def test_oanda_cost_model_fallback_for_unlisted_symbol():
    resolved = oanda_cost_model("NZD/USD")
    assert resolved.cost_model.half_spread > 0.0
    assert resolved.is_verified is False


def test_xau_usd_spread_is_not_pip_derived():
    # A pip-based model (e.g. 20 pips * 0.0001) would give ~0.002, absurdly small
    # for gold, which trades in direct USD terms. Regression guard for that bug.
    resolved = oanda_cost_model("XAU/USD")
    assert resolved.cost_model.half_spread > 0.05  # real gold spreads are tens of cents, not fractions of a cent


def test_eur_usd_is_the_only_verified_default():
    assert oanda_cost_model("EUR/USD").is_verified is True
    for symbol in ("GBP/USD", "USD/JPY", "XAU/USD", "NZD/USD"):
        assert oanda_cost_model(symbol).is_verified is False


def test_explicit_spread_override_is_never_verified():
    resolved = oanda_cost_model("EUR/USD", spread=0.0001)
    assert resolved.is_verified is False


def test_kraken_cost_model_is_never_verified():
    assert kraken_cost_model().is_verified is False
