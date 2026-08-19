import pytest

from signalforge.data.exceptions import SymbolNotFoundError
from signalforge.data.symbols import (
    from_oanda_instrument,
    to_oanda_instrument,
    validate_canonical_symbol,
)


@pytest.mark.parametrize("symbol", ["BTC/USDT", "EUR/USD", "XAU/USD", "GBP/USD"])
def test_validate_canonical_symbol_accepts_valid_symbols(symbol):
    validate_canonical_symbol(symbol)  # must not raise


@pytest.mark.parametrize("symbol", ["BTCUSDT", "btc/usdt", "BTC/USD/T", "", "BTC-USD"])
def test_validate_canonical_symbol_rejects_invalid_symbols(symbol):
    with pytest.raises(SymbolNotFoundError):
        validate_canonical_symbol(symbol)


def test_to_oanda_instrument_round_trip():
    assert to_oanda_instrument("EUR/USD") == "EUR_USD"
    assert from_oanda_instrument("EUR_USD") == "EUR/USD"
    assert from_oanda_instrument(to_oanda_instrument("USD/JPY")) == "USD/JPY"


def test_to_oanda_instrument_rejects_invalid_symbol():
    with pytest.raises(SymbolNotFoundError):
        to_oanda_instrument("EURUSD")
