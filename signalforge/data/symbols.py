"""Canonical symbol format ("BASE/QUOTE", uppercase) and generic transforms.

Kraken's canonical -> pair-code mapping is not a pure deterministic transform
(legacy assets get "X"/"Z" prefixes inconsistently, e.g. "BTC" -> "XBT"), so
it lives as a static table inside adapters/kraken.py rather than here.
OANDA's mapping is a pure deterministic string transform, so it belongs in
this shared module.
"""

from __future__ import annotations

import re

from signalforge.data.exceptions import SymbolNotFoundError

_CANONICAL_RE = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")


def validate_canonical_symbol(symbol: str) -> None:
    if not _CANONICAL_RE.match(symbol):
        raise SymbolNotFoundError(
            f"'{symbol}' is not a canonical BASE/QUOTE symbol (e.g. 'BTC/USDT')"
        )


def to_oanda_instrument(symbol: str) -> str:
    validate_canonical_symbol(symbol)
    return symbol.replace("/", "_")


def from_oanda_instrument(instrument: str) -> str:
    return instrument.replace("_", "/")
