"""Normalized data structures shared by every data adapter."""

from __future__ import annotations

from dataclasses import dataclass

TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}


@dataclass(frozen=True)
class Candle:
    symbol: str  # canonical "BASE/QUOTE", e.g. "BTC/USDT" or "EUR/USD"
    asset_class: str  # "crypto" | "forex"
    timeframe: str  # one of TIMEFRAMES
    timestamp: int  # unix epoch seconds, UTC — candle open time
    open: float
    high: float
    low: float
    close: float
    volume: float | None  # None for CoinGecko (its OHLC endpoint has no volume)
    source: str  # "coingecko" | "oanda"
