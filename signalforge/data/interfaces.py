"""Shared interface both asset-class adapters implement.

ABC rather than Protocol: we own both implementations (CoinGecko, OANDA), so
runtime enforcement of the abstract method and a common asset_class attribute
are more useful here than Protocol's structural typing, which mainly pays off
for third-party/plugin implementers we don't control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from signalforge.data.models import Candle


class DataAdapter(ABC):
    asset_class: str

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Fetch normalized OHLCV candles covering [start, end], UTC.

        `symbol` must be in canonical "BASE/QUOTE" form. `start`/`end` must be
        timezone-aware; naive datetimes raise ValueError. Returned candles are
        sorted ascending by timestamp. Raises a DataAdapterError subclass on
        failure.
        """
        raise NotImplementedError
