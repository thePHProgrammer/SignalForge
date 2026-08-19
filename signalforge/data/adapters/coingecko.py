"""CoinGecko free public API adapter (crypto).

Key constraints of the free /coins/{id}/ohlc endpoint, which shape the logic
below:
  - No `from`/`to` params — only `days`, meaning "N days back from now." A
    requested `start` further back than `days` covers is served by
    over-fetching (sized to reach `start`) and filtering client-side.
  - Granularity is automatic and not selectable on the free tier: days 1-2
    returns 30m candles, days 3-30 returns 4h candles, days 31+ returns 4d
    candles (not one of our supported timeframes at all). If the requested
    timeframe doesn't match the bucket the API will actually return, we raise
    rather than silently mislabeling data or auto-chunking requests (chunking
    is future work — it overlaps retry/backoff, out of scope for this pass).
  - The response has no volume field, so Candle.volume is always None.
  - Uses coin IDs ("bitcoin"), not tickers — there's no reliable ticker->ID
    lookup (duplicate tickers exist across chains/wrapped assets), so the
    mapping is a small hand-maintained table.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import requests

from signalforge.data.exceptions import (
    ApiRequestError,
    SymbolNotFoundError,
    UnsupportedTimeframeError,
)
from signalforge.data.interfaces import DataAdapter
from signalforge.data.models import Candle

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CoinGeckoAdapter(DataAdapter):
    asset_class = "crypto"

    BASE_URL = "https://api.coingecko.com/api/v3"

    # canonical "BASE/QUOTE" -> (coingecko coin id, vs_currency)
    SYMBOL_MAP: dict[str, tuple[str, str]] = {
        "BTC/USD": ("bitcoin", "usd"),
        "BTC/USDT": ("bitcoin", "usdt"),
        "ETH/USD": ("ethereum", "usd"),
        "ETH/USDT": ("ethereum", "usdt"),
    }

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self._api_key = api_key
        self._session = session or requests.Session()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        _require_aware(start, "start")
        _require_aware(end, "end")

        coin_id, vs_currency = self._resolve_symbol(symbol)
        now = _utcnow()
        days = max(1, math.ceil((now - start).total_seconds() / 86400))
        self._validate_timeframe(timeframe, days)

        params: dict[str, str | int] = {"vs_currency": vs_currency, "days": days}
        headers = {"x-cg-demo-api-key": self._api_key} if self._api_key else {}

        response = self._session.get(
            f"{self.BASE_URL}/coins/{coin_id}/ohlc", params=params, headers=headers
        )
        if response.status_code != 200:
            raise ApiRequestError(
                f"CoinGecko request failed with status {response.status_code}: {response.text[:200]}"
            )

        try:
            rows = response.json()
        except ValueError as exc:
            raise ApiRequestError(f"CoinGecko returned non-JSON response: {exc}") from exc

        if not isinstance(rows, list):
            raise ApiRequestError(f"CoinGecko returned unexpected response shape: {type(rows)}")

        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        candles = []
        for row in rows:
            try:
                ts_ms, o, h, l, c = row
            except (ValueError, TypeError) as exc:
                raise ApiRequestError(f"CoinGecko returned malformed OHLC row: {row}") from exc
            ts = ts_ms // 1000
            if ts < start_ts or ts > end_ts:
                continue
            candles.append(
                Candle(
                    symbol=symbol,
                    asset_class=self.asset_class,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=None,
                    source="coingecko",
                )
            )

        candles.sort(key=lambda c: c.timestamp)
        return candles

    def _resolve_symbol(self, symbol: str) -> tuple[str, str]:
        try:
            return self.SYMBOL_MAP[symbol]
        except KeyError as exc:
            raise SymbolNotFoundError(
                f"'{symbol}' is not in CoinGeckoAdapter.SYMBOL_MAP"
            ) from exc

    def _validate_timeframe(self, timeframe: str, days: int) -> None:
        if 1 <= days <= 2:
            actual = "30m"
        elif 3 <= days <= 30:
            actual = "4h"
        else:
            raise UnsupportedTimeframeError(
                f"CoinGecko free tier can't serve a {days}-day window "
                f"(granularity beyond 30 days is 4-day candles, unsupported here)"
            )
        if timeframe != actual:
            raise UnsupportedTimeframeError(
                f"Requested timeframe '{timeframe}' but CoinGecko will return "
                f"'{actual}' candles for a {days}-day window on the free tier"
            )


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"'{name}' must be a timezone-aware datetime")
