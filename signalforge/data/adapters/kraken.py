"""Kraken public REST API adapter (crypto).

GET /0/public/OHLC — no auth required for public market data.

Key quirks that shape the logic below:
  - `since` is a real anchor point (unix seconds) — unlike CoinGecko's free
    tier, Kraken lets us request an arbitrary historical start, not just "N
    days back from now." Still capped at ~720 returned rows per call
    (Kraken's internal buffer per pair/interval); auto-pagination beyond
    that is future work, alongside retry/backoff — out of scope here.
  - Errors surface as HTTP 200 with a non-empty `error` array in the body,
    not necessarily as a non-200 status — both cases are treated as
    ApiRequestError.
  - The result payload's top-level key is the pair name *as Kraken
    normalizes it* (e.g. legacy assets get "X"/"Z" prefixes), which doesn't
    always match the requested pair string, so we don't assume a key name —
    we take the single non-"last" key in `result`.
  - The last row of the OHLC array is documented as always being the
    current, not-yet-closed candle. Rather than trust position alone, we
    drop any row whose period hasn't closed yet (open time + interval >
    now) — mirrors how the OANDA adapter drops "complete": false candles,
    so a re-fetch doesn't silently rewrite a previously stored value.
  - Response rows include real traded volume, unlike CoinGecko's OHLC
    endpoint.
  - Kraken uses its own asset codes (e.g. "XBT" for BTC), not tickers, so
    canonical-symbol -> pair mapping is a small hand-maintained table, same
    reasoning as CoinGecko's coin-id map before it.
"""

from __future__ import annotations

import logging
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


class KrakenAdapter(DataAdapter):
    asset_class = "crypto"

    BASE_URL = "https://api.kraken.com"

    # canonical "BASE/QUOTE" -> Kraken pair code
    SYMBOL_MAP: dict[str, str] = {
        "BTC/USD": "XBTUSD",
        "BTC/USDT": "XBTUSDT",
        "ETH/USD": "ETHUSD",
        "ETH/USDT": "ETHUSDT",
    }

    # our timeframe -> Kraken interval, in minutes
    TIMEFRAME_MAP = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    def __init__(self, session: requests.Session | None = None):
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

        pair = self._resolve_symbol(symbol)
        interval = self.TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise UnsupportedTimeframeError(f"Kraken adapter doesn't support timeframe '{timeframe}'")

        params = {"pair": pair, "interval": interval, "since": int(start.timestamp())}
        response = self._session.get(f"{self.BASE_URL}/0/public/OHLC", params=params)
        if response.status_code != 200:
            raise ApiRequestError(
                f"Kraken request failed with status {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiRequestError(f"Kraken returned non-JSON response: {exc}") from exc

        errors = payload.get("error") or []
        if errors:
            raise ApiRequestError(f"Kraken returned an error: {errors}")

        result = payload.get("result")
        if not isinstance(result, dict):
            raise ApiRequestError(f"Kraken returned unexpected response shape: {payload}")

        result = dict(result)
        result.pop("last", None)
        if len(result) != 1:
            raise ApiRequestError(f"Kraken response 'result' had unexpected keys: {list(result)}")
        rows = next(iter(result.values()))

        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        now_ts = int(_utcnow().timestamp())
        interval_seconds = interval * 60

        candles = []
        for row in rows:
            try:
                ts, o, h, l, c = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                volume = float(row[6])
            except (IndexError, TypeError, ValueError) as exc:
                raise ApiRequestError(f"Kraken returned a malformed OHLC row: {row}") from exc

            if ts + interval_seconds > now_ts:
                continue  # candle period hasn't closed yet
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
                    volume=volume,
                    source="kraken",
                )
            )

        candles.sort(key=lambda c: c.timestamp)
        return candles

    def _resolve_symbol(self, symbol: str) -> str:
        try:
            return self.SYMBOL_MAP[symbol]
        except KeyError as exc:
            raise SymbolNotFoundError(f"'{symbol}' is not in KrakenAdapter.SYMBOL_MAP") from exc


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"'{name}' must be a timezone-aware datetime")
