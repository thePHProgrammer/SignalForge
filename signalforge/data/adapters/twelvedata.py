"""Twelve Data REST API adapter (forex).

Added as the default forex data source alongside the (kept, but no longer
default) OANDA adapter: OANDA requires opening an account, which isn't
possible from every country, while Twelve Data's free tier only needs an API
key signup. GET /time_series.

Key quirks that shape the logic below:
  - Twelve Data forex symbols are already canonical "BASE/QUOTE" form (e.g.
    "EUR/USD") -- no translation table needed, unlike Kraken's pair codes or
    OANDA's underscore instrument format. Still validated via
    validate_canonical_symbol so a malformed symbol fails fast, locally.
  - Documented, real quirk: error responses come back with HTTP 200 and a
    JSON body of the form {"status": "error", "code": ..., "message": ...}
    -- not necessarily a non-200 status. Mirrors how the Kraken adapter
    treats an in-body error array as an error regardless of HTTP status; both
    a non-200 response AND an in-body "status": "error" are treated as
    ApiRequestError here.
  - `order=ASC` is requested explicitly (the API defaults to `desc`, most
    recent first) but candles are still sorted defensively before returning,
    matching the other adapters.
  - `timezone=UTC` is requested explicitly so `datetime` values in the
    response can be parsed as UTC without a timezone suffix to strip.
  - No documented "complete"/"closed" flag for the most recent bar (unlike
    OANDA's `complete` field). As a defensive measure -- not a confirmed
    Twelve Data behavior, just the same reasoning the Kraken adapter already
    applies -- any candle whose period hasn't closed yet (open time +
    interval > now) is dropped rather than trusted, so a re-fetch can't
    silently rewrite a previously stored value.
  - Forex responses have no meaningful `volume` (spot FX is OTC, no
    centralized tape) -- the field may be absent entirely; treated as the
    existing nullable Candle.volume, same as OANDA's optional volume.
  - Free tier: 8 requests/minute, 800/day, and a request spanning
    start_date/end_date is capped at 5,000 returned rows -- auto-pagination
    beyond that is out of scope here, same call made for Kraken's ~720-row
    cap. No retry/backoff for the 8-req/min limit either; out of scope.
  - NOT verified against a live account in this sandbox (no network access
    here) -- confirmed only against Twelve Data's public documentation and
    third-party references. Re-check field names/error shape against a real
    response on first live use.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from signalforge.data.exceptions import (
    ApiRequestError,
    MissingCredentialsError,
    UnsupportedTimeframeError,
)
from signalforge.data.interfaces import DataAdapter
from signalforge.data.models import Candle
from signalforge.data.symbols import validate_canonical_symbol

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TwelveDataAdapter(DataAdapter):
    asset_class = "forex"

    BASE_URL = "https://api.twelvedata.com"

    # our timeframe -> Twelve Data interval
    TIMEFRAME_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }

    # our timeframe -> seconds, for the not-yet-closed-candle filter
    TIMEFRAME_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    def __init__(self, api_key: str | None, session: requests.Session | None = None):
        if not api_key:
            raise MissingCredentialsError("TWELVE_DATA_API_KEY is required")
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
        validate_canonical_symbol(symbol)

        interval = self.TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise UnsupportedTimeframeError(f"Twelve Data adapter doesn't support timeframe '{timeframe}'")

        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": _to_twelvedata_datetime(start),
            "end_date": _to_twelvedata_datetime(end),
            "timezone": "UTC",
            "order": "ASC",
            "apikey": self._api_key,
        }

        response = self._session.get(f"{self.BASE_URL}/time_series", params=params)
        if response.status_code != 200:
            raise ApiRequestError(
                f"Twelve Data request failed with status {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiRequestError(f"Twelve Data returned non-JSON response: {exc}") from exc

        if payload.get("status") == "error":
            raise ApiRequestError(f"Twelve Data returned an error: {payload}")

        rows = payload.get("values")
        if rows is None:
            raise ApiRequestError(f"Twelve Data response missing 'values': {payload}")

        now_ts = int(_utcnow().timestamp())
        interval_seconds = self.TIMEFRAME_SECONDS[timeframe]

        candles = []
        for row in rows:
            try:
                ts = _parse_twelvedata_datetime(row["datetime"])
                open_, high, low, close = (
                    float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
                )
                raw_volume = row.get("volume")
                volume = float(raw_volume) if raw_volume not in (None, "") else None
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiRequestError(f"Twelve Data returned a malformed candle: {row}") from exc

            if ts + interval_seconds > now_ts:
                continue  # candle period hasn't closed yet -- see module docstring

            candles.append(
                Candle(
                    symbol=symbol,
                    asset_class=self.asset_class,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    source="twelvedata",
                )
            )

        candles.sort(key=lambda c: c.timestamp)
        return candles


def _parse_twelvedata_datetime(raw: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ApiRequestError(f"Twelve Data returned an unparseable datetime: {raw!r}")


def _to_twelvedata_datetime(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"'{name}' must be a timezone-aware datetime")
