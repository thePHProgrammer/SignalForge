"""OANDA v3 REST API adapter (forex + CFDs, e.g. spot metals via OANDA).

Key quirks of GET /v3/instruments/{instrument}/candles that shape the logic
below:
  - `time` is RFC3339 UTC with 9-digit nanosecond fractional seconds, which
    Python's datetime.fromisoformat can't parse directly — truncated to
    microseconds in _parse_oanda_time before parsing.
  - The most recent candle in a range is often "complete": false (still
    forming). The normalized schema has no completeness concept, so
    in-progress candles are dropped rather than stored and silently mutating
    on a later re-fetch.
  - Authentication is a bearer token; no account ID is needed for this
    endpoint (only for account-scoped endpoints, relevant from Phase 6
    onward).
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
from signalforge.data.symbols import to_oanda_instrument

logger = logging.getLogger(__name__)


class OANDAAdapter(DataAdapter):
    asset_class = "forex"

    BASE_URLS = {
        "practice": "https://api-fxpractice.oanda.com",
        "live": "https://api-fxtrade.oanda.com",
    }

    TIMEFRAME_MAP = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D",
    }

    def __init__(
        self,
        api_token: str | None,
        environment: str = "practice",
        session: requests.Session | None = None,
    ):
        if not api_token:
            raise MissingCredentialsError("OANDA_API_TOKEN is required")
        if environment not in self.BASE_URLS:
            raise ValueError(f"Unknown OANDA environment: {environment!r}")
        self._api_token = api_token
        self._base_url = self.BASE_URLS[environment]
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

        instrument = to_oanda_instrument(symbol)
        granularity = self.TIMEFRAME_MAP.get(timeframe)
        if granularity is None:
            raise UnsupportedTimeframeError(f"OANDA adapter doesn't support timeframe '{timeframe}'")

        params = {
            "granularity": granularity,
            "price": "M",
            "from": _to_rfc3339(start),
            "to": _to_rfc3339(end),
        }
        headers = {"Authorization": f"Bearer {self._api_token}"}

        response = self._session.get(
            f"{self._base_url}/v3/instruments/{instrument}/candles",
            params=params,
            headers=headers,
        )
        if response.status_code != 200:
            raise ApiRequestError(
                f"OANDA request failed with status {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiRequestError(f"OANDA returned non-JSON response: {exc}") from exc

        raw_candles = payload.get("candles")
        if raw_candles is None:
            raise ApiRequestError(f"OANDA response missing 'candles': {payload}")

        candles = []
        for row in raw_candles:
            if not row.get("complete", False):
                continue
            try:
                ts = _parse_oanda_time(row["time"])
                mid = row["mid"]
                open_, high, low, close = (float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"]))
                volume = float(row["volume"]) if row.get("volume") is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiRequestError(f"OANDA returned a malformed candle: {row}") from exc

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
                    source="oanda",
                )
            )

        candles.sort(key=lambda c: c.timestamp)
        return candles


def _parse_oanda_time(raw: str) -> int:
    raw = raw.rstrip("Z")
    if "." in raw:
        date_part, frac = raw.split(".", 1)
        raw = f"{date_part}.{frac[:6].ljust(6, '0')}"
    dt = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"'{name}' must be a timezone-aware datetime")
