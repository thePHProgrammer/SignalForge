"""Manual CLI: fetch candles for one symbol and store them in SQLite.

Doubles as the day-to-day dev tool and the live smoke test against real
CoinGecko (no key needed) and the user's real OANDA practice account
(requires OANDA_API_TOKEN in .env). This is the only Phase 1 code path that
ever makes a real network call.

Usage:
    python -m scripts.fetch_candles --asset-class crypto --symbol BTC/USD --timeframe 4h --start 2026-08-01 --end 2026-08-19
    python -m scripts.fetch_candles --asset-class forex  --symbol EUR/USD --timeframe 1h --start 2026-08-15 --end 2026-08-19
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from signalforge.config import load_settings
from signalforge.data.adapters.coingecko import CoinGeckoAdapter
from signalforge.data.adapters.oanda import OANDAAdapter
from signalforge.data.storage.db import get_connection, init_db, upsert_candles
from signalforge.logging_config import setup_logging


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch OHLCV candles and store them in SQLite.")
    parser.add_argument("--asset-class", choices=["crypto", "forex"], required=True)
    parser.add_argument("--symbol", required=True, help='Canonical "BASE/QUOTE", e.g. BTC/USD or EUR/USD')
    parser.add_argument("--timeframe", required=True, help="e.g. 30m, 1h, 4h, 1d")
    parser.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    setup_logging(settings.log_level)

    if args.asset_class == "crypto":
        adapter = CoinGeckoAdapter(api_key=settings.coingecko_api_key)
    else:
        adapter = OANDAAdapter(api_token=settings.oanda_api_token, environment=settings.oanda_environment)

    candles = adapter.fetch_ohlcv(args.symbol, args.timeframe, args.start, args.end)

    conn = get_connection(settings.db_path)
    try:
        init_db(conn)
        written = upsert_candles(conn, candles)
    finally:
        conn.close()

    if candles:
        first, last = candles[0], candles[-1]
        print(
            f"Fetched {len(candles)} candles for {args.symbol} ({args.timeframe}), "
            f"{datetime.fromtimestamp(first.timestamp, tz=timezone.utc).isoformat()} -> "
            f"{datetime.fromtimestamp(last.timestamp, tz=timezone.utc).isoformat()}. "
            f"Wrote {written} rows to {settings.db_path}"
        )
    else:
        print(f"Fetched 0 candles for {args.symbol} ({args.timeframe}) in the requested range.")


if __name__ == "__main__":
    main()
