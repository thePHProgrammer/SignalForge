"""Manual CLI: read stored candles, compute indicators + signal, print the result.

Read-only -- no network calls, unlike fetch_candles.py. Run fetch_candles.py
first to populate the DB for whichever timeframe(s) this needs.

Without --confirm-timeframe, prints the single-timeframe signal. With it,
also fetches a faster confirmation timeframe (independently stored -- run
fetch_candles.py for both timeframes first) and prints the primary,
confirmation, and combined (confirmation-as-veto) signals side by side.

Usage:
    python -m scripts.generate_signals --asset-class crypto --symbol BTC/USD --timeframe 1h --start 2026-08-01 --end 2026-08-19
    python -m scripts.generate_signals --asset-class crypto --symbol BTC/USD --timeframe 5m --confirm-timeframe 1m --start 2026-08-15 --end 2026-08-19
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from signalforge.config import load_settings
from signalforge.data.frames import candles_to_frame
from signalforge.data.storage.db import get_connection, init_db, query_candles
from signalforge.logging_config import setup_logging
from signalforge.signals import confirm_signals, generate_signals


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute indicators/signals from stored candles.")
    parser.add_argument("--asset-class", choices=["crypto", "forex"], required=True)
    parser.add_argument("--symbol", required=True, help='Canonical "BASE/QUOTE", e.g. BTC/USD or EUR/USD')
    parser.add_argument("--timeframe", required=True, help="Primary timeframe, e.g. 5m, 15m, 1h")
    parser.add_argument("--confirm-timeframe", help="Optional faster timeframe to confirm the primary signal, e.g. 1m")
    parser.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--history", type=int, default=10, help="Rows to show in the recent-history table")
    return parser


def _load_price_and_signals(conn, args, timeframe: str):
    candles = query_candles(conn, symbol=args.symbol, asset_class=args.asset_class,
                             timeframe=timeframe, start=args.start, end=args.end)
    if not candles:
        return None, None
    price_df = candles_to_frame(candles)
    return price_df, generate_signals(price_df)


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    setup_logging(settings.log_level)

    conn = get_connection(settings.db_path)
    try:
        init_db(conn)
        primary_price, primary_signals = _load_price_and_signals(conn, args, args.timeframe)
        confirm_signals_df = None
        if args.confirm_timeframe:
            _, confirm_signals_df = _load_price_and_signals(conn, args, args.confirm_timeframe)
    finally:
        conn.close()

    if primary_signals is None:
        print(f"No stored candles for {args.symbol} ({args.timeframe}) in the requested range. "
              f"Run fetch_candles.py first.")
        return
    if args.confirm_timeframe and confirm_signals_df is None:
        print(f"No stored candles for {args.symbol} ({args.confirm_timeframe}) in the requested range. "
              f"Run fetch_candles.py for the confirmation timeframe too.")
        return

    if confirm_signals_df is None:
        _print_single_timeframe(args, primary_price.join(primary_signals))
    else:
        _print_confirmed(args, primary_signals, confirm_signals_df)


def _print_single_timeframe(args, combined) -> None:
    latest = combined.iloc[-1]
    print(f"{args.symbol} ({args.timeframe}), {len(combined)} candles, "
          f"{combined.index[0]} -> {combined.index[-1]}")
    print(f"\nLatest: close={latest['close']:.5f}  RSI={latest['rsi']:.2f}  "
          f"MACD={latest['macd']:.5f}/{latest['macd_signal']:.5f}  ATR={latest['atr']:.5f}")
    print(f"Votes: rsi={latest['rsi_vote']}  macd={latest['macd_vote']}  bb={latest['bb_vote']}  "
          f"sum={latest['vote_sum']}")
    print(f"Signal: {latest['signal'].upper()}  confidence={latest['confidence']:.2f}"
          + ("  (WARM-UP - not enough history yet)" if latest["is_warmup"] else ""))

    cols = ["close", "rsi", "macd", "macd_hist", "bb_upper", "bb_lower", "atr",
            "vote_sum", "signal", "confidence", "is_warmup"]
    print(f"\nLast {min(args.history, len(combined))} rows:")
    print(combined[cols].tail(args.history).to_string(float_format=lambda v: f"{v:.4f}"))


def _print_confirmed(args, primary, confirmation) -> None:
    combined = confirm_signals(primary, confirmation)
    latest = combined.iloc[-1]

    print(f"{args.symbol}: primary={args.timeframe} ({len(primary)} candles), "
          f"confirm={args.confirm_timeframe} ({len(confirmation)} candles)")
    print(f"\nLatest primary:      signal={latest['primary_signal'].upper():<4} "
          f"confidence={latest['primary_confidence']:.2f}")
    print(f"Latest confirmation: signal={latest['confirmation_signal'].upper():<4} "
          f"confidence={latest['confirmation_confidence']:.2f}")
    print(f"Combined:            signal={latest['signal'].upper():<4} confidence={latest['confidence']:.2f}"
          + ("  (WARM-UP)" if latest["is_warmup"] else ""))

    print(f"\nLast {min(args.history, len(combined))} rows:")
    cols = ["primary_signal", "primary_confidence", "confirmation_signal", "confirmation_confidence",
            "signal", "confidence", "is_warmup"]
    print(combined[cols].tail(args.history).to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
