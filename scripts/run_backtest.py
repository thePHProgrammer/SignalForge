"""Manual CLI: walk-forward backtest the signal generator against stored candles.

Read-only -- no network calls. Run fetch_candles.py first to populate the DB
for whichever timeframe(s) this needs (both primary and, if used,
--confirm-timeframe, over a range wide enough to cover the requested
--start/--end plus warm-up).

Usage:
    python -m scripts.run_backtest --asset-class crypto --symbol BTC/USD --timeframe 1h \\
        --start 2024-01-01 --end 2026-08-01 --train-days 90 --test-days 30

    python -m scripts.run_backtest --asset-class forex --symbol EUR/USD --timeframe 1h \\
        --start 2024-01-01 --end 2026-08-01 --train-days 90 --test-days 30 --position-mode long_short
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from signalforge.backtest.costs import default_cost_model
from signalforge.backtest.runner import run_walk_forward
from signalforge.config import load_settings
from signalforge.data.storage.db import get_connection, init_db
from signalforge.logging_config import setup_logging
from signalforge.signals import REQUIRED_WARMUP_BARS


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward backtest the signal generator.")
    parser.add_argument("--asset-class", choices=["crypto", "forex"], required=True)
    parser.add_argument("--symbol", required=True, help='Canonical "BASE/QUOTE", e.g. BTC/USD or EUR/USD')
    parser.add_argument("--timeframe", required=True, help="Primary timeframe, e.g. 5m, 15m, 1h")
    parser.add_argument("--confirm-timeframe", help="Optional faster timeframe to confirm the primary signal, e.g. 1m")
    parser.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--train-days", required=True, type=float, help="Walk-forward train window size, in days")
    parser.add_argument("--test-days", required=True, type=float, help="Walk-forward test window size, in days")
    parser.add_argument("--step-days", type=float, help="Days to advance between windows (default: --test-days)")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--position-mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--warmup-buffer-bars", type=int, default=REQUIRED_WARMUP_BARS)
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual rate, e.g. 0.02 for 2%%")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    setup_logging(settings.log_level)

    resolved_cost = default_cost_model(args.asset_class, args.symbol)
    if not resolved_cost.is_verified:
        print(f"Note: {resolved_cost.note}\n")

    conn = get_connection(settings.db_path)
    try:
        init_db(conn)
        report = run_walk_forward(
            conn, args.symbol, args.asset_class, args.timeframe, args.start, args.end,
            train_period=pd.Timedelta(days=args.train_days),
            test_period=pd.Timedelta(days=args.test_days),
            step=pd.Timedelta(days=args.step_days) if args.step_days else None,
            confirm_timeframe=args.confirm_timeframe,
            warmup_buffer_bars=args.warmup_buffer_bars,
            initial_capital=args.initial_capital,
            position_mode=args.position_mode,
            cost_model=resolved_cost.cost_model,
            risk_free_rate=args.risk_free_rate,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    finally:
        conn.close()

    if not report.windows:
        print(f"No test windows produced any results for {args.symbol} ({args.timeframe}) in the requested "
              f"range. Check that candles are stored (fetch_candles.py) covering --start/--end plus warm-up.")
        return

    rows = [
        {
            "test_start": wr.window.test_start, "test_end": wr.window.test_end,
            "trades": wr.metrics.total_trades,
            "forced_closes": sum(1 for t in wr.run.trades if t.is_forced_close),
            "sharpe": wr.metrics.sharpe_ratio, "max_dd": wr.metrics.max_drawdown,
            "win_rate": wr.metrics.win_rate, "profit_factor": wr.metrics.profit_factor,
            "final_equity": wr.metrics.final_equity,
        }
        for wr in report.windows
    ]
    print(f"{args.symbol} ({args.timeframe}" + (f", confirmed by {args.confirm_timeframe}" if args.confirm_timeframe else "")
          + f"), {len(report.windows)} test window(s):\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if report.combined_metrics is not None:
        m = report.combined_metrics
        print(f"\nCombined (stitched out-of-sample): trades={m.total_trades}  sharpe={m.sharpe_ratio:.4f}  "
              f"max_dd={m.max_drawdown:.4f}  win_rate={m.win_rate:.4f}  profit_factor={m.profit_factor:.4f}  "
              f"final_equity={m.final_equity:.2f}")
    else:
        print("\nCombined metrics unavailable -- test windows are not contiguous (step != test-days). "
              "Per-window results above are still meaningful on their own.")


if __name__ == "__main__":
    main()
