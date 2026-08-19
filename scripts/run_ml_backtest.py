"""Manual CLI: walk-forward compare the ML (LightGBM) signal against the
rule-based generate_signals() baseline, over stored candles.

Read-only -- no network calls. Run fetch_candles.py first to populate the DB
for the requested --timeframe over a range wide enough to cover --start/--end
plus warm-up and the walk-forward train window.

The ML side is ALWAYS long-only with a fixed --horizon-bar holding period
(see signalforge/ml/model.py:to_fixed_horizon_signal) -- a binary up/down
label can't support principled short decisions. --rule-position-mode governs
ONLY the rule-based baseline; it can run long_short (forex only -- Kraken is
spot-only) while the ML side stays long-only. This is a deliberate,
documented asymmetry, not a bug: the rule-based side's shorting logic comes
from real bearish indicator readings, and forcing it to long-only just to
make the comparison symmetric would understate real capability it has.

METHODOLOGY NOTE: repeatedly re-running this script against the same date
range while tuning --horizon/--buy-threshold/features is itself a form of
overfitting (you become the hyperparameter search). Reserve a final,
most-recent date range and don't run this against it until every other
decision (features, horizon, threshold, cost model) is locked in from
earlier, disjoint ranges -- then run it once there as a genuine final check.
Not enforced in code; discipline only.

Usage:
    python -m scripts.run_ml_backtest --asset-class crypto --symbol BTC/USD --timeframe 1h \\
        --start 2024-01-01 --end 2026-08-01 --train-days 60 --test-days 14

    python -m scripts.run_ml_backtest --asset-class forex --symbol EUR/USD --timeframe 1h \\
        --start 2024-01-01 --end 2026-08-01 --train-days 60 --test-days 14 --rule-position-mode long_short
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from signalforge.backtest.costs import default_cost_model
from signalforge.config import load_settings
from signalforge.data.storage.db import get_connection, init_db
from signalforge.logging_config import setup_logging
from signalforge.ml.runner import run_ml_vs_rule_comparison
from signalforge.signals import REQUIRED_WARMUP_BARS


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward compare the ML signal against the rule-based baseline.")
    parser.add_argument("--asset-class", choices=["crypto", "forex"], required=True)
    parser.add_argument("--symbol", required=True, help='Canonical "BASE/QUOTE", e.g. BTC/USD or EUR/USD')
    parser.add_argument("--timeframe", required=True, help="Primary timeframe, e.g. 5m, 15m, 1h")
    parser.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--train-days", required=True, type=float, help="Walk-forward train window size, in days")
    parser.add_argument("--test-days", required=True, type=float, help="Walk-forward test window size, in days")
    parser.add_argument("--step-days", type=float, help="Days to advance between windows (default: --test-days)")
    parser.add_argument("--horizon", type=int, default=1, help="ML fixed holding period, in bars")
    parser.add_argument("--buy-threshold", type=float, default=0.60, help="ML P(up) threshold to enter long")
    parser.add_argument("--include-rule-based-features", action="store_true",
                         help="Add the rule-based generator's vote_sum/confidence as extra ML features (default: off -- independent model)")
    parser.add_argument("--min-train-rows", type=int, default=200, help="Minimum usable post-purge training rows to fit a model")
    parser.add_argument("--rule-position-mode", choices=["long_only", "long_short"], default="long_only",
                         help="Position mode for the RULE-BASED baseline only -- the ML side is always long_only")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
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

    feature_set = "rule-aware (price/indicator features + vote_sum/confidence)" if args.include_rule_based_features else "independent (price/indicator features only)"
    print(f"ML feature set: {feature_set}")
    print(f"ML side: long-only, fixed {args.horizon}-bar holding period, buy_threshold={args.buy_threshold}")
    print(f"Rule-based baseline: position_mode={args.rule_position_mode}"
          + (" (asymmetric vs. ML -- see module docstring)\n" if args.rule_position_mode == "long_short" else "\n"))

    conn = get_connection(settings.db_path)
    try:
        init_db(conn)
        report = run_ml_vs_rule_comparison(
            conn, args.symbol, args.asset_class, args.timeframe, args.start, args.end,
            train_period=pd.Timedelta(days=args.train_days),
            test_period=pd.Timedelta(days=args.test_days),
            step=pd.Timedelta(days=args.step_days) if args.step_days else None,
            horizon=args.horizon,
            include_rule_based_features=args.include_rule_based_features,
            buy_threshold=args.buy_threshold,
            min_train_rows=args.min_train_rows,
            warmup_buffer_bars=args.warmup_buffer_bars,
            initial_capital=args.initial_capital,
            rule_position_mode=args.rule_position_mode,
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
              f"range. Check that candles are stored (fetch_candles.py) covering --start/--end plus warm-up and train period.")
        return

    rows = []
    for wr in report.windows:
        row = {
            "test_start": wr.window.test_start, "test_end": wr.window.test_end,
            "train_rows": wr.train_rows_used, "ml_skip": wr.ml_skip_reason or "",
        }
        if wr.ml_metrics is not None:
            row.update({
                "ml_trades": wr.ml_metrics.total_trades, "ml_sharpe": wr.ml_metrics.sharpe_ratio,
                "ml_win_rate": wr.ml_metrics.win_rate, "ml_profit_factor": wr.ml_metrics.profit_factor,
            })
        else:
            row.update({"ml_trades": None, "ml_sharpe": None, "ml_win_rate": None, "ml_profit_factor": None})
        if wr.ml_probability_metrics is not None:
            row.update({
                "ml_roc_auc": wr.ml_probability_metrics.roc_auc, "ml_brier": wr.ml_probability_metrics.brier_score,
                "ml_log_loss": wr.ml_probability_metrics.log_loss,
            })
        else:
            row.update({"ml_roc_auc": None, "ml_brier": None, "ml_log_loss": None})
        row.update({
            "rule_trades": wr.rule_metrics.total_trades, "rule_sharpe": wr.rule_metrics.sharpe_ratio,
            "rule_win_rate": wr.rule_metrics.win_rate, "rule_profit_factor": wr.rule_metrics.profit_factor,
        })
        rows.append(row)

    print(f"{args.symbol} ({args.timeframe}), {len(report.windows)} test window(s):\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- Combined (stitched out-of-sample) ---")
    if report.ml_combined_metrics is not None:
        m = report.ml_combined_metrics
        print(f"ML:   trades={m.total_trades}  sharpe={m.sharpe_ratio:.4f}  max_dd={m.max_drawdown:.4f}  "
              f"win_rate={m.win_rate:.4f}  profit_factor={m.profit_factor:.4f}  final_equity={m.final_equity:.2f}")
    else:
        print("ML:   unavailable -- windows are not contiguous (step != test-days), or at least one window "
              "skipped model fitting (see ml_skip column above).")

    if report.rule_combined_metrics is not None:
        m = report.rule_combined_metrics
        print(f"Rule: trades={m.total_trades}  sharpe={m.sharpe_ratio:.4f}  max_dd={m.max_drawdown:.4f}  "
              f"win_rate={m.win_rate:.4f}  profit_factor={m.profit_factor:.4f}  final_equity={m.final_equity:.2f}")
    else:
        print("Rule: unavailable -- test windows are not contiguous (step != test-days).")


if __name__ == "__main__":
    main()
