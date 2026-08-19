"""Walk-forward comparison: fits a fresh LightGBM model per window on the
purged [train_start, train_end) range, evaluates it against the existing
rule-based generate_signals() baseline on the identical [test_start, test_end)
range -- same cost model, same test window, same simulate_trades()/
compute_metrics() Phase 3 already established, so this phase can never
accidentally change how PnL is computed, only how signals are generated.

The ML side is always long-only with a fixed holding period (see
ml/model.py's to_fixed_horizon_signal) -- a binary up/down label can't
support principled short decisions (see labels.py), so ML shorting is out
of scope here. The rule-based side keeps its own configurable
position_mode (long_short by default for forex, forced long_only for
crypto, per Phase 3) -- a deliberate, documented asymmetry: forcing the
rule-based side to long-only just to make the comparison symmetric would
understate real capability it genuinely has (its shorting logic comes from
real bearish indicator readings, not a sign-only classifier).

Single-timeframe only: giving the rule-based side a confirm_timeframe
boost the ML side structurally can't use would make the comparison
asymmetric in a different, worse way -- confirm_timeframe isn't exposed
here at all.

Combined/stitched metrics correctness trap: if some windows are skipped
for ML (insufficient post-purge training rows or single-class labels)
while others aren't, a naive stitch of only the surviving ML runs would
misrepresent a gapped ML history as contiguous -- the same class of bug
Phase 3's design review caught for overlapping/gapped test windows,
resurfacing here for a different reason. ml_combined_* is therefore
populated only when windows are step-contiguous AND every window produced
a fitted ML model; rule_combined_* uses the original Phase-3-only gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from signalforge.backtest.costs import CostModel, default_cost_model
from signalforge.backtest.engine import simulate_trades
from signalforge.backtest.metrics import BacktestResult, compute_metrics
from signalforge.backtest.models import BacktestRun
from signalforge.backtest.runner import load_buffered, run_single_window, stitch_runs
from signalforge.backtest.walkforward import WalkForwardWindow, walk_forward_windows
from signalforge.ml.features import engineer_features, feature_columns
from signalforge.ml.labels import label_forward_direction
from signalforge.ml.model import ProbabilityMetrics, evaluate_predictions, fit_model, to_fixed_horizon_signal
from signalforge.signals import REQUIRED_WARMUP_BARS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLWindowResult:
    window: WalkForwardWindow
    ml_metrics: BacktestResult | None
    ml_run: BacktestRun | None
    ml_probability_metrics: ProbabilityMetrics | None
    rule_metrics: BacktestResult
    rule_run: BacktestRun
    train_rows_used: int
    ml_skip_reason: str | None  # None, "insufficient_train_rows", or "single_class_labels"


@dataclass(frozen=True)
class MLComparisonReport:
    windows: list[MLWindowResult]
    include_rule_based_features: bool
    rule_position_mode: str  # stamped for clarity, since the ML side is always long-only regardless
    ml_combined_metrics: BacktestResult | None
    ml_combined_run: BacktestRun | None
    rule_combined_metrics: BacktestResult | None
    rule_combined_run: BacktestRun | None


def _fit_and_evaluate_ml(
    conn, symbol, asset_class, timeframe, window, warmup_buffer_bars, horizon,
    include_rule_based_features, buy_threshold, min_train_rows, lgbm_params,
    initial_capital, cost_model, risk_free_rate,
):
    train_price, train_rule_signals = load_buffered(
        conn, symbol, asset_class, timeframe, window.train_start, window.train_end, warmup_buffer_bars
    )
    if train_price is None:
        logger.warning("No candles for %s %s in train range %s - %s; skipping ML for this window.",
                        symbol, timeframe, window.train_start, window.train_end)
        return None, None, None, 0, "insufficient_train_rows"

    train_price_local = train_price.loc[window.train_start:]
    train_features = engineer_features(
        train_price_local,
        rule_signals=train_rule_signals.loc[window.train_start:] if include_rule_based_features else None,
    )
    train_labels = label_forward_direction(train_price_local["open"], horizon)

    columns = feature_columns(include_rule_based_features)
    X, y = train_features[columns], train_labels
    train_rows_used = int((X.notna().all(axis=1) & y.notna()).sum())

    model = fit_model(X, y, min_train_rows=min_train_rows, lgbm_params=lgbm_params)
    if model is None:
        reason = "single_class_labels" if train_rows_used >= min_train_rows else "insufficient_train_rows"
        logger.warning("Skipping ML fit for %s %s window %s-%s: %s (train_rows=%d)",
                        symbol, timeframe, window.train_start, window.train_end, reason, train_rows_used)
        return None, None, None, train_rows_used, reason

    test_price, test_rule_signals = load_buffered(
        conn, symbol, asset_class, timeframe, window.test_start, window.test_end, warmup_buffer_bars
    )
    # Not expected to ever be None here: run_ml_vs_rule_comparison already
    # confirmed candles exist for this exact window/range via run_single_window
    # before calling this function, using the identical deterministic query.
    assert test_price is not None, "test candles must exist here since run_single_window already found them"

    test_price_local = test_price.loc[window.test_start:]
    test_features = engineer_features(
        test_price_local,
        rule_signals=test_rule_signals.loc[window.test_start:] if include_rule_based_features else None,
    )
    proba_up = model.predict_proba_up(test_features[columns])

    ml_signals = to_fixed_horizon_signal(proba_up, horizon, buy_threshold=buy_threshold)
    ml_run = simulate_trades(test_price_local, ml_signals, cost_model, initial_capital=initial_capital,
                              position_mode="long_only", force_close_at_end=True)
    ml_metrics = compute_metrics(ml_run, timeframe=timeframe, asset_class=asset_class, risk_free_rate=risk_free_rate)

    # Evaluation use of forward data (what actually happened), not a training
    # use -- no leakage concern, same as any backtest metric that necessarily
    # looks at realized outcomes.
    test_labels = label_forward_direction(test_price_local["open"], horizon)
    ml_probability_metrics = evaluate_predictions(test_labels, proba_up)

    return ml_metrics, ml_run, ml_probability_metrics, train_rows_used, None


def run_ml_vs_rule_comparison(
    conn,
    symbol: str,
    asset_class: str,
    timeframe: str,
    start,
    end,
    train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    *,
    step: pd.Timedelta | None = None,
    horizon: int = 1,
    include_rule_based_features: bool = False,
    buy_threshold: float = 0.60,
    min_train_rows: int = 200,
    lgbm_params: dict | None = None,
    warmup_buffer_bars: int = REQUIRED_WARMUP_BARS,
    initial_capital: float = 10_000.0,
    rule_position_mode: Literal["long_only", "long_short"] = "long_only",
    cost_model: CostModel | None = None,
    risk_free_rate: float = 0.0,
) -> MLComparisonReport:
    if rule_position_mode == "long_short" and asset_class == "crypto":
        raise ValueError("long_short position_mode is not supported for crypto (Kraken is spot-only in this codebase)")

    resolved_cost_model = cost_model if cost_model is not None else default_cost_model(asset_class, symbol).cost_model
    step = step if step is not None else test_period
    is_contiguous = step == test_period

    window_results: list[MLWindowResult] = []
    for window in walk_forward_windows(pd.Timestamp(start), pd.Timestamp(end), train_period, test_period, step):
        rule_run = run_single_window(conn, symbol, asset_class, timeframe, None, window,
                                      warmup_buffer_bars, initial_capital, rule_position_mode, resolved_cost_model)
        if rule_run is None:
            continue  # no candles at all for this window -- can't compare either side

        rule_metrics = compute_metrics(rule_run, timeframe=timeframe, asset_class=asset_class, risk_free_rate=risk_free_rate)

        ml_metrics, ml_run, ml_probability_metrics, train_rows_used, ml_skip_reason = _fit_and_evaluate_ml(
            conn, symbol, asset_class, timeframe, window, warmup_buffer_bars, horizon,
            include_rule_based_features, buy_threshold, min_train_rows, lgbm_params,
            initial_capital, resolved_cost_model, risk_free_rate,
        )

        window_results.append(MLWindowResult(
            window=window, ml_metrics=ml_metrics, ml_run=ml_run, ml_probability_metrics=ml_probability_metrics,
            rule_metrics=rule_metrics, rule_run=rule_run,
            train_rows_used=train_rows_used, ml_skip_reason=ml_skip_reason,
        ))

    rule_combined_run = rule_combined_metrics = None
    if is_contiguous and window_results:
        rule_combined_run = stitch_runs([r.rule_run for r in window_results], initial_capital)
        rule_combined_metrics = compute_metrics(rule_combined_run, timeframe=timeframe, asset_class=asset_class,
                                                 risk_free_rate=risk_free_rate)

    ml_combined_run = ml_combined_metrics = None
    if is_contiguous and window_results and all(r.ml_run is not None for r in window_results):
        ml_combined_run = stitch_runs([r.ml_run for r in window_results], initial_capital)
        ml_combined_metrics = compute_metrics(ml_combined_run, timeframe=timeframe, asset_class=asset_class,
                                               risk_free_rate=risk_free_rate)

    return MLComparisonReport(
        windows=window_results,
        include_rule_based_features=include_rule_based_features,
        rule_position_mode=rule_position_mode,
        ml_combined_metrics=ml_combined_metrics, ml_combined_run=ml_combined_run,
        rule_combined_metrics=rule_combined_metrics, rule_combined_run=rule_combined_run,
    )
