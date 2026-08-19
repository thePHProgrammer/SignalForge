"""LightGBM wrapper, fixed-horizon long-only signal construction, and
probabilistic prediction-quality evaluation.

Hyperparameters are fixed, not tuned -- Phase 4 explicitly excludes
hyperparameter search infrastructure.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DEFAULT_LGBM_PARAMS: dict = dict(
    objective="binary",  # explicit, not inferred -- fail loudly if y ever isn't 2-class
    n_estimators=100,
    # num_leaves/max_depth deliberately constrain model complexity relative to
    # LightGBM's own defaults (num_leaves=31, max_depth=-1) because Phase 4's
    # walk-forward windows may be relatively small -- a documented, conservative
    # choice for this project's data scale, not a claim about what those
    # defaults were designed for.
    num_leaves=15,
    max_depth=4,
    learning_rate=0.05,
    min_child_samples=30,  # up from the library default 20 -- a more conservative leaf-size floor for small data
    subsample=0.8,
    subsample_freq=1,  # required for subsample to actually take effect in the sklearn API
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbosity=-1,  # silence LightGBM's own C++ logger; this project's logging goes through logging.getLogger
)


@dataclass
class MLSignalModel:
    """Not frozen, unlike this codebase's other shared types -- wraps a
    stateful fitted estimator, so freezing would be cosmetic only."""

    classifier: LGBMClassifier
    feature_names: list[str]  # exact fit-time column order; predict_proba_up reindexes to this

    def predict_proba_up(self, X: pd.DataFrame) -> pd.Series:
        """P(class=1 / "up") per row, reindexed to self.feature_names. Rows
        with any NaN feature get NaN here -- never silently imputed by
        LightGBM's own missing-value handling; to_fixed_horizon_signal()
        treats NaN as "never triggers an entry"."""
        X = X[self.feature_names]
        mask = X.notna().all(axis=1)
        proba = pd.Series(np.nan, index=X.index, name="proba_up")
        if mask.any():
            proba.loc[mask] = self.classifier.predict_proba(X.loc[mask])[:, 1]
        return proba

    def save(self, path: Path) -> None:
        """Pickled wrapper -- simplest option while this repo is the only
        consumer. A future live-inference handoff should prefer LightGBM's
        own portable Booster.save_model()/model_to_string() text format
        (stable across LightGBM versions, language-portable) instead --
        noted as the natural next step, not built now."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MLSignalModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain an {cls.__name__}")
        return obj


def fit_model(
    X: pd.DataFrame, y: pd.Series, *, min_train_rows: int = 200, lgbm_params: dict | None = None
) -> MLSignalModel | None:
    """Fits on rows where X and y are both fully non-NaN. Returns None --
    does NOT raise or log -- if fewer than min_train_rows survive, or fewer
    than 2 label classes remain (a degenerate window LightGBM can't/shouldn't
    fit on). Mirrors backtest/runner.py's load_buffered "return None for
    nothing usable" convention: the caller, which has the window/symbol
    context for a good message, does the logger.warning + skip -- not this
    function.
    """
    mask = X.notna().all(axis=1) & y.notna()
    X_clean, y_clean = X.loc[mask], y.loc[mask]
    if len(X_clean) < min_train_rows or y_clean.nunique() < 2:
        return None
    params = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
    clf = LGBMClassifier(**params)
    clf.fit(X_clean, y_clean)
    return MLSignalModel(classifier=clf, feature_names=list(X.columns))


def to_fixed_horizon_signal(proba_up: pd.Series, horizon: int, *, buy_threshold: float = 0.60) -> pd.DataFrame:
    """Long-only, fixed-holding-period signal construction: a "buy" fires
    when proba_up crosses buy_threshold and no position is currently held
    (by this policy's own bookkeeping); the position is then force-exited
    via an automatic "sell" exactly `horizon` bars later, regardless of
    that later row's own prediction -- realizing exactly the
    entry[T+1]->exit[T+1+horizon] trade label_forward_direction scores
    during training. Deliberately stateful/sequential (mirrors engine.py's
    own reasoning for why position simulation can't be forced into
    branch-free vectorized ops).

    No sell_threshold / short-entry path: a sign-only P(up) classifier
    conflates "flat" and "meaningfully down" into the same class-0 bucket,
    a real problem specifically for deciding when to short (a short
    entered on a barely-not-up prediction is a bad trade even when the
    classifier is technically right that price didn't rise) but a
    non-issue for "should I be long," where 0/1 already means exactly
    what's acted on. ML shorting needs a genuinely different (ternary or
    magnitude-aware) target -- out of scope here, not silently
    approximated with this one.

    A position still open at the end of the input range is left as an
    ordinary open long -- simulate_trades()'s existing force_close_at_end
    handles it exactly like any other open position, no special case
    needed here.
    """
    n = len(proba_up)
    signal = np.full(n, "hold", dtype=object)
    proba_values = proba_up.to_numpy()
    bars_until_exit = 0
    for i in range(n):
        if bars_until_exit > 0:
            bars_until_exit -= 1
            if bars_until_exit == 0:
                signal[i] = "sell"
            continue
        p = proba_values[i]
        if not np.isnan(p) and p >= buy_threshold:
            signal[i] = "buy"
            bars_until_exit = horizon

    # Raw per-row model view regardless of what the fixed-horizon policy
    # actually did at that row (forced-hold rows still show their own row's
    # prediction) -- transparency, mirrors how generate_signals() exposes
    # vote_sum alongside its own derived signal. Not automatically a
    # calibrated probability (see evaluate_predictions for measuring that),
    # so this is named model_score, not confidence.
    model_score = ((proba_up - 0.5).abs() * 2).clip(upper=1.0).fillna(0.0)
    return pd.DataFrame(
        {"signal": pd.Series(signal, index=proba_up.index), "model_score": model_score, "proba_up": proba_up},
        index=proba_up.index,
    )


@dataclass(frozen=True)
class ProbabilityMetrics:
    roc_auc: float
    brier_score: float
    log_loss: float


def evaluate_predictions(y_true: pd.Series, proba_up: pd.Series) -> ProbabilityMetrics:
    """Probabilistic prediction quality (not "calibration" specifically --
    Brier score and log loss jointly reflect calibration, resolution, and
    uncertainty, not calibration alone), independent of whether the
    resulting trades made money -- those are different questions. NaN (not
    0.0 or a fabricated value) when undefined, e.g. too few rows or
    single-class realized labels in this window."""
    mask = y_true.notna() & proba_up.notna()
    y, p = y_true[mask], proba_up[mask]
    if len(y) < 2 or y.nunique() < 2:
        return ProbabilityMetrics(roc_auc=float("nan"), brier_score=float("nan"), log_loss=float("nan"))
    return ProbabilityMetrics(
        roc_auc=roc_auc_score(y, p),
        brier_score=brier_score_loss(y, p),
        log_loss=log_loss(y, p, labels=[0.0, 1.0]),
    )
