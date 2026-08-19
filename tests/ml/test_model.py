import math
import pickle

import numpy as np
import pandas as pd
import pytest

from signalforge.ml.model import (
    MLSignalModel,
    evaluate_predictions,
    fit_model,
    to_fixed_horizon_signal,
)


def _make_separable_data(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    signal_feature = rng.normal(size=n)
    noise_features = rng.normal(size=(n, 4))
    y = (signal_feature + rng.normal(scale=0.3, size=n) > 0).astype(float)
    X = pd.DataFrame({"signal_feature": signal_feature, **{f"noise_{i}": noise_features[:, i] for i in range(4)}})
    return X, pd.Series(y)


# --- to_fixed_horizon_signal ---

def test_buy_fires_on_threshold_crossing_when_flat():
    proba = pd.Series([0.3, 0.7, 0.3, 0.3])
    result = to_fixed_horizon_signal(proba, horizon=2, buy_threshold=0.6)
    assert result["signal"].iloc[0] == "hold"
    assert result["signal"].iloc[1] == "buy"


def test_forced_exit_fires_exactly_horizon_bars_later_then_reentry_allowed():
    proba = pd.Series([0.7, 0.7, 0.7, 0.7, 0.3])
    result = to_fixed_horizon_signal(proba, horizon=2, buy_threshold=0.6)
    # buy at 0; forced hold through 1; forced sell at 2 (despite proba[2]
    # still being bullish -- the exit is mechanical, not prediction-driven);
    # flat again at 3, so a fresh buy fires there too
    assert result["signal"].tolist() == ["buy", "hold", "sell", "buy", "hold"]


def test_nan_proba_never_triggers_entry():
    proba = pd.Series([float("nan"), 0.9, float("nan")])
    result = to_fixed_horizon_signal(proba, horizon=1, buy_threshold=0.6)
    assert result["signal"].iloc[0] == "hold"
    assert result["signal"].iloc[1] == "buy"


def test_buy_threshold_boundary_is_inclusive():
    # Two independent series (not one combined series) so a forced sell from
    # an earlier buy can't consume the second row being tested -- with
    # horizon=1, a buy at row 0 forces its own sell at row 1, which would
    # otherwise mask whether row 1's own threshold check ever ran.
    at_threshold = pd.Series([0.6, 0.1])
    below_threshold = pd.Series([0.5999999, 0.1])
    assert to_fixed_horizon_signal(at_threshold, horizon=1, buy_threshold=0.6)["signal"].iloc[0] == "buy"
    assert to_fixed_horizon_signal(below_threshold, horizon=1, buy_threshold=0.6)["signal"].iloc[0] == "hold"


def test_model_score_and_proba_reflect_raw_row_even_when_forced():
    proba = pd.Series([0.7, 0.9, 0.1])
    result = to_fixed_horizon_signal(proba, horizon=2, buy_threshold=0.6)
    pd.testing.assert_series_equal(result["proba_up"], proba, check_names=False)
    expected_score = ((proba - 0.5).abs() * 2).clip(upper=1.0)
    pd.testing.assert_series_equal(result["model_score"], expected_score, check_names=False)


# --- fit_model ---

def test_min_train_rows_guard_skips_below_threshold():
    X, y = _make_separable_data(n=50)
    assert fit_model(X, y, min_train_rows=200) is None


def test_min_train_rows_guard_allows_at_exact_boundary():
    X, y = _make_separable_data(n=200)
    assert fit_model(X, y, min_train_rows=200) is not None


def test_single_class_labels_returns_none():
    X, _ = _make_separable_data(n=300)
    y = pd.Series([1.0] * 300)
    assert fit_model(X, y, min_train_rows=200) is None


def test_nan_rows_excluded_from_fit_row_count():
    X, y = _make_separable_data(n=250)
    X = X.copy()
    X.iloc[0:60, 0] = float("nan")  # 60 rows become unusable -> 190 remain
    assert fit_model(X, y, min_train_rows=200) is None
    assert fit_model(X, y, min_train_rows=189) is not None


def test_fit_predict_perfectly_separable_synthetic_data():
    X, y = _make_separable_data(n=600, seed=1)
    X_train, y_train = X.iloc[:400], y.iloc[:400]
    X_test, y_test = X.iloc[400:], y.iloc[400:]
    model = fit_model(X_train, y_train, min_train_rows=200)
    assert model is not None
    proba = model.predict_proba_up(X_test)
    preds = (proba >= 0.5).astype(float)
    accuracy = (preds.to_numpy() == y_test.to_numpy()).mean()
    assert accuracy > 0.85


def test_random_uncorrelated_data_is_near_chance():
    # Guards against a pipeline artifact manufacturing fake skill -- a
    # single random sample has real sampling variance (an exact 0.5 AUC
    # assertion would be flaky), so this averages ROC-AUC across several
    # independent seeds and checks the mean lands in a chance-level band,
    # not a systematically-inflated one.
    #
    # Must be evaluated OUT-OF-SAMPLE (fit on one slice, predict on a
    # disjoint slice) -- an earlier version of this test predicted on the
    # same rows it fit on and got a mean AUC around 0.87 on pure noise.
    # That wasn't a leakage bug: a flexible enough model (100 boosting
    # rounds, no validation/early-stopping split) will partially memorize
    # any finite training sample, including one with no real signal at
    # all, so in-sample performance is inflated for ANY model with enough
    # capacity, not just one exploiting a real leak. Only held-out
    # evaluation can distinguish "the pipeline finds real signal" from
    # "the model memorized its own training rows."
    aucs = []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(rng.normal(size=(600, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randint(0, 2, size=600).astype(float))
        X_train, y_train = X.iloc[:400], y.iloc[:400]
        X_test, y_test = X.iloc[400:], y.iloc[400:]
        model = fit_model(X_train, y_train, min_train_rows=200)
        assert model is not None
        proba = model.predict_proba_up(X_test)
        aucs.append(evaluate_predictions(y_test, proba).roc_auc)
    mean_auc = sum(aucs) / len(aucs)
    assert 0.35 <= mean_auc <= 0.65


# --- evaluate_predictions ---

def test_evaluate_predictions_hand_computed():
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = pd.Series([0.9, 0.1, 0.8, 0.3])
    metrics = evaluate_predictions(y, proba)
    assert metrics.roc_auc == pytest.approx(1.0)  # positives rank strictly above negatives
    expected_brier = ((0.9 - 1) ** 2 + (0.1 - 0) ** 2 + (0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 4
    assert metrics.brier_score == pytest.approx(expected_brier)


def test_evaluate_predictions_single_class_is_nan():
    y = pd.Series([1.0, 1.0, 1.0])
    proba = pd.Series([0.9, 0.8, 0.7])
    metrics = evaluate_predictions(y, proba)
    assert math.isnan(metrics.roc_auc)
    assert math.isnan(metrics.brier_score)
    assert math.isnan(metrics.log_loss)


def test_evaluate_predictions_too_few_rows_is_nan():
    metrics = evaluate_predictions(pd.Series([1.0]), pd.Series([0.9]))
    assert math.isnan(metrics.roc_auc)


# --- save/load ---

def test_save_load_roundtrip(tmp_path):
    X, y = _make_separable_data(n=250, seed=2)
    model = fit_model(X, y, min_train_rows=200)
    assert model is not None
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = MLSignalModel.load(path)
    pd.testing.assert_series_equal(model.predict_proba_up(X), loaded.predict_proba_up(X))


def test_load_wrong_type_raises(tmp_path):
    path = tmp_path / "not_a_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"not": "a model"}, f)
    with pytest.raises(TypeError):
        MLSignalModel.load(path)
