"""Regression checks for return units, equal-date fitting, and frozen transforms."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_ensemble.forecasting import (
    FEATURE_NAMES,
    candidate_grid,
    fit_candidate,
    predict_candidate,
)
from stock_ensemble.model import SIGNALS, fit_weights


def forecast_config():
    return {
        "features": {
            "earnings_max_age": 180,
            "earnings_half_life": 45,
            "gate_half_life": 10,
        },
        "research": {
            "ridge_alphas": [1.0, 100.0],
            "boosted_depths": [2, 3],
            "boosted_iterations": 80,
            "boosted_learning_rate": 0.1,
            "boosted_min_leaf": 2,
            "seed": 7008,
        },
    }


def training_frame():
    dates = pd.bdate_range("2020-01-01", periods=60)
    x = np.tile(np.linspace(-1, 1, 4), len(dates))
    return pd.DataFrame(
        {
            "date": dates.repeat(4),
            "ticker": list("ABCD") * len(dates),
            "label_end": (dates + pd.offsets.BDay(1)).repeat(4),
            "earnings": 0.0,
            "momentum": x,
            "reversal": 0.0,
            "momentum_value": x,
            "reversal_value": 0.0,
            "volatility": 0.02,
            "dollar_volume": 1_000_000.0,
            "earnings_age": np.nan,
            "surprise": np.nan,
            "forward_return": 0.015 + 0.04 * x,
            # Deliberately the opposite ordering: ML must not fit this rank label.
            "target": -x,
        }
    )


def candidate(config, name):
    return next(item for item in candidate_grid(config) if item.name == name)


def test_candidate_grid_is_stable_and_fixes_boosting_protocol():
    config = forecast_config()
    grid = candidate_grid(config)
    assert [item.name for item in grid] == [
        "momentum", "equal_signal", "rank_ensemble",
        "ridge_alpha_1", "ridge_alpha_100", "boosted_depth_2", "boosted_depth_3",
    ]
    assert grid[2].parameters == {"ridge": 0.05}
    for item in grid[-2:]:
        assert item.family == "boosted"
        assert item.parameters["early_stopping"] is False
        assert item.parameters["max_bins"] == 63
        assert item.parameters["l2_regularization"] == 1
        assert item.parameters["random_state"] == 7008
    config["research"]["ridge_alphas"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="unique"):
        candidate_grid(config)


def test_ridge_learns_forward_return_fractions_not_rank_targets():
    config, training = forecast_config(), training_frame()
    original = training.copy(deep=True)
    fitted = fit_candidate(candidate(config, "ridge_alpha_1"), training, config)
    scored = predict_candidate(fitted, training, config)
    np.testing.assert_allclose(scored["predicted_return"], training["forward_return"], atol=0.002)
    np.testing.assert_array_equal(scored["score"], scored["predicted_return"])
    assert np.corrcoef(scored["predicted_return"], training["target"])[0, 1] < -0.99
    assert_frame_equal(training, original)
    assert_frame_equal(scored[original.columns], original)
    assert fitted.metadata["feature_names"] == list(FEATURE_NAMES)
    json.dumps(fitted.metadata, allow_nan=False)


def test_boosting_learns_nonlinear_returns_reproducibly_without_early_stopping():
    config, training = forecast_config(), training_frame()
    training["forward_return"] = np.where(training["momentum_value"] > 0, 0.08, -0.04)
    specification = candidate(config, "boosted_depth_2")
    fitted = fit_candidate(specification, training, config)
    repeated = fit_candidate(specification, training, config)
    held_out = training.iloc[[0, 3]].copy()
    held_out["momentum_value"] = [-0.8, 0.8]
    scored = predict_candidate(fitted, held_out, config)
    np.testing.assert_allclose(scored["predicted_return"], [-0.04, 0.08], atol=0.005)
    assert_frame_equal(scored, predict_candidate(repeated, held_out, config))
    assert fitted.model.early_stopping is False
    assert fitted.model.n_iter_ == config["research"]["boosted_iterations"]
    json.dumps(fitted.metadata, allow_nan=False)


def test_training_weights_and_scaler_give_each_date_equal_mass():
    config = forecast_config()
    training = training_frame().iloc[:2].copy()
    training["date"] = pd.to_datetime(["2020-01-01", "2020-01-02"])
    training["momentum_value"] = [0.0, 1.0]
    training["forward_return"] = [0.0, 0.2]
    crowded = pd.concat([training.iloc[[0]], pd.concat([training.iloc[[1]]] * 9)], ignore_index=True)
    specification = candidate(config, "ridge_alpha_1")
    original = fit_candidate(specification, training, config)
    fitted = fit_candidate(specification, crowded, config)
    assert fitted.training_mean == pytest.approx(0.1)
    assert fitted.training_mean != pytest.approx(crowded["forward_return"].mean())
    position = FEATURE_NAMES.index("momentum_value")
    assert fitted.metadata["scaler"]["mean"][position] == pytest.approx(0.5)
    assert fitted.metadata["scaler"]["variance"][position] == pytest.approx(0.25)
    np.testing.assert_allclose(
        predict_candidate(original, training, config)["predicted_return"],
        predict_candidate(fitted, training, config)["predicted_return"],
        atol=1e-14,
    )
    np.testing.assert_allclose(predict_candidate(fitted, training, config)["null_prediction"], 0.1)


@pytest.mark.parametrize("name", ["ridge_alpha_1", "boosted_depth_2"])
def test_future_rows_and_training_source_mutations_do_not_change_fitted_forecast(name):
    config, training = forecast_config(), training_frame()
    fitted = fit_candidate(candidate(config, name), training, config)
    current = training.iloc[:4].copy()
    expected = predict_candidate(fitted, current, config)
    metadata = copy.deepcopy(fitted.metadata)
    future = current.copy()
    future["date"] = pd.Timestamp("2099-01-01")
    future["momentum_value"] = 1e12
    future["forward_return"] = -1e12
    future["target"] = 1e12
    combined = predict_candidate(fitted, pd.concat([current, future], ignore_index=True), config)
    assert_frame_equal(combined.iloc[:4].reset_index(drop=True), expected.reset_index(drop=True))
    # Neither estimator nor its scaler may retain mutable views into the caller's panel.
    training.loc[:, "momentum_value"] = -1e9
    training.loc[:, "forward_return"] = 1e9
    assert_frame_equal(expected, predict_candidate(fitted, current, config))
    assert fitted.metadata == metadata


@pytest.mark.parametrize("name", ["momentum", "equal_signal", "rank_ensemble"])
def test_controls_match_legacy_scores_without_fabricating_return_forecasts(name):
    config, training = forecast_config(), training_frame()
    training["earnings"] = np.sin(np.arange(len(training)))
    training["reversal"] = np.cos(np.arange(len(training)))
    fitted = fit_candidate(candidate(config, name), training, config)
    scored = predict_candidate(fitted, training, config)
    if name == "momentum":
        expected = training["momentum"]
    elif name == "equal_signal":
        expected = training[list(SIGNALS)].mean(axis=1)
    else:
        expected = training[list(SIGNALS)].to_numpy() @ fit_weights(training, ridge=0.05)
        np.testing.assert_allclose(fitted.metadata["weights"], fit_weights(training, ridge=0.05))
    np.testing.assert_allclose(scored["score"], expected)
    assert scored["predicted_return"].isna().all()
    np.testing.assert_allclose(scored["null_prediction"], 0.015)
    json.dumps(fitted.metadata, allow_nan=False)


def test_missing_and_stale_earnings_are_neutral_but_availability_is_explicit():
    config, training = forecast_config(), training_frame()
    training.loc[0, ["earnings_age", "surprise", "earnings"]] = [1, -2, -0.7]
    training.loc[0, "reversal_value"] = 2.0
    training.loc[1, ["earnings_age", "surprise", "earnings"]] = [181, -2, -0.7]
    fitted = fit_candidate(candidate(config, "ridge_alpha_1"), training, config)
    means = dict(zip(FEATURE_NAMES, fitted.metadata["scaler"]["mean"]))
    assert means["earnings_available"] == pytest.approx(1 / len(training))
    assert means["earnings"] == pytest.approx(-0.7 / len(training))
    assert means["earnings_freshness"] == pytest.approx(np.exp2(-1 / 45) / len(training))
    assert means["bad_news_rebound"] == pytest.approx(2 * np.tanh(2) * np.exp2(-1 / 10) / len(training))
    stale = training.iloc[[1]].copy()
    missing = stale.copy()
    missing[["earnings_age", "surprise"]] = np.nan
    missing["earnings"] = 0.0
    np.testing.assert_array_equal(
        predict_candidate(fitted, stale, config)["predicted_return"],
        predict_candidate(fitted, missing, config)["predicted_return"],
    )


@pytest.mark.parametrize("column", ["momentum_value", "volatility", "dollar_volume", "surprise"])
def test_nonfinite_predictors_fail_explicitly(column):
    config, training = forecast_config(), training_frame()
    fitted = fit_candidate(candidate(config, "ridge_alpha_1"), training, config)
    bad = training.copy()
    bad.loc[0, column] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_candidate(candidate(config, "ridge_alpha_1"), bad, config)
    with pytest.raises(ValueError, match="finite"):
        predict_candidate(fitted, bad, config)


def test_missing_labels_or_dates_fail_instead_of_dropping_rows():
    config, training = forecast_config(), training_frame()
    specification = candidate(config, "ridge_alpha_1")
    training.loc[0, "forward_return"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_candidate(specification, training, config)
    training.loc[0, "forward_return"] = 0.0
    training.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match="dates"):
        fit_candidate(specification, training, config)
