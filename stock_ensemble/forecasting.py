"""Training-only, date-balanced return forecasts and explicitly ordinal controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .model import SIGNALS, fit_weights


FEATURE_NAMES = (
    "earnings",
    "momentum_value",
    "reversal_value",
    "volatility",
    "log_dollar_volume",
    "earnings_available",
    "earnings_freshness",
    "bad_news_rebound",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameters: dict


@dataclass(frozen=True)
class FittedForecast:
    candidate: Candidate
    model: object
    scaler: StandardScaler | None
    training_mean: float
    metadata: dict


def _finite_number(value, name, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _positive_integer(value, name):
    result = _finite_number(value, name, positive=True)
    if not result.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def candidate_grid(config: dict) -> list[Candidate]:
    """Return the prespecified order used for deterministic selection tie-breaks."""
    settings = config["research"]
    result = [
        Candidate("momentum", "momentum", {}),
        Candidate("equal_signal", "equal_signal", {}),
        Candidate("rank_ensemble", "rank_ensemble", {"ridge": 0.05}),
    ]
    for value in settings["ridge_alphas"]:
        alpha = _finite_number(value, "ridge alpha", positive=True)
        result.append(Candidate(f"ridge_alpha_{alpha:.12g}", "ridge", {"alpha": alpha}))
    for value in settings["boosted_depths"]:
        depth = _positive_integer(value, "boosted depth")
        result.append(
            Candidate(
                f"boosted_depth_{depth}",
                "boosted",
                {
                    "max_depth": depth,
                    "max_iter": _positive_integer(settings["boosted_iterations"], "boosted iterations"),
                    "learning_rate": _finite_number(
                        settings["boosted_learning_rate"], "boosted learning rate", positive=True
                    ),
                    "min_samples_leaf": _positive_integer(settings["boosted_min_leaf"], "boosted minimum leaf"),
                    "random_state": int(settings["seed"]),
                    "early_stopping": False,
                    "max_bins": 63,
                    "l2_regularization": 1.0,
                },
            )
        )
    if not settings["ridge_alphas"] or not settings["boosted_depths"]:
        raise ValueError("The candidate grid requires Ridge alphas and boosted depths")
    if len({candidate.name for candidate in result}) != len(result):
        raise ValueError("Candidate names must be unique; remove duplicate grid values")
    return result


def _numeric(frame: pd.DataFrame, columns) -> np.ndarray:
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(f"Missing forecast inputs: {', '.join(missing)}")
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Forecast inputs must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Forecast inputs must be finite: {', '.join(columns)}")
    return values


def _design(frame: pd.DataFrame, config: dict) -> np.ndarray:
    """Eight observable predictors; no cross-date statistics are computed here."""
    values = _numeric(frame, ("earnings", "momentum_value", "reversal_value", "volatility", "dollar_volume"))
    if (values[:, 3] <= 0).any() or (values[:, 4] <= 0).any():
        raise ValueError("Forecast volatility and dollar volume must be positive")
    for column in ("earnings_age", "surprise"):
        if column not in frame:
            raise ValueError(f"Missing forecast inputs: {column}")
    try:
        age = frame["earnings_age"].to_numpy(dtype=float, copy=True)
        surprise = frame["surprise"].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Earnings age and surprise must be numeric or missing") from error
    if np.isinf(age).any() or np.isinf(surprise).any():
        raise ValueError("Earnings age and surprise must be finite or missing")
    settings = config["features"]
    max_age = _finite_number(settings["earnings_max_age"], "earnings maximum age", positive=True)
    half_life = _finite_number(settings["earnings_half_life"], "earnings half life", positive=True)
    gate_half_life = _finite_number(settings["gate_half_life"], "gate half life", positive=True)
    # Filing age is in calendar days; missing/stale filings contribute no earnings signal.
    active = np.isfinite(age) & np.isfinite(surprise) & (age >= 1) & (age <= max_age)
    freshness = np.zeros(len(frame))
    freshness[active] = np.exp2(-age[active] / half_life)
    bad_news = np.zeros(len(frame))
    bad_news[active] = np.maximum(-np.tanh(surprise[active]), 0) * np.exp2(-age[active] / gate_half_life)
    values[:, 0] = np.where(active, values[:, 0], 0.0)
    # Reversal is the raw market-relative shock in prior-volatility units, not its rank.
    result = np.column_stack(
        (
            values[:, :4],
            np.log(values[:, 4]),
            active.astype(float),
            freshness,
            np.maximum(values[:, 2], 0) * bad_news,
        )
    )
    if not np.isfinite(result).all():
        raise ValueError("Derived economic predictors must be finite")
    return result


def fit_candidate(candidate: Candidate, training: pd.DataFrame, config: dict) -> FittedForecast:
    """Fit only the supplied, already matured training observations.

    Weights sum to one per session (not per stock), so Ridge's penalty is relative
    to the number of training sessions. Forward returns are nominal fractions.
    """
    candidate = deepcopy(candidate)
    if training.empty:
        raise ValueError("No training observations")
    if "date" not in training or training["date"].isna().any():
        raise ValueError("Training dates must be present and nonmissing")
    y = _numeric(training, ("forward_return",)).ravel()
    counts = training.groupby("date")["date"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    training_mean = float(np.average(y, weights=weights))
    scaler = None
    metadata = {
        "algorithm": candidate.family,
        "parameters": deepcopy(candidate.parameters),
        "feature_names": [],
        "training_mean": training_mean,
        "target_units": "nominal_forward_return_fraction",
        "observation_weight": "1 / stock_count_for_training_date",
    }
    # Bound BLAS and OpenMP pools: parallel study folds must not oversubscribe CPUs.
    with threadpool_limits(limits=1):
        if candidate.family in ("ridge", "boosted"):
            x = _design(training, config)
            metadata["feature_names"] = list(FEATURE_NAMES)
            if candidate.family == "ridge":
                alpha = _finite_number(candidate.parameters["alpha"], "ridge alpha", positive=True)
                scaler = StandardScaler().fit(x, sample_weight=weights)
                model = Ridge(alpha=alpha, solver="cholesky").fit(
                    scaler.transform(x), y, sample_weight=weights
                )
                metadata.update(
                    algorithm="Ridge",
                    coefficients=model.coef_.tolist(),
                    intercept=float(model.intercept_),
                    scaler={
                        "mean": scaler.mean_.tolist(),
                        "variance": scaler.var_.tolist(),
                        "scale": scaler.scale_.tolist(),
                    },
                )
            else:
                parameters = deepcopy(candidate.parameters)
                # Never allow sklearn's default random validation/early-stopping split.
                parameters.update(early_stopping=False, max_bins=63, l2_regularization=1.0)
                model = HistGradientBoostingRegressor(**parameters).fit(x, y, sample_weight=weights)
                metadata.update(algorithm="HistGradientBoostingRegressor", parameters=parameters)
        elif candidate.family in ("momentum", "equal_signal", "rank_ensemble"):
            columns = ("momentum",) if candidate.family == "momentum" else SIGNALS
            _numeric(training, columns)
            metadata.update(feature_names=list(columns), target_units="ordinal_score")
            if candidate.family == "rank_ensemble":
                _numeric(training, ("target",))
                model = fit_weights(training, ridge=0.05)
                metadata.update(algorithm="simplex_rank_ensemble", parameters={"ridge": 0.05}, weights=model.tolist())
            else:
                model = None
        else:
            raise ValueError(f"Unknown forecast family: {candidate.family}")
    return FittedForecast(candidate, model, scaler, training_mean, metadata)


def predict_candidate(fitted: FittedForecast, frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Score close-of-session observables without fitting or using future labels."""
    result = frame.copy()
    family = fitted.candidate.family
    with threadpool_limits(limits=1):
        if family in ("ridge", "boosted"):
            x = _design(frame, config)
            if fitted.scaler is not None:
                x = fitted.scaler.transform(x) if len(x) else x
            prediction = fitted.model.predict(x) if len(x) else np.empty(0)
            result["predicted_return"] = prediction
            result["score"] = prediction
        elif family in ("momentum", "equal_signal", "rank_ensemble"):
            columns = ("momentum",) if family == "momentum" else SIGNALS
            values = _numeric(frame, columns)
            if family == "momentum":
                scores = values[:, 0]
            elif family == "equal_signal":
                scores = values.mean(axis=1)
            else:
                scores = values @ fitted.model
            result["score"] = scores
            # Ranks are not calibrated expected returns; do not invent percentages.
            result["predicted_return"] = np.nan
        else:
            raise ValueError(f"Unknown forecast family: {family}")
    if not np.isfinite(result["score"].to_numpy(dtype=float)).all():
        raise ValueError("Forecast scores must be finite")
    result["null_prediction"] = fitted.training_mean
    return result
