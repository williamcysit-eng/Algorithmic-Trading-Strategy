"""Past-only risk estimates and exact bounded allocations for at most five stocks."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _bounds(count, lower, upper):
    if not 1 <= count <= 5:
        raise ValueError("Risk allocations require one to five stocks")
    if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper <= 1:
        raise ValueError("Invalid stock-weight bounds")
    if count * lower > 1 + 1e-14 or count * upper < 1 - 1e-14:
        raise ValueError("Stock-weight bounds are infeasible")


def bounded_simplex(values, lower=0.05, upper=0.4):
    """Euclidean projection; never normalize a clipped vector beyond its bounds."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Projection requires a finite vector")
    _bounds(len(values), lower, upper)
    low, high = float(values.min() - upper), float(values.max() - lower)
    for _ in range(100):
        midpoint = (low + high) / 2
        weights = np.clip(values - midpoint, lower, upper)
        if weights.sum() > 1:
            low = midpoint
        else:
            high = midpoint
    weights = np.clip(values - (low + high) / 2, lower, upper)
    remainder = 1 - weights.sum()
    for position in range(len(values)):
        adjustment = np.clip(remainder, lower - weights[position], upper - weights[position])
        weights[position] += adjustment
        remainder -= adjustment
    return weights


def minimum_variance(covariance, lower=0.05, upper=0.4):
    """Solve all faces of the bounded simplex; at most 3**5 small KKT systems.

    A stock is free, at its lower bound, or at its upper bound. The convex
    quadratic's global minimum appears among these equality-constrained faces.
    Scale the matrix for conditioning without changing its optimum.
    """
    covariance = np.asarray(covariance, dtype=float)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError("Covariance must be a finite symmetric square matrix")
    count = len(covariance)
    _bounds(count, lower, upper)
    equal = np.full(count, 1 / count)
    scale = np.max(np.abs(covariance))
    if scale == 0:
        return equal
    quadratic = covariance / scale
    if np.linalg.eigvalsh(quadratic).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite")
    best = equal
    best_objective = float(best @ quadratic @ best)
    for status in product((0, 1, 2), repeat=count):
        status = np.asarray(status)
        free, fixed = np.flatnonzero(status == 0), np.flatnonzero(status != 0)
        weights = np.where(status == 1, lower, upper).astype(float)
        remaining = 1 - weights[fixed].sum()
        if remaining < len(free) * lower - 1e-12 or remaining > len(free) * upper + 1e-12:
            continue
        if not len(free):
            if abs(remaining) > 1e-12:
                continue
        else:
            kkt = np.zeros((len(free) + 1, len(free) + 1))
            kkt[:-1, :-1] = quadratic[np.ix_(free, free)]
            kkt[-1, :-1] = kkt[:-1, -1] = 1
            rhs = np.r_[-quadratic[np.ix_(free, fixed)] @ weights[fixed], remaining]
            try:
                solution = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            if not np.allclose(kkt @ solution, rhs, atol=1e-10, rtol=1e-10):
                continue
            weights[free] = solution[:-1]
            if (weights < lower - 1e-12).any() or (weights > upper + 1e-12).any():
                continue
        weights = bounded_simplex(weights, lower, upper)
        objective = float(weights @ quadratic @ weights)
        # Preserve the equal-weight tie-break on numerically flat faces.
        if objective < best_objective - 1e-12 * max(1., abs(best_objective)):
            best, best_objective = weights, objective
    return best


def dated_covariance(factors, tickers, signal_date, window=252):
    """Use exact calendar rows through the signal close, before next-close trades."""
    if (
        isinstance(window, bool)
        or int(window) != window
        or window < 2
        or signal_date not in factors.index
    ):
        raise ValueError("Covariance needs an exact signal session and integer window >= 2")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Risk tickers must be unique and nonempty")
    position = factors.index.get_loc(signal_date)
    if position + 1 < window:
        raise ValueError("Insufficient covariance history")
    history = factors.loc[:, list(tickers)].iloc[position - window + 1 : position + 1] - 1
    if len(history) != window or not np.isfinite(history.to_numpy()).all():
        raise ValueError("Missing covariance observations; cannot fill or compress sessions")
    estimator = LedoitWolf(store_precision=False).fit(history.to_numpy(copy=True))
    covariance = estimator.covariance_
    if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
        raise ValueError("Covariance must have positive finite stock variances")
    metadata = {
        "signal_date": pd.Timestamp(signal_date),
        "history_start": history.index[0],
        "history_end": history.index[-1],
        "n_observations": len(history),
        "tickers": list(tickers),
        "shrinkage": float(estimator.shrinkage_),
        "covariance": covariance.tolist(),
    }
    return covariance, metadata


def allocation_weights(covariance, lower=0.05, upper=0.4):
    count = len(covariance)
    _bounds(count, lower, upper)
    inverse = 1 / np.sqrt(np.diag(covariance))
    if not np.isfinite(inverse).all() or (inverse <= 0).any():
        raise ValueError("Inverse-volatility allocation needs positive variances")
    equal = np.full(count, 1 / count)
    minimum = minimum_variance(covariance, lower, upper)
    return {
        "equal": equal,
        "inverse_volatility": bounded_simplex(inverse / inverse.sum(), lower, upper),
        "minimum_variance": minimum,
        "balanced": (equal + minimum) / 2,
    }
