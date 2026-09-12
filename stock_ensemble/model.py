"""Date-balanced, simplex-constrained least-squares signal ensemble."""

from itertools import combinations

import numpy as np
import pandas as pd

SIGNALS = ("earnings", "momentum", "reversal")


def fit_weights(training: pd.DataFrame, columns=SIGNALS, ridge: float = 0.05) -> np.ndarray:
    """Minimize mean(date mean squared error) plus ridge distance to equal weights.

    Every face of the simplex is solved, including its vertices. With three
    signals this is seven small equality-constrained quadratic problems.
    Rows with unavailable labels or signals are not training observations.
    """
    columns = tuple(columns)
    if len(columns) != 3 or len(set(columns)) != 3:
        raise ValueError("The ensemble requires three distinct signal columns")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    frame = training.loc[:, ["date", *columns, "target"]].copy()
    values = frame.loc[:, [*columns, "target"]].to_numpy(dtype=float)
    frame = frame.loc[np.isfinite(values).all(axis=1) & frame["date"].notna()]
    if frame.empty:
        raise ValueError("No finite training observations")
    x = frame.loc[:, list(columns)].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=float)
    counts = frame.groupby("date")["target"].transform("size").to_numpy(dtype=float)
    observation_weight = 1.0 / (counts * frame["date"].nunique())
    center = np.full(3, 1.0 / 3.0)
    quadratic = x.T @ (observation_weight[:, None] * x) + ridge * np.eye(3)
    linear = x.T @ (observation_weight * y) + ridge * center
    best = None
    best_objective = np.inf
    for size in range(1, 4):
        for active_tuple in combinations(range(3), size):
            active = np.asarray(active_tuple)
            kkt = np.zeros((size + 1, size + 1))
            kkt[:size, :size] = quadratic[np.ix_(active, active)]
            kkt[:size, size] = 1.0
            kkt[size, :size] = 1.0
            rhs = np.r_[linear[active], 1.0]
            try:
                solution = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            if not np.allclose(kkt @ solution, rhs, atol=1e-9, rtol=1e-9):
                continue
            if np.any(solution[:size] < -1e-10):
                continue
            candidate = np.zeros(3)
            candidate[active] = np.maximum(solution[:size], 0.0)
            candidate /= candidate.sum()
            objective = candidate @ quadratic @ candidate - 2.0 * linear @ candidate
            if objective < best_objective:
                best_objective = objective
                best = candidate
    if best is None:
        raise ArithmeticError("No feasible simplex solution")
    return best
