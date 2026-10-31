"""Exact selection of equal-weight stock subsets under ordinal and sector limits."""

from itertools import combinations

import numpy as np
from sklearn.covariance import LedoitWolf

METHODS = ("incumbent", "sector_cap", "variance_guarded", "joint")


def shortlist_covariance(factors, tickers, signal, window):
    if isinstance(window, bool) or int(window) != window or window < 2:
        raise ValueError("Covariance window must be an integer >= 2")
    if signal not in factors.index or not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Covariance requires an exact signal date and unique tickers")
    position = factors.index.get_loc(signal)
    if position + 1 < window:
        raise ValueError("Insufficient dated covariance history")
    history = factors.loc[:, tickers].iloc[position - window + 1 : position + 1] - 1
    if len(history) != window or not np.isfinite(history.to_numpy()).all():
        raise ValueError("Missing covariance observations; cannot fill or compress sessions")
    model = LedoitWolf(store_precision=False).fit(history.to_numpy(copy=True))
    if not np.isfinite(model.covariance_).all() or (np.diag(model.covariance_) <= 0).any():
        raise ValueError("Invalid covariance estimate")
    return model.covariance_, {
        "signal_date": signal,
        "history_start": history.index[0],
        "history_end": history.index[-1],
        "n_observations": window,
        "tickers": tickers,
        "shrinkage": float(model.shrinkage_),
        "covariance": model.covariance_.tolist(),
    }


def choose_subsets(ranking, covariance, *, top_n=5, sector_cap=2, score_budget=0.05):
    """Enumerate the declared finite feasible set, never interpret score as return.

    Covariance is aligned with the input row order. Ticker ordering makes exact
    objective ties reproducible independently of provider or input row ordering.
    """
    if not {"ticker", "score", "sector"}.issubset(ranking):
        raise ValueError("Selection requires tickers, ordinal scores, and sector labels")
    if (
        ranking.ticker.isna().any()
        or ranking.ticker.duplicated().any()
        or ranking.sector.isna().any()
        or ranking.sector.eq("").any()
        or not np.isfinite(ranking.score).all()
    ):
        raise ValueError("Missing or duplicate selection identifiers or nonfinite scores")
    if not 1 <= top_n <= 5 or int(top_n) != top_n or len(ranking) < top_n:
        raise ValueError("Selection requires one to five companies and sufficient candidates")
    if int(sector_cap) != sector_cap or not 1 <= sector_cap <= top_n:
        raise ValueError("Invalid sector cap")
    if not np.isfinite(score_budget) or score_budget < 0:
        raise ValueError("Score budget must be finite and nonnegative")
    covariance = np.asarray(covariance, dtype=float)
    if (
        covariance.shape != (len(ranking), len(ranking))
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, atol=1e-14, rtol=1e-10)
        or np.linalg.eigvalsh(covariance).min() < -1e-12
    ):
        raise ValueError("Covariance must be aligned, finite, symmetric, positive semidefinite")
    ranked = ranking.sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
    incumbent_names = set(ranked.iloc[:top_n].ticker)
    candidates = ranking.reset_index(drop=True).copy()
    sets = []
    for positions in combinations(range(len(candidates)), top_n):
        indices = np.asarray(positions)
        selected = candidates.iloc[indices]
        sets.append(
            {
                "indices": indices,
                "tickers": tuple(sorted(selected.ticker)),
                "mean_score": float(selected.score.mean()),
                "max_sector_count": int(selected.sector.value_counts().max()),
                "variance": float(covariance[np.ix_(indices, indices)].sum() / top_n**2),
            }
        )
    original = next(row for row in sets if set(row["tickers"]) == incumbent_names)
    feasible_cap = max(sector_cap, min(row["max_sector_count"] for row in sets))
    capped = [row for row in sets if row["max_sector_count"] <= feasible_cap]
    sector = min(capped, key=lambda row: (-row["mean_score"], row["tickers"]))

    def least_variance(pool):
        best_variance = min(row["variance"] for row in pool)
        tied = [row for row in pool if row["variance"] <= best_variance + 1e-14]
        return min(tied, key=lambda row: (-row["mean_score"], row["tickers"]))

    variance = least_variance(
        [row for row in sets if row["mean_score"] >= original["mean_score"] - score_budget - 1e-12]
    )
    joint = least_variance(
        [row for row in capped if row["mean_score"] >= sector["mean_score"] - score_budget - 1e-12]
    )
    choices, diagnostics = {}, []
    for method, row in zip(METHODS, (original, sector, variance, joint)):
        selected = candidates.iloc[row["indices"]].sort_values("ticker").copy()
        selected["target_weight"] = 1 / top_n
        choices[method] = selected
        diagnostics.append(
            {
                "strategy": method,
                "mean_score": row["mean_score"],
                "incumbent_mean_score": original["mean_score"],
                "capped_optimal_mean_score": sector["mean_score"],
                "score_sacrifice": original["mean_score"] - row["mean_score"],
                "predicted_daily_variance": row["variance"],
                "max_sector_weight": row["max_sector_count"] / top_n,
                "requested_sector_cap": sector_cap,
                "effective_sector_cap": feasible_cap,
                "cap_relaxed": feasible_cap > sector_cap,
                "shared_incumbent_names": len(set(row["tickers"]) & incumbent_names),
            }
        )
    return choices, diagnostics
