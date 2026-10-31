"""Return–risk preferences and paired uncertainty for fixed allocation methods."""

import numpy as np
import pandas as pd

METHODS = ("equal", "inverse_volatility", "minimum_variance", "balanced")
KEY = ["fold", "entry_date", "exit_date"]


def utility(returns, gamma):
    returns = np.asarray(returns, dtype=float)
    if (
        not np.isfinite(returns).all()
        or (returns <= -1).any()
        or not np.isfinite(gamma)
        or gamma < 0
    ):
        raise ValueError("CRRA utility needs finite returns > -1 and nonnegative risk aversion")
    if gamma == 1:
        return np.log1p(returns)
    return np.expm1((1 - gamma) * np.log1p(returns)) / (1 - gamma)


def inverse_utility(mean_utility, gamma):
    mean_utility = np.asarray(mean_utility, dtype=float)
    if not np.isfinite(mean_utility).all() or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Invalid utility mean or risk aversion")
    if gamma == 1:
        return np.expm1(mean_utility)
    base = 1 + (1 - gamma) * mean_utility
    if (base <= 0).any():
        raise ValueError("Utility mean outside the CRRA domain")
    return np.expm1(np.log1p((1 - gamma) * mean_utility) / (1 - gamma))


def certainty_equivalent(returns, gamma):
    if not np.asarray(returns).size:
        raise ValueError("Certainty equivalents need observed returns")
    return float(inverse_utility(utility(returns, gamma).mean(), gamma))


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def paired_panel(episodes, horizon):
    frame = episodes[episodes.horizon == horizon]
    if frame.empty or frame.duplicated([*KEY, "strategy"]).any():
        raise ValueError("Missing or duplicate risk-study episodes")
    panel = frame.pivot(index=KEY, columns="strategy", values="price_return").sort_index()
    if (
        set(panel.columns) != set(METHODS)
        or panel.isna().any().any()
        or not np.isfinite(panel.to_numpy()).all()
    ):
        raise ValueError("Unmatched or nonfinite risk-study episodes")
    return panel.loc[:, list(METHODS)]


def bootstrap_indices(panel, horizon, settings, block_sessions):
    draws = settings["bootstrap_draws"]
    rng = np.random.default_rng(settings["seed"])
    block = int(np.ceil(block_sessions / horizon))
    folds = panel.index.get_level_values("fold").to_numpy()
    parts = []
    for fold in sorted(set(folds)):
        positions = np.flatnonzero(folds == fold)
        count = len(positions)
        starts = rng.integers(0, count, size=(draws, (count + block - 1) // block))
        local = ((starts[:, :, None] + np.arange(block)) % count).reshape(draws, -1)[:, :count]
        parts.append(positions[local])
    return np.concatenate(parts, axis=1)


def episode_metrics(episodes, gammas, holdout_start):
    metrics, preferences, annual = [], [], []
    for period, frame in (
        ("all", episodes),
        ("inspected_2023_onward", episodes[episodes.entry_date >= pd.Timestamp(holdout_start)]),
    ):
        for (horizon, strategy), group in frame.groupby(["horizon", "strategy"]):
            values = group.price_return.to_numpy()
            fifth = float(np.quantile(values, 0.05))
            metrics.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "strategy": strategy,
                    "n_episodes": len(values),
                    "mean_return": values.mean(),
                    "median_return": np.median(values),
                    "volatility": values.std(ddof=1),
                    "loss_fraction": np.mean(values < 0),
                    "fifth_percentile": fifth,
                    "tail_mean": values[values <= fifth].mean(),
                    "worst_episode": values.min(),
                }
            )
            for gamma in gammas:
                preferences.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "strategy": strategy,
                        "gamma": gamma,
                        "certainty_equivalent": certainty_equivalent(values, gamma),
                        "mean_utility": utility(values, gamma).mean(),
                        "n_episodes": len(values),
                    }
                )
    for (horizon, fold, strategy), group in episodes.groupby(["horizon", "fold", "strategy"]):
        values = group.price_return.to_numpy()
        annual.append(
            {
                "horizon": horizon,
                "fold": fold,
                "strategy": strategy,
                "n_episodes": len(values),
                "mean_return": values.mean(),
                "volatility": values.std(ddof=1),
                "loss_fraction": np.mean(values < 0),
                **{f"ce_gamma_{gamma:g}": certainty_equivalent(values, gamma) for gamma in gammas},
            }
        )
    return pd.DataFrame(metrics), pd.DataFrame(preferences), pd.DataFrame(annual)


def _comparison(panel, strategy, gamma, indices, horizon, block_sessions):
    first, equal = utility(panel[strategy], gamma), utility(panel["equal"], gamma)
    first_mean, equal_mean = first.mean(), equal.mean()
    first_boot, equal_boot = first[indices].mean(axis=1), equal[indices].mean(axis=1)
    difference = first_mean - equal_mean
    centered = (first_boot - equal_boot) - difference
    pvalue = (1 + np.count_nonzero(centered >= difference)) / (len(indices) + 1)
    ce_boot = inverse_utility(first_boot, gamma) - inverse_utility(equal_boot, gamma)
    low, high = np.quantile(ce_boot, [0.025, 0.975])
    return {
        "horizon": horizon,
        "strategy": strategy,
        "gamma": gamma,
        "n_episodes": len(panel),
        "mean_utility_difference": difference,
        "ce_difference": float(
            inverse_utility(first_mean, gamma) - inverse_utility(equal_mean, gamma)
        ),
        "ce_ci_lower": low,
        "ce_ci_upper": high,
        "pvalue_greater": pvalue,
        "block_sessions": block_sessions,
        "block_episodes": int(np.ceil(block_sessions / horizon)),
        "draws": len(indices),
    }


def risk_inference(episodes, config):
    settings = config["risk_study"]
    metrics, preferences, annual = episode_metrics(
        episodes, settings["gammas"], config["backtest"]["holdout_start"]
    )
    comparisons, sensitivity = [], []
    for horizon in config["backtest"]["horizons"]:
        panel = paired_panel(episodes, horizon)
        block = settings["bootstrap_block_sessions"]
        indices = bootstrap_indices(panel, horizon, settings, block)
        for strategy in METHODS[1:]:
            for gamma in settings["gammas"]:
                comparisons.append(_comparison(panel, strategy, gamma, indices, horizon, block))
        for block in settings["bootstrap_sensitivity_sessions"]:
            indices = bootstrap_indices(panel, horizon, settings, block)
            sensitivity.append(
                _comparison(
                    panel,
                    settings["primary_policy"],
                    settings["primary_gamma"],
                    indices,
                    horizon,
                    block,
                )
            )
    comparisons = pd.DataFrame(comparisons)
    comparisons["pvalue_holm_all_preferences"] = holm(comparisons.pvalue_greater)
    primary = (
        comparisons[
            comparisons.strategy.eq(settings["primary_policy"])
            & comparisons.gamma.eq(settings["primary_gamma"])
        ]
        .copy()
        .reset_index(drop=True)
    )
    primary["pvalue_holm_primary_horizons"] = holm(primary.pvalue_greater)
    main_metrics = metrics[metrics.period.eq("all")].set_index(["horizon", "strategy"])
    adoption = []
    for row in primary.to_dict("records"):
        horizon = row["horizon"]
        first = main_metrics.loc[(horizon, settings["primary_policy"])]
        equal = main_metrics.loc[(horizon, "equal")]
        annual_loss = max(0.0, (equal.mean_return - first.mean_return) * 252 / horizon)
        row.update(
            {
                "equal_mean": equal.mean_return,
                "balanced_mean": first.mean_return,
                "equal_volatility": equal.volatility,
                "balanced_volatility": first.volatility,
                "equal_tail_mean": equal.tail_mean,
                "balanced_tail_mean": first.tail_mean,
                "annualized_mean_sacrifice": annual_loss,
                "passes_utility": row["mean_utility_difference"] > 0
                and row["pvalue_holm_primary_horizons"] <= 0.05,
                "passes_volatility": first.volatility < equal.volatility,
                "passes_tail": first.tail_mean >= equal.tail_mean,
                "passes_return_budget": annual_loss
                <= settings["annualized_return_sacrifice_limit"],
            }
        )
        row["qualifies"] = all(
            row[key]
            for key in (
                "passes_utility",
                "passes_volatility",
                "passes_tail",
                "passes_return_budget",
            )
        )
        if settings["status"] == "demonstration_only":
            row["qualifies"] = False
        adoption.append(row)
    return {
        "episode_metrics": metrics,
        "preferences": preferences,
        "fold_metrics": annual,
        "comparisons": comparisons,
        "sensitivity": pd.DataFrame(sensitivity),
        "adoption": pd.DataFrame(adoption),
    }
