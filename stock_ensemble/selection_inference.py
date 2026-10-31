"""Paired episode utility inference for frozen joint stock-selection policies."""

import numpy as np
import pandas as pd

from .selection import METHODS


def utility(values, gamma):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any() or gamma < 0:
        raise ValueError("Utility needs finite returns above -1 and nonnegative risk aversion")
    if gamma == 1:
        return np.log1p(values)
    return np.expm1((1 - gamma) * np.log1p(values)) / (1 - gamma)


def inverse_utility(values, gamma):
    values = np.asarray(values, dtype=float)
    if gamma == 1:
        return np.expm1(values)
    return np.expm1(np.log1p((1 - gamma) * values) / (1 - gamma))


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid p-value")
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return result


def paired_panel(episodes, horizon):
    frame = episodes[episodes.horizon == horizon]
    keys = ["fold", "entry_date", "exit_date"]
    if frame.empty or frame.duplicated([*keys, "strategy"]).any():
        raise ValueError("Missing or duplicate selection episodes")
    panel = frame.pivot(index=keys, columns="strategy", values="price_return").sort_index()
    if set(panel.columns) != set(METHODS) or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Unmatched or nonfinite selection episodes")
    return panel


def block_indices(panel, horizon, settings, block):
    rng = np.random.default_rng(settings["seed"])
    draws = settings["bootstrap_draws"]
    length = int(np.ceil(block / horizon))
    folds = panel.index.get_level_values("fold").to_numpy()
    parts = []
    for fold in sorted(set(folds)):
        positions = np.flatnonzero(folds == fold)
        count = len(positions)
        starts = rng.integers(0, count, size=(draws, (count + length - 1) // length))
        local = ((starts[:, :, None] + np.arange(length)) % count).reshape(draws, -1)[:, :count]
        parts.append(positions[local])
    return np.concatenate(parts, axis=1)


def comparison(panel, method, gamma, indices, horizon, block):
    first, incumbent = utility(panel[method], gamma), utility(panel.incumbent, gamma)
    first_boot, incumbent_boot = first[indices].mean(axis=1), incumbent[indices].mean(axis=1)
    difference = (first - incumbent).mean()
    centered = (first_boot - incumbent_boot) - difference
    pvalue = (1 + np.count_nonzero(centered >= difference)) / (len(indices) + 1)
    ce_boot = inverse_utility(first_boot, gamma) - inverse_utility(incumbent_boot, gamma)
    lower, upper = np.quantile(ce_boot, [0.025, 0.975])
    return {
        "horizon": horizon,
        "strategy": method,
        "gamma": gamma,
        "n_episodes": len(panel),
        "mean_utility_difference": difference,
        "ce_difference": float(
            inverse_utility(first.mean(), gamma) - inverse_utility(incumbent.mean(), gamma)
        ),
        "ce_ci_lower": lower,
        "ce_ci_upper": upper,
        "pvalue_greater": pvalue,
        "block_sessions": block,
        "block_episodes": int(np.ceil(block / horizon)),
        "draws": len(indices),
    }


def selection_inference(episodes, config):
    settings = config["selection_study"]
    metrics, preferences, annual, comparisons, sensitivities = [], [], [], [], []
    for period, frame in (
        ("all", episodes),
        (
            "inspected_2023_onward",
            episodes[episodes.entry_date >= pd.Timestamp(config["backtest"]["holdout_start"])],
        ),
    ):
        for (horizon, method), group in frame.groupby(["horizon", "strategy"]):
            values = group.price_return.to_numpy()
            fifth = np.quantile(values, 0.05)
            metrics.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "strategy": method,
                    "n_episodes": len(values),
                    "mean_return": values.mean(),
                    "volatility": values.std(ddof=1),
                    "loss_fraction": np.mean(values < 0),
                    "median_return": np.median(values),
                    "fifth_percentile": fifth,
                    "tail_mean": values[values <= fifth].mean(),
                    "worst_episode": values.min(),
                }
            )
            for gamma in settings["gammas"]:
                preferences.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "strategy": method,
                        "gamma": gamma,
                        "certainty_equivalent": float(
                            inverse_utility(utility(values, gamma).mean(), gamma)
                        ),
                        "mean_utility": utility(values, gamma).mean(),
                    }
                )
    for (horizon, fold, method), group in episodes.groupby(["horizon", "fold", "strategy"]):
        values = group.price_return.to_numpy()
        annual.append(
            {
                "horizon": horizon,
                "fold": fold,
                "strategy": method,
                "n_episodes": len(values),
                "mean_return": values.mean(),
                "volatility": values.std(ddof=1),
                "loss_fraction": np.mean(values < 0),
                **{
                    f"ce_gamma_{gamma:g}": float(
                        inverse_utility(utility(values, gamma).mean(), gamma)
                    )
                    for gamma in settings["gammas"]
                },
            }
        )
    for horizon in config["backtest"]["horizons"]:
        panel = paired_panel(episodes, horizon)
        block = settings["bootstrap_block_sessions"]
        indices = block_indices(panel, horizon, settings, block)
        for method in METHODS[1:]:
            for gamma in settings["gammas"]:
                comparisons.append(comparison(panel, method, gamma, indices, horizon, block))
        for block in settings["bootstrap_sensitivity_sessions"]:
            indices = block_indices(panel, horizon, settings, block)
            sensitivities.append(
                comparison(panel, "joint", settings["primary_gamma"], indices, horizon, block)
            )
    metrics = pd.DataFrame(metrics)
    comparisons = pd.DataFrame(comparisons)
    comparisons["pvalue_holm_all_preferences"] = holm(comparisons.pvalue_greater)
    primary = comparisons[
        comparisons.strategy.eq("joint") & comparisons.gamma.eq(settings["primary_gamma"])
    ].copy()
    primary["pvalue_holm_primary_horizons"] = holm(primary.pvalue_greater)
    main_metrics = metrics[metrics.period.eq("all")].set_index(["horizon", "strategy"])
    adoption = []
    for row in primary.to_dict("records"):
        horizon = row["horizon"]
        first, incumbent = (
            main_metrics.loc[(horizon, "joint")],
            main_metrics.loc[(horizon, "incumbent")],
        )
        sacrifice = max(0.0, (incumbent.mean_return - first.mean_return) * 252 / horizon)
        row.update(
            {
                "incumbent_mean": incumbent.mean_return,
                "joint_mean": first.mean_return,
                "incumbent_volatility": incumbent.volatility,
                "joint_volatility": first.volatility,
                "incumbent_tail_mean": incumbent.tail_mean,
                "joint_tail_mean": first.tail_mean,
                "annualized_mean_sacrifice": sacrifice,
                "passes_utility": row["mean_utility_difference"] > 0
                and row["pvalue_holm_primary_horizons"] <= 0.05,
                "passes_volatility": first.volatility < incumbent.volatility,
                "passes_tail": first.tail_mean >= incumbent.tail_mean,
                "passes_return_budget": sacrifice <= settings["annualized_return_sacrifice_limit"],
            }
        )
        row["qualifies"] = (
            all(
                row[key]
                for key in (
                    "passes_utility",
                    "passes_volatility",
                    "passes_tail",
                    "passes_return_budget",
                )
            )
            and settings["status"] != "demonstration_only"
        )
        adoption.append(row)
    return {
        "episode_metrics": metrics,
        "preferences": pd.DataFrame(preferences),
        "fold_metrics": pd.DataFrame(annual),
        "comparisons": comparisons,
        "sensitivity": pd.DataFrame(sensitivities),
        "adoption": pd.DataFrame(adoption),
    }
