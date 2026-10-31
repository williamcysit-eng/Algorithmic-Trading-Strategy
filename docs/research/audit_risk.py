"""Audit risk-study artifacts independently of experimental production modules.

Run from repository root: uv run --frozen python docs/research/audit_risk.py
[results/risk_study]. Requires the exact normalized snapshot in run_manifest.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

ROOT = Path.cwd()
DIRECTORY = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/risk_study")
METHODS = ["equal", "inverse_volatility", "minimum_variance", "balanced"]


def crra(values, gamma):
    wealth = 1 + np.asarray(values)
    return np.log(wealth) if gamma == 1 else (wealth ** (1 - gamma) - 1) / (1 - gamma)


def equivalent(mean_utility, gamma):
    return (
        np.exp(mean_utility) - 1
        if gamma == 1
        else ((1 + (1 - gamma) * mean_utility) ** (1 / (1 - gamma)) - 1)
    )


def adjust(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    for rank, position in enumerate(order):
        result[position] = min(
            1, max((len(order) - earlier) * pvalues[order[earlier]] for earlier in range(rank + 1))
        )
    return result


def main():
    manifest = json.loads((DIRECTORY / "run_manifest.json").read_text())
    config = manifest["config"]
    settings = config["risk_study"]
    prices = pd.read_csv(Path(config["data_dir"]) / "prices.csv", parse_dates=["date"])
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    splits = prices.pivot(index="date", columns="ticker", values="split").reindex_like(closes)
    factors = closes / closes.shift() * splits
    unit_value = closes * splits.cumprod()
    holdings = pd.read_csv(
        DIRECTORY / "holdings.csv.gz", parse_dates=["entry_date", "signal_date", "refit_date"]
    )
    daily = pd.read_csv(DIRECTORY / "returns.csv", parse_dates=["date"])
    episodes = pd.read_csv(
        DIRECTORY / "episodes.csv", parse_dates=["entry_date", "exit_date", "signal_date"]
    )
    fits = pd.read_csv(DIRECTORY / "fits.csv", parse_dates=["refit_date", "train_end", "label_end"])
    risk_models = json.loads((DIRECTORY / "risk_models.json").read_text())
    keys = ["horizon", "fold", "strategy"]
    decisions = {
        key: {entry: frame for entry, frame in group.groupby("entry_date")}
        for key, group in holdings.groupby(keys)
    }
    assert fits.label_end.lt(fits.refit_date).all()
    assert fits.train_end.lt(fits.refit_date).all()
    assert holdings.signal_date.lt(holdings.entry_date).all()
    assert holdings.refit_date.le(holdings.signal_date).all()
    np.testing.assert_allclose(
        holdings.groupby(keys + ["entry_date"]).target_weight.sum(), 1, atol=1e-12
    )
    selected = (
        holdings.groupby(["horizon", "fold", "entry_date", "strategy"])
        .ticker.apply(lambda x: tuple(sorted(x)))
        .unstack("strategy")
    )
    assert selected.apply(lambda row: len(set(row)), axis=1).eq(1).all()
    assert holdings.target_weight.between(settings["min_weight"], settings["max_weight"]).all()
    sets_by_risk = {}
    for (risk_id, strategy), group in holdings.groupby(["risk_id", "strategy"]):
        one = group.drop_duplicates("ticker").sort_values("ticker")
        # Cached covariance ids must map to identical stock weights even across horizons.
        for _, repeated in group.groupby(["horizon", "entry_date"]):
            np.testing.assert_allclose(
                repeated.sort_values("ticker").target_weight, one.target_weight, atol=1e-12
            )
        sets_by_risk[(risk_id, strategy)] = one.target_weight.to_numpy()
    max_covariance_error, max_optimality_gap = 0.0, 0.0
    for model in risk_models:
        signal = pd.Timestamp(model["signal_date"])
        end = factors.index.get_loc(signal)
        window = settings["covariance_window"]
        frame = factors.loc[:, model["tickers"]].iloc[end - window + 1 : end + 1] - 1
        assert len(frame) == window and np.isfinite(frame.to_numpy()).all()
        assert frame.index[0] == pd.Timestamp(model["history_start"])
        assert frame.index[-1] == pd.Timestamp(model["history_end"]) == signal
        x = frame.to_numpy()
        x = x - x.mean(axis=0)
        n, count = x.shape
        empirical = x.T @ x / n
        mu = np.trace(empirical) / count
        # Direct Ledoit-Wolf moment expression, independent of sklearn's estimator.
        delta = np.sum((empirical - mu * np.eye(count)) ** 2) / count
        beta = (np.mean(np.sum(x**2, axis=1) ** 2) - np.sum(empirical**2)) / (n * count)
        shrinkage = np.clip(beta / delta, 0, 1) if delta > 0 else 0.0
        covariance = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(count)
        stored = np.asarray(model["covariance"])
        np.testing.assert_allclose(stored, covariance, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(model["shrinkage"], shrinkage, rtol=1e-9, atol=1e-12)
        max_covariance_error = max(max_covariance_error, float(np.max(np.abs(stored - covariance))))
        index = model["risk_id"]
        equal = sets_by_risk[(index, "equal")]
        minimum = sets_by_risk[(index, "minimum_variance")]
        balanced = sets_by_risk[(index, "balanced")]
        np.testing.assert_allclose(equal, 1 / count)
        np.testing.assert_allclose(balanced, (equal + minimum) / 2, atol=1e-12)
        inverse = 1 / np.sqrt(np.diag(covariance))
        raw_inverse = inverse / inverse.sum()
        # Independent optimization checks both QP and bounded inverse-volatility projection.
        normalized = covariance / np.max(np.abs(covariance))
        for method in ["minimum_variance", "inverse_volatility"]:
            if method == "minimum_variance":

                def objective(w, matrix):
                    return w @ matrix @ w

                args = (normalized,)
            else:

                def objective(w, target):
                    return np.sum((w - target) ** 2)

                args = (raw_inverse,)
            solved = minimize(
                objective,
                equal,
                args=args,
                method="SLSQP",
                bounds=[(settings["min_weight"], settings["max_weight"])] * count,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            assert solved.success, solved.message
            actual = sets_by_risk[(index, method)]
            gap = objective(actual, *args) - solved.fun
            assert gap <= 1e-9, (index, method, gap)
            max_optimality_gap = max(max_optimality_gap, gap)
    print(
        f"{len(risk_models)} past-only covariance windows and independent allocation solves pass",
        flush=True,
    )
    # Baseline reconciliation with the previous study's matched incumbent.
    matched_incumbent = 0
    archive = ROOT / "results/nested_full"
    if settings["status"] != "demonstration_only" and (archive / "holdings.csv.gz").exists():
        old = pd.read_csv(archive / "holdings.csv.gz", parse_dates=["entry_date", "signal_date"])
        old = old[old.strategy == "incumbent"].sort_values(["horizon", "entry_date", "ticker"])
        new = holdings[holdings.strategy == "equal"].sort_values(
            ["horizon", "entry_date", "ticker"]
        )
        columns = [
            "horizon",
            "fold",
            "entry_date",
            "signal_date",
            "ticker",
            "target_weight",
            "score",
        ]
        pd.testing.assert_frame_equal(
            old[columns].reset_index(drop=True),
            new[columns].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
        matched_incumbent = len(new)
        old_episodes = pd.read_csv(
            archive / "episodes.csv", parse_dates=["entry_date", "exit_date"]
        )
        columns = ["horizon", "fold", "entry_date", "exit_date", "price_return"]
        old_episodes = old_episodes[old_episodes.strategy == "incumbent"].sort_values(columns[:-1])
        new_episodes = episodes[episodes.strategy == "equal"].sort_values(columns[:-1])
        pd.testing.assert_frame_equal(
            old_episodes[columns].reset_index(drop=True),
            new_episodes[columns].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
    max_episode_error = 0.0
    for row in episodes.itertuples(index=False):
        selected = decisions[(row.horizon, row.fold, row.strategy)][row.entry_date]
        growth = (
            unit_value.loc[row.exit_date, selected.ticker].to_numpy()
            / unit_value.loc[row.entry_date, selected.ticker].to_numpy()
        )
        expected = growth @ selected.target_weight.to_numpy() - 1
        np.testing.assert_allclose(row.price_return, expected, rtol=1e-10, atol=1e-12)
        max_episode_error = max(max_episode_error, abs(row.price_return - expected))
    print(
        f"{len(episodes)} nominal holding episodes and {matched_incumbent} archived baseline holdings pass",
        flush=True,
    )
    max_nav_error = 0.0
    capital = config["backtest"]["initial_capital"]
    for key, group in daily.groupby(keys + ["cost_bps"], sort=False):
        horizon, fold, strategy, cost_bps = key
        group = group.sort_values("date").reset_index(drop=True)
        choices = decisions[(horizon, fold, strategy)]
        fee = cost_bps / 10000
        shares = np.zeros(len(closes.columns))
        nav = capital
        for index, row in enumerate(group.itertuples(index=False)):
            shares *= splits.loc[row.date].to_numpy()
            quote = closes.loc[row.date].to_numpy()
            before = shares * quote
            marked = before.sum() if index else capital
            gross = marked / nav - 1
            cost_dollars = traded = 0.0
            if index == len(group) - 1:
                traded = before.sum()
                cost_dollars = traded * fee
                wealth = marked - cost_dollars
                shares[:] = 0
            elif row.date in choices:
                selected = choices[row.date]
                target = np.zeros(len(closes.columns))
                target[closes.columns.get_indexer(selected.ticker)] = selected.target_weight

                def residual(amount, rate, weights, current, value):
                    return amount + rate * np.abs(amount * weights - current).sum() - value

                amount = (
                    brentq(
                        residual,
                        0,
                        marked,
                        args=(fee, target, before, marked),
                        xtol=1e-8,
                        rtol=1e-14,
                    )
                    if fee
                    else marked
                )
                after = amount * target
                traded = np.abs(after - before).sum()
                cost_dollars = fee * traded
                shares = after / quote
                wealth = after.sum()
            else:
                wealth = marked
            np.testing.assert_allclose(
                [group.iloc[index]["return"], row.gross_return, row.cost, row.turnover],
                [wealth / nav - 1, gross, cost_dollars / nav, traded / nav],
                atol=1e-11,
                rtol=1e-9,
            )
            np.testing.assert_allclose(row.nav, wealth, atol=1e-7, rtol=1e-10)
            max_nav_error = max(max_nav_error, abs(row.nav / wealth - 1))
            nav = wealth
    print(f"{len(daily)} daily share values and funded transaction costs pass", flush=True)
    preferences = pd.read_csv(DIRECTORY / "preferences.csv")
    for row in preferences.itertuples(index=False):
        selected = episodes[(episodes.horizon == row.horizon) & episodes.strategy.eq(row.strategy)]
        if row.period != "all":
            selected = selected[
                selected.entry_date >= pd.Timestamp(config["backtest"]["holdout_start"])
            ]
        value = equivalent(crra(selected.price_return, row.gamma).mean(), row.gamma)
        np.testing.assert_allclose(row.certainty_equivalent, value, rtol=1e-9, atol=1e-12)
    comparisons = pd.read_csv(DIRECTORY / "comparisons.csv")
    sensitivity = pd.read_csv(DIRECTORY / "sensitivity.csv")
    adoption = pd.read_csv(DIRECTORY / "adoption.csv")
    for horizon in config["backtest"]["horizons"]:
        panel = (
            episodes[episodes.horizon == horizon]
            .pivot(
                index=["fold", "entry_date", "exit_date"], columns="strategy", values="price_return"
            )
            .sort_index()
        )
        for block_sessions in [
            settings["bootstrap_block_sessions"],
            *settings["bootstrap_sensitivity_sessions"],
        ]:
            rng = np.random.default_rng(settings["seed"])
            draws = settings["bootstrap_draws"]
            block = int(np.ceil(block_sessions / horizon))
            fold_values = panel.index.get_level_values("fold").to_numpy()
            sampled = []
            for fold in sorted(set(fold_values)):
                positions = np.flatnonzero(fold_values == fold)
                count = len(positions)
                starts = rng.integers(0, count, size=(draws, (count + block - 1) // block))
                indices = np.empty((draws, count), dtype=int)
                for draw in range(draws):
                    local = np.concatenate(
                        [(np.arange(block) + start) % count for start in starts[draw]]
                    )[:count]
                    indices[draw] = positions[local]
                sampled.append(indices)
            indices = np.concatenate(sampled, axis=1)
            source = (
                comparisons
                if block_sessions == settings["bootstrap_block_sessions"]
                else sensitivity
            )
            subset = source[(source.horizon == horizon) & source.block_sessions.eq(block_sessions)]
            for row in subset.itertuples(index=False):
                first, equal = crra(panel[row.strategy], row.gamma), crra(panel["equal"], row.gamma)
                observed = (first - equal).mean()
                first_boot, equal_boot = first[indices].mean(axis=1), equal[indices].mean(axis=1)
                differences = equivalent(first_boot, row.gamma) - equivalent(equal_boot, row.gamma)
                pvalue = (
                    1 + np.count_nonzero((first_boot - equal_boot) - observed >= observed)
                ) / (draws + 1)
                low, high = np.quantile(differences, [0.025, 0.975])
                np.testing.assert_allclose(
                    [row.ce_difference, row.ce_ci_lower, row.ce_ci_upper, row.pvalue_greater],
                    [
                        equivalent(first.mean(), row.gamma) - equivalent(equal.mean(), row.gamma),
                        low,
                        high,
                        pvalue,
                    ],
                    rtol=1e-8,
                    atol=1e-12,
                )
    np.testing.assert_allclose(
        comparisons.pvalue_holm_all_preferences, adjust(comparisons.pvalue_greater), atol=1e-12
    )
    np.testing.assert_allclose(
        adoption.pvalue_holm_primary_horizons, adjust(adoption.pvalue_greater), atol=1e-12
    )
    for row in adoption.itertuples(index=False):
        expected = (
            row.mean_utility_difference > 0
            and row.pvalue_holm_primary_horizons <= 0.05
            and row.balanced_volatility < row.equal_volatility
            and row.balanced_tail_mean >= row.equal_tail_mean
            and row.annualized_mean_sacrifice <= settings["annualized_return_sacrifice_limit"]
        )
        if settings["status"] == "demonstration_only":
            expected = False
        assert row.qualifies == expected
    report = {
        "status": "passed",
        "source_commit": manifest["git_commit"],
        "risk_windows": len(risk_models),
        "independent_allocation_solves": len(risk_models) * 2,
        "fit_boundaries": len(fits),
        "episodes": len(episodes),
        "daily_observations": len(daily),
        "archived_baseline_holdings": matched_incumbent,
        "covariance_max_absolute_error": max_covariance_error,
        "optimizer_max_objective_gap": max_optimality_gap,
        "episode_max_absolute_error": max_episode_error,
        "nav_max_relative_error": max_nav_error,
        "preference_checks": len(preferences),
        "comparison_checks": len(comparisons) + len(sensitivity),
        "primary_adoption_checks": len(adoption),
    }
    (DIRECTORY / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
