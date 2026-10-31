"""Independent numerical validation of nested study output and frozen decisions.

Run from the repository root after the full study finishes. This reconstructs
nominal share holdings and uses an independent root solver for fee funding.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path.cwd()
DIRECTORY = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/nested_full")


def main():
    manifest = json.loads((DIRECTORY / "run_manifest.json").read_text())
    config = manifest["config"]
    prices = pd.read_csv(Path(config["data_dir"]) / "prices.csv", parse_dates=["date"])
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    split = prices.pivot(index="date", columns="ticker", values="split").reindex_like(close)
    unit_value = close * split.cumprod()
    daily = pd.read_csv(DIRECTORY / "returns.csv", parse_dates=["date"])
    episodes = pd.read_csv(
        DIRECTORY / "episodes.csv", parse_dates=["entry_date", "exit_date", "signal_date"]
    )
    holdings = pd.read_csv(
        DIRECTORY / "holdings.csv.gz",
        parse_dates=["entry_date", "signal_date", "refit_date", "selection_date"],
    )
    selections = json.loads((DIRECTORY / "selections.json").read_text())
    fits = json.loads((DIRECTORY / "fits.json").read_text())
    objectives = pd.read_csv(
        DIRECTORY / "selection_scores.csv", parse_dates=["selection_date", "inner_exit"]
    )
    n_episode = 0
    max_episode_error = 0.0
    max_nav_relative_error = 0.0
    key = ["horizon", "fold", "strategy"]
    decisions = {
        group_key: {entry: frame for entry, frame in group.groupby("entry_date")}
        for group_key, group in holdings.groupby(key)
    }
    for row in episodes.itertuples(index=False):
        selected = decisions[(row.horizon, row.fold, row.strategy)][row.entry_date]
        factors = (
            unit_value.loc[row.exit_date, selected.ticker].to_numpy()
            / unit_value.loc[row.entry_date, selected.ticker].to_numpy()
        )
        expected = factors @ selected.target_weight.to_numpy() - 1
        np.testing.assert_allclose(row.price_return, expected, rtol=1e-10, atol=1e-12)
        max_episode_error = max(max_episode_error, abs(row.price_return - expected))
        n_episode += 1
    print(
        f"{n_episode} complete nominal holding episodes reconstructed from closes and splits",
        flush=True,
    )
    np.testing.assert_allclose(
        holdings.groupby(key + ["entry_date"]).target_weight.sum(), 1, atol=1e-12
    )
    assert holdings.groupby(key + ["entry_date"]).size().eq(config["backtest"]["top_n"]).all()
    assert holdings.signal_date.lt(holdings.entry_date).all()
    assert holdings.refit_date.le(holdings.signal_date).all()
    assert objectives.inner_exit.lt(objectives.selection_date).all()
    for fit in fits:
        assert pd.Timestamp(fit["label_end"]) < pd.Timestamp(fit["refit_date"])
        assert pd.Timestamp(fit["train_end"]) < pd.Timestamp(fit["refit_date"])
        expected_settings = config["model"] if fit["legacy"] else config["research"]
        assert (
            expected_settings["min_train_dates"]
            <= fit["train_dates"]
            <= expected_settings["train_window"]
        )
    # Recreate every annual policy choice from its emitted earlier inner objectives.
    order = [
        "momentum",
        "equal_signal",
        "rank_ensemble",
        "ridge_equal",
        "boosted_equal",
        "ridge_regularized",
        "boosted_regularized",
    ]
    for selection in selections:
        sample = objectives[
            (objectives.horizon == selection["horizon"]) & (objectives.fold == selection["fold"])
        ]
        choices = selection["choices"]
        means = {}
        for policy in order:
            spec = choices[policy]
            rows = sample[(sample.policy == policy) & (sample.candidate == spec["candidate"])]
            if spec["penalty"] is not None:
                rows = rows[np.isclose(rows.penalty, spec["penalty"])]
            assert len(rows) == 1, (policy, len(rows))
            means[policy] = rows.mean_return.item()
        assert max(order, key=lambda name: means[name]) == selection["champion"]
        for family in ["ridge", "boosted"]:
            alternatives = sample[sample.policy == family + "_equal"]
            expected = alternatives.iloc[np.argmax(alternatives.mean_return.to_numpy())].candidate
            assert choices[family + "_equal"]["candidate"] == expected
            alternatives = sample[
                (sample.policy == family + "_regularized") & (sample.candidate == expected)
            ]
            expected_penalty = alternatives.iloc[
                np.argmax(alternatives.mean_return.to_numpy())
            ].penalty
            assert choices[family + "_regularized"]["candidate"] == expected
            assert choices[family + "_regularized"]["penalty"] == expected_penalty
    # Confirm the comparator's fitted coefficients are the existing legacy
    # vintages, not a newly favorable version of the benchmark.
    incumbent_matches = 0
    if config["research"].get("status") != "demonstration_only":
        saved = pd.read_csv(ROOT / "results/current/weights.csv", parse_dates=["refit_date"])
        saved = saved[saved.strategy == "ensemble"].set_index(["horizon", "refit_date"])
        for fit in fits:
            vintage_key = (fit["horizon"], pd.Timestamp(fit["refit_date"]))
            if fit["legacy"] and vintage_key in saved.index:
                expected = saved.loc[
                    vintage_key, ["w_earnings", "w_momentum", "w_reversal"]
                ].to_numpy(dtype=float)
                np.testing.assert_allclose(
                    fit["model"]["weights"], expected, rtol=1e-10, atol=1e-12
                )
                incumbent_matches += 1
        assert incumbent_matches > 0
    print(
        f"{len(fits)} fit boundaries, {len(selections)} annual choices, and {incumbent_matches} saved incumbent vintages validated",
        flush=True,
    )
    np.testing.assert_allclose(daily["return"], daily.gross_return - daily.cost, atol=1e-12)
    np.testing.assert_allclose(daily.cost, daily.turnover * daily.cost_bps / 10000, atol=1e-12)
    capital = config["backtest"]["initial_capital"]
    n_daily = 0
    for group_key, observed in daily.groupby(key + ["cost_bps"], sort=False):
        horizon, fold, strategy, cost_bps = group_key
        observed = observed.sort_values("date").reset_index(drop=True)
        fee = cost_bps / 10000
        shares = np.zeros(len(close.columns))
        nav = capital
        selected_by_entry = decisions[(horizon, fold, strategy)]
        assert sorted(selected_by_entry) == observed.date.iloc[:-1:horizon].tolist()
        for index, row in enumerate(observed.itertuples(index=False)):
            quote = close.loc[row.date].to_numpy()
            shares *= split.loc[row.date].to_numpy()
            pretrade = shares * quote
            marked = pretrade.sum() if index else capital
            gross = marked / nav - 1
            traded = fee_dollars = 0.0
            if index == len(observed) - 1:
                traded = pretrade.sum()
                fee_dollars = fee * traded
                wealth = marked - fee_dollars
                shares[:] = 0
            elif row.date in selected_by_entry:
                selected = selected_by_entry[row.date]
                target = np.zeros(len(close.columns))
                target[close.columns.get_indexer(selected.ticker)] = selected.target_weight
                invested = (
                    brentq(
                        lambda amount, rate, weights, before, wealth: (
                            amount + rate * np.abs(amount * weights - before).sum() - wealth
                        ),
                        0,
                        marked,
                        args=(fee, target, pretrade, marked),
                        xtol=1e-8,
                        rtol=1e-14,
                    )
                    if fee
                    else marked
                )
                positions = invested * target
                traded = np.abs(positions - pretrade).sum()
                fee_dollars = fee * traded
                shares = positions / quote
                wealth = positions.sum()
            else:
                wealth = marked
            expected = [wealth / nav - 1, gross, fee_dollars / nav, traded / nav]
            actual = [observed.iloc[index]["return"], row.gross_return, row.cost, row.turnover]
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(row.nav, wealth, rtol=1e-10, atol=1e-7)
            max_nav_relative_error = max(max_nav_relative_error, abs(row.nav / wealth - 1))
            nav = wealth
            n_daily += 1
    print(
        f"{n_daily} daily portfolio marks and self-financing trades independently replayed",
        flush=True,
    )
    # Independent scalar-draw bootstrap reproduces the vectorized inference.
    comparisons = pd.read_csv(DIRECTORY / "comparisons.csv")
    adoption = pd.read_csv(DIRECTORY / "adoption.csv")
    actual_pvalues = []
    for horizon in config["backtest"]["horizons"]:
        frame = episodes[episodes.horizon == horizon]
        wide = frame.pivot(
            index=["fold", "entry_date", "exit_date"], columns="strategy", values="price_return"
        ).sort_index()
        for block_sessions in [
            config["research"]["bootstrap_block_sessions"],
            *config["research"]["bootstrap_sensitivity_sessions"],
        ]:
            rng = np.random.default_rng(config["research"]["seed"])
            draws = config["research"]["bootstrap_draws"]
            block = int(np.ceil(block_sessions / horizon))
            sampled = []
            folds = wide.index.get_level_values("fold").to_numpy()
            for fold in sorted(set(folds)):
                positions = np.flatnonzero(folds == fold)
                length = len(positions)
                starts = rng.integers(0, length, size=(draws, (length + block - 1) // block))
                local = np.empty((draws, length), dtype=int)
                for draw in range(draws):
                    local[draw] = positions[
                        np.concatenate(
                            [(np.arange(block) + start) % length for start in starts[draw]]
                        )[:length]
                    ]
                sampled.append(local)
            indices = np.concatenate(sampled, axis=1)
            subset = comparisons[
                (comparisons.horizon == horizon) & (comparisons.block_sessions == block_sessions)
            ]
            for row in subset.itertuples(index=False):
                values = (wide[row.first] - wide[row.second]).to_numpy()
                boot = values[indices].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
                pvalue = (1 + (np.abs(boot - values.mean()) >= abs(values.mean())).sum()) / (
                    draws + 1
                )
                np.testing.assert_allclose(
                    [row.mean_difference, row.ci_lower, row.ci_upper, row.pvalue],
                    [values.mean(), low, high, pvalue],
                    rtol=1e-9,
                    atol=1e-12,
                )
            if block_sessions == config["research"]["bootstrap_block_sessions"]:
                values = (wide.nested - wide.incumbent).to_numpy()
                boot = values[indices].mean(axis=1)
                pvalue = (1 + (boot - values.mean() >= values.mean()).sum()) / (draws + 1)
                row = adoption[adoption.horizon == horizon].iloc[0]
                np.testing.assert_allclose(row.pvalue, pvalue, atol=1e-12)
                actual_pvalues.append(pvalue)
    # Independent Holm closed-testing expression for the adoption family.
    order = np.argsort(actual_pvalues)
    adjusted = np.empty(len(order))
    for rank, position in enumerate(order):
        adjusted[position] = min(
            1,
            max(
                (len(order) - earlier) * actual_pvalues[order[earlier]]
                for earlier in range(rank + 1)
            ),
        )
    np.testing.assert_allclose(adoption.pvalue_holm_horizons, adjusted, atol=1e-12)
    report = {
        "episode_rows": n_episode,
        "daily_rows": n_daily,
        "fit_boundaries": len(fits),
        "annual_selections": len(selections),
        "saved_incumbent_vintages_matched": incumbent_matches,
        "maximum_episode_absolute_error": max_episode_error,
        "maximum_nav_relative_error": max_nav_relative_error,
        "bootstrap_comparisons": len(comparisons),
        "adoption_checks": len(adoption),
        "status": "passed",
        "source_commit": manifest["git_commit"],
    }
    (DIRECTORY / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
