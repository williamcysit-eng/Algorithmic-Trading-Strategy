"""Independent audit of joint stock selection, accounting, and utility inference.

Run from repository root: uv run --frozen python docs/research/audit_selection.py
[results/selection_study]. No experimental selector or inference module is imported.
"""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path.cwd()
DIRECTORY = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/selection_study")
METHODS = ["incumbent", "sector_cap", "variance_guarded", "joint"]


def crra(values, gamma):
    wealth = 1 + np.asarray(values)
    return np.log(wealth) if gamma == 1 else (wealth ** (1 - gamma) - 1) / (1 - gamma)


def ce(value, gamma):
    return np.exp(value) - 1 if gamma == 1 else (1 + (1 - gamma) * value) ** (1 / (1 - gamma)) - 1


def adjusted_pvalues(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    for rank, position in enumerate(order):
        adjusted[position] = min(
            1, max((len(order) - earlier) * values[order[earlier]] for earlier in range(rank + 1))
        )
    return adjusted


def main():
    manifest = json.loads((DIRECTORY / "run_manifest.json").read_text())
    config = manifest["config"]
    settings = config["selection_study"]
    prices = pd.read_csv(Path(config["data_dir"]) / "prices.csv", parse_dates=["date"])
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    split = prices.pivot(index="date", columns="ticker", values="split").reindex_like(close)
    factors, unit_value = close / close.shift() * split, close * split.cumprod()
    holdings = pd.read_csv(
        DIRECTORY / "holdings.csv.gz", parse_dates=["entry_date", "signal_date", "refit_date"]
    )
    shortlists = pd.read_csv(
        DIRECTORY / "shortlists.csv.gz", parse_dates=["entry_date", "signal_date", "refit_date"]
    )
    episodes = pd.read_csv(DIRECTORY / "episodes.csv", parse_dates=["entry_date", "exit_date"])
    daily = pd.read_csv(DIRECTORY / "returns.csv", parse_dates=["date"])
    fits = pd.read_csv(DIRECTORY / "fits.csv", parse_dates=["refit_date", "train_end", "label_end"])
    diagnostics = pd.read_csv(DIRECTORY / "selection_diagnostics.csv")
    risk_models = json.loads((DIRECTORY / "risk_models.json").read_text())
    assert fits.label_end.lt(fits.refit_date).all() and fits.train_end.lt(fits.refit_date).all()
    assert holdings.signal_date.lt(holdings.entry_date).all()
    assert holdings.refit_date.le(holdings.signal_date).all()
    count = config["backtest"]["top_n"]
    np.testing.assert_allclose(holdings.target_weight, 1 / count)
    keys = ["horizon", "fold", "strategy"]
    choices = {
        key: {entry: group for entry, group in frame.groupby("entry_date")}
        for key, frame in holdings.groupby(keys)
    }
    assert holdings.groupby(keys + ["entry_date"]).size().eq(count).all()
    # Reconstruct covariance directly from sample moments rather than sklearn.
    covariances, covariance_error = {}, 0.0
    for record in risk_models:
        signal = pd.Timestamp(record["signal_date"])
        end = factors.index.get_loc(signal)
        history = (
            factors.loc[:, record["tickers"]].iloc[
                end - settings["covariance_window"] + 1 : end + 1
            ]
            - 1
        )
        assert (
            len(history) == settings["covariance_window"] and np.isfinite(history.to_numpy()).all()
        )
        assert history.index[0] == pd.Timestamp(record["history_start"])
        assert history.index[-1] == pd.Timestamp(record["history_end"]) == signal
        x = history.to_numpy()
        x -= x.mean(axis=0)
        n, k = x.shape
        empirical = x.T @ x / n
        mu = np.trace(empirical) / k
        delta = np.sum((empirical - mu * np.eye(k)) ** 2) / k
        beta = (np.mean(np.sum(x**2, axis=1) ** 2) - np.sum(empirical**2)) / (n * k)
        shrinkage = np.clip(beta / delta, 0, 1) if delta > 0 else 0.0
        covariance = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(k)
        stored = np.asarray(record["covariance"])
        np.testing.assert_allclose(stored, covariance, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(record["shrinkage"], shrinkage, rtol=1e-9, atol=1e-12)
        covariance_error = max(covariance_error, float(np.max(np.abs(stored - covariance))))
        covariances[record["risk_id"]] = stored
    print(f"{len(risk_models)} dated shortlist covariance windows pass", flush=True)
    # Enumerate every admissible subset independently from emitted candidates.
    n_selections = 0
    for (horizon, fold, entry), frame in shortlists.groupby(["horizon", "fold", "entry_date"]):
        frame = frame.sort_values("ticker")
        assert len(frame) <= settings["shortlist_size"]
        assert set(frame["rank"]) == set(range(1, len(frame) + 1))
        matrix = covariances[frame.risk_id.iloc[0]]
        names, scores, sectors = (
            frame.ticker.to_numpy(),
            frame.score.to_numpy(),
            frame.sector.to_numpy(),
        )
        original = tuple(
            sorted(
                frame.sort_values(["score", "ticker"], ascending=[False, True]).iloc[:count].ticker
            )
        )
        sets = []
        for indices in combinations(range(len(frame)), count):
            weights = np.zeros(len(frame))
            weights[list(indices)] = 1 / count
            sets.append(
                {
                    "names": tuple(names[list(indices)]),
                    "score": np.mean(scores[list(indices)]),
                    "cap": max(Counter(sectors[list(indices)]).values()),
                    "variance": float(weights @ matrix @ weights),
                }
            )
        incumbent = next(row for row in sets if row["names"] == original)
        cap = max(settings["sector_cap"], min(row["cap"] for row in sets))
        capped = [row for row in sets if row["cap"] <= cap]
        sector = min(capped, key=lambda row: (-row["score"], row["names"]))
        pools = {
            "variance_guarded": [
                row
                for row in sets
                if row["score"] >= incumbent["score"] - settings["score_budget"] - 1e-12
            ],
            "joint": [
                row
                for row in capped
                if row["score"] >= sector["score"] - settings["score_budget"] - 1e-12
            ],
        }
        expected = {"incumbent": incumbent, "sector_cap": sector}
        for method, pool in pools.items():
            best = min(row["variance"] for row in pool)
            expected[method] = min(
                [row for row in pool if row["variance"] <= best + 1e-14],
                key=lambda row: (-row["score"], row["names"]),
            )
        for method, expected_row in expected.items():
            selected = choices[(horizon, fold, method)][entry]
            assert tuple(sorted(selected.ticker)) == expected_row["names"]
            row = diagnostics[
                (diagnostics.horizon == horizon)
                & (diagnostics.fold == fold)
                & diagnostics.entry_date.eq(str(entry.date()))
                & diagnostics.strategy.eq(method)
            ].iloc[0]
            np.testing.assert_allclose(row.mean_score, expected_row["score"], atol=1e-12)
            np.testing.assert_allclose(
                row.predicted_daily_variance, expected_row["variance"], atol=1e-14
            )
            assert row.effective_sector_cap == cap
            assert row.cap_relaxed == (cap > settings["sector_cap"])
            n_selections += 1
    baseline_matches = 0
    archive = ROOT / "results/risk_study"
    if settings["status"] != "demonstration_only" and (archive / "holdings.csv.gz").exists():
        old = pd.read_csv(
            archive / "holdings.csv.gz", parse_dates=["entry_date", "signal_date", "refit_date"]
        )
        columns = [
            "horizon",
            "fold",
            "entry_date",
            "signal_date",
            "refit_date",
            "ticker",
            "score",
            "target_weight",
        ]
        old = old[old.strategy.eq("equal")].sort_values(["horizon", "entry_date", "ticker"])
        new = holdings[holdings.strategy.eq("incumbent")].sort_values(
            ["horizon", "entry_date", "ticker"]
        )
        pd.testing.assert_frame_equal(
            old[columns].reset_index(drop=True),
            new[columns].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
        baseline_matches = len(new)
    print(
        f"{n_selections} exact subset choices and {baseline_matches} archived incumbent holdings pass",
        flush=True,
    )
    episode_error = 0.0
    for row in episodes.itertuples(index=False):
        selected = choices[(row.horizon, row.fold, row.strategy)][row.entry_date]
        growth = (
            unit_value.loc[row.exit_date, selected.ticker].to_numpy()
            / unit_value.loc[row.entry_date, selected.ticker].to_numpy()
        )
        expected = np.mean(growth) - 1
        np.testing.assert_allclose(row.price_return, expected, atol=1e-12, rtol=1e-10)
        episode_error = max(episode_error, abs(row.price_return - expected))
    nav_error = 0.0
    for key, frame in daily.groupby(keys + ["cost_bps"], sort=False):
        horizon, fold, method, cost_bps = key
        frame = frame.sort_values("date").reset_index(drop=True)
        selected_by_date = choices[(horizon, fold, method)]
        fee = cost_bps / 10000
        shares = np.zeros(len(close.columns))
        nav = config["backtest"]["initial_capital"]
        for index, row in enumerate(frame.itertuples(index=False)):
            shares *= split.loc[row.date].to_numpy()
            quote = close.loc[row.date].to_numpy()
            before = shares * quote
            marked = before.sum() if index else nav
            gross = marked / nav - 1
            traded = cost_dollars = 0.0
            if index == len(frame) - 1:
                traded = before.sum()
                cost_dollars = fee * traded
                wealth = marked - cost_dollars
                shares[:] = 0
            elif row.date in selected_by_date:
                target = np.zeros(len(close.columns))
                target[close.columns.get_indexer(selected_by_date[row.date].ticker)] = 1 / count

                def residual(amount, rate, weights, previous, total):
                    return amount + rate * np.abs(amount * weights - previous).sum() - total

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
                shares, wealth = after / quote, after.sum()
            else:
                wealth = marked
            np.testing.assert_allclose(
                [frame.iloc[index]["return"], row.gross_return, row.cost, row.turnover],
                [wealth / nav - 1, gross, cost_dollars / nav, traded / nav],
                atol=1e-11,
                rtol=1e-9,
            )
            np.testing.assert_allclose(row.nav, wealth, atol=1e-7, rtol=1e-10)
            nav_error = max(nav_error, abs(row.nav / wealth - 1))
            nav = wealth
    print(f"{len(episodes)} episodes and {len(daily)} daily share/fee marks pass", flush=True)
    preferences = pd.read_csv(DIRECTORY / "preferences.csv")
    for row in preferences.itertuples(index=False):
        selected = episodes[(episodes.horizon == row.horizon) & episodes.strategy.eq(row.strategy)]
        if row.period != "all":
            selected = selected[
                selected.entry_date >= pd.Timestamp(config["backtest"]["holdout_start"])
            ]
        np.testing.assert_allclose(
            row.certainty_equivalent,
            ce(crra(selected.price_return, row.gamma).mean(), row.gamma),
            atol=1e-12,
            rtol=1e-9,
        )
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
        for block in [
            settings["bootstrap_block_sessions"],
            *settings["bootstrap_sensitivity_sessions"],
        ]:
            rng = np.random.default_rng(settings["seed"])
            length = int(np.ceil(block / horizon))
            draws = settings["bootstrap_draws"]
            folds = panel.index.get_level_values("fold").to_numpy()
            parts = []
            for fold in sorted(set(folds)):
                positions = np.flatnonzero(folds == fold)
                size = len(positions)
                starts = rng.integers(0, size, size=(draws, (size + length - 1) // length))
                indices = np.empty((draws, size), dtype=int)
                for draw in range(draws):
                    indices[draw] = positions[
                        np.concatenate(
                            [(np.arange(length) + start) % size for start in starts[draw]]
                        )[:size]
                    ]
                parts.append(indices)
            indices = np.concatenate(parts, axis=1)
            table = comparisons if block == settings["bootstrap_block_sessions"] else sensitivity
            for row in table[
                (table.horizon == horizon) & table.block_sessions.eq(block)
            ].itertuples(index=False):
                first, base = crra(panel[row.strategy], row.gamma), crra(panel.incumbent, row.gamma)
                mean = (first - base).mean()
                first_boot, base_boot = first[indices].mean(axis=1), base[indices].mean(axis=1)
                pvalue = (1 + np.count_nonzero((first_boot - base_boot) - mean >= mean)) / (
                    draws + 1
                )
                low, high = np.quantile(
                    ce(first_boot, row.gamma) - ce(base_boot, row.gamma), [0.025, 0.975]
                )
                np.testing.assert_allclose(
                    [row.ce_difference, row.ce_ci_lower, row.ce_ci_upper, row.pvalue_greater],
                    [ce(first.mean(), row.gamma) - ce(base.mean(), row.gamma), low, high, pvalue],
                    atol=1e-12,
                    rtol=1e-8,
                )
    np.testing.assert_allclose(
        comparisons.pvalue_holm_all_preferences,
        adjusted_pvalues(comparisons.pvalue_greater),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adoption.pvalue_holm_primary_horizons, adjusted_pvalues(adoption.pvalue_greater), atol=1e-12
    )
    for row in adoption.itertuples(index=False):
        expected = (
            row.mean_utility_difference > 0
            and row.pvalue_holm_primary_horizons <= 0.05
            and row.joint_volatility < row.incumbent_volatility
            and row.joint_tail_mean >= row.incumbent_tail_mean
            and row.annualized_mean_sacrifice <= settings["annualized_return_sacrifice_limit"]
        )
        if settings["status"] == "demonstration_only":
            expected = False
        assert row.qualifies == expected
    report = {
        "status": "passed",
        "source_commit": manifest["git_commit"],
        "risk_windows": len(risk_models),
        "exact_subset_choices": n_selections,
        "fit_boundaries": len(fits),
        "archived_incumbent_holdings": baseline_matches,
        "episodes": len(episodes),
        "daily_observations": len(daily),
        "covariance_max_absolute_error": covariance_error,
        "episode_max_absolute_error": episode_error,
        "nav_max_relative_error": nav_error,
        "preference_checks": len(preferences),
        "inference_checks": len(comparisons) + len(sensitivity),
        "adoption_checks": len(adoption),
    }
    (DIRECTORY / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
