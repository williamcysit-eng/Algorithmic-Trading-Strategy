"""Exact subset objectives, time availability, and shared accounting invariants."""

import copy
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_ensemble.cli import main
from stock_ensemble.features import build_features
from stock_ensemble.selection import METHODS, choose_subsets, shortlist_covariance
from stock_ensemble.selection_inference import (
    holm,
    inverse_utility,
    paired_panel,
    selection_inference,
    utility,
)
from stock_ensemble.selection_reporting import write_selection_results
from stock_ensemble.selection_study import run_selection_study


@pytest.fixture
def market():
    dates = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(7008)
    rows = []
    common = rng.normal(0, 0.008, len(dates))
    for index, ticker in enumerate("ABCDEF"):
        close = 100 * np.cumprod(1 + common + rng.normal(0.0004, 0.006 + 0.001 * index, len(dates)))
        rows.extend(
            {
                "date": date,
                "ticker": ticker,
                "close": value,
                "adj_close": value,
                "volume": 1_000_000,
                "dividend": 0.0,
                "split": 1.0,
            }
            for date, value in zip(dates, close)
        )
    prices = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    earnings = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "filed_date": dates[470],
                "fiscal_end": dates[450],
                "surprise": index - 3,
                "seasonal_change": 0.01,
                "source_accn": ticker,
            }
            for index, ticker in enumerate("ABCDEF")
        ]
    )
    universe = pd.DataFrame({"ticker": list("ABCDEF"), "sector": ["X", "X", "Y", "Y", "Z", "Z"]})
    config = {
        "features": {
            "momentum_lookback": 10,
            "momentum_skip": 2,
            "volatility_window": 5,
            "earnings_half_life": 45,
            "earnings_max_age": 180,
            "gate_half_life": 10,
            "min_price": 5,
            "min_dollar_volume": 0,
            "min_cross_section": 6,
        },
        "model": {"train_window": 100, "min_train_dates": 40, "ridge": 0.05},
        "backtest": {
            "start": "2020-01-01",
            "holdout_start": "2020-01-01",
            "horizons": [1, 5, 19],
            "top_n": 3,
            "cost_bps": 0.0,
            "return_mode": "price",
            "initial_capital": 1000.0,
        },
        "selection_study": {
            "status": "exploratory_retrospective",
            "shortlist_size": 6,
            "sector_cap": 1,
            "score_budget": 0.05,
            "covariance_window": 60,
            "primary_policy": "joint",
            "primary_gamma": 3.0,
            "gammas": [0.0, 1.0, 3.0, 5.0, 10.0],
            "bootstrap_draws": 100,
            "bootstrap_block_sessions": 20,
            "bootstrap_sensitivity_sessions": [10, 40],
            "seed": 7008,
            "cost_scenarios_bps": [0.0, 10.0],
            "annualized_return_sacrifice_limit": 0.02,
        },
    }
    return dates, prices, earnings, universe, config


def test_joint_selector_matches_independent_enumeration_and_row_permutations():
    rng = np.random.default_rng(7008)
    ranking = pd.DataFrame(
        {
            "ticker": list("ABCDEFGHIJ"),
            "score": np.linspace(1.0, 0.55, 10),
            "sector": ["X"] * 4 + ["Y"] * 3 + ["Z"] * 3,
        }
    )
    for _ in range(15):
        x = rng.normal(size=(10, 10))
        covariance = x.T @ x + np.eye(10)
        choices, detail = choose_subsets(ranking, covariance)
        alternatives = []
        for names in combinations(ranking.ticker, 5):
            positions = [ranking.index[ranking.ticker.eq(name)][0] for name in names]
            selected = ranking.iloc[positions]
            cap = selected.sector.value_counts().max()
            variance = sum(covariance[i, j] for i in positions for j in positions) / 25
            alternatives.append((names, selected.score.mean(), cap, variance))
        capped = [row for row in alternatives if row[2] <= 2]
        best_score = max(row[1] for row in capped)
        allowed = [row for row in capped if row[1] >= best_score - 0.05 - 1e-12]
        expected = min(allowed, key=lambda row: (row[3], -row[1], row[0]))
        assert tuple(choices["joint"].ticker) == expected[0]
        assert choices["incumbent"].ticker.tolist() == list("ABCDE")
        assert choices["joint"].sector.value_counts().max() <= 2
        assert choices["joint"].score.mean() >= best_score - 0.05 - 1e-12
        permuted = rng.permutation(10)
        repeated, _ = choose_subsets(ranking.iloc[permuted], covariance[np.ix_(permuted, permuted)])
        for method in METHODS:
            assert_frame_equal(
                choices[method].reset_index(drop=True), repeated[method].reset_index(drop=True)
            )
        assert len(detail) == 4
    # A single-sector shortlist must declare relaxation, not quietly drop names.
    choices, detail = choose_subsets(ranking.assign(sector="X"), np.eye(10))
    assert all(row["effective_sector_cap"] == 5 and row["cap_relaxed"] for row in detail)
    assert all(len(frame) == 5 for frame in choices.values())
    # With identical scores and risks, deterministic ticker ties remain stable.
    choices, _ = choose_subsets(ranking.assign(score=0.0, sector="X"), np.eye(10))
    assert all(frame.ticker.tolist() == list("ABCDE") for frame in choices.values())


def test_covariance_cutoff_and_missing_history_fail_closed():
    rng = np.random.default_rng(7008)
    dates = pd.bdate_range("2020-01-01", periods=120)
    factors = pd.DataFrame(1 + rng.normal(0, 0.01, (120, 6)), index=dates, columns=list("ABCDEF"))
    matrix, metadata = shortlist_covariance(factors, list("ABCDEF"), dates[80], 60)
    changed = factors.copy()
    changed.loc[changed.index > dates[80]] = 50
    repeated, repeated_meta = shortlist_covariance(changed, list("ABCDEF"), dates[80], 60)
    np.testing.assert_array_equal(matrix, repeated)
    assert metadata == repeated_meta
    assert metadata["history_start"] == dates[21] and metadata["history_end"] == dates[80]
    factors.loc[dates[40], "A"] = np.nan
    with pytest.raises(ValueError, match="cannot fill or compress"):
        shortlist_covariance(factors, list("ABCDEF"), dates[80], 60)


def test_study_selects_equal_weight_subsets_and_reconciles_realized_paths(market, tmp_path):
    _dates, prices, earnings, universe, config = market
    result = run_selection_study(prices, build_features(prices, earnings, config), universe, config)
    holdings, fits = result["holdings"], result["fits"]
    assert fits.label_end.lt(fits.refit_date).all() and fits.train_end.lt(fits.refit_date).all()
    assert holdings.signal_date.lt(holdings.entry_date).all()
    assert holdings.refit_date.le(holdings.signal_date).all()
    keys = ["horizon", "fold", "strategy", "entry_date"]
    np.testing.assert_allclose(holdings.groupby(keys).target_weight.sum(), 1)
    np.testing.assert_allclose(holdings.target_weight, 1 / 3)
    assert holdings.groupby(keys).size().eq(3).all()
    assert set(holdings.strategy) == set(METHODS)
    rows = result["selection_diagnostics"]
    assert rows[rows.strategy.eq("joint")].max_sector_weight.le(1 / 3 + 1e-12).all()
    daily = result["returns"]
    np.testing.assert_allclose(daily["return"], daily.gross_return - daily.cost, atol=1e-12)
    np.testing.assert_allclose(daily.cost, daily.turnover * daily.cost_bps / 10000, atol=1e-12)
    for row in result["episodes"].itertuples(index=False):
        period = daily[
            (daily.horizon == row.horizon)
            & (daily.fold == row.fold)
            & daily.strategy.eq(row.strategy)
            & daily.cost_bps.eq(0)
            & daily.date.gt(row.entry_date)
            & daily.date.le(row.exit_date)
        ]
        assert len(period) == row.horizon
        assert row.price_return == pytest.approx((1 + period.gross_return).prod() - 1, abs=1e-12)
    config["output_dir"] = str(tmp_path)
    inference = write_selection_results(result, {"fixture": True}, config)
    assert len(inference["comparisons"]) == 45 and len(inference["adoption"]) == 3
    assert (tmp_path / "selection_summary.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()


def test_future_prices_filings_and_truncation_cannot_revise_selections(market):
    dates, prices, earnings, universe, config = market
    config = copy.deepcopy(config)
    config["backtest"]["horizons"] = [5]
    cutoff = dates[550]
    config["backtest"]["end"] = str(cutoff.date())
    expected = run_selection_study(
        prices, build_features(prices, earnings, config), universe, config
    )
    altered = prices.copy()
    altered.loc[altered.date > cutoff, ["close", "adj_close"]] *= 100
    later = earnings.assign(filed_date=dates[560], fiscal_end=dates[555], surprise=-5)
    actual = run_selection_study(
        altered, build_features(altered, pd.concat([earnings, later]), config), universe, config
    )
    for name in ["holdings", "shortlists", "episodes", "returns", "selection_diagnostics", "fits"]:
        assert_frame_equal(expected[name], actual[name])
    truncated = prices[prices.date <= cutoff]
    live = run_selection_study(
        truncated, build_features(truncated, earnings, config), universe, config
    )
    assert_frame_equal(expected["holdings"], live["holdings"])
    assert expected["risk_models"] == live["risk_models"]


def test_utility_inference_can_accept_or_reject_and_requires_matched_episodes(market, capsys):
    config = market[-1]
    rows = []
    for horizon in config["backtest"]["horizons"]:
        for index, entry in enumerate(pd.bdate_range("2020-01-01", periods=40)):
            for method in METHODS:
                value = 0.005 + 0.025 * np.sin(index)
                if method == "joint":
                    value = 0.025 + 0.01 * np.sin(index)
                rows.append(
                    {
                        "horizon": horizon,
                        "fold": 2020,
                        "entry_date": entry,
                        "exit_date": entry + pd.offsets.BDay(horizon),
                        "strategy": method,
                        "price_return": value,
                    }
                )
    episodes = pd.DataFrame(rows)
    expected = selection_inference(episodes, config)
    assert expected["adoption"].qualifies.all()
    actual = selection_inference(episodes.sample(frac=1, random_state=7), config)
    assert_frame_equal(expected["adoption"], actual["adoption"])
    assert_frame_equal(expected["comparisons"], actual["comparisons"])
    episodes.loc[episodes.strategy.eq("joint"), "price_return"] -= 0.05
    assert not selection_inference(episodes, config)["adoption"].qualifies.any()
    damaged = episodes.drop(episodes[episodes.strategy.eq("incumbent")].index[0])
    with pytest.raises(ValueError, match="Unmatched"):
        paired_panel(damaged, 1)
    values = np.array([-0.1, 0.12])
    assert inverse_utility(utility(values, 0).mean(), 0) == pytest.approx(values.mean())
    assert inverse_utility(utility(values, 1).mean(), 1) == pytest.approx(np.sqrt(0.9 * 1.12) - 1)
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert main(["--config", "samples/selection.json", "backtest"]) == 1
    assert "require the selection-study command" in capsys.readouterr().err
