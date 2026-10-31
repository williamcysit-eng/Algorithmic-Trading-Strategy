"""Behavioral checks for risk cutoffs, bounded QP, utilities, and isolated allocation."""

import copy
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.optimize import minimize

from stock_ensemble.cli import main
from stock_ensemble.features import build_features
from stock_ensemble.risk import (
    allocation_weights,
    bounded_simplex,
    dated_covariance,
    minimum_variance,
)
from stock_ensemble.risk_inference import (
    METHODS,
    bootstrap_indices,
    certainty_equivalent,
    holm,
    inverse_utility,
    paired_panel,
    risk_inference,
    utility,
)
from stock_ensemble.risk_reporting import write_risk_results
from stock_ensemble.risk_study import run_risk_study


@pytest.fixture
def risk_market():
    dates = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(7008)
    rows = []
    common = rng.normal(0, 0.008, len(dates))
    for index, ticker in enumerate("ABCDEF"):
        values = 100 * np.cumprod(
            1 + common + rng.normal(0.0004, 0.004 + 0.002 * index, len(dates))
        )
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
            for date, value in zip(dates, values)
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
        "risk_study": {
            "status": "exploratory_retrospective",
            "covariance_window": 60,
            "min_weight": 0.05,
            "max_weight": 0.6,
            "primary_policy": "balanced",
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
    return dates, prices, earnings, config


def test_bounded_optimizer_matches_independent_scipy_solver():
    rng = np.random.default_rng(7008)
    for _ in range(80):
        factor = rng.normal(size=(5, 5))
        covariance = factor.T @ factor + 0.01 * np.eye(5)
        actual = minimum_variance(covariance)
        equal = np.full(5, 0.2)
        result = minimize(
            lambda w, matrix: w @ matrix @ w,
            equal,
            args=(covariance,),
            jac=lambda w, matrix: 2 * matrix @ w,
            method="SLSQP",
            bounds=[(0.05, 0.4)] * 5,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        assert result.success
        assert actual @ covariance @ actual <= result.fun + 1e-9
        assert actual @ covariance @ actual <= equal @ covariance @ equal + 1e-12
        assert actual.min() >= 0.05 and actual.max() <= 0.4
        assert actual.sum() == pytest.approx(1, abs=1e-12)
        np.testing.assert_allclose(minimum_variance(covariance * 1e-8), actual, atol=1e-10)
    inverse_variance = 1 / np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(
        minimum_variance(np.diag([1.0, 4.0, 9.0]), upper=0.9),
        inverse_variance / inverse_variance.sum(),
        atol=1e-12,
    )
    np.testing.assert_allclose(minimum_variance(np.ones((5, 5))), 0.2)
    np.testing.assert_allclose(
        bounded_simplex([2.0, 0.0, 0.0, 0.0, 0.0]), [0.4, 0.15, 0.15, 0.15, 0.15]
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        minimum_variance(np.diag([-1.0, 1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="infeasible"):
        minimum_variance(np.eye(2))


def test_covariance_uses_exact_past_rows_and_allocation_limits():
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(7008)
    factors = pd.DataFrame(1 + rng.normal(0, 0.02, (120, 5)), index=dates, columns=list("ABCDE"))
    covariance, metadata = dated_covariance(factors, list("ABCDE"), dates[80], 60)
    changed = factors.copy()
    changed.loc[changed.index > dates[80]] = 2
    repeated, repeated_metadata = dated_covariance(changed, list("ABCDE"), dates[80], 60)
    np.testing.assert_array_equal(covariance, repeated)
    assert metadata == repeated_metadata
    assert metadata["history_end"] == dates[80]
    assert metadata["history_start"] == dates[21]
    assert metadata["n_observations"] == 60
    allocations = allocation_weights(covariance)
    np.testing.assert_allclose(
        allocations["balanced"], (allocations["equal"] + allocations["minimum_variance"]) / 2
    )
    assert allocations["balanced"].min() >= 0.125
    assert allocations["balanced"].max() <= 0.3 + 1e-12
    factors.loc[dates[30], "A"] = np.nan
    with pytest.raises(ValueError, match="cannot fill or compress"):
        dated_covariance(factors, list("ABCDE"), dates[80], 60)


def test_crra_preferences_are_correct_and_distinguish_mean_from_risk():
    returns = np.array([-0.1, 0.12])
    assert certainty_equivalent(returns, 0) == pytest.approx(0.01)
    assert certainty_equivalent(returns, 1) == pytest.approx(np.sqrt(0.9 * 1.12) - 1)
    equivalents = [certainty_equivalent(returns, gamma) for gamma in [0, 1, 3, 5, 10]]
    assert all(first > second for first, second in pairwise(equivalents))
    for gamma in [0, 1, 3, 5, 10]:
        np.testing.assert_allclose(
            inverse_utility(utility(returns, gamma), gamma), returns, atol=1e-14
        )
        assert certainty_equivalent(np.full(20, 0.02), gamma) == pytest.approx(0.02)
        assert certainty_equivalent(returns, gamma) <= returns.mean() + 1e-14
    with pytest.raises(ValueError, match="returns > -1"):
        utility([-1, 0.1], 3)
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_risk_study_keeps_stocks_purges_fits_and_reconciles_episodes(risk_market, tmp_path):
    _dates, prices, earnings, config = risk_market
    result = run_risk_study(prices, build_features(prices, earnings, config), config)
    holdings = result["holdings"]
    fits = result["fits"]
    assert fits.label_end.lt(fits.refit_date).all()
    assert fits.train_end.lt(fits.refit_date).all()
    assert holdings.signal_date.lt(holdings.entry_date).all()
    assert holdings.refit_date.le(holdings.signal_date).all()
    keys = ["horizon", "fold", "entry_date"]
    selected = holdings.groupby(keys + ["strategy"]).ticker.apply(tuple).unstack("strategy")
    assert selected.apply(lambda row: len(set(row)), axis=1).eq(1).all()
    np.testing.assert_allclose(holdings.groupby(keys + ["strategy"]).target_weight.sum(), 1)
    assert holdings.target_weight.between(0.05, 0.6).all()
    returns = result["returns"]
    np.testing.assert_allclose(returns["return"], returns.gross_return - returns.cost, atol=1e-12)
    np.testing.assert_allclose(
        returns.cost, returns.turnover * returns.cost_bps / 10000, atol=1e-12
    )
    for row in result["episodes"].itertuples(index=False):
        daily = returns[
            (returns.horizon == row.horizon)
            & (returns.strategy == row.strategy)
            & (returns.fold == row.fold)
            & (returns.cost_bps == 0)
            & (returns.date > row.entry_date)
            & (returns.date <= row.exit_date)
        ]
        assert len(daily) == row.horizon
        expected = (1 + daily.gross_return).prod() - 1
        assert row.price_return == pytest.approx(expected, abs=1e-12)
    config["output_dir"] = str(tmp_path)
    inference = write_risk_results(result, {"fixture": True}, config)
    assert len(inference["comparisons"]) == 45
    assert len(inference["adoption"]) == 3
    assert (tmp_path / "risk_preferences.pdf").exists()
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "allocation_metrics.csv").exists()


def test_future_prices_cannot_revise_risk_study_decisions(risk_market):
    dates, prices, earnings, config = risk_market
    config = copy.deepcopy(config)
    config["backtest"]["horizons"] = [5]
    cutoff = dates[550]
    config["backtest"]["end"] = str(cutoff.date())
    expected = run_risk_study(prices, build_features(prices, earnings, config), config)
    changed = prices.copy()
    changed.loc[changed.date > cutoff, ["close", "adj_close"]] *= 50
    actual = run_risk_study(changed, build_features(changed, earnings, config), config)
    for name in ["returns", "episodes", "holdings", "fits"]:
        assert_frame_equal(expected[name], actual[name])
    assert expected["risk_models"] == actual["risk_models"]
    truncated = prices[prices.date <= cutoff]
    live = run_risk_study(truncated, build_features(truncated, earnings, config), config)
    assert_frame_equal(expected["holdings"], live["holdings"])
    assert expected["risk_models"] == live["risk_models"]


def test_paired_utility_inference_and_rejection_cannot_cherry_pick(risk_market, capsys):
    config = risk_market[-1]
    rows = []
    for horizon in config["backtest"]["horizons"]:
        for index, entry in enumerate(pd.bdate_range("2020-01-01", periods=40)):
            for strategy in METHODS:
                value = 0.005 + 0.025 * np.sin(index)
                if strategy == "balanced":
                    value = 0.005 + 0.01 * np.sin(index)
                rows.append(
                    {
                        "horizon": horizon,
                        "fold": 2020,
                        "entry_date": entry,
                        "exit_date": entry + pd.offsets.BDay(horizon),
                        "strategy": strategy,
                        "price_return": value,
                    }
                )
    episodes = pd.DataFrame(rows)
    expected = risk_inference(episodes, config)
    shuffled = risk_inference(episodes.sample(frac=1, random_state=7), config)
    assert_frame_equal(expected["comparisons"], shuffled["comparisons"])
    assert_frame_equal(expected["adoption"], shuffled["adoption"])
    panel = paired_panel(episodes, 5)
    assert bootstrap_indices(panel, 5, config["risk_study"], 20).shape == (100, 40)
    assert (expected["adoption"].balanced_volatility < expected["adoption"].equal_volatility).all()
    # A deterministic gain plus lower variability should pass every safeguard.
    episodes.loc[episodes.strategy == "balanced", "price_return"] += .02
    assert risk_inference(episodes, config)["adoption"].qualifies.all()
    # A small quiet return is not acceptable when it sacrifices too much mean.
    episodes.loc[episodes.strategy == "balanced", "price_return"] -= .04
    assert not risk_inference(episodes, config)["adoption"].qualifies.any()
    damaged = episodes.drop(episodes[episodes.strategy == "equal"].index[0])
    with pytest.raises(ValueError, match="Unmatched"):
        paired_panel(damaged, 1)
    assert main(["--config", "samples/risk.json", "backtest"]) == 1
    assert "require the risk-study command" in capsys.readouterr().err
