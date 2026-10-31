"""Integration defenses for nested selection, paired inference, and rollback gates."""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_ensemble.cli import main
from stock_ensemble.features import build_features
from stock_ensemble.inference import (
    block_indices,
    holm_adjust,
    paired_difference,
    paired_panel,
    research_inference,
)
from stock_ensemble.research import inner_blocks, run_research
from stock_ensemble.research_reporting import forecast_metrics, write_research


@pytest.fixture
def research_market():
    dates = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(7008)
    rows = []
    for ticker in "ABCDEF":
        close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(dates)))
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
        "research": {
            "outer_start": "2020-01-01",
            "train_window": 180,
            "min_train_dates": 60,
            "inner_folds": 2,
            "inner_sessions": 40,
            "ridge_alphas": [1.0, 100.0],
            "boosted_depths": [2, 3],
            "boosted_iterations": 6,
            "boosted_learning_rate": 0.05,
            "boosted_min_leaf": 10,
            "allocation_penalties": [0.01, 0.1],
            "min_weight": 0.05,
            "max_weight": 0.4,
            "cost_scenarios_bps": [0.0, 10.0],
            "bootstrap_draws": 100,
            "bootstrap_block_sessions": 20,
            "bootstrap_sensitivity_sessions": [10, 40],
            "seed": 7008,
        },
    }
    return dates, prices, earnings, config


def test_nested_pipeline_has_purged_choices_same_stock_ablations_and_shared_accounting(
    research_market, tmp_path
):
    _dates, prices, earnings, config = research_market
    features = build_features(prices, earnings, config)
    result = run_research(prices, features, config)
    assert result["episodes"].strategy.nunique() == 9
    fits = result["fits"]
    assert all(row["label_end"] < row["refit_date"] for row in fits)
    assert all(row["train_end"] < row["refit_date"] for row in fits)
    assert any(row["train_dates"] > 100 for row in fits if not row["legacy"])
    assert all(row["train_dates"] <= 100 for row in fits if row["legacy"])
    assert result["selection_scores"].inner_exit.lt(result["selection_scores"].selection_date).all()
    holdings = result["holdings"]
    keys = ["horizon", "fold", "strategy", "entry_date"]
    np.testing.assert_allclose(holdings.groupby(keys).target_weight.sum(), 1)
    assert holdings.groupby(keys).size().eq(3).all()
    assert holdings.signal_date.lt(holdings.entry_date).all()
    assert holdings.refit_date.le(holdings.signal_date).all()
    for family in ("ridge", "boosted"):
        equal = holdings[holdings.strategy == family + "_equal"]
        regular = holdings[holdings.strategy == family + "_regularized"]
        compare = ["horizon", "fold", "entry_date", "ticker", "candidate", "predicted_return"]
        assert_frame_equal(
            equal[compare].reset_index(drop=True), regular[compare].reset_index(drop=True)
        )
        assert regular.target_weight.between(0.05, 0.4).all()
    returns = result["returns"]
    np.testing.assert_allclose(returns["return"], returns.gross_return - returns.cost, atol=1e-12)
    np.testing.assert_allclose(
        returns.cost, returns.turnover * returns.cost_bps / 10000, atol=1e-12
    )
    for _, group in returns.groupby(["horizon", "fold", "strategy", "cost_bps"]):
        assert group.iloc[0].gross_return == 0
        assert group.iloc[-1].turnover > 0
    # Independently compound daily marks between complete entry and exit closes.
    for (horizon, strategy), group in result["episodes"].groupby(["horizon", "strategy"]):
        daily = (
            returns[
                (returns.horizon == horizon)
                & (returns.strategy == strategy)
                & (returns.cost_bps == 0)
            ]
            .set_index("date")
            .gross_return
        )
        for row in group.itertuples():
            observed = (
                1 + daily.loc[(daily.index > row.entry_date) & (daily.index <= row.exit_date)]
            ).prod() - 1
            assert row.price_return == pytest.approx(observed, abs=1e-12)
    assert np.isfinite(forecast_metrics(result["forecasts"]).mse).all()
    config["output_dir"] = str(tmp_path)
    write_research(result, {"fixture": True}, config)
    assert (tmp_path / "adoption.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "forecast_metrics.csv").exists()


def test_future_data_cannot_revise_research_selection_or_earlier_holdings(research_market):
    dates, prices, earnings, config = research_market
    config = copy.deepcopy(config)
    config["backtest"]["horizons"] = [5]
    cutoff = dates[550]
    config["backtest"]["end"] = str(cutoff.date())
    original = run_research(prices, build_features(prices, earnings, config), config)
    altered = prices.copy()
    altered.loc[altered.date > cutoff, ["close", "adj_close"]] *= 100
    extra = earnings.assign(filed_date=dates[560], fiscal_end=dates[555], surprise=-5)
    changed_features = build_features(altered, pd.concat([earnings, extra]), config)
    changed = run_research(altered, changed_features, config)
    assert original["selections"] == changed["selections"]
    for name in ("holdings", "episodes", "returns", "selection_scores", "forecasts"):
        assert_frame_equal(original[name], changed[name])
    truncated = prices[prices.date <= cutoff]
    live = run_research(truncated, build_features(truncated, earnings, config), config)
    assert original["selections"] == live["selections"]
    assert_frame_equal(original["holdings"], live["holdings"])
    # Outer-year outcomes may change the realized score, never its prior choice.
    selection = original["selections"][0]["selection_date"]
    changed_outer = prices.copy()
    mask = changed_outer.date > selection
    changed_outer.loc[mask & changed_outer.ticker.eq("A"), ["close", "adj_close"]] *= 3
    outer = run_research(changed_outer, build_features(changed_outer, earnings, config), config)
    assert original["selections"] == outer["selections"]
    assert_frame_equal(original["selection_scores"], outer["selection_scores"])


def test_inner_boundaries_and_configuration_cannot_be_silently_ignored(research_market, capsys):
    dates, prices, earnings, config = research_market
    blocks = inner_blocks(dates, dates[500], 3, 40)
    assert blocks[-1][-1] < dates[500]
    assert all(len(block) == 41 for block in blocks)
    assert all(blocks[i][-1] == blocks[i + 1][0] for i in range(2))
    assert main(["--config", "samples/research.json", "backtest"]) == 1
    assert "require the research command" in capsys.readouterr().err
    bad = copy.deepcopy(config)
    bad["research"]["min_train_dates"] = 999
    with pytest.raises(ValueError, match="minimum training dates"):
        run_research(prices, build_features(prices, earnings, config), bad)


def episode_fixture():
    rows = []
    strategies = [
        "ridge_equal",
        "boosted_equal",
        "ridge_regularized",
        "boosted_regularized",
        "nested",
        "momentum",
        "equal_signal",
        "rank_ensemble",
        "incumbent",
    ]
    for horizon in (1, 5, 19):
        for index, entry in enumerate(pd.bdate_range("2020-01-01", periods=30)):
            for strategy in strategies:
                value = 0.01 + 0.02 * np.sin(index)
                if strategy == "nested":
                    value += 0.005
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
    return pd.DataFrame(rows)


def test_paired_bootstrap_holm_and_adoption_reject_false_improvement(research_market):
    episodes = episode_fixture()
    config = research_market[-1]
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    panel = paired_panel(episodes, 5)
    indices = block_indices(panel, 5, config["research"], 20)
    positive = paired_difference(panel, "nested", "incumbent", indices, one_sided=True)
    assert positive["ci_lower"] == pytest.approx(0.005)
    zero = paired_difference(panel, "ridge_equal", "equal_signal", indices)
    assert zero["pvalue"] == 1 and zero["mean_difference"] == 0
    result = research_inference(episodes, config)
    assert result["adoption"].qualifies.all()
    assert len(result["comparisons"]) == 54
    shuffled = research_inference(episodes.sample(frac=1, random_state=7), config)
    assert_frame_equal(result["comparisons"], shuffled["comparisons"])
    assert_frame_equal(result["adoption"], shuffled["adoption"])
    # A negative shift cannot pass a greater-than-zero test or descriptive safeguards.
    episodes.loc[episodes.strategy == "nested", "price_return"] -= 0.02
    failed = research_inference(episodes, config)
    assert not failed["adoption"].qualifies.any()
    damaged = episodes.drop(episodes[episodes.strategy == "incumbent"].index[0])
    with pytest.raises(ValueError, match="Unmatched"):
        paired_panel(damaged, 1)
