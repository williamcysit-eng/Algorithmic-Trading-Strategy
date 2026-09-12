"""Behavioral defenses against plausible look-ahead and portfolio-accounting bugs."""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_ensemble.backtest import rank_as_of, run_backtests
from stock_ensemble.data import _yahoo_prices, earnings_from_companyfacts
from stock_ensemble.features import build_features
from stock_ensemble.model import fit_weights


def market_fixture():
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(42)
    rows = []
    for index, ticker in enumerate("ABCDEF"):
        values = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(dates)))
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
                "filed_date": dates[60],
                "fiscal_end": dates[50],
                "surprise": index - 3,
                "seasonal_change": 0.01,
                "source_accn": f"filing-{ticker}",
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
        "model": {"train_window": 40, "min_train_dates": 10, "ridge": 0.05},
        "backtest": {
            "start": str(dates[55].date()),
            "holdout_start": str(dates[75].date()),
            "horizons": [1, 5],
            "top_n": 2,
            "cost_bps": 10,
            "return_mode": "price",
            "initial_capital": 1000,
        },
    }
    return dates, prices, earnings, config


def test_future_prices_and_filings_cannot_change_past_rankings():
    dates, prices, earnings, config = market_fixture()
    cutoff = dates[80]
    features = build_features(prices, earnings, config)
    ranking, fitted = rank_as_of(prices, features, config, cutoff, 5)
    altered = prices.copy()
    future = altered["date"] > cutoff
    altered.loc[future, ["close", "adj_close"]] *= 100
    later = earnings.copy()
    later["filed_date"] = dates[90]
    later["fiscal_end"] = dates[85]
    later["surprise"] = -5
    changed_features = build_features(altered, pd.concat([earnings, later]), config)
    changed_ranking, changed_fit = rank_as_of(altered, changed_features, config, cutoff, 5)
    assert_frame_equal(
        features[features["date"] <= cutoff], changed_features[changed_features["date"] <= cutoff]
    )
    assert_frame_equal(ranking, changed_ranking)
    assert fitted == changed_fit
    assert fitted["label_end"] < fitted["refit_date"]
    # Live scoring cannot require next session's price or the future label.
    truncated = prices[prices["date"] <= cutoff]
    live, _ = rank_as_of(truncated, build_features(truncated, earnings, config), config, cutoff, 5)
    assert_frame_equal(ranking, live)


def test_filing_availability_and_bad_news_gate():
    dates, prices, earnings, config = market_fixture()
    # A temporary loser with freshly bad earnings must get less rebound credit.
    stock = (prices["ticker"] == "A") & (prices["date"] == dates[61])
    prices.loc[stock, ["close", "adj_close"]] *= 0.80
    features = build_features(prices, earnings, config).set_index(["date", "ticker"])
    assert features.loc[(dates[60], "A"), "earnings"] == 0
    rebound = features.loc[(dates[61], "A")]
    assert rebound["reversal_raw"] > 0
    assert 0 <= rebound["reversal"] < rebound["reversal_raw"]
    assert rebound["earnings_age"] == (dates[61] - dates[60]).days
    # Good news must not trigger the absolute-negative-news gate.
    assert features.loc[(dates[61], "F"), "gate"] == 1


def test_momentum_skip_window_ignores_latest_shock():
    dates, prices, earnings, config = market_fixture()
    original = build_features(prices, earnings, config).set_index(["date", "ticker"])
    prices.loc[
        (prices["date"] == dates[80]) & (prices["ticker"] == "A"), ["close", "adj_close"]
    ] *= 1.5
    changed = build_features(prices, earnings, config).set_index(["date", "ticker"])
    assert changed.loc[(dates[80], "A"), "momentum"] == original.loc[(dates[80], "A"), "momentum"]
    assert (
        changed.loc[(dates[80], "A"), "reversal_raw"]
        < original.loc[(dates[80], "A"), "reversal_raw"]
    )


def test_execution_lag_full_investment_and_cost_accounting():
    dates, prices, earnings, config = market_fixture()
    features = build_features(prices, earnings, config)
    net = run_backtests(prices, features, config)
    free_config = copy.deepcopy(config)
    free_config["backtest"]["cost_bps"] = 0
    free = run_backtests(prices, features, free_config)
    holdings = net["holdings"]
    grouped = holdings.groupby(["entry_date", "horizon", "strategy"])
    np.testing.assert_allclose(grouped["target_weight"].sum(), 1)
    stock_holdings = holdings[holdings["strategy"] != "universe"]
    assert stock_holdings.groupby(["entry_date", "horizon", "strategy"]).size().max() <= 2
    assert (holdings["signal_date"] < holdings["entry_date"]).all()
    assert (net["weights"]["label_end"] < net["weights"]["refit_date"]).all()
    np.testing.assert_allclose(
        net["returns"]["gross_return"], free["returns"]["return"], atol=1e-14
    )
    np.testing.assert_allclose(
        net["returns"]["return"],
        net["returns"]["gross_return"] - net["returns"]["cost"],
        atol=1e-14,
    )
    for (horizon, strategy), series in net["returns"].groupby(["horizon", "strategy"]):
        assert series.iloc[0]["gross_return"] == 0  # no return before first entry close
        assert series.iloc[0]["nav"] == pytest.approx(1000 / 1.001)
        assert series.iloc[-1]["turnover"] > 0  # terminal liquidation charged
        free_series = free["returns"][
            (free["returns"]["horizon"] == horizon) & (free["returns"]["strategy"] == strategy)
        ]
        assert series.iloc[-1]["nav"] < free_series.iloc[-1]["nav"]
    first_entry = dates[55]
    picks = holdings[
        (holdings["entry_date"] == first_entry)
        & (holdings["horizon"] == 5)
        & (holdings["strategy"] == "ensemble")
    ]["ticker"]
    close = prices.pivot(index="date", columns="ticker", values="close")
    expected = (close.loc[dates[59], picks] / close.loc[first_entry, picks]).mean()
    observed = free["returns"].query("horizon == 5 and strategy == 'ensemble'")
    assert observed.loc[observed["date"] == dates[59], "nav"].item() == pytest.approx(
        1000 * expected
    )


def test_missing_held_price_is_not_dropped_from_the_portfolio():
    dates, prices, earnings, config = market_fixture()
    features = build_features(prices, earnings, config)
    result = run_backtests(prices, features, config)
    pick = result["holdings"].query("horizon == 5 and strategy == 'ensemble'").iloc[0]
    missing_date = dates[dates.get_loc(pick["entry_date"]) + 1]
    damaged = prices[~((prices["ticker"] == pick["ticker"]) & (prices["date"] == missing_date))]
    with pytest.raises(ValueError, match="Missing held price/return"):
        run_backtests(damaged, features, config)


def test_share_split_is_not_a_loss_and_dividends_have_explicit_return_basis():
    dates, prices, earnings, config = market_fixture()
    features = build_features(prices, earnings, config)
    original = run_backtests(prices, features, config)["returns"]
    # A 2:1 split changes nominal shares and price, but not economic price returns.
    split_date = dates[70]
    prices.loc[prices["date"] >= split_date, "close"] /= 2
    prices.loc[prices["date"] == split_date, "split"] = 2
    split_result = run_backtests(prices, features, config)["returns"]
    np.testing.assert_allclose(original["return"], split_result["return"], atol=1e-14)
    # A cash distribution reduces nominal prices, not adjusted total-return wealth.
    prices.loc[prices["date"] >= dates[85], "close"] *= 0.99
    prices.loc[prices["date"] == dates[85], "dividend"] = 0.5
    price_result = run_backtests(prices, features, config)["returns"]
    config["backtest"]["return_mode"] = "total"
    total_result = run_backtests(prices, features, config)["returns"]
    price_day = price_result[price_result["date"] == dates[85]]["gross_return"]
    total_day = total_result[total_result["date"] == dates[85]]["gross_return"]
    np.testing.assert_allclose((1 + price_day.to_numpy()) / (1 + total_day.to_numpy()), 0.99)


def test_constrained_fit_prefers_predictive_signal_without_negative_bets():
    training = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2020-01-01", periods=4), 3),
            "earnings": np.tile([-1.0, 0.0, 1.0], 4),
        }
    )
    training["momentum"] = -training["earnings"]
    training["reversal"] = 0.0
    training["target"] = training["earnings"]
    fitted = fit_weights(training, ridge=0)
    np.testing.assert_allclose(fitted, [1, 0, 0], atol=1e-10)


def companyfacts_fixture():
    incomes, assets = [], []
    quarters = pd.period_range("2015Q1", periods=16, freq="Q")
    for index, quarter in enumerate(quarters):
        start, end = quarter.start_time.normalize(), quarter.end_time.normalize()
        filed = (end + pd.Timedelta(days=35)).date().isoformat()
        fact = {
            "start": str(start.date()),
            "end": str(end.date()),
            "filed": filed,
            "val": 100 + index**2,
            "accn": f"original-{index}",
            "form": "10-Q",
        }
        incomes.append(fact)
        assets.append(
            {key: value for key, value in {**fact, "val": 10000}.items() if key != "start"}
        )
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": incomes}},
                "Assets": {"units": {"USD": assets}},
            }
        }
    }


def test_late_restatements_cannot_rewrite_old_earnings_or_reset_freshness():
    payload = companyfacts_fixture()
    original = earnings_from_companyfacts(payload, "A")
    changed = copy.deepcopy(payload)
    restatement = {
        **changed["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"][3],
        "filed": "2025-01-01",
        "accn": "later-restatement",
        "val": -1000000,
        "form": "10-K/A",
    }
    changed["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(restatement)
    assert_frame_equal(original, earnings_from_companyfacts(changed, "A"))
    first = original.iloc[0]
    # Seasonal changes at indices4..7 are16,24,32,40 divided by assets; current is48.
    assert first["surprise"] == pytest.approx(48 / np.std([16, 24, 32, 40], ddof=1))


def test_instant_income_context_is_not_misread_as_quarterly_earnings():
    payload = companyfacts_fixture()
    expected = earnings_from_companyfacts(payload, "GS")
    payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(
        {
            "end": "2008-12-31",
            "val": -780000000,
            "accn": "0000950123-11-020067",
            "fy": 2010,
            "fp": "FY",
            "form": "10-K",
            "filed": "2011-03-01",
            "frame": "CY2008Q4I",
        }
    )
    assert_frame_equal(expected, earnings_from_companyfacts(payload, "GS"))


def test_fourth_quarter_deaccumulation_preserves_earnings_signal():
    payload = companyfacts_fixture()
    direct = earnings_from_companyfacts(payload, "A")
    facts = payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    replacements = []
    for offset in range(0, 16, 4):
        annual = facts[offset : offset + 4]
        replacements.extend(annual[:3])
        replacements.append(
            {
                **annual[2],
                "start": annual[0]["start"],
                "val": sum(fact["val"] for fact in annual[:3]),
            }
        )
        replacements.append(
            {
                **annual[3],
                "start": annual[0]["start"],
                "form": "10-K",
                "val": sum(fact["val"] for fact in annual),
            }
        )
    payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = replacements
    deaccumulated = earnings_from_companyfacts(payload, "A")
    assert_frame_equal(
        direct.drop(columns="source_accn"), deaccumulated.drop(columns="source_accn")
    )


def test_alternate_earnings_tag_is_not_overwritten_by_future_preferred_tag():
    payload = companyfacts_fixture()
    tags = payload["facts"]["us-gaap"]
    tags["NetIncomeLossAvailableToCommonStockholdersBasic"] = tags.pop("NetIncomeLoss")
    expected = earnings_from_companyfacts(payload, "CAT")
    assert len(expected) == 8
    later = copy.deepcopy(tags["NetIncomeLossAvailableToCommonStockholdersBasic"])
    for fact in later["units"]["USD"]:
        fact["filed"] = str((pd.Timestamp(fact["filed"]) + pd.DateOffset(years=10)).date())
        fact["val"] *= -100
    tags["NetIncomeLoss"] = later
    assert_frame_equal(expected, earnings_from_companyfacts(payload, "CAT"))


def test_retail_sixteen_week_fourth_quarters_are_not_lost():
    income, assets, ends = [], [], []
    start = pd.Timestamp("2015-01-04")
    for index in range(16):
        end = start + pd.Timedelta(weeks=16 if index % 4 == 3 else 12) - pd.Timedelta(days=1)
        fact = {
            "start": str(start.date()),
            "end": str(end.date()),
            "filed": str((end + pd.Timedelta(days=35)).date()),
            "val": 100 + index**2,
            "accn": f"retail-{index}",
            "form": "10-Q",
        }
        income.append(fact)
        assets.append(
            {key: value for key, value in {**fact, "val": 10000}.items() if key != "start"}
        )
        ends.append(end)
        start = end + pd.Timedelta(days=1)
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": income}},
                "Assets": {"units": {"USD": assets}},
            }
        }
    }
    events = earnings_from_companyfacts(payload, "COST")
    assert events["fiscal_end"].tolist() == ends[8:]


def test_split_after_requested_end_still_reconstructs_nominal_price():
    dates = pd.date_range("2020-08-27", periods=5, tz="America/New_York") + pd.Timedelta(hours=16)
    stamps = [int(date.timestamp()) for date in dates]
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "instrumentType": "EQUITY",
                        "currency": "USD",
                        "symbol": "A",
                        "exchangeTimezoneName": "America/New_York",
                    },
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [{"close": [100, 101, 102, 103, 104], "volume": [4000] * 5}],
                        "adjclose": [{"adjclose": [100, 101, 102, 103, 104]}],
                    },
                    "events": {
                        "splits": {"split": {"date": stamps[-1], "numerator": 4, "denominator": 1}}
                    },
                }
            ],
            "error": None,
        }
    }
    rows = _yahoo_prices(payload, "A", pd.Timestamp("2020-08-27"), pd.Timestamp("2020-08-29"))
    np.testing.assert_allclose(rows["close"], [400, 404])
    np.testing.assert_allclose(rows["volume"], [1000, 1000])
