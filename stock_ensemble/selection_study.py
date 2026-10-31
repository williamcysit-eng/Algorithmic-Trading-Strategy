"""Chronological joint stock selection, keeping fitted scores and weights fixed."""

import numpy as np
import pandas as pd

from . import portfolio
from .backtest import _fit, _ranking
from .selection import METHODS, choose_subsets, shortlist_covariance


def validate_study(config):
    settings = config["selection_study"]
    for key in (
        "shortlist_size",
        "sector_cap",
        "covariance_window",
        "bootstrap_draws",
        "bootstrap_block_sessions",
    ):
        value = settings[key]
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"selection_study.{key} must be a positive integer")
    count, _, _ = portfolio.portfolio_settings(config)
    if not count <= settings["shortlist_size"] <= 12 or settings["sector_cap"] > count:
        raise ValueError("Invalid shortlist size or sector cap")
    if (
        settings["covariance_window"] < 2
        or not np.isfinite(settings["score_budget"])
        or settings["score_budget"] < 0
    ):
        raise ValueError("Invalid covariance window or score budget")
    if config["backtest"].get("return_mode", "price") != "price":
        raise ValueError("Selection study requires nominal price mode")
    horizons = config["backtest"]["horizons"]
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(isinstance(h, bool) or int(h) != h or h < 1 for h in horizons)
    ):
        raise ValueError("Invalid selection horizons")
    if settings["primary_policy"] != "joint":
        raise ValueError("Only the frozen joint selection can qualify")
    gammas = settings["gammas"]
    if (
        not gammas
        or len(set(gammas)) != len(gammas)
        or any(not np.isfinite(gamma) or gamma < 0 for gamma in gammas)
        or settings["primary_gamma"] not in gammas
    ):
        raise ValueError("Invalid risk-preference grid")
    costs = settings["cost_scenarios_bps"]
    if (
        not costs
        or costs[0] != 0
        or len(set(costs)) != len(costs)
        or any(not np.isfinite(cost) or not 0 <= cost < 10000 for cost in costs)
    ):
        raise ValueError("Invalid cost scenarios")
    for value in settings["bootstrap_sensitivity_sessions"]:
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError("Invalid bootstrap sensitivity")
    if (
        not np.isfinite(settings["annualized_return_sacrifice_limit"])
        or settings["annualized_return_sacrifice_limit"] < 0
    ):
        raise ValueError("Invalid return sacrifice limit")
    return settings


def run_selection_study(prices, features, universe, config, progress=None):
    settings = validate_study(config)
    market = portfolio.prepare_market(prices, features, config)
    top_n, capital, _ = portfolio.portfolio_settings(config)
    if universe.ticker.duplicated().any() or universe.sector.isna().any():
        raise ValueError("Universe sector labels must be nonmissing and unique by ticker")
    sectors = universe.set_index("ticker").sector
    sessions = market.sessions
    first_signal = portfolio.first_signal(market, config)
    dates = sessions[sessions.get_loc(first_signal) + 1 :]
    end = portfolio.normalize_date(config["backtest"].get("end", sessions[-1]))
    dates = dates[dates <= end]
    if len(dates) < 2:
        raise ValueError("Selection study needs at least two evaluation sessions")
    covariance_cache, risk_records = {}, []
    fits, diagnostics, candidates, holdings, episodes, returns = [], [], [], [], [], []
    for horizon in config["backtest"]["horizons"]:
        labels = portfolio.forward_labels(market, horizon)
        outcomes = labels.set_index(["date", "ticker"])["forward_return"]
        last_signal = sessions[sessions.get_loc(dates[-1]) - 1]
        refits = portfolio.refit_dates(market, first_signal, last_signal)
        fitted = {}
        for refit in refits:
            weights, metadata = _fit(labels, market, refit, horizon, "ensemble", config)
            fitted[refit] = weights
            fits.append(metadata)
        for year in sorted(set(dates.year)):
            fold_dates = dates[dates.year == year]
            if len(fold_dates) < 2:
                continue
            decisions = {method: {} for method in METHODS}
            for offset in range(0, len(fold_dates) - 1, horizon):
                entry = fold_dates[offset]
                signal = sessions[sessions.get_loc(entry) - 1]
                refit = refits[refits.searchsorted(signal, side="right") - 1]
                frame = portfolio.eligible_as_of(market, signal, top_n)
                ranking = _ranking(frame, "ensemble", {"ensemble": fitted[refit]})
                shortlist = ranking.iloc[: settings["shortlist_size"]][
                    ["ticker", "score", "rank"]
                ].copy()
                shortlist["sector"] = shortlist.ticker.map(sectors)
                if shortlist.sector.isna().any():
                    raise ValueError("Missing sector for shortlisted company")
                shortlist = shortlist.sort_values("ticker").reset_index(drop=True)
                tickers = shortlist.ticker.tolist()
                cache_key = (signal, tuple(tickers))
                if cache_key not in covariance_cache:
                    covariance, metadata = shortlist_covariance(
                        market.factors, tickers, signal, settings["covariance_window"]
                    )
                    metadata["risk_id"] = len(risk_records)
                    risk_records.append(metadata)
                    covariance_cache[cache_key] = (covariance, metadata["risk_id"])
                covariance, risk_id = covariance_cache[cache_key]
                choices, details = choose_subsets(
                    shortlist,
                    covariance,
                    top_n=top_n,
                    sector_cap=settings["sector_cap"],
                    score_budget=settings["score_budget"],
                )
                common = {
                    "horizon": horizon,
                    "fold": int(year),
                    "entry_date": entry,
                    "signal_date": signal,
                    "refit_date": refit,
                    "risk_id": risk_id,
                }
                candidates.append(shortlist.assign(**common))
                diagnostics.extend({**row, **common} for row in details)
                exit_date = (
                    fold_dates[offset + horizon] if offset + horizon < len(fold_dates) else None
                )
                for method, selected in choices.items():
                    decisions[method][entry] = selected
                    holdings.append(selected.assign(strategy=method, **common))
                    if exit_date is not None:
                        lookup = pd.MultiIndex.from_arrays(
                            [np.repeat(signal, top_n), selected.ticker], names=["date", "ticker"]
                        )
                        forward = outcomes.reindex(lookup).to_numpy()
                        if not np.isfinite(forward).all():
                            raise ValueError("Missing selected-stock episode outcome")
                        episodes.append(
                            {
                                "horizon": horizon,
                                "fold": int(year),
                                "strategy": method,
                                "entry_date": entry,
                                "exit_date": exit_date,
                                "signal_date": signal,
                                "price_return": float(forward @ selected.target_weight.to_numpy()),
                            }
                        )
            for method, selected in decisions.items():
                for cost_bps in settings["cost_scenarios_bps"]:
                    rows = portfolio.simulate_portfolio(
                        market, fold_dates, horizon, method, selected, capital, cost_bps / 10000
                    )
                    returns.extend(dict(row, fold=int(year), cost_bps=cost_bps) for row in rows)
            if progress:
                progress(f"h={horizon}, outer={year}: four fixed equal-weight selection policies")
    if not episodes:
        raise ValueError("No complete selection episodes")
    return {
        "returns": pd.DataFrame(returns),
        "episodes": pd.DataFrame(episodes),
        "holdings": pd.concat(holdings, ignore_index=True),
        "shortlists": pd.concat(candidates, ignore_index=True),
        "fits": pd.DataFrame(fits),
        "selection_diagnostics": pd.DataFrame(diagnostics),
        "risk_models": risk_records,
    }
