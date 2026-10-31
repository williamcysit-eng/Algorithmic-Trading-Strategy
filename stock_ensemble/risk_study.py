"""Fixed-stock chronological allocation comparison; no historical winner fitting."""

import numpy as np
import pandas as pd

from . import portfolio
from .backtest import _fit, _ranking
from .risk import allocation_weights, dated_covariance


def validate_study(config):
    if config["backtest"].get("return_mode", "price") != "price":
        raise ValueError("Risk study requires nominal price returns")
    settings = config.get("risk_study", {})
    for key in ("covariance_window", "bootstrap_draws", "bootstrap_block_sessions"):
        value = settings.get(key, 0)
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 2:
            raise ValueError(f"risk_study.{key} must be an integer >= 2")
    horizons = config["backtest"]["horizons"]
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(isinstance(h, bool) or int(h) != h or h < 1 for h in horizons)
    ):
        raise ValueError("Risk-study horizons must be distinct positive integers")
    if settings["primary_policy"] != "balanced":
        raise ValueError("Only the frozen balanced candidate may qualify for adoption")
    gammas = settings["gammas"]
    if (
        not gammas
        or len(set(gammas)) != len(gammas)
        or any(not np.isfinite(gamma) or gamma < 0 for gamma in gammas)
        or settings["primary_gamma"] not in gammas
    ):
        raise ValueError(
            "Risk aversion values must be unique, finite, nonnegative and include primary"
        )
    if settings["annualized_return_sacrifice_limit"] < 0:
        raise ValueError("Return-sacrifice tolerance must be nonnegative")
    costs = settings["cost_scenarios_bps"]
    if (
        not costs
        or costs[0] != 0
        or len(set(costs)) != len(costs)
        or any(not np.isfinite(cost) or not 0 <= cost < 10000 for cost in costs)
    ):
        raise ValueError("Cost scenarios must start with zero and be valid and unique")
    for block in settings["bootstrap_sensitivity_sessions"]:
        if not np.isfinite(block) or int(block) != block or block < 1:
            raise ValueError("Sensitivity blocks must be positive integer session lengths")
    count, _, _ = portfolio.portfolio_settings(config)
    allocation_weights(np.eye(count), settings["min_weight"], settings["max_weight"])
    return settings


def run_risk_study(prices, features, config, progress=None):
    settings = validate_study(config)
    market = portfolio.prepare_market(prices, features, config)
    top_n, capital, _ = portfolio.portfolio_settings(config)
    sessions = market.sessions
    first_signal = portfolio.first_signal(market, config)
    start = sessions.get_loc(first_signal) + 1
    dates = sessions[start:]
    end = portfolio.normalize_date(config["backtest"].get("end", sessions[-1]))
    dates = dates[dates <= end]
    if len(dates) < 2:
        raise ValueError("Risk study needs at least two evaluation sessions")
    risk_cache = {}
    risk_records, fit_records, return_rows, holdings, episodes = [], [], [], [], []
    for horizon in config["backtest"]["horizons"]:
        labels = portfolio.forward_labels(market, horizon)
        outcome = labels.set_index(["date", "ticker"])["forward_return"]
        last_signal = sessions[sessions.get_loc(dates[-1]) - 1]
        refits = portfolio.refit_dates(market, first_signal, last_signal)
        fitted = {}
        for refit in refits:
            weights, metadata = _fit(labels, market, refit, horizon, "ensemble", config)
            fitted[refit] = weights
            fit_records.append(metadata)
        for year in sorted(set(dates.year)):
            fold_dates = dates[dates.year == year]
            if len(fold_dates) < 2:
                continue
            decisions = {
                method: {}
                for method in ("equal", "inverse_volatility", "minimum_variance", "balanced")
            }
            for offset in range(0, len(fold_dates) - 1, horizon):
                entry = fold_dates[offset]
                signal = sessions[sessions.get_loc(entry) - 1]
                refit = refits[refits.searchsorted(signal, side="right") - 1]
                frame = portfolio.eligible_as_of(market, signal, top_n)
                ranking = _ranking(frame, "ensemble", {"ensemble": fitted[refit]})
                selected = ranking.iloc[:top_n].sort_values("ticker").copy()
                tickers = selected.ticker.tolist()
                cache_key = (signal, tuple(tickers))
                if cache_key not in risk_cache:
                    covariance, metadata = dated_covariance(
                        market.factors, tickers, signal, settings["covariance_window"]
                    )
                    weights = allocation_weights(
                        covariance, settings["min_weight"], settings["max_weight"]
                    )
                    metadata["risk_id"] = len(risk_records)
                    risk_records.append(metadata)
                    risk_cache[cache_key] = (covariance, weights, metadata["risk_id"])
                covariance, weights, risk_id = risk_cache[cache_key]
                exit_date = (
                    fold_dates[offset + horizon] if offset + horizon < len(fold_dates) else None
                )
                forward = None
                if exit_date is not None:
                    lookup = pd.MultiIndex.from_arrays(
                        [np.repeat(signal, top_n), tickers], names=["date", "ticker"]
                    )
                    forward = outcome.reindex(lookup).to_numpy()
                    if not np.isfinite(forward).all():
                        raise ValueError("Missing selected-stock episode outcome")
                for method, target in weights.items():
                    decision = selected[["ticker", "score"]].copy()
                    decision["target_weight"] = target
                    decisions[method][entry] = decision
                    predicted_daily_variance = float(target @ covariance @ target)
                    holding = decision.assign(
                        horizon=horizon,
                        strategy=method,
                        fold=int(year),
                        entry_date=entry,
                        signal_date=signal,
                        refit_date=refit,
                        risk_id=risk_id,
                        predicted_daily_variance=predicted_daily_variance,
                    )
                    holdings.append(holding)
                    if exit_date is not None:
                        episodes.append(
                            {
                                "horizon": horizon,
                                "strategy": method,
                                "fold": int(year),
                                "signal_date": signal,
                                "entry_date": entry,
                                "exit_date": exit_date,
                                "price_return": float(forward @ target),
                                "predicted_daily_variance": predicted_daily_variance,
                            }
                        )
            for method, choices in decisions.items():
                for cost_bps in settings["cost_scenarios_bps"]:
                    result = portfolio.simulate_portfolio(
                        market, fold_dates, horizon, method, choices, capital, cost_bps / 10000
                    )
                    return_rows.extend(
                        dict(row, fold=int(year), cost_bps=cost_bps) for row in result
                    )
            if progress:
                progress(f"h={horizon}, outer={year}: same stocks, four dated allocations")
    if not episodes:
        raise ValueError("Risk study has no complete episodes")
    return {
        "returns": pd.DataFrame(return_rows),
        "episodes": pd.DataFrame(episodes),
        "holdings": pd.concat(holdings, ignore_index=True),
        "risk_models": risk_records,
        "fits": pd.DataFrame(fit_records),
    }
