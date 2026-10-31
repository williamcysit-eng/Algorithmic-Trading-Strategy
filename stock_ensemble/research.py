"""Nested chronological research; selection and scoring never consume outer outcomes.

The shared portfolio engine remains responsible for fees, missing held marks, and
liquidation. This module owns only dated fitting, nested selection, and episodes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import portfolio
from .forecasting import Candidate, candidate_grid, fit_candidate, predict_candidate

POLICY_ORDER = (
    "momentum",
    "equal_signal",
    "rank_ensemble",
    "ridge_equal",
    "boosted_equal",
    "ridge_regularized",
    "boosted_regularized",
)


@dataclass(frozen=True)
class Policy:
    name: str
    candidate: Candidate
    method: str = "equal"
    penalty: float | None = None


def validate_research(config):
    settings = config.get("research", {})
    for name in (
        "train_window",
        "min_train_dates",
        "inner_folds",
        "inner_sessions",
        "bootstrap_draws",
        "bootstrap_block_sessions",
    ):
        value = settings.get(name, 0)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"research.{name} must be a positive integer")
    if settings["min_train_dates"] > settings["train_window"]:
        raise ValueError("Research minimum training dates exceed window")
    if config["backtest"].get("return_mode", "price") != "price":
        raise ValueError("Research targets require nominal price return mode")
    horizons = config["backtest"]["horizons"]
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(isinstance(h, bool) or int(h) != h or h < 1 for h in horizons)
    ):
        raise ValueError("Research horizons must be distinct positive integers")
    if settings["inner_sessions"] < max(horizons):
        raise ValueError("Inner blocks must contain complete episodes at every horizon")
    for value in settings["allocation_penalties"]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Allocation penalties must be positive and finite")
    if not settings["allocation_penalties"] or len(set(settings["allocation_penalties"])) != len(
        settings["allocation_penalties"]
    ):
        raise ValueError("Allocation penalties must be nonempty and unique")
    top_n, _, _ = portfolio.portfolio_settings(config)
    portfolio.regularized_weights(
        np.zeros(top_n),
        settings["allocation_penalties"][0],
        settings["min_weight"],
        settings["max_weight"],
    )
    costs = settings["cost_scenarios_bps"]
    if not costs or costs[0] != 0 or len(set(costs)) != len(costs):
        raise ValueError("Cost scenarios must start with zero and be unique")
    if any(not np.isfinite(cost) or not 0 <= cost < 10000 for cost in costs):
        raise ValueError("Invalid research cost scenario")
    return candidate_grid(config)


def inner_blocks(sessions, selection_date, count, length):
    """Blocks share an endpoint; their final exit strictly precedes selection."""
    endpoint = sessions.get_loc(selection_date) - 1
    first = endpoint - count * length
    if first < 1:
        raise ValueError("Insufficient historical sessions for inner blocks and prior signals")
    return [sessions[first + i * length : first + (i + 1) * length + 1] for i in range(count)]


class ResearchEngine:
    def __init__(self, market, config, horizon):
        self.market, self.config, self.horizon = market, config, horizon
        self.labels = portfolio.forward_labels(market, horizon)
        self.outcomes = self.labels.set_index(["date", "ticker"])["forward_return"]
        self.frames = {date: frame for date, frame in market.features.groupby("date", sort=False)}
        self.fitted = {}
        self.scored = {}
        self.fit_records = []
        self.forecast_records = []
        self.settings = config["research"]
        self.top_n, _, _ = portfolio.portfolio_settings(config)

    def training(self, refit, legacy=False):
        settings = self.config["model"] if legacy else self.settings
        position = self.market.sessions.get_loc(refit)
        start = self.market.sessions[max(0, position - settings["train_window"])]
        training = self.labels[
            self.labels.date.ge(start)
            & self.labels.date.lt(refit)
            & self.labels.label_end.lt(refit)
        ].copy()
        n_dates = training.date.nunique()
        if n_dates < settings["min_train_dates"]:
            raise ValueError(
                f"Insufficient matured research training at {refit.date()}, "
                f"h={self.horizon}: {n_dates} < {settings['min_train_dates']}"
            )
        return training

    def predictions(self, candidate, signals, anchor, *, legacy=False):
        """Fit each admissible vintage once and batch its strictly observable inputs."""
        signals = pd.DatetimeIndex(signals).sort_values().unique()
        refits = portfolio.refit_dates(self.market, anchor, signals[-1])
        grouped = {}
        for date in signals:
            refit = refits[refits.searchsorted(date, side="right") - 1]
            grouped.setdefault(refit, []).append(date)
        result = {}
        for refit, dates in grouped.items():
            model_key = (candidate.name, refit, legacy)
            if model_key not in self.fitted:
                training = self.training(refit, legacy)
                fitted = fit_candidate(candidate, training, self.config)
                record = {
                    "candidate": candidate.name,
                    "legacy": legacy,
                    "horizon": self.horizon,
                    "refit_date": refit,
                    "train_start": training.date.min(),
                    "train_end": training.date.max(),
                    "label_end": training.label_end.max(),
                    "train_dates": training.date.nunique(),
                    "train_rows": len(training),
                    "model": fitted.metadata,
                }
                self.fitted[model_key] = fitted
                self.fit_records.append(record)
            pending = [
                date for date in dates if (candidate.name, refit, date, legacy) not in self.scored
            ]
            if pending:
                frames = []
                for date in pending:
                    frame = self.frames[date]
                    frame = frame.loc[frame.eligible].copy()
                    if len(frame) < self.top_n:
                        raise ValueError(f"Insufficient eligible stocks on {date}")
                    frames.append(frame)
                scored = predict_candidate(
                    self.fitted[model_key], pd.concat(frames, ignore_index=True), self.config
                )
                scored["refit_date"] = refit
                for date, frame in scored.groupby("date", sort=False):
                    self.scored[(candidate.name, refit, date, legacy)] = frame
            for date in dates:
                result[date] = self.scored[(candidate.name, refit, date, legacy)]
        return result

    def allocation(self, policy, frame):
        return (
            portfolio.allocate_ranked(
                frame,
                self.top_n,
                method=policy.method,
                penalty=policy.penalty,
                lower=self.settings["min_weight"],
                upper=self.settings["max_weight"],
            )
            .iloc[: self.top_n]
            .copy()
        )

    def episode(self, selected, signal, entry, exit_date):
        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(signal, len(selected)),
                selected.ticker.to_numpy(),
            ],
            names=["date", "ticker"],
        )
        outcomes = self.outcomes.reindex(index).to_numpy(dtype=float)
        if not np.isfinite(outcomes).all():
            raise ValueError(f"Missing held outcome for {entry.date()} to {exit_date.date()}")
        expected_end = self.market.sessions[self.market.sessions.get_loc(signal) + self.horizon + 1]
        if expected_end != exit_date:
            raise ValueError("Episode exit does not match the configured label horizon")
        return float(outcomes @ selected.target_weight.to_numpy())

    def inner_mean(self, policy, blocks, predictions):
        values = []
        for dates in blocks:
            for offset in range(0, len(dates) - self.horizon, self.horizon):
                entry, exit_date = dates[offset], dates[offset + self.horizon]
                signal = self.market.sessions[self.market.sessions.get_loc(entry) - 1]
                selected = self.allocation(policy, predictions[signal])
                values.append(self.episode(selected, signal, entry, exit_date))
        if not values:
            raise ValueError("No complete inner episodes")
        return float(np.mean(values)), len(values)


def select_policies(engine, grid, selection_date):
    settings, sessions = engine.settings, engine.market.sessions
    blocks = inner_blocks(
        sessions, selection_date, settings["inner_folds"], settings["inner_sessions"]
    )
    signals = []
    for block in blocks:
        entries = block[: len(block) - engine.horizon : engine.horizon]
        signals.extend(sessions[sessions.get_indexer(entries) - 1])
    signals = pd.DatetimeIndex(signals).sort_values().unique()
    anchor = signals[0]
    predictions = {
        candidate.name: engine.predictions(candidate, signals, anchor) for candidate in grid
    }
    records = []

    def evaluate(policy):
        mean, count = engine.inner_mean(policy, blocks, predictions[policy.candidate.name])
        records.append(
            {
                "selection_date": selection_date,
                "horizon": engine.horizon,
                "policy": policy.name,
                "candidate": policy.candidate.name,
                "allocation": policy.method,
                "penalty": policy.penalty,
                "mean_return": mean,
                "n_episodes": count,
                "inner_start": blocks[0][0],
                "inner_exit": blocks[-1][-1],
            }
        )
        return mean

    policies = [Policy(candidate.name, candidate) for candidate in grid[:3]]
    objectives = {policy.name: evaluate(policy) for policy in policies}
    for family in ("ridge", "boosted"):
        candidates = [candidate for candidate in grid if candidate.family == family]
        alternatives = [Policy(family + "_equal", candidate) for candidate in candidates]
        means = [evaluate(policy) for policy in alternatives]
        best = int(np.argmax(means))
        equal_policy = alternatives[best]
        policies.append(equal_policy)
        objectives[equal_policy.name] = means[best]
        regularized = [
            Policy(family + "_regularized", equal_policy.candidate, "regularized", p)
            for p in settings["allocation_penalties"]
        ]
        means = [evaluate(policy) for policy in regularized]
        best = int(np.argmax(means))
        policies.append(regularized[best])
        objectives[regularized[best].name] = means[best]
    by_name = {policy.name: policy for policy in policies}
    policies = [by_name[name] for name in POLICY_ORDER]
    champion = max(policies, key=lambda policy: objectives[policy.name])
    selection = {
        "horizon": engine.horizon,
        "selection_date": selection_date,
        "inner_start": blocks[0][0],
        "inner_exit": blocks[-1][-1],
        "champion": champion.name,
        "choices": {
            policy.name: {
                "candidate": policy.candidate.name,
                "method": policy.method,
                "penalty": policy.penalty,
            }
            for policy in policies
        },
    }
    return policies, champion, selection, records


def run_research(prices, features, config, progress=None):
    """Return all outer ablations, the nested policy, and a matched incumbent."""
    grid = validate_research(config)
    market = portfolio.prepare_market(prices, features, config)
    sessions, settings = market.sessions, config["research"]
    _top_n, capital, _ = portfolio.portfolio_settings(config)
    start = portfolio.normalize_date(settings["outer_start"])
    end = portfolio.normalize_date(config["backtest"].get("end", sessions[-1]))
    dates = sessions[(sessions >= start) & (sessions <= end)]
    if len(dates) < 2 or sessions.get_loc(dates[0]) < 1:
        raise ValueError("Research needs historical training and at least two outer sessions")
    legacy_anchor = portfolio.first_signal(market, config)
    returns, episodes, holdings, selections, selection_scores = [], [], [], [], []
    fits, forecast_records = [], []
    for horizon in config["backtest"]["horizons"]:
        engine = ResearchEngine(market, config, int(horizon))
        for year in sorted(set(dates.year)):
            fold_dates = dates[dates.year == year]
            if len(fold_dates) <= horizon:
                continue
            selection_date = sessions[sessions.get_loc(fold_dates[0]) - 1]
            policies, champion, selection, scores = select_policies(engine, grid, selection_date)
            selection["fold"] = int(year)
            selections.append(selection)
            selection_scores.extend(dict(row, fold=int(year)) for row in scores)
            all_policies = policies + [
                Policy("nested", champion.candidate, champion.method, champion.penalty)
            ]
            incumbent = Policy(
                "incumbent", Candidate("incumbent", "rank_ensemble", {"ridge": 0.05})
            )
            all_policies.append(incumbent)
            entries = fold_dates[:-1:horizon]
            signals = sessions[sessions.get_indexer(entries) - 1]
            predictions = {}
            for policy in all_policies:
                name = policy.candidate.name
                if name not in predictions:
                    predictions[name] = engine.predictions(
                        policy.candidate,
                        signals,
                        legacy_anchor if policy.name == "incumbent" else selection_date,
                        legacy=policy.name == "incumbent",
                    )
            # Audit all emitted predictor forecasts only on complete outer episodes.
            for policy in all_policies:
                decisions = {}
                for entry, signal in zip(entries, signals):
                    frame = predictions[policy.candidate.name][signal]
                    selected = engine.allocation(policy, frame)
                    decisions[entry] = selected
                    holding = selected[
                        ["ticker", "score", "predicted_return", "target_weight"]
                    ].copy()
                    holding = holding.assign(
                        horizon=horizon,
                        fold=int(year),
                        strategy=policy.name,
                        candidate=policy.candidate.name,
                        signal_date=signal,
                        entry_date=entry,
                        refit_date=frame.refit_date.iloc[0],
                        selection_date=selection_date,
                    )
                    holdings.append(holding)
                    exit_position = sessions.get_loc(entry) + horizon
                    if exit_position <= sessions.get_loc(fold_dates[-1]):
                        exit_date = sessions[exit_position]
                        value = engine.episode(selected, signal, entry, exit_date)
                        episodes.append(
                            {
                                "horizon": horizon,
                                "fold": int(year),
                                "strategy": policy.name,
                                "entry_date": entry,
                                "exit_date": exit_date,
                                "signal_date": signal,
                                "candidate": policy.candidate.name,
                                "allocation": policy.method,
                                "penalty": policy.penalty,
                                "price_return": value,
                            }
                        )
                        if policy.name in ("ridge_equal", "boosted_equal"):
                            lookup = pd.MultiIndex.from_arrays(
                                [np.repeat(signal, len(frame)), frame.ticker],
                                names=["date", "ticker"],
                            )
                            forecast = frame[
                                ["date", "ticker", "predicted_return", "null_prediction", "score"]
                            ].copy()
                            forecast["forward_return"] = engine.outcomes.reindex(lookup).to_numpy()
                            forecast_records.append(
                                forecast.assign(
                                    horizon=horizon,
                                    fold=int(year),
                                    strategy=policy.name,
                                    entry_date=entry,
                                    exit_date=exit_date,
                                )
                            )
                for cost_bps in settings["cost_scenarios_bps"]:
                    rows = portfolio.simulate_portfolio(
                        market,
                        fold_dates,
                        horizon,
                        policy.name,
                        decisions,
                        capital,
                        cost_bps / 10000,
                    )
                    returns.extend(dict(row, fold=int(year), cost_bps=cost_bps) for row in rows)
            if progress:
                progress(
                    f"h={horizon}, outer={year}, nested={champion.name}; "
                    f"inner exits through {selection['inner_exit'].date()}"
                )
        fits.extend(engine.fit_records)
    if not episodes:
        raise ValueError("No complete research episodes")
    return {
        "returns": pd.DataFrame(returns),
        "episodes": pd.DataFrame(episodes),
        "holdings": pd.concat(holdings, ignore_index=True),
        "selections": selections,
        "selection_scores": pd.DataFrame(selection_scores),
        "fits": fits,
        "forecasts": pd.concat(forecast_records, ignore_index=True),
    }
