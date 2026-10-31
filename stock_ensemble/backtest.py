"""Purged monthly fitting and self-financing, daily marked stock portfolios.

Signals observed at a session's close trade only at the next session's close.
The final shared evaluation session liquidates even a partially completed final
holding interval; its complete marked history and liquidation fee are retained.
Costs and turnover are fractions of NAV before that day's price mark. NAV is in
initial-capital dollars. Sharpe ratios use daily returns and zero risk-free rate.
"""

import numpy as np
import pandas as pd

from . import portfolio
from .model import SIGNALS, fit_weights

STRATEGIES = ("ensemble", "equal_signal", "ungated", "earnings", "momentum", "reversal", "universe")
RAW_SIGNALS = ("earnings", "momentum", "reversal_raw")


def _fit(labels, market, refit_date, horizon, strategy, config):
    """Fit one vintage using only labels realized before the refit session."""
    columns = SIGNALS if strategy == "ensemble" else RAW_SIGNALS
    model = config["model"]
    window = int(model["train_window"])
    minimum = int(model["min_train_dates"])
    if window < 1 or minimum < 1 or minimum > window:
        raise ValueError("Require 1 <= min_train_dates <= train_window")
    index = int(market.sessions.get_loc(refit_date))
    window_start = market.sessions[max(0, index - window)]
    # Purge by label exit as well as signal date: older signals can still have
    # unrealized outcomes, especially at longer holding horizons.
    training = labels.loc[
        (labels["date"] >= window_start)
        & (labels["date"] < refit_date)
        & (labels["label_end"] < refit_date)
    ].copy()
    finite = np.isfinite(training.loc[:, [*columns, "target"]].to_numpy(dtype=float)).all(axis=1)
    training = training.loc[finite]
    train_dates = training["date"].nunique()
    if train_dates < minimum:
        raise ValueError(
            f"Insufficient matured training at {refit_date.date()} for {strategy}, "
            f"horizon={horizon}: {train_dates} dates, require {minimum}. "
            "Supply earlier history or explicitly move evaluation start."
        )
    weights = fit_weights(training, columns=columns, ridge=float(model.get("ridge", 0.05)))
    metadata = {
        "refit_date": refit_date,
        "horizon": horizon,
        "strategy": strategy,
        "w_earnings": weights[0],
        "w_momentum": weights[1],
        "w_reversal": weights[2],
        "train_start": training["date"].min(),
        "train_end": training["date"].max(),
        "label_end": training["label_end"].max(),
        "train_dates": train_dates,
        "train_rows": len(training),
    }
    return weights, metadata


def _ranking(frame, strategy, fitted):
    if strategy == "ensemble":
        values = frame.loc[:, list(SIGNALS)].to_numpy(dtype=float) @ fitted["ensemble"]
    elif strategy == "ungated":
        values = frame.loc[:, list(RAW_SIGNALS)].to_numpy(dtype=float) @ fitted["ungated"]
    elif strategy == "equal_signal":
        values = frame.loc[:, list(SIGNALS)].mean(axis=1).to_numpy()
    elif strategy == "universe":
        values = np.zeros(len(frame))
    else:
        values = frame["reversal_raw" if strategy == "reversal" else strategy].to_numpy(dtype=float)
    result = frame.copy()
    result["score"] = values
    # Ticker tie-breaks make equally scored portfolios deterministic.
    result = result.sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
    result["rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def summarize_returns(returns: pd.DataFrame, holdout_start: str) -> pd.DataFrame:
    """Summarize chronological daily returns; Sharpe uses a zero risk-free rate."""
    holdout = portfolio.normalize_date(holdout_start)
    rows = []
    for (horizon, strategy), series in returns.groupby(["horizon", "strategy"], sort=False):
        series = series.sort_values("date", kind="stable")
        periods = {
            "all": series,
            "development": series.loc[series["date"] < holdout],
            "holdout": series.loc[series["date"] >= holdout],
        }
        for period, frame in periods.items():
            row = {
                "horizon": horizon,
                "strategy": strategy,
                "period": period,
                "start": frame["date"].min(),
                "end": frame["date"].max(),
                "n_days": len(frame),
            }
            if frame.empty:
                row.update(
                    {
                        key: np.nan
                        for key in (
                            "total_return",
                            "cagr",
                            "volatility",
                            "sharpe",
                            "max_drawdown",
                            "mean_daily_turnover",
                        )
                    }
                )
            else:
                r = frame["return"].to_numpy(dtype=float)
                # Starting wealth makes an initial loss count toward drawdown.
                path = np.r_[1.0, np.cumprod(1.0 + r)]
                std = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
                row.update(
                    total_return=path[-1] - 1.0,
                    cagr=path[-1] ** (252.0 / len(r)) - 1.0,
                    volatility=std * np.sqrt(252.0),
                    sharpe=float(np.mean(r)) / std * np.sqrt(252.0) if std > 0 else np.nan,
                    max_drawdown=float(np.min(path / np.maximum.accumulate(path) - 1.0)),
                    mean_daily_turnover=frame["turnover"].mean(),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def run_backtests(prices, features, config):
    """Return daily net/gross NAV evidence, allocations, fits, scores and metrics.

    All seven strategies and horizons share evaluation dates. Each starts in
    cash, purchases at the first evaluation close, trades every h sessions, and
    liquidates at the last close rather than discarding an unfinished interval.
    The universe strategy alone is exempt from the five-stock constraint.
    """
    market = portfolio.prepare_market(prices, features, config)
    top_n, capital, fee = portfolio.portfolio_settings(config)
    horizons = config["backtest"].get("horizons", [1, 5, 20])
    if (
        not horizons
        or any(int(h) != h or h < 1 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be distinct positive integers")
    first_signal = portfolio.first_signal(market, config)
    first_entry_index = int(market.sessions.get_loc(first_signal)) + 1
    end = portfolio.normalize_date(config["backtest"].get("end", market.sessions[-1]))
    dates = market.sessions[first_entry_index:]
    dates = dates[dates <= end]
    if len(dates) < 2:
        raise ValueError("Evaluation needs at least two sessions for entry and liquidation")
    last_signal = market.sessions[market.sessions.get_loc(dates[-1]) - 1]
    refits = portfolio.refit_dates(market, first_signal, last_signal)
    return_rows, holding_frames, weight_rows, score_frames = [], [], [], []
    for horizon_value in horizons:
        horizon = int(horizon_value)
        labels = portfolio.forward_labels(market, horizon)
        fitted_by_date = {}
        for refit_date in refits:
            fitted = {}
            for strategy in ("ensemble", "ungated"):
                fitted[strategy], metadata = _fit(
                    labels, market, refit_date, horizon, strategy, config
                )
                weight_rows.append(metadata)
            fitted_by_date[refit_date] = fitted
        decisions = {strategy: {} for strategy in STRATEGIES}
        # Never open a position at the terminal liquidation close.
        for offset in range(0, len(dates) - 1, horizon):
            entry_date = dates[offset]
            signal_date = market.sessions[market.sessions.get_loc(entry_date) - 1]
            # Use the refit known at signal close, not at the later execution close.
            refit_date = refits[refits.searchsorted(signal_date, side="right") - 1]
            fitted = fitted_by_date[refit_date]
            frame = portfolio.eligible_as_of(market, signal_date, top_n)
            for strategy in STRATEGIES:
                ranking = _ranking(frame, strategy, fitted)
                columns = ["date", "ticker", "score", *SIGNALS, "reversal_raw"]
                scored = ranking.loc[:, columns].copy()
                scored["horizon"], scored["strategy"] = horizon, strategy
                score_frames.append(scored)
                selected = (ranking if strategy == "universe" else ranking.iloc[:top_n]).copy()
                selected["target_weight"] = 1.0 / len(selected)
                decisions[strategy][entry_date] = selected
                holding = selected.loc[:, ["ticker", "score"]].copy()
                holding["entry_date"], holding["signal_date"] = entry_date, signal_date
                holding["horizon"], holding["strategy"] = horizon, strategy
                holding["target_weight"] = 1.0 / len(selected)
                holding_frames.append(holding)
        for strategy in STRATEGIES:
            return_rows.extend(
                portfolio.simulate_portfolio(
                    market, dates, horizon, strategy, decisions[strategy], capital, fee
                )
            )
    returns = pd.DataFrame(return_rows)
    return {
        "returns": returns,
        "holdings": pd.concat(holding_frames, ignore_index=True),
        "weights": pd.DataFrame(weight_rows),
        "metrics": summarize_returns(returns, config["backtest"]["holdout_start"]),
        "scores": pd.concat(score_frames, ignore_index=True),
    }


def rank_as_of(prices, features, config, signal_date, horizon):
    """Rank an exact signal session, even when its execution/future is unknown.

    Fitting uses the same initial/monthly schedule and strict label maturity as
    the historical simulator. The date must be in its configured OOS period.
    Metadata's w_reversal is the gated-reversal coefficient for this ensemble.
    """
    market = portfolio.prepare_market(prices, features, config)
    top_n, _, _ = portfolio.portfolio_settings(config)
    signal_date = portfolio.normalize_date(signal_date)
    if signal_date not in market.sessions:
        raise ValueError(f"{signal_date.date()} is not an exact available signal session")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)
    first_signal = portfolio.first_signal(market, config)
    if signal_date < first_signal:
        raise ValueError("signal_date precedes the configured out-of-sample signal period")
    refit_date = portfolio.refit_dates(market, first_signal, signal_date)[-1]
    labels = portfolio.forward_labels(market, horizon)
    weights, metadata = _fit(labels, market, refit_date, horizon, "ensemble", config)
    ranking = _ranking(
        portfolio.eligible_as_of(market, signal_date, top_n), "ensemble", {"ensemble": weights}
    )
    ranking["target_weight"] = np.where(ranking["rank"] <= top_n, 1.0 / top_n, 0.0)
    metadata.update(
        signal_date=signal_date,
        horizon=horizon,
        return_mode=config["backtest"].get("return_mode", "price"),
        weights=dict(zip(SIGNALS, weights.tolist())),
        execution="next available trading session close; no same-close execution",
    )
    columns = ["ticker", "score", "rank", *SIGNALS, "reversal_raw", "target_weight"]
    return ranking.loc[:, columns], metadata
