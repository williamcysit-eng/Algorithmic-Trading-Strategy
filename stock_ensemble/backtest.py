"""Purged monthly fitting and self-financing, daily marked stock portfolios.

Signals observed at a session's close trade only at the next session's close.
The final shared evaluation session liquidates even a partially completed final
holding interval; its complete marked history and liquidation fee are retained.
Costs and turnover are fractions of NAV before that day's price mark. NAV is in
initial-capital dollars. Sharpe ratios use daily returns and zero risk-free rate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import SIGNALS, fit_weights

STRATEGIES = ("ensemble", "equal_signal", "ungated", "earnings", "momentum", "reversal", "universe")
RAW_SIGNALS = ("earnings", "momentum", "reversal_raw")


@dataclass
class _Market:
    sessions: pd.DatetimeIndex
    prices: pd.DataFrame
    factors: pd.DataFrame
    features: pd.DataFrame


def _date(value):
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _prepare(prices, features, config):
    mode = config["backtest"].get("return_mode", "price")
    if mode not in ("price", "total"):
        raise ValueError("return_mode must be 'price' or 'total'")
    prices = prices.copy()
    features = features.copy()
    for name, frame in (("prices", prices), ("features", features)):
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        if frame[["date", "ticker"]].isna().any().any():
            raise ValueError(f"Missing date or ticker in {name}")
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError(f"Duplicate ticker/date in {name}")
    close_column = "close" if mode == "price" else "adj_close"
    closes = prices.pivot(index="date", columns="ticker", values=close_column).sort_index()
    if closes.empty or len(closes) < 2:
        raise ValueError("At least two price sessions are required")
    bad = closes.notna() & ((closes <= 0) | ~np.isfinite(closes))
    if bad.any().any():
        raise ValueError("Prices must be positive and finite where provided")
    factors = closes / closes.shift(1)
    if mode == "price":
        splits = prices.pivot(index="date", columns="ticker", values="split").reindex_like(closes)
        if (closes.notna() & (splits.isna() | (splits <= 0) | ~np.isfinite(splits))).any().any():
            raise ValueError("Every observed nominal close requires a positive finite split ratio")
        factors *= splits
    required = ["eligible", *SIGNALS, "reversal_raw"]
    missing = set(required).difference(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if not features["date"].isin(closes.index).all():
        raise ValueError("Feature dates must be exact price sessions")
    if not features["ticker"].isin(closes.columns).all():
        raise ValueError("Feature tickers must exist in prices")
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)
    return _Market(closes.index, closes, factors, features)


def _labels(market, horizon):
    # For signal i: enter i+1, exit i+1+h. Multiplication propagates any missing
    # intermediate mark; an unavailable label never removes a prediction row.
    forward = pd.DataFrame(1.0, index=market.sessions, columns=market.prices.columns)
    for step in range(1, horizon + 1):
        forward *= market.factors.shift(-(step + 1))
    forward -= 1.0
    forward.index.name = "date"
    forward.columns.name = "ticker"
    target = forward.stack().rename("forward_return").reset_index()
    frame = market.features.loc[
        market.features["eligible"].fillna(False).astype(bool),
        ["date", "ticker", *SIGNALS, "reversal_raw"],
    ]
    frame = frame.merge(target, on=["date", "ticker"], how="inner", validate="one_to_one")
    frame = frame.loc[np.isfinite(frame["forward_return"])].copy()
    count = frame.groupby("date")["forward_return"].transform("size")
    rank = frame.groupby("date")["forward_return"].rank(method="average")
    frame["target"] = np.where(count > 1, 2.0 * (rank - 1.0) / (count - 1.0) - 1.0, 0.0)
    label_ends = pd.Series(market.sessions, index=market.sessions).shift(-(horizon + 1))
    frame["label_end"] = frame["date"].map(label_ends)
    return frame


def _first_signal(market, config):
    start = _date(config["backtest"]["start"])
    entry_index = int(market.sessions.searchsorted(start))
    if entry_index >= len(market.sessions):
        raise ValueError("Evaluation start is after the last price session")
    if entry_index == 0:
        raise ValueError("Evaluation requires a prior signal session and historical training")
    return market.sessions[entry_index - 1]


def _refit_dates(market, first_signal, last_signal):
    dates = market.sessions[(market.sessions >= first_signal) & (market.sessions <= last_signal)]
    # This tests against the full calendar, not just rebalance/entry sessions.
    months = market.sessions.to_period("M")
    month_first = market.sessions[np.r_[True, months[1:] != months[:-1]]]
    return dates[(dates == first_signal) | dates.isin(month_first)]


def _fit(labels, market, refit_date, horizon, strategy, config):
    columns = SIGNALS if strategy == "ensemble" else RAW_SIGNALS
    model = config["model"]
    window = int(model["train_window"])
    minimum = int(model["min_train_dates"])
    if window < 1 or minimum < 1 or minimum > window:
        raise ValueError("Require 1 <= min_train_dates <= train_window")
    index = int(market.sessions.get_loc(refit_date))
    window_start = market.sessions[max(0, index - window)]
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


def _eligible(market, signal_date, top_n):
    frame = market.features.loc[
        (market.features["date"] == signal_date)
        & market.features["eligible"].fillna(False).astype(bool)
    ].copy()
    if len(frame) < top_n:
        raise ValueError(
            f"Only {len(frame)} eligible stocks on {signal_date.date()}; require {top_n}"
        )
    finite = np.isfinite(frame.loc[:, [*SIGNALS, "reversal_raw"]].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        bad = frame.loc[~finite, "ticker"].tolist()
        raise ValueError(f"Nonfinite eligible signals on {signal_date.date()}: {bad}")
    return frame


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
    result = result.sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
    result["rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def _settings(config):
    backtest = config["backtest"]
    top_n = int(backtest.get("top_n", 5))
    if not 1 <= top_n <= 5:
        raise ValueError("Stock portfolios must contain one to five companies")
    capital = float(backtest.get("initial_capital", 1_000_000.0))
    fee = float(backtest.get("cost_bps", 0.0)) / 10_000.0
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("initial_capital must be positive and finite")
    if not np.isfinite(fee) or not 0 <= fee < 1:
        raise ValueError("cost_bps must be finite and in [0, 10000)")
    return top_n, capital, fee


def _self_financing(pretrade, target, wealth, fee):
    """Return post-fee holdings, fee dollars and traded dollars (no borrowing)."""
    if fee == 0:
        after = wealth
    else:
        low, high = 0.0, wealth
        # f<1 makes this continuous piecewise-linear equation strictly increasing.
        for _ in range(64):
            mid = (low + high) / 2.0
            residual = mid + fee * np.abs(mid * target - pretrade).sum() - wealth
            if residual > 0:
                high = mid
            else:
                low = mid
        after = (low + high) / 2.0
    holdings = after * target
    traded = float(np.abs(holdings - pretrade).sum())
    return holdings, fee * traded, traded


def _check_prices(market, date, selected):
    observed = market.prices.loc[date, selected]
    missing = observed.index[observed.isna() | ~np.isfinite(observed) | (observed <= 0)]
    if len(missing):
        raise ValueError(f"Missing trade price on {date.date()} for {', '.join(missing)}")


def _simulate(market, dates, horizon, strategy, decisions, capital, fee):
    tickers = market.prices.columns
    positions = np.zeros(len(tickers))
    cash = capital
    nav = capital
    rows = []
    for date in dates:
        previous_nav = nav
        held = positions > 0
        if held.any():
            factors = market.factors.loc[date].to_numpy(dtype=float)
            bad = held & (~np.isfinite(factors) | (factors <= 0))
            if bad.any():
                names = ", ".join(tickers[bad])
                raise ValueError(f"Missing held price/return on {date.date()} for {names}")
            positions[held] *= factors[held]
        marked = float(positions.sum() + cash)
        gross_return = marked / previous_nav - 1.0
        fee_dollars = traded = 0.0
        if date == dates[-1]:
            traded = float(positions.sum())
            fee_dollars = fee * traded
            positions.fill(0.0)
            cash = marked - fee_dollars
        elif date in decisions:
            selection = decisions[date]
            _check_prices(market, date, selection["ticker"])
            target = np.zeros(len(tickers))
            target[tickers.get_indexer(selection["ticker"])] = 1.0 / len(selection)
            positions, fee_dollars, traded = _self_financing(positions, target, marked, fee)
            cash = 0.0
        nav = float(positions.sum() + cash)
        rows.append(
            {
                "date": date,
                "horizon": horizon,
                "strategy": strategy,
                "return": nav / previous_nav - 1.0,
                "gross_return": gross_return,
                "cost": fee_dollars / previous_nav,
                "turnover": traded / previous_nav,
                "nav": nav,
            }
        )
    return rows


def summarize_returns(returns: pd.DataFrame, holdout_start: str) -> pd.DataFrame:
    """Summarize daily tradable returns; Sharpe uses a zero risk-free rate."""
    holdout = _date(holdout_start)
    rows = []
    for (horizon, strategy), series in returns.groupby(["horizon", "strategy"], sort=False):
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
    market = _prepare(prices, features, config)
    top_n, capital, fee = _settings(config)
    horizons = config["backtest"].get("horizons", [1, 5, 20])
    if (
        not horizons
        or any(int(h) != h or h < 1 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be distinct positive integers")
    first_signal = _first_signal(market, config)
    first_entry_index = int(market.sessions.get_loc(first_signal)) + 1
    end = _date(config["backtest"].get("end", market.sessions[-1]))
    dates = market.sessions[first_entry_index:]
    dates = dates[dates <= end]
    if len(dates) < 2:
        raise ValueError("Evaluation needs at least two sessions for entry and liquidation")
    last_signal = market.sessions[market.sessions.get_loc(dates[-1]) - 1]
    refits = _refit_dates(market, first_signal, last_signal)
    return_rows, holding_frames, weight_rows, score_frames = [], [], [], []
    for horizon_value in horizons:
        horizon = int(horizon_value)
        labels = _labels(market, horizon)
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
            refit_date = refits[refits.searchsorted(signal_date, side="right") - 1]
            fitted = fitted_by_date[refit_date]
            frame = _eligible(market, signal_date, top_n)
            for strategy in STRATEGIES:
                ranking = _ranking(frame, strategy, fitted)
                columns = ["date", "ticker", "score", *SIGNALS, "reversal_raw"]
                scored = ranking.loc[:, columns].copy()
                scored["horizon"], scored["strategy"] = horizon, strategy
                score_frames.append(scored)
                selected = ranking if strategy == "universe" else ranking.iloc[:top_n]
                decisions[strategy][entry_date] = selected
                holding = selected.loc[:, ["ticker", "score"]].copy()
                holding["entry_date"], holding["signal_date"] = entry_date, signal_date
                holding["horizon"], holding["strategy"] = horizon, strategy
                holding["target_weight"] = 1.0 / len(selected)
                holding_frames.append(holding)
        for strategy in STRATEGIES:
            return_rows.extend(
                _simulate(market, dates, horizon, strategy, decisions[strategy], capital, fee)
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
    market = _prepare(prices, features, config)
    top_n, _, _ = _settings(config)
    signal_date = _date(signal_date)
    if signal_date not in market.sessions:
        raise ValueError(f"{signal_date.date()} is not an exact available signal session")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)
    first_signal = _first_signal(market, config)
    if signal_date < first_signal:
        raise ValueError("signal_date precedes the configured out-of-sample signal period")
    refit_date = _refit_dates(market, first_signal, signal_date)[-1]
    labels = _labels(market, horizon)
    weights, metadata = _fit(labels, market, refit_date, horizon, "ensemble", config)
    ranking = _ranking(_eligible(market, signal_date, top_n), "ensemble", {"ensemble": weights})
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
