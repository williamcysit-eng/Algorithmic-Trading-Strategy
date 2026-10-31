"""Shared signal calendars, forward labels and self-financing portfolio accounting.

Signals observed at a session's close execute at the next session's close. Dollar
holdings drift between trades; fees are funded from wealth, never borrowing.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import SIGNALS


@dataclass
class Market:
    """Session-aligned prices and gross growth factors, plus signal-time features."""

    sessions: pd.DatetimeIndex
    prices: pd.DataFrame
    factors: pd.DataFrame
    features: pd.DataFrame


def normalize_date(value):
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def prepare_market(prices, features, config):
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
    # Each factor marks existing dollar holdings from the prior close to this close.
    factors = closes / closes.shift(1)
    if mode == "price":
        splits = prices.pivot(index="date", columns="ticker", values="split").reindex_like(closes)
        if (closes.notna() & (splits.isna() | (splits <= 0) | ~np.isfinite(splits))).any().any():
            raise ValueError("Every observed nominal close requires a positive finite split ratio")
        # A split changes shares held, offsetting the nominal-price discontinuity.
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
    return Market(closes.index, closes, factors, features)


def forward_labels(market, horizon):
    """Attach next-close forward returns and centered ranks to eligible features."""
    # For signal i: enter i+1, exit i+1+h. Multiplication propagates any missing
    # intermediate mark. Retain every original predictor for forecast fitting.
    forward = pd.DataFrame(1.0, index=market.sessions, columns=market.prices.columns)
    for step in range(1, horizon + 1):
        forward *= market.factors.shift(-(step + 1))
    forward -= 1.0
    forward.index.name = "date"
    forward.columns.name = "ticker"
    target = forward.stack().rename("forward_return").reset_index()
    frame = market.features.loc[
        market.features["eligible"].fillna(False).astype(bool)
    ]
    frame = frame.merge(target, on=["date", "ticker"], how="inner", validate="one_to_one")
    frame = frame.loc[np.isfinite(frame["forward_return"])].copy()
    count = frame.groupby("date")["forward_return"].transform("size")
    rank = frame.groupby("date")["forward_return"].rank(method="average")
    frame["target"] = np.where(count > 1, 2.0 * (rank - 1.0) / (count - 1.0) - 1.0, 0.0)
    label_ends = pd.Series(market.sessions, index=market.sessions).shift(-(horizon + 1))
    frame["label_end"] = frame["date"].map(label_ends)
    return frame


def first_signal(market, config):
    start = normalize_date(config["backtest"]["start"])
    entry_index = int(market.sessions.searchsorted(start))
    if entry_index >= len(market.sessions):
        raise ValueError("Evaluation start is after the last price session")
    if entry_index == 0:
        raise ValueError("Evaluation requires a prior signal session and historical training")
    return market.sessions[entry_index - 1]


def refit_dates(market, first_signal, last_signal):
    dates = market.sessions[(market.sessions >= first_signal) & (market.sessions <= last_signal)]
    # This tests against the full calendar, not just rebalance/entry sessions.
    months = market.sessions.to_period("M")
    month_first = market.sessions[np.r_[True, months[1:] != months[:-1]]]
    return dates[(dates == first_signal) | dates.isin(month_first)]


def eligible_as_of(market, signal_date, top_n):
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


def portfolio_settings(config):
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
    """Return post-fee dollar holdings, fee dollars and traded dollars.

    Post-fee invested wealth A must satisfy:
    A + fee * sum(abs(A * target - pretrade)) = wealth.
    Fees reduce the investment budget instead of creating borrowing.
    """
    if fee == 0:
        after = wealth
    else:
        low, high = 0.0, wealth
        # With fee < 1, the residual strictly increases with post-fee wealth.
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


def simulate_portfolio(market, dates, horizon, strategy, decisions, capital, fee):
    """Mark before trading; require explicit, fully invested long-only targets.

    Decision dates are execution closes, not signal closes. Callers must use
    information observed no later than the preceding session's close.
    """
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be positive and finite")
    if not np.isfinite(fee) or not 0 <= fee < 1:
        raise ValueError("fee must be finite and in [0, 1)")
    for selection in decisions.values():
        if not {"ticker", "target_weight"}.issubset(selection.columns):
            raise ValueError("Every decision requires ticker and target_weight")
        if selection.empty or selection["ticker"].isna().any():
            raise ValueError("Every decision requires nonmissing selected tickers")
        if selection["ticker"].duplicated().any():
            raise ValueError("Selected tickers must be unique")
        if not selection["ticker"].isin(market.prices.columns).all():
            raise ValueError("Selected tickers must be known market tickers")
        weights = selection["target_weight"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any() or (weights > 1).any():
            raise ValueError("target_weight must be positive and finite without borrowing")
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("target_weight must sum to one for full investment")
    tickers = market.prices.columns
    positions = np.zeros(len(tickers))
    cash = capital
    nav = capital
    rows = []
    for date in dates:
        previous_nav = nav
        held = positions > 0
        # Only positions held before today's close earn today's price movement.
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
        # Terminal liquidation takes precedence over a scheduled rebalance.
        if date == dates[-1]:
            traded = float(positions.sum())
            fee_dollars = fee * traded
            positions.fill(0.0)
            cash = marked - fee_dollars
        elif date in decisions:
            selection = decisions[date]
            _check_prices(market, date, selection["ticker"])
            target = np.zeros(len(tickers))
            target[tickers.get_indexer(selection["ticker"])] = selection["target_weight"].to_numpy(
                dtype=float
            )
            positions, fee_dollars, traded = _self_financing(positions, target, marked, fee)
            cash = 0.0
        nav = float(positions.sum() + cash)
        # Costs and turnover use prior NAV, matching both return denominators.
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


def regularized_weights(predicted_returns, penalty, lower=0.05, upper=0.4):
    """Maximize nominal expected return minus quadratic equal-weight deviation.

    The strictly concave objective has a unique bounded-simplex solution.
    Center the projection at the marginal forecast of the linear optimum: its
    Lagrange multiplier lies in [-upper, -lower], even for extreme forecasts.
    This avoids bisection over an arbitrarily wide forecast-dependent interval.
    """
    forecasts = np.asarray(predicted_returns, dtype=float)
    if forecasts.ndim != 1 or not forecasts.size or not np.isfinite(forecasts).all():
        raise ValueError("predicted_returns must be a nonempty finite one-dimensional vector")
    penalty, lower, upper = float(penalty), float(lower), float(upper)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be positive and finite")
    if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper <= 1:
        raise ValueError("Require finite bounds with 0 <= lower <= upper <= 1")
    count = len(forecasts)
    if count * lower > 1 or count * upper < 1:
        raise ValueError("Weight bounds are infeasible for the selected stock count")
    if count * lower == 1:
        return np.full(count, lower)
    if count * upper == 1:
        return np.full(count, upper)
    marginal = min(int((1.0 - count * lower) / (upper - lower)), count - 1)
    center = np.sort(forecasts)[::-1][marginal]
    # Extended precision and clipping also handle finite forecasts whose scaled
    # differences overflow ordinary floats; their weights simply hit a bound.
    with np.errstate(over="ignore"):
        scaled = (forecasts.astype(np.longdouble) - np.longdouble(center)) / (
            2 * np.longdouble(penalty)
        )
    low, high = -upper, -lower
    for _ in range(128):
        midpoint = (low + high) / 2.0
        weights = np.clip(scaled - midpoint, lower, upper)
        mass = weights.sum()
        if abs(mass - 1.0) <= 1e-15:
            break
        if mass > 1.0:
            low = midpoint
        else:
            high = midpoint
    weights = np.asarray(weights, dtype=float)
    # Remove only roundoff, without rescaling and violating an active bound.
    remainder = 1.0 - weights.sum()
    for index in range(count):
        adjustment = np.clip(remainder, lower - weights[index], upper - weights[index])
        weights[index] += adjustment
        remainder -= adjustment
        if remainder == 0:
            break
    return weights


def allocate_ranked(ranking, top_n, *, method="equal", penalty=None, lower=0.05, upper=0.4):
    """Return the full deterministic ranking with explicit selected-stock weights.

    Ordinal scores select companies but never act as expected return forecasts.
    Regularization therefore consumes predicted_return, not score.
    """
    if int(top_n) != top_n or not 1 <= top_n <= len(ranking):
        raise ValueError("top_n must be a positive integer no larger than the ranking")
    top_n = int(top_n)
    if not {"ticker", "score"}.issubset(ranking.columns):
        raise ValueError("Ranking requires ticker and score")
    if ranking["ticker"].isna().any() or ranking["ticker"].duplicated().any():
        raise ValueError("Ranking tickers must be nonmissing and unique")
    if not np.isfinite(ranking["score"].to_numpy(dtype=float)).all():
        raise ValueError("Ranking scores must be finite")
    result = ranking.sort_values(
        ["score", "ticker"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True).copy()
    result["rank"] = np.arange(1, len(result) + 1)
    if method == "equal":
        weights = np.full(top_n, 1.0 / top_n)
    elif method == "regularized":
        if penalty is None or "predicted_return" not in result.columns:
            raise ValueError("Regularized allocation requires penalty and predicted_return")
        weights = regularized_weights(
            result.iloc[:top_n]["predicted_return"].to_numpy(dtype=float),
            penalty,
            lower,
            upper,
        )
    else:
        raise ValueError("Allocation method must be 'equal' or 'regularized'")
    result["target_weight"] = 0.0
    result.loc[: top_n - 1, "target_weight"] = weights
    return result
