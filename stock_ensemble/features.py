"""Close-of-session experts; no forward returns or future eligibility filters.

Earnings is a filing-date seasonal unexpected-profit proxy, not analyst surprise.
Momentum uses the conventional 12-minus-1-month formation window. Reversal
ranks the negative one-session market-relative return in prior-volatility units.
Our testable extension discounts *positive* rebound credit after bad earnings;
it does not assume that every sharp fall is temporary price pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def centered_rank(values: pd.Series) -> pd.Series:
    """Map average ranks to [-1, 1], preserving ties; singleton is neutral."""
    count = values.notna().sum()
    if count < 2:
        return values * 0.0
    return 2.0 * (values.rank(method="average") - 1.0) / (count - 1.0) - 1.0


def build_features(prices: pd.DataFrame, earnings: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return the full price panel, including ineligible rows for honest auditing.

    A filing dated d is not usable at close d: only filed_date < signal_date
    joins. The simulator adds a further next-close execution lag. Windows count
    exchange sessions on the union price calendar, not each stock's available
    rows, so missing prices cannot compress time or fabricate momentum.
    """
    settings = config["features"]
    lookback = int(settings["momentum_lookback"])
    skip = int(settings["momentum_skip"])
    vol_window = int(settings["volatility_window"])
    if not 0 < skip < lookback or vol_window < 2:
        raise ValueError("Require 0 < momentum_skip < momentum_lookback and volatility_window >= 2")
    for key in ("earnings_half_life", "earnings_max_age", "gate_half_life"):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate ticker/session price rows")
    frame = prices.sort_values(["date", "ticker"]).copy()
    adjusted = frame.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    close = frame.pivot(index="date", columns="ticker", values="close").reindex_like(adjusted)
    volume = frame.pivot(index="date", columns="ticker", values="volume").reindex_like(adjusted)
    returns = adjusted.pct_change(fill_method=None)
    volatility = returns.shift(1).rolling(vol_window, min_periods=vol_window).std(ddof=1)
    momentum = adjusted.shift(skip) / adjusted.shift(lookback) - 1.0
    # The cross-sectional market proxy is contemporaneously observable; no ETF is held.
    residual = returns.sub(returns.mean(axis=1), axis=0)
    reversal = -residual / volatility
    dollar_volume = (close * volume).rolling(vol_window, min_periods=vol_window).mean()
    complete_window = adjusted.rolling(lookback + 1, min_periods=lookback + 1).count()
    eligible = (
        (complete_window == lookback + 1)
        & (close >= settings["min_price"])
        & (dollar_volume >= settings["min_dollar_volume"])
        & (volatility > 0)
        & np.isfinite(momentum)
        & np.isfinite(reversal)
    )
    eligible.loc[eligible.sum(axis=1) < settings["min_cross_section"], :] = False

    for name, wide in (
        ("momentum_value", momentum),
        ("reversal_value", reversal),
        ("volatility", volatility),
        ("dollar_volume", dollar_volume),
        ("eligible", eligible),
    ):
        frame = frame.join(wide.stack(future_stack=True).rename(name), on=["date", "ticker"])
    frame["eligible"] = frame["eligible"].fillna(False).astype(bool)

    if not earnings.empty:
        if earnings.duplicated(["ticker", "filed_date"]).any():
            raise ValueError("Earnings must contain at most one event per ticker/filing date")
        if (earnings["fiscal_end"] > earnings["filed_date"]).any():
            raise ValueError("Earnings fiscal end cannot be later than public filing date")
        frame = pd.merge_asof(
            frame.sort_values("date"),
            earnings.sort_values("filed_date"),
            left_on="date",
            right_on="filed_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=False,
        )
    else:
        frame["filed_date"] = pd.NaT
        frame["fiscal_end"] = pd.NaT
        frame["surprise"] = np.nan
        frame["seasonal_change"] = np.nan
        frame["source_accn"] = None
    frame["earnings_age"] = (frame["date"] - frame["filed_date"]).dt.days
    active = (
        frame["earnings_age"].between(1, settings["earnings_max_age"]) & frame["surprise"].notna()
    )
    freshness = np.exp2(-frame["earnings_age"] / settings["earnings_half_life"]).where(active, 0)
    earnings_value = frame["surprise"].where(active & frame["eligible"])
    frame["earnings"] = (
        earnings_value.groupby(frame["date"]).transform(centered_rank).fillna(0.0) * freshness
    )
    for source, target in (("momentum_value", "momentum"), ("reversal_value", "reversal_raw")):
        frame[target] = (
            frame[source].where(frame["eligible"]).groupby(frame["date"]).transform(centered_rank)
        )
    # Absolute bad news, not merely below-peer earnings, activates the gate.
    bad_news = (-np.tanh(frame["surprise"])).clip(lower=0).where(active, 0.0)
    gate_freshness = np.exp2(-frame["earnings_age"] / settings["gate_half_life"]).where(active, 0)
    frame["gate"] = (1.0 - bad_news * gate_freshness).clip(0, 1)
    frame["reversal"] = (
        frame["reversal_raw"].clip(upper=0) + frame["reversal_raw"].clip(lower=0) * frame["gate"]
    )
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)
