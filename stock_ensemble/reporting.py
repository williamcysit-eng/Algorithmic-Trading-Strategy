"""Machine-generated research artifacts, not the assessed human-written report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCES = [
    {
        "authors": "Bernard and Thomas (1989)",
        "title": "Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?",
        "url": "https://doi.org/10.2307/2491062",
        "use": "Motivates earnings drift; our seasonal net-income/assets proxy is not their exact SUE.",
    },
    {
        "authors": "Jegadeesh and Titman (1993)",
        "title": "Returns to Buying Winners and Selling Losers",
        "url": "https://doi.org/10.1111/j.1540-6261.1993.tb04702.x",
        "use": "Intermediate-horizon price continuation; not evidence that all winners keep rising.",
    },
    {
        "authors": "Kenneth R. French Data Library",
        "title": "Momentum Factor Construction",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html",
        "use": "Prior 2-12 month momentum motivates skipping the latest month.",
    },
    {
        "authors": "Lehmann (1990)",
        "title": "Fads, Martingales, and Market Efficiency",
        "url": "https://www.nber.org/papers/w2533",
        "use": "Short-horizon reversal; our one-day implementation differs from the weekly study.",
    },
    {
        "authors": "Chan, Jegadeesh, and Lakonishok (1996)",
        "title": "Momentum Strategies",
        "url": "https://www.nber.org/papers/w5375",
        "use": "Earnings and price momentum contain distinct information.",
    },
    {
        "authors": "Da, Liu, and Schaumburg (2014)",
        "title": "A Closer Look at the Short-Term Return Reversal",
        "url": "https://academicweb.nd.edu/~zda/Reversal.pdf",
        "use": "Separating fundamental news from reversal motivates, but does not validate, our gate.",
    },
]

LIMITATIONS = [
    (
        "The hand-selected universe contains present-day survivors, not historical index membership. "
        "No delisted-stock returns: this is a reproducible survivor-biased pilot, not market-wide alpha."
    ),
    (
        "SEC filing dates are later than many earnings releases. The signal is seasonal unexpected "
        "profit scaled by assets, not analyst-consensus surprise or exact replication of PEAD; "
        "net income and income available to common shareholders use separate historical scales."
    ),
    (
        "Missing or stale earnings are neutral, not imputed from current fundamentals. Free SEC "
        "companyfacts coverage, tags, reorganizations, and fiscal-year changes can limit coverage."
    ),
    (
        "Yahoo adjusted prices are vendor-revised. Splits are handled; spin-offs, rights, mergers, "
        "special distributions and vendor corrections are not a complete corporate-action ledger."
    ),
    (
        "Signals use close d, filings strictly before d, and execute at close d+1. A one-day reversal "
        "may dissipate before execution; next-open results are not claimed."
    ),
    (
        "Closing-auction fills and fractional shares are assumed. Ten basis points per traded dollar "
        "is a sensitivity scenario, not calibrated spread/market impact or a capacity estimate."
    ),
    (
        "Price-mode performance excludes dividends, matching the assignment; total-return mode is "
        "available separately. Sharpe uses zero risk-free return; it is not a factor-model alpha."
    ),
    (
        "Holdout is chronological prequential evaluation: the fixed procedure may refit using already "
        "realized earlier holdout labels, never later labels. Hyperparameters are not retuned on it."
    ),
    (
        "The earnings gate is a project-specific synthesis, not a claim of new academic discovery. "
        "Compare it with the ungated ablation; do not assume an improvement."
    ),
    (
        "Bootstrap intervals are paired 20-session circular-block intervals under stationarity, "
        "not multiple-testing-adjusted discovery claims or guarantees of future performance."
    ),
]


def save_json(path: Path, payload: object) -> None:
    def convert(value):
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    path.write_text(json.dumps(payload, indent=2, default=convert, allow_nan=False) + "\n")


def paired_intervals(returns: pd.DataFrame, holdout_start: str) -> pd.DataFrame:
    """Intervals for annualized *arithmetic mean* paired daily return differences."""
    rng = np.random.default_rng(7008)
    rows = []
    for horizon, group in returns[returns["date"] >= pd.Timestamp(holdout_start)].groupby(
        "horizon"
    ):
        wide = group.pivot(index="date", columns="strategy", values="return").sort_index()
        for comparator in ("universe", "equal_signal", "ungated"):
            paired = wide[["ensemble", comparator]].dropna()
            if len(paired) < 40:
                continue
            difference = (paired["ensemble"] - paired[comparator]).to_numpy()
            size = len(difference)
            block = min(20, size)
            means = np.empty(1000)
            offsets = np.arange(block)
            for draw in range(len(means)):
                starts = rng.integers(0, size, size=(size + block - 1) // block)
                indices = ((starts[:, None] + offsets) % size).ravel()[:size]
                means[draw] = difference[indices].mean() * 252
            lower, upper = np.quantile(means, [0.025, 0.975])
            rows.append(
                {
                    "horizon": horizon,
                    "comparator": comparator,
                    "n_days": size,
                    "block_sessions": block,
                    "draws": len(means),
                    "annualized_mean_difference": difference.mean() * 252,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _plot_curves(returns: pd.DataFrame, output: Path, holdout_start: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(returns["horizon"].unique())
    colors = {
        "ensemble": "#005f73",
        "equal_signal": "#94a3b8",
        "ungated": "#ca6702",
        "momentum": "#6a4c93",
        "universe": "#222222",
    }
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3.4 * len(horizons)), squeeze=False)
    for ax, horizon in zip(axes[:, 0], horizons):
        group = returns[returns["horizon"] == horizon]
        for strategy, color in colors.items():
            curve = group[group["strategy"] == strategy].sort_values("date")
            ax.plot(
                curve["date"], curve["nav"] / 1_000_000, label=strategy, color=color, linewidth=1.25
            )
        boundary = pd.Timestamp(holdout_start)
        if group["date"].min() <= boundary <= group["date"].max():
            ax.axvline(boundary, color="#777777", linestyle="--", linewidth=0.8)
        ax.set(
            title=f"{horizon}-session holding period", ylabel="Net wealth ($m, log)", yscale="log"
        )
        ax.grid(alpha=0.15)
        ax.legend(ncol=5, fontsize=8, loc="upper left")
    fig.suptitle("Chronological evaluation | fixed survivor-selected universe", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "equity_curves.png", dpi=160)
    plt.close(fig)


def write_results(result: dict, features: pd.DataFrame, metadata: dict, config: dict) -> None:
    from .backtest import summarize_returns

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            suffix = ".csv.gz" if name in ("scores", "holdings") else ".csv"
            frame.to_csv(output / f"{name}{suffix}", index=False)
    features.to_csv(output / "features.csv.gz", index=False)
    returns = result["returns"]
    zero_cost = returns.copy()
    zero_cost["return"] = zero_cost["gross_return"]
    zero_cost["cost"] = 0.0
    zero_cost["nav"] = (
        config["backtest"]["initial_capital"]
        * (1 + zero_cost["return"]).groupby([zero_cost["horizon"], zero_cost["strategy"]]).cumprod()
    )
    zero_metrics = summarize_returns(zero_cost, config["backtest"]["holdout_start"])
    # Gross marks recover fee-free performance exactly; configured-fee dollar
    # turnover is not the fee-free traded notional, so do not mislabel it.
    zero_metrics = zero_metrics.drop(columns="mean_daily_turnover")
    zero_metrics.to_csv(output / "zero_cost_metrics.csv", index=False)
    paired_intervals(returns, config["backtest"]["holdout_start"]).to_csv(
        output / "paired_holdout_intervals.csv", index=False
    )
    annual = returns.assign(year=returns["date"].dt.year, factor=1 + returns["return"])
    annual = annual.groupby(["horizon", "strategy", "year"])["factor"].prod().sub(1)
    annual.rename("return").to_csv(output / "annual_returns.csv")
    active = features["earnings_age"].between(1, config["features"]["earnings_max_age"])
    coverage = (
        features.assign(has_earnings=active & features["surprise"].notna())
        .groupby("ticker")
        .agg(
            price_sessions=("date", "size"),
            eligible_sessions=("eligible", "sum"),
            sessions_with_earnings=("has_earnings", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
    )
    coverage.to_csv(output / "coverage.csv")
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    save_json(
        output / "run_manifest.json",
        {
            "config": config,
            "source_sha256": source_hashes,
            "data_provenance": metadata,
            "references": REFERENCES,
            "limitations": LIMITATIONS,
            "score_interpretation": "Relative ranking score, NOT expected return or a probability.",
            "zero_cost_metrics": "Same decisions and gross marks, with execution fees removed.",
            "reported_intervals": "Annualized arithmetic mean differences, not CAGR differences.",
        },
    )
    _plot_curves(returns, output, config["backtest"]["holdout_start"])
    selected = result["metrics"]
    selected = selected[
        (selected["period"] == "holdout")
        & selected["strategy"].isin(["ensemble", "ungated", "momentum", "universe"])
    ]
    columns = ["horizon", "strategy", "cagr", "sharpe", "max_drawdown"]
    print(selected[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved evaluation artifacts to {output}")
    print("Caution: survivor-selected pilot; no claim of unbiased alpha or future profitability.")
