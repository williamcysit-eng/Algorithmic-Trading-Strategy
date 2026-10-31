"""Evidence and preference curves for the fixed-stock risk-allocation experiment."""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import summarize_returns
from .reporting import save_json
from .risk_inference import METHODS, risk_inference


def _figure(metrics, preferences, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(metrics.horizon.unique())
    colors = {
        "equal": "#333333",
        "balanced": "#005f73",
        "inverse_volatility": "#ca6702",
        "minimum_variance": "#6a4c93",
    }
    fig, axes = plt.subplots(len(horizons), 2, figsize=(11, 3.5 * len(horizons)), squeeze=False)
    for index, horizon in enumerate(horizons):
        frame = metrics[(metrics.horizon == horizon) & (metrics.period == "all")]
        for method in METHODS:
            row = frame[frame.strategy == method].iloc[0]
            axes[index, 0].scatter(
                row.volatility * 100,
                row.mean_return * 100,
                color=colors[method],
                label=method,
                s=45,
            )
            curve = preferences[
                (preferences.horizon == horizon)
                & (preferences.period == "all")
                & preferences.strategy.eq(method)
            ]
            axes[index, 1].plot(
                curve.gamma,
                curve.certainty_equivalent * 100,
                marker="o",
                color=colors[method],
                label=method,
            )
        axes[index, 0].set(
            title=f"{horizon} sessions: observed mean and variability",
            xlabel="Episode standard deviation (%)",
            ylabel="Mean episode return (%)",
        )
        axes[index, 1].set(
            title=f"{horizon} sessions: preference changes the tradeoff",
            xlabel="Relative risk aversion (gamma)",
            ylabel="Certainty-equivalent return (%)",
        )
        for axis in axes[index]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    fig.suptitle("Retrospective risk study: fixed stocks, different weights", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "risk_preferences.png", dpi=170)
    fig.savefig(output / "risk_preferences.pdf")
    plt.close(fig)


def write_risk_results(result, metadata, config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    inference = risk_inference(result["episodes"], config)
    for name, frame in {**result, **inference}.items():
        if isinstance(frame, pd.DataFrame):
            suffix = ".csv.gz" if name == "holdings" else ".csv"
            frame.to_csv(output / (name + suffix), index=False)
    save_json(output / "risk_models.json", result["risk_models"])
    path_metrics = []
    for cost_bps, frame in result["returns"].groupby("cost_bps"):
        metrics = summarize_returns(frame, config["backtest"]["holdout_start"])
        path_metrics.append(metrics.assign(cost_bps=cost_bps))
    pd.concat(path_metrics, ignore_index=True).to_csv(output / "path_metrics.csv", index=False)
    holdings = result["holdings"]
    grouped = holdings.groupby(["horizon", "fold", "strategy", "entry_date"])
    allocation_metrics = grouped.agg(
        max_weight=("target_weight", "max"),
        min_weight=("target_weight", "min"),
        predicted_daily_variance=("predicted_daily_variance", "first"),
        signal_date=("signal_date", "first"),
    ).reset_index()
    hhi = grouped.target_weight.apply(lambda w: float(np.sum(w**2))).rename("weight_hhi")
    allocation_metrics = allocation_metrics.merge(
        hhi.reset_index(), on=["horizon", "fold", "strategy", "entry_date"]
    )
    allocation_metrics.to_csv(output / "allocation_metrics.csv", index=False)
    metrics = inference["episode_metrics"]
    dominance = []
    for horizon, group in metrics[metrics.period == "all"].groupby("horizon"):
        for row in group.itertuples(index=False):
            dominates = group[
                (group.mean_return >= row.mean_return)
                & (group.volatility <= row.volatility)
                & ((group.mean_return > row.mean_return) | (group.volatility < row.volatility))
            ]
            dominance.append(
                {
                    "horizon": horizon,
                    "strategy": row.strategy,
                    "mean_volatility_dominated": not dominates.empty,
                    "dominated_by": ",".join(dominates.strategy),
                    "interpretation": "observed only, among four fixed methods",
                }
            )
    pd.DataFrame(dominance).to_csv(output / "dominance.csv", index=False)
    source_root = Path(__file__).resolve().parent
    protocol = source_root.parent / "docs/research/risk_protocol.md"
    save_json(
        output / "run_manifest.json",
        {
            "config": config,
            "data_provenance": metadata,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source_root.glob("*.py"))
            },
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "matplotlib")
            },
            "status": config["risk_study"]["status"],
            "covariance_units": "daily nominal split-aware simple price returns through signal close",
            "horizon_risk": "daily covariance is a proxy; realized multi-session risk is measured separately",
            "certainty_equivalent": "inverse CRRA of empirical mean utility, return per complete episode",
            "primary_preference": "gamma=3 is an analysis convention, not an elicited user preference",
            "inference": "paired, outer-year-stratified circular whole-episode blocks; conditional",
            "limitations": [
                "previously inspected history; retrospective exploratory evidence",
                "survivor-selected stock universe; incomplete corporate-action ledger",
                "empirical risk preference, not a forecast or guarantee",
                "testing several preferences does not identify a universal optimum",
            ],
            "references": [
                "https://doi.org/10.1016/S0047-259X(03)00096-4",
                "https://doi.org/10.1093/rfs/hhm075",
            ],
        },
    )
    _figure(inference["episode_metrics"], inference["preferences"], output)
    print(inference["episode_metrics"].query("period == 'all'").to_string(index=False))
    print(inference["adoption"].to_string(index=False))
    print(f"Saved retrospective risk-allocation evidence to {output}")
    return inference
