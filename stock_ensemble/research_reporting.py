"""Numeric research evidence, exact provenance, and transparent rejection reasons."""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import summarize_returns
from .inference import research_inference
from .reporting import REFERENCES, save_json


def forecast_metrics(forecasts):
    rows = []
    for (horizon, strategy, fold), group in forecasts.groupby(["horizon", "strategy", "fold"]):
        if (
            not np.isfinite(group[["forward_return", "predicted_return", "null_prediction"]])
            .all()
            .all()
        ):
            raise ValueError("Nonfinite outer prediction or outcome")
        error = (
            group.assign(
                squared_error=(group.predicted_return - group.forward_return) ** 2,
                null_squared_error=(group.null_prediction - group.forward_return) ** 2,
            )
            .groupby("date")[["squared_error", "null_squared_error"]]
            .mean()
            .mean()
        )
        correlations = []
        for _, frame in group.groupby("date"):
            left, right = frame.predicted_return.rank(), frame.forward_return.rank()
            if left.std() > 0 and right.std() > 0:
                correlations.append(left.corr(right))
        rows.append(
            {
                "horizon": horizon,
                "strategy": strategy,
                "fold": fold,
                "mse": error.squared_error,
                "null_mse": error.null_squared_error,
                "oos_r_squared": 1 - error.squared_error / error.null_squared_error,
                "mean_rank_correlation": np.mean(correlations) if correlations else np.nan,
                "n_dates": group.date.nunique(),
            }
        )
    return pd.DataFrame(rows)


def write_research(result, metadata, config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    inference = research_inference(result["episodes"], config)
    for name, frame in {**result, **inference}.items():
        if isinstance(frame, pd.DataFrame):
            suffix = ".csv.gz" if name in ("holdings", "forecasts") else ".csv"
            frame.to_csv(output / (name + suffix), index=False)
    for name in ("fits", "selections"):
        save_json(output / (name + ".json"), result[name])
    daily = result["returns"]
    metrics = []
    for cost_bps, frame in daily.groupby("cost_bps"):
        metric = summarize_returns(frame, config["backtest"]["holdout_start"])
        metrics.append(metric.assign(cost_bps=cost_bps))
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(output / "path_metrics.csv", index=False)
    forecast_metrics(result["forecasts"]).to_csv(output / "forecast_metrics.csv", index=False)
    save_json(
        output / "run_manifest.json",
        {
            "config": config,
            "data_provenance": metadata,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "protocol_sha256": hashlib.sha256(
                (
                    Path(__file__).resolve().parent.parent
                    / "docs/research/implementation_protocol.md"
                ).read_bytes()
            ).hexdigest(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl")
            },
            "references": REFERENCES,
            "evidence_status": "exploratory retrospective; outcomes previously inspected",
            "bootstrap": "paired complete episodes; circular blocks within outer years; conditional",
            "selection": "inner mean return; family hyperparameter, then allocation, then policy",
            "incumbent": "legacy window and vintage schedule; same annually reset execution dates",
            "path_accounting": "each fold starts with configured capital; period metrics compound returns",
            "adoption": "only nested; Holm across horizons plus observed loss-frequency/tail safeguards",
            "warning": "No positive future return or universal optimum is established by this study.",
        },
    )
    primary = inference["episode_metrics"]
    primary = primary[(primary.period == "all") & primary.strategy.isin(["nested", "incumbent"])]
    print(primary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(inference["adoption"].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved retrospective research evidence to {output}")
    return inference
