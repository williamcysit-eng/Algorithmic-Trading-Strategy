"""Numerical evidence and provenance for joint stock selection."""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import pandas as pd

from .backtest import summarize_returns
from .reporting import save_json
from .selection_inference import selection_inference


def write_selection_results(result, metadata, config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    inference = selection_inference(result["episodes"], config)
    for name, frame in {**result, **inference}.items():
        if isinstance(frame, pd.DataFrame):
            suffix = ".csv.gz" if name in ("holdings", "shortlists") else ".csv"
            frame.to_csv(output / (name + suffix), index=False)
    save_json(output / "risk_models.json", result["risk_models"])
    summaries = []
    for cost_bps, group in result["returns"].groupby("cost_bps"):
        summaries.append(
            summarize_returns(group, config["backtest"]["holdout_start"]).assign(cost_bps=cost_bps)
        )
    pd.concat(summaries, ignore_index=True).to_csv(output / "path_metrics.csv", index=False)
    diagnostics = result["selection_diagnostics"]
    diagnostics.groupby(["horizon", "strategy"]).agg(
        n_decisions=("entry_date", "size"),
        mean_score_sacrifice=("score_sacrifice", "mean"),
        mean_shared_incumbent=("shared_incumbent_names", "mean"),
        mean_max_sector_weight=("max_sector_weight", "mean"),
        worst_sector_weight=("max_sector_weight", "max"),
        cap_relaxations=("cap_relaxed", "sum"),
        mean_predicted_variance=("predicted_daily_variance", "mean"),
    ).reset_index().to_csv(output / "selection_summary.csv", index=False)
    source = Path(__file__).resolve().parent
    protocol = source.parent / "docs/research/selection_protocol.md"
    save_json(
        output / "run_manifest.json",
        {
            "config": config,
            "data_provenance": metadata,
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source.glob("*.py"))
            },
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn")
            },
            "status": config["selection_study"]["status"],
            "sector_basis": "static labels from supplied universe; historical reclassification unavailable",
            "objective_units": "ordinal score budget, nominal daily covariance; no score-as-return assumption",
            "inference": "paired whole episodes; circular blocks stratified by outer year; conditional",
            "information": "announcement ledger separate; no newly captured event enters historical models",
            "limitations": [
                "previously inspected survivor-selected universe",
                "static sector labels",
                "small fixed candidate set; no global optimum",
                "no guarantee of future returns",
            ],
        },
    )
    print(inference["episode_metrics"].query("period == 'all'").to_string(index=False))
    print(inference["adoption"].to_string(index=False))
    print(f"Saved exploratory stock-selection evidence to {output}")
    return inference
