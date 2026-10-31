"""Paired episode inference, prespecified multiplicity, and adoption safeguards."""

import numpy as np
import pandas as pd

COMPARISONS = (
    ("ridge_equal", "equal_signal"),
    ("boosted_equal", "equal_signal"),
    ("ridge_regularized", "ridge_equal"),
    ("boosted_regularized", "boosted_equal"),
    ("nested", "equal_signal"),
    ("nested", "momentum"),
)
EPISODE_KEY = ["fold", "entry_date", "exit_date"]


def holm_adjust(pvalues):
    """Holm step-down adjustment, valid under arbitrary dependence."""
    pvalues = np.asarray(pvalues, dtype=float)
    if (
        pvalues.ndim != 1
        or not np.isfinite(pvalues).all()
        or (pvalues < 0).any()
        or (pvalues > 1).any()
    ):
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def paired_panel(episodes, horizon):
    frame = episodes[episodes.horizon == horizon]
    if frame.empty or frame.duplicated([*EPISODE_KEY, "strategy"]).any():
        raise ValueError("Missing or duplicate paired episodes")
    panel = frame.pivot(index=EPISODE_KEY, columns="strategy", values="price_return").sort_index()
    if panel.isna().any().any() or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Unmatched or nonfinite paired episodes; cannot drop inconvenient returns")
    return panel


def block_indices(panel, horizon, settings, block_sessions):
    """Resample paired episodes within each observed year, preserving its mass."""
    draws = int(settings["bootstrap_draws"])
    if draws < 1 or block_sessions < 1 or horizon < 1:
        raise ValueError("Bootstrap sizes must be positive")
    rng = np.random.default_rng(settings["seed"])
    block = max(1, int(np.ceil(block_sessions / horizon)))
    folds = panel.index.get_level_values("fold").to_numpy()
    parts = []
    for fold in sorted(set(folds)):
        positions = np.flatnonzero(folds == fold)
        count = len(positions)
        starts = rng.integers(0, count, size=(draws, (count + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % count).reshape(draws, -1)[:, :count]
        parts.append(positions[indices])
    return np.concatenate(parts, axis=1)


def paired_difference(panel, first, second, indices, *, one_sided=False):
    if first not in panel or second not in panel:
        raise ValueError(f"Missing comparator: {first}, {second}")
    difference = (panel[first] - panel[second]).to_numpy()
    mean = difference.mean()
    resampled = difference[indices].mean(axis=1)
    lower, upper = np.quantile(resampled, [0.025, 0.975])
    centered = resampled - mean
    # Center under H0 rather than mistake percentile mass for a null p-value.
    exceedances = (centered >= mean) if one_sided else (np.abs(centered) >= abs(mean))
    pvalue = (np.count_nonzero(exceedances) + 1) / (len(resampled) + 1)
    return {
        "first": first,
        "second": second,
        "n_episodes": len(difference),
        "mean_difference": float(mean),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "pvalue": float(pvalue),
        "alternative": "greater" if one_sided else "two-sided",
    }


def episode_summaries(episodes, holdout_start):
    rows = []
    for period, frame in (
        ("all", episodes),
        ("inspected_2023_onward", episodes[episodes.entry_date >= pd.Timestamp(holdout_start)]),
    ):
        for (horizon, strategy), group in frame.groupby(["horizon", "strategy"]):
            values = group.price_return.to_numpy(dtype=float)
            q05 = float(np.quantile(values, 0.05))
            rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "strategy": strategy,
                    "n_episodes": len(values),
                    "mean_return": values.mean(),
                    "median_return": np.median(values),
                    "loss_fraction": np.mean(values < 0),
                    "fifth_percentile": q05,
                    "mean_below_fifth_percentile": values[values <= q05].mean(),
                    "standard_deviation": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def research_inference(episodes, config):
    settings = config["research"]
    primary_block = settings["bootstrap_block_sessions"]
    blocks = [primary_block, *settings["bootstrap_sensitivity_sessions"]]
    summaries = episode_summaries(episodes, config["backtest"]["holdout_start"])
    comparisons, adoption = [], []
    for horizon in config["backtest"]["horizons"]:
        panel = paired_panel(episodes, horizon)
        for block in blocks:
            indices = block_indices(panel, horizon, settings, block)
            rows = [
                dict(
                    paired_difference(panel, first, second, indices),
                    horizon=horizon,
                    block_sessions=block,
                    block_episodes=int(np.ceil(block / horizon)),
                    draws=settings["bootstrap_draws"],
                    primary=block == primary_block,
                )
                for first, second in COMPARISONS
            ]
            adjusted = holm_adjust([row["pvalue"] for row in rows])
            for row, pvalue in zip(rows, adjusted):
                row["pvalue_holm_six"] = pvalue
            comparisons.extend(rows)
            if block == primary_block:
                row = dict(
                    paired_difference(panel, "nested", "incumbent", indices, one_sided=True),
                    horizon=horizon,
                )
                for strategy in ("nested", "incumbent"):
                    values = panel[strategy].to_numpy()
                    row[strategy + "_loss_fraction"] = float(np.mean(values < 0))
                    row[strategy + "_fifth_percentile"] = float(np.quantile(values, 0.05))
                adoption.append(row)
    adjusted = holm_adjust([row["pvalue"] for row in adoption])
    for row, pvalue in zip(adoption, adjusted):
        row["pvalue_holm_horizons"] = pvalue
        row["passes_return"] = row["mean_difference"] > 0 and pvalue <= 0.05
        row["passes_loss_frequency"] = row["nested_loss_fraction"] <= row["incumbent_loss_fraction"]
        row["passes_lower_tail"] = (
            row["nested_fifth_percentile"] >= row["incumbent_fifth_percentile"]
        )
        row["qualifies"] = (
            row["passes_return"] and row["passes_loss_frequency"] and row["passes_lower_tail"]
        )
    return {
        "episode_metrics": summaries,
        "comparisons": pd.DataFrame(comparisons),
        "adoption": pd.DataFrame(adoption),
    }
