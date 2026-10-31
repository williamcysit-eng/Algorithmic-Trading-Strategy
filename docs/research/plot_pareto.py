"""Plot the empirical Pareto set of already-tested risk allocations.

Run from the repository root:
    uv run --frozen python docs/research/plot_pareto.py

The staircase is max(mean return) among tested methods whose standard deviation
is at most the x-axis value. Only markers are tested portfolios. No interpolation,
new strategy, expected-return forecast, or covariance optimization is introduced.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "equal": "Equal weights",
    "inverse_volatility": "Inverse volatility",
    "minimum_variance": "Minimum variance",
    "balanced": "Balanced",
}
COLORS = {
    "equal": "#243746",
    "inverse_volatility": "#007f73",
    "minimum_variance": "#7056a1",
    "balanced": "#c47a19",
}


def classify_points(frame):
    """Strict mean/volatility dominance, separately within each horizon.

    Equal coordinates do not dominate each other. The two comparisons must
    both be weakly favorable, with at least one strictly favorable comparison.
    """
    required = {"period", "horizon", "strategy", "mean_return", "volatility", "n_episodes"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing or empty episode metrics")
    if frame.duplicated(["period", "horizon", "strategy"]).any():
        raise ValueError("Duplicate policy metrics")
    if frame.period.nunique() != 1:
        raise ValueError("Plot one evaluation period at a time")
    if not np.isfinite(frame[["mean_return", "volatility", "n_episodes"]]).all().all():
        raise ValueError("Nonfinite episode metrics")
    if (frame.volatility < 0).any() or (frame.n_episodes < 2).any():
        raise ValueError("Invalid standard deviation or episode count")
    rows = []
    for horizon, group in frame.groupby("horizon", sort=True):
        if set(group.strategy) != set(LABELS):
            raise ValueError("Each horizon requires the four tested allocation methods")
        if group.n_episodes.nunique() != 1:
            raise ValueError("Methods must use the same holding episodes")
        for row in group.sort_values(["volatility", "mean_return", "strategy"]).to_dict("records"):
            dominating = group[
                (group.volatility <= row["volatility"])
                & (group.mean_return >= row["mean_return"])
                & (
                    (group.volatility < row["volatility"])
                    | (group.mean_return > row["mean_return"])
                )
            ]
            row.update(
                {
                    "horizon": horizon,
                    "pareto_efficient": dominating.empty,
                    "dominated_by": ",".join(sorted(dominating.strategy)),
                    "standard_deviation_percent": 100 * row["volatility"],
                    "mean_return_percent": 100 * row["mean_return"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def validate_frontier(points, archived_dominance):
    """Compare pairwise classification with an independent sorted running maximum."""
    for horizon, group in points.groupby("horizon"):
        best_mean = -np.inf
        scanned = set()
        for _, tied_risk in group.sort_values("volatility").groupby("volatility", sort=True):
            best_at_risk = tied_risk.mean_return.max()
            if best_at_risk > best_mean:
                scanned.update(tied_risk.loc[tied_risk.mean_return == best_at_risk, "strategy"])
                best_mean = best_at_risk
        pairwise = set(group.loc[group.pareto_efficient, "strategy"])
        if scanned != pairwise:
            raise ValueError(f"Pareto classification mismatch for horizon {horizon}")
    if points.period.iloc[0] == "all" and archived_dominance.exists():
        archived = pd.read_csv(archived_dominance)
        check = points.merge(archived, on=["horizon", "strategy"], validate="one_to_one")
        if (
            len(check) != len(points)
            or not (check.pareto_efficient == ~check.mean_volatility_dominated).all()
        ):
            raise ValueError("Classification disagrees with the archived risk-study audit")


def render(points, output, period):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    horizons = sorted(points.horizon.unique())
    titles = {1: "Daily: 1 session", 5: "Weekly: 5 sessions", 19: "Monthly: 19 sessions"}
    period_label = {
        "all": "2018-01-02 to 2026-10-21",
        "inspected_2023_onward": "2023-01-03 to 2026-10-21",
    }[period]
    fig, axes = plt.subplots(1, len(horizons), figsize=(15.2, 6.1), squeeze=False)
    for axis, horizon in zip(axes[0], horizons):
        frame = points[points.horizon == horizon]
        efficient = frame[frame.pareto_efficient].sort_values("volatility")
        x = efficient.standard_deviation_percent.to_numpy()
        y = efficient.mean_return_percent.to_numpy()
        x_all = frame.standard_deviation_percent.to_numpy()
        y_all = frame.mean_return_percent.to_numpy()
        dx = max(np.ptp(x_all), 0.01)
        dy = max(np.ptp(y_all), 0.001)
        x_low, x_high = x_all.min() - 0.15 * dx, x_all.max() + 0.18 * dx
        y_low, y_high = y_all.min() - 0.27 * dy, y_all.max() + 0.40 * dy
        axis.step(
            np.r_[x, x_high],
            np.r_[y, y[-1]],
            where="post",
            color="#63817c",
            linewidth=1.6,
            linestyle="--",
            zorder=1,
        )
        for row in frame.itertuples(index=False):
            efficient_point = row.pareto_efficient
            marker = "s" if row.strategy == "equal" else ("o" if efficient_point else "X")
            axis.scatter(
                row.standard_deviation_percent,
                row.mean_return_percent,
                marker=marker,
                s=85 if efficient_point else 110,
                color=COLORS[row.strategy],
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
            label = LABELS[row.strategy]
            if not efficient_point:
                label += "\n(dominated)"
            offsets = {
                "minimum_variance": (7, 12),
                "balanced": (7, -24),
                "inverse_volatility": (-7, 14),
                "equal": (-7, 13),
            }
            offset = offsets[row.strategy]
            alignment = "right" if offset[0] < 0 else "left"
            axis.annotate(
                label,
                (row.standard_deviation_percent, row.mean_return_percent),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                fontsize=9,
                color=COLORS[row.strategy],
            )
        n = int(frame.n_episodes.iloc[0])
        axis.set(
            title=f"{titles.get(horizon, str(horizon) + ' sessions')}\n{n:,} complete episodes",
            xlabel="Holding-period standard deviation (%)",
            ylabel="Mean holding-period return (%)",
            xlim=(x_low, x_high),
            ylim=(y_low, y_high),
        )
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if horizon == 1 else "%.2f"))
    fig.suptitle("Historical Pareto frontier among four tested allocations", fontsize=17, y=0.97)
    fig.text(
        0.5,
        0.885,
        f"{period_label}  |  Same five stock selections and execution dates  |  "
        "Zero-cost nominal price returns",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="#007f73",
            markersize=8,
            label="Efficient tested allocation",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="none",
            markerfacecolor=COLORS["equal"],
            markersize=8,
            label="Current equal-weight allocation",
        ),
        Line2D(
            [],
            [],
            marker="X",
            color="none",
            markerfacecolor=COLORS["balanced"],
            markersize=9,
            label="Dominated tested allocation",
        ),
        Line2D(
            [],
            [],
            color="#63817c",
            linestyle="--",
            label="Best tested return within a volatility limit",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.045,
        "Lower volatility is left; higher return is up. Only markers are tested portfolios; "
        "the steps are a volatility-budget envelope.\n"
        "Historical point estimates with sampling uncertainty. Separate horizon scales. "
        "Dividends excluded. No new strategy is fitted or selected.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.285, wspace=0.33)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"pareto_frontier.{extension}", dpi=200)
    plt.close(fig)


def main():
    default = Path(__file__).resolve().parent / "risk_2026_10_31"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default / "episode_metrics.csv")
    parser.add_argument("--period", choices=["all", "inspected_2023_onward"], default="all")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    points = classify_points(frame[frame.period == args.period].copy())
    validate_frontier(points, args.input.parent / "dominance.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    columns = [
        "period",
        "horizon",
        "strategy",
        "n_episodes",
        "mean_return",
        "volatility",
        "standard_deviation_percent",
        "mean_return_percent",
        "pareto_efficient",
        "dominated_by",
    ]
    points[columns].to_csv(args.output / "pareto_points.csv", index=False)
    render(points, args.output, args.period)
    manifest = {
        "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "period": args.period,
        "units": "mean return and sample standard deviation per complete holding period",
        "risk_direction": "minimize standard deviation",
        "return_direction": "maximize mean",
        "candidate_scope": "four previously tested stock-allocation rules",
        "frontier_counts": {
            str(h): int(group.pareto_efficient.sum()) for h, group in points.groupby("horizon")
        },
        "validation": "pairwise dominance equals independent sorted scan and archived classification",
        "steps": "maximum tested mean return among policies meeting each volatility limit",
        "limits": "historical point estimates; no fitted frontier across all feasible stock portfolios",
    }
    (args.output / "pareto_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(points[columns].to_string(index=False))
    print(f"Saved PNG, PDF, SVG, point data, and provenance to {args.output}")


if __name__ == "__main__":
    main()
