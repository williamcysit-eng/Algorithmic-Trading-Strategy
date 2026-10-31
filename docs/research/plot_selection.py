"""Draw the fixed stock-selection study's historical return–risk comparison."""

from pathlib import Path

import pandas as pd


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    folder = Path(__file__).resolve().parent / "selection_2026_10_31"
    metrics = pd.read_csv(folder / "episode_metrics.csv")
    metrics = metrics[metrics.period.eq("all")]
    methods = {
        "incumbent": ("Original top five", "#243746", "s"),
        "sector_cap": ("Sector cap", "#007f73", "o"),
        "variance_guarded": ("Variance with score budget", "#7056a1", "o"),
        "joint": ("Joint rule (primary)", "#c47a19", "D"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    for axis, horizon in zip(axes, [1, 5, 19]):
        frame = metrics[metrics.horizon.eq(horizon)]
        for row in frame.itertuples(index=False):
            label, color, marker = methods[row.strategy]
            axis.scatter(
                100 * row.volatility,
                100 * row.mean_return,
                label=label,
                color=color,
                marker=marker,
                s=75,
            )
            offset = (-7, 10) if row.strategy in ("incumbent", "sector_cap") else (7, -20)
            if horizon == 19:
                offset = (-7, 10) if row.strategy == "variance_guarded" else (7, -20)
            axis.annotate(
                label,
                (100 * row.volatility, 100 * row.mean_return),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                ha="right" if offset[0] < 0 else "left",
                color=color,
            )
        axis.set(
            title=f"{horizon}-session holdings | {int(frame.n_episodes.iloc[0]):,} episodes",
            xlabel="Holding-period standard deviation (%)",
            ylabel="Mean holding-period return (%)",
        )
        axis.margins(x=0.28, y=0.35)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if horizon == 1 else "%.2f"))
    fig.suptitle(
        "Joint stock selection: diversification reduced risk but evidence remains unresolved",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.035,
        "2018–October 2026; equal stock weights, zero costs, nominal price returns. "
        "Historical point estimates on inspected data.\n"
        "No primary horizon qualified; alternative point gains do not justify switching policies after viewing results.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.23, wspace=0.38)
    fig.savefig(folder / "selection_tradeoffs.png", dpi=180)
    fig.savefig(folder / "selection_tradeoffs.pdf")
    plt.close(fig)
    print(f"Saved selection plots to {folder}")


if __name__ == "__main__":
    main()
