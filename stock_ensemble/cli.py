"""Download, evaluate, rank, and allocate the three assignment portfolios."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_dataset, load_dataset
from .features import build_features
from .reporting import save_json, write_results


def _config(path: str) -> dict:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text())
    for key in ("universe_path", "data_dir", "output_dir"):
        config[key] = str((config_path.parent / config[key]).resolve())
    if not 1 <= config["backtest"]["top_n"] <= 5:
        raise ValueError("Assignment portfolios must contain between one and five companies")
    if config["backtest"]["initial_capital"] <= 0:
        raise ValueError("Initial capital must be positive")
    if config["features"]["min_cross_section"] < config["backtest"]["top_n"]:
        raise ValueError("min_cross_section must be at least top_n")
    return config


def _rank(prices, features, universe, config, as_of, output):
    from .backtest import rank_as_of

    rankings, fits = [], []
    for horizon in config["backtest"]["horizons"]:
        ranking, fit = rank_as_of(prices, features, config, as_of, horizon)
        ranking = ranking.merge(universe[["ticker", "name", "sector"]], on="ticker", how="left")
        ranking["horizon"] = horizon
        ranking["signal_date"] = as_of
        rankings.append(ranking)
        fits.append({"horizon": horizon, "fit": fit})
        print(f"\n{horizon}-session horizon; information through {pd.Timestamp(as_of).date()}")
        print(
            ranking.iloc[: config["backtest"]["top_n"]][
                ["rank", "ticker", "score", "earnings", "momentum", "reversal"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
    combined = pd.concat(rankings, ignore_index=True)
    combined.to_csv(output / "rankings.csv", index=False)
    save_json(output / "ranking_models.json", fits)
    return combined


def _allocate(prices, features, universe, config, args, output):
    entry = pd.Timestamp(args.entry_date).normalize()
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    if entry not in dates or dates.get_loc(entry) == 0:
        raise ValueError(
            "Entry date must have a recorded market close and a preceding signal session"
        )
    entry_position = dates.get_loc(entry)
    signal_date = dates[entry_position - 1]
    ranked = _rank(prices, features, universe, config, signal_date, output)
    capital = args.capital if args.capital is not None else config["backtest"]["initial_capital"]
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Capital must be positive and finite")
    horizons = config["backtest"]["horizons"]
    if args.exit_dates and len(args.exit_dates) != len(horizons):
        raise ValueError(
            "Supply exactly one exit date for each configured horizon, in horizon order"
        )
    allocations, dividend_rows = [], []
    indexed = prices.set_index(["date", "ticker"])
    for index, horizon in enumerate(horizons):
        picks = ranked[ranked["horizon"] == horizon].iloc[: config["backtest"]["top_n"]]
        if picks.empty:
            raise ValueError(f"No eligible selections for horizon {horizon}")
        if args.exit_dates:
            exit_date = pd.Timestamp(args.exit_dates[index]).normalize()
            if exit_date <= entry:
                raise ValueError("Exit dates must be after entry date")
            if exit_date <= dates[-1] and exit_date not in dates:
                raise ValueError(
                    f"Exit date {exit_date.date()} has no market close; choose a session"
                )
        else:
            exit_position = entry_position + horizon
            exit_date = dates[exit_position] if exit_position < len(dates) else pd.NaT
        for pick in picks.itertuples():
            if (entry, pick.ticker) not in indexed.index:
                raise ValueError(
                    f"Missing execution close for selected {pick.ticker} on {entry.date()}"
                )
            close = float(indexed.loc[(entry, pick.ticker), "close"])
            dollars = capital / len(picks)
            shares = dollars / close
            row = {
                "horizon": horizon,
                "signal_date": signal_date,
                "entry_date": entry,
                "evaluation_date": exit_date,
                "ticker": pick.ticker,
                "name": pick.name,
                "score": pick.score,
                "target_weight": 1 / len(picks),
                "close": close,
                "dollars": dollars,
                "shares": shares,
                "evaluation_status": "pending",
                "price_return": np.nan,
                "dividend_cash": np.nan,
                "ending_value_excl_dividends": np.nan,
            }
            if pd.notna(exit_date) and exit_date <= dates[-1]:
                if (exit_date, pick.ticker) not in indexed.index:
                    raise ValueError(
                        f"Missing evaluation close for {pick.ticker} on {exit_date.date()}"
                    )
                period = prices[
                    (prices["ticker"] == pick.ticker)
                    & (prices["date"] > entry)
                    & (prices["date"] <= exit_date)
                ].sort_values("date")
                expected = dates[(dates > entry) & (dates <= exit_date)]
                if len(period) != len(expected):
                    raise ValueError(f"Missing held-session data for {pick.ticker}; cannot skip it")
                held_shares = shares * period["split"].cumprod()
                dividends = held_shares * period["dividend"]
                ending = (
                    shares
                    * period["split"].prod()
                    * float(indexed.loc[(exit_date, pick.ticker), "close"])
                )
                row.update(
                    evaluation_status="evaluated",
                    price_return=ending / dollars - 1,
                    dividend_cash=float(dividends.sum()),
                    ending_value_excl_dividends=ending,
                )
                for position in np.flatnonzero(period["dividend"].to_numpy() != 0):
                    event = period.iloc[position]
                    dividend_rows.append(
                        {
                            "horizon": horizon,
                            "ticker": pick.ticker,
                            "ex_date": event["date"],
                            "dividend_per_share": event["dividend"],
                            "entitled_shares": held_shares.iloc[position],
                            "dividend_cash": dividends.iloc[position],
                        }
                    )
            allocations.append(row)
    allocation = pd.DataFrame(allocations)
    allocation.to_csv(output / "allocations.csv", index=False)
    pd.DataFrame(
        dividend_rows,
        columns=[
            "horizon",
            "ticker",
            "ex_date",
            "dividend_per_share",
            "entitled_shares",
            "dividend_cash",
        ],
    ).to_csv(output / "dividend_notifications.csv", index=False)
    save_json(
        output / "allocation_assumptions.json",
        {
            "capital_per_horizon": capital,
            "cost_bps": 0,
            "fractional_shares": True,
            "price_basis": "Nominal unadjusted close; splits change shares; dividends excluded from grade return.",
            "signal_date": signal_date,
            "entry_date": entry,
            "no_lookahead": "Selection uses previous-close features, not entry-close returns or future outcomes.",
            "horizons": "Training uses 1/5/20 sessions; supplied exit dates override only evaluation dates.",
            "dividends": "Historical ex-dates only; future dividends are unknown, not assumed absent.",
        },
    )
    print("\nFully invested, zero-cost assignment allocations (fractional shares):")
    print(
        allocation[["horizon", "ticker", "dollars", "shares", "evaluation_status"]].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"Saved allocations and ex-dividend notifications to {output}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.json", help="JSON config; paths relative to this file"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch/cache real SEC filings and Yahoo prices")
    download.add_argument("--start", help="Inclusive price history start")
    download.add_argument("--end", help="Exclusive price history end, e.g. 2026-09-12")
    download.add_argument(
        "--sec-user-agent",
        default=os.getenv("SEC_USER_AGENT", "STAT7008 university investment research"),
        help="SEC identifying User-Agent; set your project name and contact email",
    )
    evaluate = commands.add_parser(
        "backtest", help="Run chronological comparison and write artifacts"
    )
    evaluate.add_argument("--cost-bps", type=float, help="One-way fee per traded dollar")
    evaluate.add_argument("--return-mode", choices=["price", "total"])
    evaluate.add_argument("--output", help="Override output directory")
    rank = commands.add_parser("rank", help="Rank stocks at a historical/latest signal close")
    rank.add_argument("--as-of", help="Exact signal session; defaults to latest downloaded close")
    rank.add_argument("--output", help="Override output directory")
    allocate = commands.add_parser("allocate", help="Produce the three $1m assignment portfolios")
    allocate.add_argument(
        "--entry-date", required=True, help="Exact close used to buy, after signal day"
    )
    allocate.add_argument(
        "--exit-dates", nargs="+", help="Daily, weekly, monthly exact evaluation dates"
    )
    allocate.add_argument("--capital", type=float, help="Budget per horizon; default $1m")
    allocate.add_argument("--output", help="Override output directory")
    args = parser.parse_args(argv)
    try:
        config = _config(args.config)
        if args.command == "download":
            manifest = download_dataset(
                config["universe_path"],
                args.start or config["data_start"],
                args.end or config["data_end"],
                config["data_dir"],
                args.sec_user_agent,
            )
            print(f"Historical data saved to {config['data_dir']}")
            print(json.dumps(manifest, default=str, indent=2))
            return 0
        if args.output:
            config["output_dir"] = str(Path(args.output).resolve())
        if args.command == "backtest":
            if args.cost_bps is not None:
                config["backtest"]["cost_bps"] = args.cost_bps
            if args.return_mode:
                config["backtest"]["return_mode"] = args.return_mode
        prices, earnings, universe, metadata = load_dataset(config["data_dir"])
        features = build_features(prices, earnings, config)
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if args.command == "backtest":
            from .backtest import run_backtests

            result = run_backtests(prices, features, copy.deepcopy(config))
            write_results(result, features, metadata, config)
        elif args.command == "rank":
            as_of = pd.Timestamp(args.as_of) if args.as_of else prices["date"].max()
            _rank(prices, features, universe, config, as_of, output)
            print("Ranks are relative scores, not predicted returns. Execution is the next close.")
        else:
            _allocate(prices, features, universe, config, args, output)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
