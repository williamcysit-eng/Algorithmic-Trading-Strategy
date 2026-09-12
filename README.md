# Earnings–momentum stock-ranking ensemble

Python implementation for the STAT7008 investment exercises. This is technical usage documentation, **not the assessed written report**. The assignment permits AI-written code but requires students to write the report themselves.

## Run

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required for these commands. `uv.lock` pins the environment. NumPy is constrained below 2.4 for compatibility with the pandas 2.x date operations used here.

```sh
uv sync --frozen --extra dev

# Complete offline demonstration using the included small, real-data sample:
uv run stock-ensemble --config samples/config.json backtest

# Full 30-stock experiment; download is unnecessary if data/ is already present:
uv run stock-ensemble download
uv run stock-ensemble backtest

# Latest DOWNLOADED signal session, not necessarily today's market:
uv run stock-ensemble rank --output results/latest

# Regression checks:
uv run --extra dev pytest -q
```

Default full history: **2010-01-04 through 2025-12-31**. Evaluation: 2018–2025; reserved chronological evaluation: 2023–2025. The offline sample uses ten alphabetically selected tickers, 2022–2025 prices, and 2025 evaluation. Its results are a different experiment, not substitutes for the full results.

`--config` precedes the subcommand. Config paths are relative to the JSON file. CLI `--output` overrides are relative to the current directory.

For new observations, set your own SEC identification string and extend the exclusive end date. Both price and filing caches refresh when their snapshots cannot cover the new endpoint:

```sh
export SEC_USER_AGENT='Your project name your-contact-email'
uv run stock-ensemble download --end 2026-09-12
uv run stock-ensemble rank --output results/latest
```

Only completed sessions before `--end` are accepted. An unchanged date range reuses cached source snapshots for reproducibility. Keep a copy of the data and results before extending an experiment. Raw snapshot and normalized CSV checksums are recorded in `data/manifest.json`.

## Recorded evaluation

The verified full-universe run uses 120,720 daily price observations and 1,732 earnings events across 30 stocks. The table records the ensemble's price-return CAGR over **2023-01-03 through 2025-12-31**, not the smaller offline sample.

| Holding period | Zero transaction costs | 10 bps per dollar traded |
|---|---:|---:|
| 1 session | 18.19% | -8.66% |
| 5 sessions | 27.37% | 20.64% |
| 20 sessions | 21.60% | 19.79% |

These are survivor-selected pilot results, not established alpha. Momentum alone had higher net CAGR at all three horizons in this price-return run. The paired bootstrap intervals for the earnings gate's incremental benefit included zero. Reproduce the tables with the full configuration; changed data snapshots or settings require updating this recorded result.

## Three experts and the learned score

- **Earnings:** seasonal change in quarterly GAAP profit divided by quarter-end assets, standardized by the sample standard deviation of up to eight previously observable changes (at least four). Net income and earnings available to common shareholders are processed separately; definitions are never mixed within a difference or scale. The earliest usable disclosure wins. `source_concept` and `source_accn` identify the inputs. Quarter-four profit can be annual minus year-to-date profit; 52/53-week retail calendars are supported. This is **not analyst-consensus surprise**.
- **Long-term momentum:** adjusted close 21 sessions ago divided by adjusted close 252 sessions ago, minus one. Skipping the latest month separates continuation from short-run reversal.
- **Short-term reversal:** negative one-session stock return relative to the contemporaneous equal-weight stock-universe return, divided by volatility estimated from the preceding 20 sessions, excluding the current shock.

Scores are cross-sectional centered ranks. Earnings rank decays with a 45-calendar-day half-life and expires after 180 days; unavailable earnings are neutral. Eligibility requires a complete 252-session lookback, a nominal price of at least $5, and mean 20-session dollar volume of at least $10 million. No future return or future survival test controls eligibility.

**Project-specific hypothesis:** positive rebound credit should be discounted after negative earnings news. The gate is `1 - max(-tanh(surprise), 0) * 2**(-earnings_age/10)`. It discounts only positive reversal scores, not recent winners' negative scores. This is an academically motivated synthesis, not a claim of new academic discovery or demonstrated superiority. `ungated` is a separately fitted ablation.

Each 1/5/20-session horizon learns three non-negative weights summing to one. The model minimizes date-balanced squared error against subsequent return ranks, with a fixed ridge penalty toward equal weights. It solves a three-variable constrained quadratic problem; **no neural network**. Scores are relative rankings, not predicted percentage returns or probabilities.

## Chronology and portfolio accounting

1. A filing is visible only on signal dates **strictly after its filing date**, never its fiscal-period end. Filing dates conservatively lag actual earnings announcements.
2. Signals observed at close `d` execute at close `d+1`. A horizon `h` label ends at `d+1+h`.
3. Models refit monthly on a rolling 756-session window. Only labels ending **strictly before** the refit date enter training; at least 252 training dates are required.
4. Hyperparameters are fixed, not selected on the holdout. This is **prequential evaluation**: later refits can use already realized earlier holdout outcomes. It is not a frozen-model holdout.
5. The top five stocks receive equal dollar allocations. Holdings drift between non-overlapping rebalances and are marked every session. Missing held prices cause an explicit error, not deletion of a losing stock.
6. Fees are funded from portfolio wealth without borrowing; purchases, drift-aware rebalances, and terminal liquidation are charged. The final partial holding interval is marked and liquidated rather than discarded.

Default evaluation uses split-aware **nominal price returns**, excluding dividends, to match the assignment. Yahoo split-adjusted closes are converted back to historical nominal closes using subsequent split events. Features use adjusted returns to avoid false split/dividend shocks. `--return-mode total` instead evaluates vendor-adjusted total returns; keep its output separate:

```sh
uv run stock-ensemble backtest --return-mode total --output results/total_return
```

The primary cost scenario is 10 basis points **per dollar bought or sold**, not per rebalance. `zero_cost_metrics.csv` removes fees from the same decisions and gross daily marks, matching the assignment's zero-cost rule. Costs are a sensitivity assumption, not a calibrated market-impact model.

## Assignment allocations

The three portfolios each invest $1 million, long-only, in at most five companies. Fractional shares are assumed so the entire budget is invested; no rounding residual or leverage is hidden. Allocation always uses the preceding session's signals and the requested entry date's nominal close.

A verified historical example:

```sh
uv run stock-ensemble allocate --entry-date 2025-11-06 \
  --exit-dates 2025-11-07 2025-11-13 2025-12-04 \
  --output results/example_2025
```

Once the **2026-11-06 close is actually available**, refresh data and use the assignment's exact dates:

```sh
uv run stock-ensemble download --end 2026-11-07
uv run stock-ensemble allocate --entry-date 2026-11-06 \
  --exit-dates 2026-11-09 2026-11-13 2026-12-04 \
  --output results/assignment_2026
```

Future market data are not fabricated. Pending evaluations remain pending. Rerun after the exit dates to evaluate realized returns and produce `dividend_notifications.csv` for ex-dividend events. Archive the original rankings, allocations and raw data; later vendor revisions can affect a rerun. Exact exit dates override evaluation dates, while model training horizons remain 1/5/20 sessions.

## Files and outputs

| File | Purpose |
|---|---|
| `config.json`, `universe.csv` | Fixed settings and explicit stock universe |
| `stock_ensemble/data.py` | Yahoo/SEC ingestion, filing vintages, corporate actions |
| `stock_ensemble/features.py` | Three experts and earnings-conditioned reversal |
| `stock_ensemble/model.py` | Constrained, regularized rank ensemble |
| `stock_ensemble/backtest.py` | Purged monthly fits, portfolios, daily accounting |
| `stock_ensemble/cli.py` | Download, backtest, rank and allocate commands |
| `stock_ensemble/reporting.py` | Numeric artifacts, charts, uncertainty and citations |
| `samples/` | Small, offline-runnable **real** data, config and provenance |

The full run writes:

- `results/metrics.csv`, `zero_cost_metrics.csv`, `annual_returns.csv`: all-period, development and holdout performance; 252-session CAGR/volatility; zero-risk-free Sharpe; daily maximum drawdown.
- `returns.csv`, `holdings.csv.gz`, `weights.csv`, `scores.csv.gz`: daily dollar NAV, costs, selections, ranks and auditable training/label-end dates. Cost and traded-notional fractions use previous-session NAV. Zero-cost metrics intentionally omit turnover rather than mislabel fee-dependent dollar turnover.
- `features.csv.gz`, `coverage.csv`: signal inputs, gates, eligibility and active earnings coverage.
- `paired_holdout_intervals.csv`: paired circular 20-session block-bootstrap intervals for annualized **arithmetic mean return differences**, not CAGR differences; fixed seed, 1,000 draws.
- `equity_curves.png`, `run_manifest.json`: comparison chart, exact configuration, source hashes, data provenance, references and limitations.

Comparators: learned gated ensemble, equal-weight signal blend, learned ungated ensemble, each single-signal top-five portfolio, and an equal-weight eligible-universe benchmark. **Only the benchmark** exceeds five stocks; it is not an assignment portfolio or an ETF.

## Repository hygiene

- Commit source, tests, `README.md`, `.gitignore`, configuration, `uv.lock`, and the small real-data files in `samples/`. The sample is sufficient for the offline demonstration from a fresh checkout.
- Full downloads in `/data/`, generated artifacts in `/results/`, the local course brief in `/Information/`, virtual environments, caches, and build products stay out of Git. Full-run output links are available after running the experiment, not as checked-in results.
- Keep local credentials in environment variables or ignored `.env` files. Only a sanitized `.env.example` may be committed; the CLI reads the process environment and does not automatically load `.env`.
- When commands, data conventions, dependencies, or outputs change, update this README and the corresponding ignore rules in the same change. Keep the sample configuration aligned with the supported data schema.

Before committing code changes:

```sh
uv run --extra dev ruff check stock_ensemble tests
uv run --extra dev pytest -q
uv run stock-ensemble --config samples/config.json backtest
```

## Interpretation limits and academic sources

The fixed list consists of present-day survivors and lacks delisted stocks/historical membership. Results are a **survivor-biased pilot**, not unbiased US-market alpha. Free vendor histories can be revised. Splits are handled, but spin-offs, rights, mergers and special distributions do not have a complete event ledger here. The one-day reversal may dissipate before next-close execution. Neither positive returns nor bootstrap intervals establish future profitability; intervals are not multiple-testing-adjusted. The gate's measured improvement is not assumed to be significant.

Academic motivation, not exact replications:

- [Bernard & Thomas (1989), earnings-announcement drift](https://doi.org/10.2307/2491062).
- [Jegadeesh & Titman (1993), intermediate-horizon momentum](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x), with [French's prior 2–12 month convention](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html).
- [Lehmann (1990), short-horizon reversal](https://www.nber.org/papers/w2533).
- [Chan, Jegadeesh & Lakonishok (1996), earnings and price momentum](https://www.nber.org/papers/w5375).
- [Da, Liu & Schaumburg (2014), fundamental news versus reversal](https://www.newyorkfed.org/research/staff_reports/sr513.html).
