# Earnings–momentum stock-ranking ensemble

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

For new observations, use `config.current.json` to keep the refreshed snapshot separate from the recorded experiment. Set your own SEC identification string and extend the exclusive end date when needed. Both price and filing caches refresh when their snapshots cannot cover the new endpoint:

```sh
export SEC_USER_AGENT='Your project name your-contact-email'
uv run stock-ensemble --config config.current.json download --end 2026-10-22
uv run stock-ensemble --config config.current.json backtest
uv run stock-ensemble --config config.current.json rank
```

Only completed sessions before `--end` are accepted. An unchanged date range reuses cached source snapshots for reproducibility. Keep a copy of the data and results before extending an experiment. Raw snapshot and normalized CSV checksums are recorded in the selected dataset's `manifest.json`.

The refreshed snapshot contains 126,780 daily price observations and 1,824 earnings events across 30 stocks, through **2026-10-21**. Its data and outputs are in `data/current/` and `results/current/`; `config.json`, `data/`'s original normalized files and the recorded historical tables remain unchanged. On later refreshes, set `--end` to that day's **New York date** to include only prior completed sessions. Treat refreshed rankings and extended backtests as a separate experiment, not replacements for the recorded 2018–2025 evaluation.

## Recorded evaluation

The verified full-universe run uses 120,720 daily price observations and 1,732 earnings events across 30 stocks. The table records the ensemble's price-return CAGR over **2023-01-03 through 2025-12-31**, not the smaller offline sample.

| Holding period | Zero transaction costs | 10 bps per dollar traded |
|---|---:|---:|
| 1 session | 18.19% | -8.66% |
| 5 sessions | 27.37% | 20.64% |
| 20 sessions | 21.60% | 19.79% |

These are survivor-selected pilot results, not established alpha. Momentum alone had higher net CAGR at all three horizons in this price-return run. The paired bootstrap intervals for the earnings gate's incremental benefit included zero. Reproduce the tables with the full configuration; changed data snapshots or settings require updating this recorded result.

### Code and result validation

The review reproduced all ten numeric artifacts from both the full experiment and the offline sample. Independent calculations from the daily returns matched the saved CAGR, volatility, Sharpe, drawdown and per-dollar trading costs. The historical allocation example was also reconciled directly to nominal entry/exit closes, split-adjusted share counts and dividend cash. Recorded data hashes, strict label purging and fitted-weight constraints were checked; these checks validate the implementation on the supplied snapshots, not the accuracy of every vendor observation or future investment performance.

`summarize_returns` now orders each portfolio's observations by session before computing period statistics. Previously, shuffled input could understate maximum drawdown even though the ordinary backtest already produced chronological rows. The regression covers both subperiod and full-path drawdowns; the recorded results are unchanged.

Allocation audit metadata also reports the configured training horizons rather than hard-coding `1/5/20`, so a 19-session monthly model is not mislabeled as a 20-session fit.

Compatible dependencies have been refreshed in `uv.lock`, retaining the NumPy `<2.4` and pandas `<3` bounds. Reproduce the environment with `uv sync --frozen --extra dev`. To check for newer compatible releases deliberately, use `uv lock --upgrade`, sync again, and run the checks below before accepting the new lockfile. Do not remove compatibility bounds or change the experiment's settings merely to obtain newer version numbers.

```sh
uv run --frozen --extra dev ruff check stock_ensemble tests
uv run --frozen --extra dev pytest -q
uv run --frozen stock-ensemble backtest --output results/review
uv run --frozen stock-ensemble --config samples/config.json backtest --output results/review_sample
```

## Empirical horizon diagnostics and fitness for purpose

The full backtest was rerun on the recorded data snapshot, together with two behavioral checks targeted at the main threats to a multi-horizon study: `test_future_prices_and_filings_cannot_change_past_rankings` and `test_execution_lag_full_investment_and_cost_accounting`. Both checks passed. They establish that later prices and filings cannot revise an earlier ranking, that labels mature before model fitting, and that next-close execution, full investment and trading costs are accounted for consistently. These are implementation-validity checks; they do not establish predictive validity.

### Assignment-consistent quantitative results

The primary estimand is the close-to-close nominal price return of an equal-dollar, long-only, top-five portfolio. The table uses the untouched 2023–2025 chronological holdout and zero costs, as required by the assignment. “Episode” means one complete, non-overlapping entry-to-exit holding period; the final incomplete interval is excluded from episode statistics but remains marked and liquidated in the compounded path statistics. CAGR, volatility and Sharpe annualize daily observations using 252 sessions, and Sharpe uses a zero risk-free rate.

| Horizon | Complete episodes | Mean episode return | Positive episodes | Holdout total return | CAGR | Sharpe | Maximum drawdown | CAGR at 10 bps/side |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 session | 751 | 0.076% | 53.9% | 64.66% | 18.19% | 0.99 | -25.58% | -8.66% |
| 5 sessions | 150 | 0.504% | 61.3% | 105.83% | 27.37% | 1.37 | -28.41% | 20.64% |
| 20 sessions | 37 | 1.679% | 62.2% | 79.23% | 21.60% | 1.12 | -24.21% | 19.79% |

The episode mean is the quantity closest to the one-off course exercise; CAGR instead describes repeatedly reinvested, non-overlapping historical portfolios. Neither is a forecast for the single 2026 realization. The much smaller number of complete monthly episodes also makes the 20-session estimate materially less precise than the daily estimate.

Descriptively, the learned ensemble was the best tested zero-cost strategy only at five sessions. The fixed equal-signal blend produced the highest zero-cost holdout CAGR at one session (29.04%) and 20 sessions (25.53%), while the learned ensemble led at five sessions (27.37%). Selecting among these variants after observing the holdout would itself be data snooping, so these rankings are diagnostic rather than a rule for choosing the live model.

### Paired uncertainty test

To avoid treating serially dependent daily marks as independent observations, the same fixed-seed circular block bootstrap used by the reporting pipeline was applied to **gross** holdout returns: 1,000 paired resamples, 20-session blocks, and the ensemble minus comparator return on each common date. The estimand below is the annualized arithmetic mean difference, not a CAGR difference.

| Horizon | Comparator | Ensemble difference | Paired 95% bootstrap interval |
|---|---|---:|---:|
| 1 session | Eligible-universe benchmark | +1.96 pp/year | [-11.01, 15.02] |
| 1 session | Equal-signal blend | -8.90 pp/year | [-15.52, -2.24] |
| 5 sessions | Eligible-universe benchmark | +9.50 pp/year | [-5.03, 23.72] |
| 5 sessions | Equal-signal blend | +0.53 pp/year | [-9.51, 8.84] |
| 20 sessions | Eligible-universe benchmark | +4.97 pp/year | [-8.05, 18.62] |
| 20 sessions | Equal-signal blend | -3.26 pp/year | [-11.08, 4.10] |

Only the one-session comparison with the equal-signal blend excludes zero, and it favors the simpler blend. None of the ensemble-versus-universe intervals excludes zero. The bad-news gate is likewise unresolved: its annualized gross-return differences relative to the ungated fit were +0.17, +0.32 and +0.03 percentage points for 1, 5 and 20 sessions, respectively, and all three paired intervals included zero. These intervals are not adjusted for the several horizons and comparators examined. The standard `paired_holdout_intervals.csv` applies the same procedure to the configured 10-bps return series; this zero-cost diagnostic instead substitutes `gross_return`, matching the assignment.

### Cost, horizon and calendar stress tests

Turnover explains the daily strategy's apparent contradiction between a positive assignment result and a negative cost-aware result. Mean daily traded notional was 102.31%, 21.52% and 5.91% of prior NAV for the 1-, 5- and 20-session ensembles. Applying 10 basis points to every dollar bought or sold reduced holdout CAGR by 26.85, 6.73 and 1.80 percentage points, respectively. Zero cost is therefore the correct contractual assumption for grading, but the one-session result is not economically robust to even the stated illustrative fee.

The exact 2026 calendar also matters. From the November 6 close through the December 4 close there are 19 post-entry NYSE sessions, because Thanksgiving Day, November 26, is closed; the November 27 early close remains a session under the [NYSE calendar](https://www.nyse.com/trade/hours-calendars). Thus 20 sessions is a conventional trading-month proxy rather than the exact assignment span. Whole-backtest sensitivity runs at 19 and 21 sessions produced:

| Horizon convention | Ensemble zero-cost CAGR | Equal-signal CAGR | Universe CAGR | Ensemble-minus-universe 95% interval |
|---|---:|---:|---:|---:|
| 19 sessions | 26.27% | 31.06% | 16.97% | [-3.22, 21.43] pp/year |
| 20 sessions | 21.60% | 25.53% | 16.91% | [-8.05, 18.62] pp/year |
| 21 sessions | 18.68% | 19.45% | 17.10% | [-9.62, 14.85] pp/year |

Changing the horizon changes both the forward-return label and the non-overlapping rebalance calendar, so this is an operational sensitivity test rather than a pure one-parameter causal comparison. The sign of the ensemble's point excess return is stable, but no interval excludes zero and the magnitude is schedule-sensitive. The exact-date allocation command correctly evaluates the supplied December 4 exit; its ranking model still uses the configured training horizon. The 19-session exact convention or the 20-session conventional proxy should therefore be specified **before** producing the live ranking, not selected after comparing outcomes.

Horizon differentiation is also weak in the current fit. Across holdout refits, mean ensemble weights were approximately 56% earnings, 17–18% momentum and 26% reversal at every horizon. On the latest downloaded signal session, all three models consequently selected the same top five stocks. This is permitted by the assignment, but the portfolios should not be described as three economically independent bets merely because three models were fitted.

As a one-event operational check, the historical November 6, 2025 allocation in `results/example_2025/` returned -0.534% after one session, -0.047% after five sessions and +1.365% through December 4, leaving $994,658, $999,535 and $1,013,648 before dividend cash. That example confirms the exact-date calculation path; one event supplies no reliable estimate of expected performance.

**Assessment.** The repository is fit for the course goal of constructing, executing and auditing separate daily, weekly and monthly zero-cost portfolios: horizons are explicit, chronology is purged, nominal closes and exact exits are supported, and the investment constraints are enforced. It is not evidence that the learned ensemble reliably maximizes return. The weekly and monthly point estimates are encouraging but statistically unresolved; the daily learned fit is inferior to the equal blend and exceptionally cost-sensitive; the gate has no detected incremental benefit; and the 30-stock survivor universe limits external validity. The defensible characterization is therefore a horizon-aligned exploratory decision system, not a validated alpha model.

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
uv run stock-ensemble --config config.current.json download --end 2026-11-07
uv run stock-ensemble --config config.current.json allocate --entry-date 2026-11-06 \
  --exit-dates 2026-11-09 2026-11-13 2026-12-04 \
  --output results/assignment_2026
```

Future market data are not fabricated. Pending evaluations remain pending. Rerun after the exit dates to evaluate realized returns and produce `dividend_notifications.csv` for ex-dividend events. Archive the original rankings, allocations and raw data; later vendor revisions can affect a rerun. Exact exit dates override evaluation dates, while model training horizons remain 1/5/20 sessions.

## Files and outputs

| File | Purpose |
|---|---|
| `config.json`, `config.current.json`, `universe.csv` | Separate historical/current settings and explicit stock universe |
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
- Comments explain information availability, return/share units, constraint equations and execution order. Avoid line-by-line narration; update annotations alongside the calculations they explain.

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
