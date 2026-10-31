# Nested forecasting experiment: rejected

The new forecasting/allocation procedure did not satisfy the adoption rule committed before its outcomes were run. The experimental integration is rejected and reverted; the legacy ranking and allocation procedure remains active. This is a technical experiment record, not the assessed course report.

## Design and provenance

The protocol is [`implementation_protocol.md`](implementation_protocol.md), committed as `d4590d7` before the new model outcomes. The tested implementation is recoverable at `f9effe3c46131484de9b3136d5e206fa699086ba`. It connects return forecasts, purged training, inner selection, allocations, shared portfolio accounting, and episode inference. Rejection is not a finding that those software components are numerically invalid.

The full run used 30 stocks and the existing snapshot through 2026-10-21: 126,780 price rows and 1,824 earnings events. The outer interval is 2018-01-02 through 2026-10-21, with nine annual folds and separate 1-, 5-, and 19-session targets. Zero fees and nominal price returns match grading; dividends are excluded. The universe is survivor-selected and the historical observations were already inspected, so this is retrospective exploratory research rather than an untouched confirmatory holdout.

The candidate grid stayed unchanged: Ridge alphas 1/100, histogram boosting depths 2/3 with 60 iterations, fixed momentum/equal-signal/rank controls, and regularization penalties 0.01/0.1 with 5%–40% stock bounds. Three earlier 126-session inner blocks selected a predictor's hyperparameter under equal weights, then its allocation penalty, then the policy. All outcomes used for fitting or selection matured strictly before the relevant boundary. There were no new predictors, additional architecture searches, or outcome-driven reruns of the grid.

All nine outer series, including the matched incumbent, restart their rebalance phase together each year. This isolates policy decisions from calendar differences. The incumbent retains its original 756-session training window and monthly vintage schedule. The research models use the frozen 1,260-session window. Incomplete terminal intervals remain in daily accounting but are excluded from primary episode statistics. These series therefore differ from the old uninterrupted 2018–2025 path; compare the matched rows below rather than treating changes in the old headline CAGR as model gains.

## Primary result

Percentages in the mean columns are nominal returns per complete holding episode, not annual returns. The loss quantile is an observed fifth percentile, not a guaranteed future bound.

| Horizon | Episodes | Incumbent mean | Nested mean | Difference, percentage points | 95% paired interval for difference, pp | Holm p-value for improvement |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2,204 | 0.0711% | 0.0491% | -0.0220 | [-0.0588, 0.0158] | 0.9034 |
| 5 | 438 | 0.3043% | 0.3711% | +0.0668 | [-0.1964, 0.3234] | 0.8878 |
| 19 | 114 | 0.7788% | 0.8400% | +0.0612 | [-0.8405, 0.9871] | 0.9034 |

The intervals are pointwise paired percentile intervals. The one-sided p-values use centered null resamples and Holm correction across the three adoption comparisons. Resampling uses 5,000 circular-block draws within outer years; the requested 20-session block rounds up to 20, 4, and 2 whole episodes at the respective horizons. The monthly primary block therefore spans 38 sessions. Requested 10- and 40-session sensitivities are reported separately and do not alter the adoption threshold.

| Horizon | Incumbent losing episodes | Nested losing episodes | Incumbent fifth percentile | Nested fifth percentile |
|---|---:|---:|---:|---:|
| 1 | 44.60% | 45.24% | -2.23% | -2.51% |
| 5 | 42.69% | 40.87% | -4.47% | -6.90% |
| 19 | 43.86% | 42.11% | -7.62% | -10.45% |

No horizon passes the corrected return test. Every horizon has a worse observed lower tail, which also fails the frozen safeguard. Weekly/monthly loss frequency improves slightly, but that does not offset the larger adverse outcomes under the declared decision rule.

The matched zero-cost path CAGRs are 16.59% versus 9.21% at one session, 13.97% versus 14.74% at five, and 8.49% versus 8.21% at nineteen, incumbent versus nested. A higher arithmetic mean can coexist with lower compound growth when variability rises. Zero-cost maximum drawdown worsens from -35.98% to -44.09%, -30.89% to -51.49%, and -33.49% to -40.77%. Cost sensitivities are archived, but zero costs remain primary.

Some ablations have better mean-return point estimates. In particular, Ridge/equal has higher mean returns than the incumbent at all three horizons. This does not permit selecting it after observing outer results. Its prespecified comparisons with the fixed equal-signal blend do not exclude zero; no primary six-comparison research family produces an adjusted p-value below 0.05. The nested procedure also underperforms the simpler momentum control in mean episodes at every horizon. These findings are retained rather than hidden behind a favorable variant or period.

## What the failed experiment teaches

Return prediction remains weak with this feature set. Date-weighted out-of-sample R-squared against a training-only constant mean is approximately 0.00027, 0.00171, and 0.00188 for Ridge at 1/5/19 sessions. Boosting's corresponding values are negative: -0.00089, -0.00489, and -0.00973. These tiny or adverse gains do not support the assumption that a nonlinear model automatically improves the signal.

Selecting among policies using about 18 months of inner episodes did not reliably identify the subsequently better policy. The monthly inner blocks contain only eighteen complete 19-session episodes in total, limiting selection precision. Forecast-driven allocations can amplify estimation error and correlation exposure. The implemented concentration penalty does not model covariance risk. These mechanisms are explanations to investigate, not grounds to change the grid and reuse the same outer results as confirmation.

A future study should freeze a new protocol before outcomes, evaluate simpler or more stable policy aggregation, and separate return-forecast error from covariance and event exposure. Better timestamped earnings information and historical universe coverage are more defensible additions than a large architecture search on this 30-stock panel. None of these future ideas has been established as an improvement here.

## Numerical and behavioral validation

The experimental revision passed all 33 tests and Ruff. Four new integration cases exercise nested choices, training maturity, future-data invariance, same-stock allocation ablations, fees, episode reconciliation, configuration dispatch, paired bootstrap behavior, and the rejection rule. The original offline sample's ten numeric artifacts remained unchanged. The research sample ran separately and supplied software evidence only.

Rollback commit `72f6806` removed the experimental strategy, inference, and CLI integration. The restored active code passed all 29 legacy tests and Ruff. Full and sample backtests reproduced all twenty corresponding numeric artifacts. Active source matches the starting revision except for two import-formatting corrections; configurations and the lockfile are unchanged. These checks are recorded in `nested_2026_10_31/rollback_verification.json`.

An independent audit reconstructed 24,804 complete episode returns from nominal closes and splits, then replayed 119,502 daily portfolio observations with explicit shares and an independent SciPy root solver for fee funding. Maximum episode absolute difference was 7.08e-16; maximum relative NAV difference was 8.22e-14. It checked 3,166 fitted-model boundaries, all 27 annual choices, and 214 saved incumbent vintages. It independently reproduced all 54 bootstrap comparisons and three adoption checks.

Compact evidence is versioned under [`nested_2026_10_31/`](nested_2026_10_31/):

- `episode_metrics.csv`, `fold_episode_metrics.csv`: every policy, including failed ablations and the inspected 2023-onward subset.
- `adoption.csv`, `comparisons.csv`: raw and corrected tests, sensitivity blocks, and explicit rejection flags.
- `forecast_metrics.csv`, `forecast_summary.csv`: forecast error against a training-only mean and rank association.
- `selection_scores.csv`, `selections.json`: earlier objectives and frozen annual decisions.
- `path_metrics.csv`: zero-cost and 10-bps path statistics, labeled separately.
- `provenance.json`, `independent_audit.json`, `rollback_verification.json`: exact source/config/data hashes, numerical audit counts, and confirmation that the original strategy was restored.

The larger daily returns, holdings, forecasts, fitted metadata, and downloaded datasets remain ignored under `results/nested_full/` and `data/current/`. The independent audit script is [`audit_nested.py`](audit_nested.py); run it from the repository root against the full local output directory:

```sh
uv run --frozen python docs/research/audit_nested.py results/nested_full
```

To reproduce the experimental model run, export revision `f9effe3` to a separate directory, provide the normalized `data/current/{prices,earnings,universe}.csv` and `manifest.json` matching `provenance.json`, and run:

```sh
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest -q
uv run --frozen stock-ensemble --config samples/research.json research
uv run --frozen stock-ensemble --config config.research.json research --output results/nested_full
```

The active revision intentionally omits the rejected research command. `config.research.json` and `samples/research.json` remain protocol records rather than executable research studies on the active code. Legacy commands continue to use only their legacy settings.
