# Prospective implementation rules for the retrospective comparison

This document is committed before executing the new model outcomes. It supplements the candidate grid frozen in `config.research.json`; it is a local protocol record, not independent preregistration. Prices through 2026-10-21 and legacy results have already been inspected. All historical evidence remains exploratory.

## Question and fixed alternatives

Does an annually selected, horizon-specific forecasting/allocation procedure improve complete nominal holding-period returns over the existing gated rank ensemble, with no greater observed loss frequency or lower-tail loss?

Use the existing features and data snapshots without adding predictors or searching architectures. Horizons are 1, 5, and 19 NYSE sessions to match the assignment. Fit Ridge (alpha 1 or 100), shallow histogram boosting (depth 2 or 3, 60 iterations), and the three existing controls. Use the previously frozen 1,260-session window, 504 matured training dates, date-balanced weights, and monthly model updates. The original ensemble comparator retains its 756-session/252-date training settings and original monthly vintage schedule.

Seven candidate policies are momentum, equal-signal, rank-ensemble, Ridge/equal, boosting/equal, Ridge/regularized, and boosting/regularized. The eighth research series is the annually selected policy. The incumbent ensemble is an additional audit comparator, never selected using outer outcomes. Ordinal controls do not receive fabricated percentage forecasts. Regularized allocations retain the frozen penalties 0.01/0.1 and bounds 5%/40%; the same stocks and forecasts underpin their equal-weight ablations.

## Time ordering and selection

At the previous close before each outer year's first entry, use three preceding blocks, each lasting 126 session-to-session intervals. Each block's endpoint precedes selection strictly. Adjacent blocks share only an endpoint. Only complete, non-overlapping h-session episodes enter inner objectives. Every inner predictor is fitted on earlier labels ending strictly before its refit; inner models may update monthly after outcomes mature.

Select each ML family's hyperparameter by mean pooled inner episode return with equal weights. For that same candidate, select its allocation penalty by the same objective. Then select among the seven policies. Ties follow configuration/policy order, with no hindsight tie-breaking. Freeze these choices for the outer calendar year. Model parameters may update monthly using only matured labels.

Each outer year restarts the shared rebalance phase at its first entry and ends at its last available close. Complete episodes enter primary statistics; any remaining interval is marked and liquidated for path and cost diagnostics. The incumbent trades the same dates, isolating model decisions from schedule changes. These yearly resets differ from the old uninterrupted path, so direct comparison of the new CAGR with a previously recorded CAGR is not the primary test.

## Inference

Primary estimation uses paired zero-cost complete episode returns, stratified by outer year. Resample circular blocks of `ceil(20 / h)` episodes, with 5,000 draws and seed 7008. Ten- and forty-session equivalents are descriptive sensitivities. Pair on horizon, fold, entry, and exit, and fail on unmatched episodes. Intervals are pointwise percentiles; centered paired resamples give two-sided p-values for the six research comparisons, with Holm correction within horizon. The comparisons are Ridge/equal minus equal-signal, boosting/equal minus equal-signal, Ridge/regularized minus Ridge/equal, boosting/regularized minus boosting/equal, nested minus equal-signal, and nested minus momentum. Conditional bootstrap uncertainty excludes researcher choices, vendor changes, and new regimes.

Forecast diagnostics use date-balanced out-of-sample squared error against a training-only constant mean, plus cross-sectional rank association. Risk diagnostics include episode loss frequency, fifth percentile, average return below that percentile, and daily drawdown. Zero costs match grading; fixed decisions are also replayed at 10 bps per dollar traded, including purchases, drift-aware rebalances, and liquidation.

## Retention and rollback rule

Only the nested procedure can qualify for adoption; outer ablations cannot be cherry-picked. Test nested minus incumbent at all three horizons using one-sided centered paired block-bootstrap p-values and Holm correction across the three horizons. A horizon qualifies only if all these conditions hold on the full outer evaluation:

1. Mean paired nominal episode return is strictly positive.
2. Holm-adjusted p-value is at most 0.05.
3. Observed losing-episode frequency is no greater than the incumbent's.
4. Observed fifth-percentile episode return is no lower than the incumbent's.

The latter two are descriptive safeguards, not certified risk guarantees. The 2023-onward inspected subset and cost sensitivity are reported without changing the decision. Keep the legacy strategy for any horizon that does not qualify. If no horizon qualifies, revert the experimental code and CLI integration through a normal Git revert, while retaining this protocol and a compact, reproducible record of the failed experiment. Commit/push experimental work and the rollback on `main` so the evidence and implementation remain recoverable in history. Do not tune and rerun the same experiment to obtain acceptance.

Sample outcomes validate software only and never qualify a strategy. Keep generated full datasets and outputs ignored. Archive the exact source/data/config hashes, decisions, selections, fit boundaries, and numerical audit results.
