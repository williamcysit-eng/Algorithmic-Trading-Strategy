# Risk-allocation study: frozen design and rollback rule

This is a new retrospective exploratory study after the rejected nested forecasting experiment. Existing prices, legacy results, and the previous experiment through 2026-10-21 have already been inspected. This local protocol is committed before computing the new covariance-allocation outcomes; it is not independent preregistration or a new untouched holdout. The aim is to learn about the return–risk tradeoff among a small declared set, not claim a universal optimum.

## Isolation and fixed methods

Keep the legacy gated ensemble, original features, 756-session model-training window, purged monthly vintages, top five stocks, next-close execution, and fully invested long-only $1 million budgets. Use 1/5/19-session horizons to match the assignment calendar. All methods trade the same dates and stocks; only stock weights differ. Each outer calendar year restarts on its first available session and liquidates on its last, matching the archived nested-study incumbent calendar. Incomplete terminal episodes remain in daily accounting but are excluded from complete-episode statistics. Zero nominal-price costs are primary; 10 bps per dollar traded is a fixed-decision sensitivity.

At each signal close, estimate risk using exactly the last 252 complete daily split-aware nominal price returns of the five selected stocks, ending at that close. Fail on missing history; never fill or compress missing sessions. Fit a Ledoit–Wolf shrunk covariance matrix on that dated panel, with no expected-return fitting or parameter selection. The covariance's daily scale is a risk proxy; multiplying by h does not change minimum-variance weights and does not assert that multi-session returns are independent.

Four policies, fixed before results:

1. `equal`: 20% each, matching the existing incumbent.
2. `inverse_volatility`: normalize the inverse square roots of the shrunk covariance diagonal, then project onto the bounded simplex.
3. `minimum_variance`: minimize w'Cov w on the same bounded simplex.
4. `balanced`: the 50/50 weight average of equal and minimum-variance allocations. This is the sole primary replacement candidate, chosen ex ante to moderate covariance-estimation error. It effectively keeps weights between 12.5% and 30% for five stocks.

Bounds are 5%–40% per selected stock, and weights sum to one. They are robustness choices, not assignment rules or fitted optima. Use exact deterministic bounded quadratic optimization in five dimensions and check it independently. No new features, expected returns, stock-selection constraints, covariance-window searches, or adaptive historical winner selection are introduced. The four allocations may choose the same stocks across horizons; forced differentiation is not an objective.

## Preference curve and estimands

For each horizon report complete episode mean return, standard deviation, losing fraction, fifth percentile, average return at or below the fifth percentile, and certainty-equivalent returns under constant-relative-risk-aversion utility (CRRA). Use relative risk aversion gamma in [0, 1, 3, 5, 10]. Gamma=0 is mean return; gamma=1 is log utility. For gamma != 1, normalized utility is `((1+r)^(1-gamma)-1)/(1-gamma)`. Certainty equivalent is the inverse utility of mean utility minus the initial wealth unit. These are empirical preference summaries, not guaranteed returns or forecasts. Gamma=3 is the declared primary analysis convention, not an elicited user preference. Do not change it after seeing results.

Report each method's utility across the whole preference grid, observed Pareto dominance in mean/volatility, predicted risk, concentration, turnover, daily drawdown, and cost sensitivity. Show annual folds and the previously inspected 2023-onward subset without using them to choose a new winner. Do not rank policies solely by Sharpe or confuse lower loss frequency with smaller severe losses. Dividends remain excluded from nominal grading returns.

## Inference

Pair complete episodes on horizon, year, entry, and exit. Do not drop unmatched policies. Resample whole paired episodes in circular blocks of `ceil(20/h)` episodes within outer calendar years, with 5,000 draws and seed 7008. At h=19 the main block spans 38 sessions. Report descriptive 10/40-session sensitivities for the primary comparison. Intervals condition on observed years/procedure and exclude researcher choices, vendor revisions, or new regimes.

The primary family is balanced versus equal at gamma=3 for the three horizons. Use centered paired mean-utility differences for one-sided bootstrap p-values and Holm correction across those three tests. Certainty-equivalent difference intervals are paired percentile intervals. Also report descriptive utility comparisons for all 3 alternatives x 3 horizons x 5 preferences, with one Holm family across all 45 tests. The primary family is separately declared rather than selected from that table. Sensitivity intervals and observed risk safeguards are not simultaneous guarantees.

## Adoption and rollback

The primary balanced policy qualifies for a horizon only if all these conditions hold on the full 2018–2026 outer evaluation:

- Positive mean-utility difference at gamma=3 with primary Holm-adjusted p <= 0.05.
- Observed episode standard deviation is strictly below equal weights.
- Observed average return at/below its own fifth percentile is no worse than equal weights (less negative is better).
- Mean-return sacrifice, expressed as `max(0, (mean_equal-mean_balanced)*252/h)`, is no more than 0.02, i.e. two percentage points per year on this arithmetic scaling. This is a declared tolerance, not a calibrated user preference or CAGR bound.

These criteria explicitly allow some loss of raw expected return in exchange for risk reduction, unlike the prior experiment. They do not certify future protection. Only the fixed balanced policy may qualify for adoption; an attractive inverse-volatility or full minimum-variance ablation cannot be substituted after viewing results. Any qualified integration stays explicit and optional until a risk preference is chosen; the default legacy rule remains equal weights. If no horizon qualifies, revert the experimental production/CLI changes using Git revert and retain this protocol, all method summaries, failed comparisons, independent audit, and reproduction revision. No tuning-and-rerunning to obtain acceptance.

## Verification and provenance

Run the original tests and lint, add behavioral checks for covariance cutoffs, future-data invariance, identical stock selections, bounds, exact optimizer optimality, utility formulas, paired resampling, cost identities, and complete-episode accounting. Reconcile the baseline with archived incumbent holdings and returns. Independently reconstruct prices/shares, daily fees, utility, and inference. Run the real-data offline sample as a software demonstration only. Keep full generated artifacts/data ignored; commit compact numerical evidence, references, manifests, and findings. Commit and push reviewable stages on main, including any rollback.

Academic motivation: Ledoit and Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* (2004), DOI 10.1016/S0047-259X(03)00096-4; DeMiguel, Garlappi and Uppal, *Optimal Versus Naive Diversification* (2009), DOI 10.1093/rfs/hhm075. These motivate covariance regularization and the equal-weight control. They do not establish that these allocations improve this repository's portfolios.
