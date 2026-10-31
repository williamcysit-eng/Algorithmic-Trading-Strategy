# Joint stock-selection experiment and event-data implementation

This protocol is recorded before computing the new selection outcomes. It follows two rejected experiments on the same inspected history and is explicitly exploratory retrospective research, not a new untouched holdout. The prior results motivate the hypotheses but do not constitute independent confirmation.

## Fixed selection comparison

Retain the current gated ensemble, its features, strictly matured 756-session monthly model fits, next-close execution, fully invested top-five equal weights, and 1/5/19-session targets. Use the 30-stock snapshot through 2026-10-21. Every method shares the same annual-reset trading calendar as the archived incumbent studies. Complete episodes are primary; final partial intervals remain marked and liquidated in daily paths. Zero nominal-price fees are primary; 10 bps per traded dollar is sensitivity. No new return predictors or expected-return fitting are introduced.

For each signal close rank eligible stocks with the unchanged ensemble. The shortlist contains the first ten (or every eligible stock if fewer); enumerate every five-company subset. Sector labels use the supplied universe and are static, so sector reclassification is a disclosed limitation. Estimate a Ledoit–Wolf covariance matrix from exactly 252 split-aware nominal daily returns of the shortlist, ending at the signal close. Missing observations fail; no future prices or compressed histories enter estimates.

Four fixed policies, with ticker order breaking remaining ties:

- `incumbent`: the original top five by score, equal weighted.
- `sector_cap`: maximize the sum of existing ordinal scores with at most two names per sector inside the shortlist. If infeasible, use the smallest feasible integer cap above two; record every relaxation.
- `variance_guarded`: minimize equal-weight estimated portfolio variance among shortlist subsets whose mean ordinal score is at least the incumbent mean minus 0.05. Break variance ties by greater score and then ticker. This score budget is an arbitrary, declared robustness choice, not five percent expected return.
- `joint`: the primary candidate. Apply the sector cap/fallback first. Find the maximum mean score under that cap, then minimize covariance risk among capped subsets whose mean score is within 0.05 of that capped optimum. Break ties by score and ticker.

The shortlist size, sector cap, and score budget are fixed, not selected by historical returns. There is no assumption that a score unit equals a return percentage. The capped optimum supplies the joint policy's score reference so infeasibility never silently turns it into the incumbent. Stock weights remain 20%, isolating selection from allocation.

## Outcomes, inference, and rollback

Report episode mean, standard deviation, loss frequency, fifth percentile, lower-tail mean, annual folds, daily CAGR/drawdown, stock overlap, sector concentration, score sacrifice, cap relaxations, turnover, and costs. Repeat the previously declared empirical CRRA preferences gamma=[0,1,3,5,10]. Gamma=3 is the primary analysis convention, not an elicited user preference.

Pair on horizon, fold, entry, and exit. Use 5,000 seed-7008 circular block resamples within outer years with ceil(20/h) whole episodes; 10/40-session equivalents are descriptive sensitivities. The primary family is joint minus incumbent at gamma=3 with one-sided centered utility bootstrap p-values and Holm correction across horizons. Report all 45 alternative/horizon/preference comparisons with a separate Holm family and preserve failures. Confidence intervals are conditional and pointwise; they exclude researcher search, survivor bias, vendor revisions, and regime changes.

Only joint can qualify for a horizon, requiring all four: positive mean-utility difference at gamma=3 with primary adjusted p<=0.05, lower observed episode standard deviation, no worse average return among each method's worst 5% of episodes, and mean-return sacrifice scaled by 252/h no greater than 0.02. These are unchanged tradeoff criteria from the risk-allocation study. Other variants cannot be substituted after seeing outcomes. Sample runs only validate software. Keep any qualified implementation optional with explicit settings; the legacy default remains unchanged. If none qualifies, Git-revert the experimental selection/CLI commit and preserve protocol, compact results, independent audit, and reproduction revision. No tune-and-rerun loop.

## Independent event-data foundation

Implement source snapshots and an append-only earnings-event ledger separately from the selection experiment. Capture original source bytes, URL, publication precision, retrieval time, identity/revision metadata, and checksums. Treat collection time as the earliest independently observed availability; newly collected historical releases cannot enter old decisions. Date-only publication metadata remains date-only and receives a conservative next-day bound. Corrections append new records and never rewrite earlier as-of views. Validate cutoff timezone, checksum integrity, duplicate handling, and revision ordering.

Provide capture/import and as-of inspection commands, tests, and a small demonstration using actual issuer release pages if reachable. Do not invent earnings values, publication times, analyst consensus, or backdate collection. SEC collection requires the caller's identifying user agent; current issuer pages are acceptable source captures, not a historical point-in-time archive. The ledger is not automatically connected to forecasts until an independently designed feature study exists. This tested data infrastructure can be retained even if the selection strategy fails; it makes no profitability claim.

## Validation and Git

Commit protocol first, then the independently useful event foundation, then the tested selection experiment before outcomes. Push each reviewable stage on main. Check the original sample artifacts, future-data invariance, exact sector/score constrained subset choice, strict fitting boundaries, missing prices, episode/daily accounting, utility and multiplicity. Independently enumerate selections, reproduce covariance moments, rebuild shares and costs, and recompute inference. Keep full generated data ignored and compact evidence versioned. After a rollback rerun tests/lint and reconcile baseline numeric artifacts. These are technical artifacts, not the assessed human-written course report.
