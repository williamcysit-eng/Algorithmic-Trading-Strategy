# Risk-allocation study: what the tradeoff actually costs

The fixed risk allocations reduced observed volatility but also reduced mean returns. The predeclared balanced policy did not pass the corrected utility comparison at any horizon, so the experimental production/CLI integration is rejected and reverted. Equal weights remain the active default. This finding does not establish equal weights as universally optimal; the preference curves show why a more risk-averse investor can reasonably prefer a different point on the observed tradeoff.

These are technical research notes and reproducible evidence, not the assessed human-written course report.

## Design

The protocol in [`risk_protocol.md`](risk_protocol.md) was committed as `b20f3ff` before the new allocation outcomes were computed. The tested implementation is revision `9aa1256d9411d2bc888cfe5cdae13ce946d3bcc4`. The candidate set, risk window, preferences, inference, and acceptance thresholds were fixed before seeing these outcomes. Existing history and previous experiments had already been inspected, making this an exploratory retrospective study rather than independent preregistration or a fresh holdout.

The legacy gated ranking model chose the same top five stocks for every method. The study retained its 756-session training window, strictly purged monthly fitting, next-close execution, and fully invested long-only portfolios. It covered 2018-01-02 through 2026-10-21 using the 30-stock snapshot with 126,780 price rows and 1,824 earnings events. The horizons were 1, 5, and 19 sessions. Each outer year restarted its trading phase and liquidated at its final close, matching the preceding study's incumbent rather than changing calendars to obtain favorable comparisons.

Risk estimates used exactly 252 split-aware nominal daily returns of the selected stocks through the signal close. Ledoit–Wolf shrinkage stabilized the five-stock covariance matrix. No expected-return model, hyperparameter selection, additional predictors, or stock substitutions were introduced. The four fixed allocation methods were:

- **Equal:** 20% per stock, the incumbent.
- **Inverse volatility:** inverse shrunk stock volatilities, normalized and projected onto the bounded simplex.
- **Minimum variance:** the exact bounded minimum of portfolio covariance risk.
- **Balanced:** 50% of equal weights plus 50% of minimum-variance weights, the sole predeclared adoption candidate.

The bounds were 5%–40% per stock. Balanced weights consequently lie between 12.5% and 30% with five stocks. Covariance determines only allocations, not stock selection. Daily covariance is a risk proxy, not an assertion that 19-session returns are independent daily draws.

Primary returns exclude dividends and fees to match the assignment. The same decisions were replayed at 10 bps per dollar traded. Only complete episodes enter the primary return/risk comparisons; incomplete final intervals remain in daily portfolio accounting.

## Observed price of lower volatility

Every number below is per complete holding episode. Mean returns and standard deviations are percentages. Volatility reductions are relative to equal weights. Annualizing episode means by 252/h is a descriptive arithmetic scaling, not a CAGR forecast.

| Horizon | Policy | Mean return | Standard deviation | Volatility reduction |
|---|---|---:|---:|---:|
| 1 | Equal | 0.0711% | 1.4060% | — |
| 1 | Inverse volatility | 0.0695% | 1.3327% | 5.2% |
| 1 | Balanced | 0.0655% | 1.3080% | 7.0% |
| 1 | Minimum variance | 0.0599% | 1.2487% | 11.2% |
| 5 | Equal | 0.3043% | 2.9220% | — |
| 5 | Inverse volatility | 0.2992% | 2.7910% | 4.5% |
| 5 | Balanced | 0.2906% | 2.7513% | 5.8% |
| 5 | Minimum variance | 0.2769% | 2.6573% | 9.1% |
| 19 | Equal | 0.7788% | 5.3055% | — |
| 19 | Inverse volatility | 0.6988% | 5.0554% | 4.7% |
| 19 | Balanced | 0.6398% | 5.0598% | 4.6% |
| 19 | Minimum variance | 0.5007% | 4.9688% | 6.3% |

The sample sizes were 2,204 daily, 438 weekly, and 114 monthly episodes. All three alternatives traded some return for reduced variability. Inverse volatility was the gentlest tradeoff: about 0.40, 0.26, and 1.06 percentage points/year of arithmetic mean return sacrificed at the respective horizons. Balanced sacrificed about 1.41, 0.69, and 1.84 points/year. Full minimum variance sacrificed about 2.83, 1.38, and 3.69 points/year.

None of the four methods observedly dominated another in both mean and volatility at daily/weekly horizons. At 19 sessions, inverse volatility had both higher mean and slightly lower volatility than balanced. This is dominance among the four tested historical series only. It is not statistical certification or evidence that a global optimal portfolio has been found.

## Risk has several dimensions

Average return among each method's worst 5% of episodes improved under balanced weights: daily -3.40% to -3.15%, weekly -6.81% to -6.49%, and monthly -11.36% to -11.32%. These are empirical tail means, with sparse monthly observations, not guaranteed loss bounds.

The improvements did not make every outcome safer. Losing-episode frequencies for equal versus balanced were 44.60% versus 44.96%, 42.69% versus 42.69%, and 43.86% versus 47.37%. The worst observed weekly episode moved from -13.47% to -13.58%; the worst monthly episode moved from -18.09% to -18.80%. Minimizing estimated variance did not minimize realized extreme losses.

Zero-cost daily-path drawdowns showed the same distinction. Full minimum variance reduced daily episode volatility by 11.2% but had a slightly worse daily-path maximum drawdown than equal weights, -36.11% versus -35.98%. At five sessions, minimum variance improved drawdown from -30.89% to -29.31%. Drawdown depends on the sequence of gains and losses, not only their dispersion.

## Preference changes the ranking

A certainty-equivalent return is the fixed return with the same utility as the observed distribution under a specified preference. It is an empirical decision summary, not a guaranteed or predicted payoff. The study used constant-relative-risk-aversion utility with gamma = 0, 1, 3, 5, and 10. Gamma=0 ranks mean returns, gamma=1 ranks log utility, and higher values increasingly penalize adverse wealth outcomes. Gamma=3 was the declared primary reference, not an elicited user preference.

The observed maximizers within the fixed candidate set were:

| Risk preference | Daily | Weekly | Monthly |
|---|---|---|---|
| Gamma 0: mean return | Equal | Equal | Equal |
| Gamma 1: log utility | Equal | Equal | Equal |
| Gamma 3 | Inverse volatility | Inverse volatility | Equal |
| Gamma 5 | Inverse volatility | Inverse volatility | Equal |
| Gamma 10 | Minimum variance | Minimum variance | Inverse volatility |

For example, at gamma=3 the daily certainty equivalent is 0.0412% under equal weights and 0.0427% under inverse volatility. Weekly it is 0.1746% versus 0.1807%. These small point differences do not establish dependable improvements. At monthly gamma=3 equal weights give 0.3421%, inverse volatility 0.3006%, balanced 0.2403%, and minimum variance 0.1149%: the observed return sacrifice outweighs the variance reduction for this reference preference.

At gamma=10 all tested certainty equivalents are negative. An observed preference for the least adverse fully invested equity portfolio is not an assurance of positive utility relative to a hypothetical zero-return alternative. The assignment's full-investment constraint prevents using cash as a general solution to that problem.

The [standalone PDF](risk_2026_10_31/risk_preferences.pdf) and [PNG](risk_2026_10_31/risk_preferences.png) show both the mean/volatility tradeoff and the preference curves. All exact values, including the previously inspected 2023-onward subset, are in `preferences.csv` and `preference_maximizers.csv`.

## What the statistical evidence supports

The primary balanced-versus-equal comparison used paired mean-utility differences at gamma=3, with 5,000 circular block resamples within outer years and Holm correction across the three horizons. The main block rounds 20 requested sessions up to whole episodes: 20, 4, and 2 episodes respectively, so the monthly block spans 38 sessions. Ten- and forty-session sensitivities are descriptive.

| Horizon | Balanced minus equal certainty equivalent, pp/episode | Paired 95% interval, pp/episode | Corrected improvement p-value |
|---|---:|---:|---:|
| 1 | -0.0016 | [-0.0109, 0.0076] | 1.0000 |
| 5 | +0.0008 | [-0.0394, 0.0422] | 1.0000 |
| 19 | -0.1018 | [-0.2566, 0.0475] | 1.0000 |

The balanced policy passed the observed volatility, average-tail, and return-sacrifice safeguards at every horizon, but it failed the corrected utility test at all three. The new rule explicitly allowed some mean-return loss for less risk; rejection therefore was not simply caused by insisting on greater raw profitability.

All 45 secondary method/preference/horizon comparisons were also reported in one Holm family. None was significant after correction. The most favorable unadjusted finding was daily inverse volatility at gamma=10, with p approximately 0.021; its corrected p was approximately 0.954. Selecting it based on that isolated unadjusted result would overstate the evidence. Conditional uncertainty excludes the full researcher search history, survivor selection, vendor revisions, and new regimes.

Year-by-year utility results also varied. At gamma=3 inverse volatility exceeded equal weights in five of nine daily years and four of nine weekly and monthly years. The inspected 2023-onward subset often shows greater risk reduction, but it is not an untouched test and was not used to change the adoption decision.

## Execution costs matter to the balance

Risk-weighted portfolios traded more. Mean daily traded notional at the one-session horizon increased from 100.11% of prior NAV under equal weights to 105.47% under inverse volatility, 108.21% under balanced, and 116.63% under minimum variance. Changing desired weights creates transactions even when stock membership stays fixed.

At zero fees, daily inverse volatility produced CAGR 16.42% versus equal weights' 16.59%, while its Sharpe rose from 0.801 to 0.826. At 10 bps per dollar traded, their CAGRs were -10.74% versus -9.39%, respectively. Weekly inverse volatility produced zero-fee CAGR 13.91% versus 13.97%, but 10-bps CAGR 7.75% versus 8.10%. The assignment uses zero costs; cost-aware deployment would need additional evidence that risk improvements justify the added turnover.

These CAGRs compound the matched annual-reset paths through 2026-10-21. They are distinct from the old uninterrupted 2018–2025 headline backtest and must be compared within this study.

## What we learned and what to retain

1. **Risk can be reduced without claiming stronger alpha.** Past-only covariance allocation produced lower realized volatility on the same stock decisions. The measurable price was reduced mean return, plus higher trading activity.
2. **A modest risk tilt was more economical than full optimization in this sample.** Inverse volatility retained more return for a similar reduction in risk than the more aggressive covariance optimizer. This is a descriptive result, not permission to replace the frozen primary candidate after seeing outcomes.
3. **The daily/weekly and monthly decisions have different economics.** Moderate risk aversion favored inverse volatility's daily/weekly point estimates, while equal weights retained the best monthly utility through gamma=5. A single weighting rule need not suit all horizons.
4. **Volatility is not certainty.** Loss frequency, severe episodes, expected-tail outcomes, drawdown, and confidence in estimated averages remain different objects. None of the allocation methods guarantees positive outcomes under the full-investment rule.
5. **The active incumbent remains reasonable, not proven optimal.** The statistical evidence does not support adopting a replacement from this experiment. Equal weights retained the highest mean and log utility among these four methods. More risk-averse preferences can favor alternatives in point estimates, which merits prospective monitoring rather than an after-the-fact promotion.

A further study should target what this comparison held fixed: correlated stock selection, sector/event concentration, and whether turnover-aware shrinkage adds value. It should declare a new protocol, report the accumulating researcher search, and freeze decisions before future observations. Searching more covariance windows or preference values on this same inspected data would not manufacture a confirmatory result. The user's actual willingness to sacrifice return remains a preference to establish before choosing a live risk tilt.

## Validation and reproduction

The experimental revision passed 35 tests with warnings treated as errors and passed Ruff. Rollback commit `7bed5a1` restored all active source and tests exactly to the starting revision `3d1bae6`. The restored baseline passed all 29 tests with warnings treated as errors and passed Ruff; full and sample backtests reproduced all twenty corresponding numeric artifacts. `risk_2026_10_31/rollback_verification.json` records those checks. Six new test cases cover exact bounded optimization, signal-time covariance, utility identities, unchanged stock selections, training/label chronology, fees and episodes, future-data invariance, paired inference, and both acceptance/rejection behavior. The real-data offline demonstration ran separately and cannot trigger adoption. The original sample's ten numeric artifacts matched its saved outputs.

The independent audit in [`audit_risk.py`](audit_risk.py) does not import the experimental allocation or inference modules. It reconstructs Ledoit–Wolf covariance using direct sample-moment formulas, solves allocation and projection problems with SciPy, rebuilds split-aware share holdings and fee-funded trades, and independently computes utility and resampling results. The full audit checked:

- 2,262 dated risk windows and 4,524 independent allocation solves.
- 321 strictly purged ranking-model boundaries.
- 11,024 complete episode returns and 53,112 daily portfolio observations.
- 13,860 equal-weight holdings matching the previous study's incumbent, with matching baseline episode returns.
- 120 preference summaries, 51 inference comparisons including sensitivities, and all three adoption decisions.

Maximum covariance absolute error was 4.34e-19; maximum episode error was 7.60e-16; maximum relative NAV error was 7.19e-14. The direct optimizer's maximum objective gap from the independent solver was 5.00e-16. These checks validate calculations on the supplied snapshot rather than every vendor observation or future performance.

Compact evidence, figures, and exact source/config/data hashes are versioned in [`risk_2026_10_31/`](risk_2026_10_31/). Full generated returns, holdings, covariance metadata, and fit records remain ignored in `results/risk_study/`. To audit those local artifacts on the active revision:

```sh
uv run --frozen python docs/research/audit_risk.py results/risk_study
```

To reproduce the experiment, export revision `9aa1256` to a separate directory and supply the normalized snapshot and manifest matching `risk_2026_10_31/provenance.json`, then run:

```sh
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest -q -W error
uv run --frozen stock-ensemble --config samples/risk.json risk-study
uv run --frozen stock-ensemble --config config.risk.json risk-study
```

`config.risk.json` and `samples/risk.json` remain archived study settings. The active CLI intentionally omits the rejected `risk-study` integration; passing those configurations to a legacy command does not run this experiment.

Academic motivation: Ledoit and Wolf (2004), [A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://doi.org/10.1016/S0047-259X(03)00096-4), motivates covariance shrinkage. DeMiguel, Garlappi and Uppal (2009), [Optimal Versus Naive Diversification](https://doi.org/10.1093/rfs/hhm075), motivates taking equal weights seriously when estimates are uncertain. CRRA utility is an explicitly declared preference model, not an empirically proven description of the user's preferences.
