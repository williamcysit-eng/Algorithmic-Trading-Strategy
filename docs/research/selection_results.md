# Joint stock selection and the event-data foundation

The requested implementation has two separately reviewable results. The source-snapshot/event ledger is retained as tested data infrastructure. The new stock-selection integration was implemented and tested, but none of its three horizons met the adoption rule, so it was reverted. Trading defaults remain the original top-five equal-weight ensemble. No new profitability improvement is claimed.

These are technical experiment notes and evidence, not the assessed human-written course report.

## What was implemented

Protocol [`selection_protocol.md`](selection_protocol.md) was committed as `a955a93` before computing new selection outcomes. The independent event foundation was committed as `637db04`. The tested selection implementation is recoverable at `a87b4537c0a64181efdb020289da6fd87984cfe5`. All implementation and evidence stages were pushed to main.

The event ledger captures original HTTPS source bytes, source checksums, publication metadata with its actual precision, UTC collection timestamps, conservative availability, and append-only revisions. The `stock-events capture`, `import`, and `as-of` commands verify content and revision integrity. A historical document collected now remains absent from earlier decisions. Corrections do not overwrite the information visible at an old cutoff. This conservative foundation supports future announcement research; it does not infer numerical earnings, consensus, guidance, or expected returns.

A real Apple earnings-release source was captured from its original newsroom URL. The page's October 30, 2025 publication date was verified in its content, and the captured bytes pass checksum verification. The actual collection time determines first availability. An as-of query at the 2025 assignment signal cutoff correctly returns no record; the source was not independently collected then. [`../event_capture_example.json`](../event_capture_example.json) preserves the demonstration metadata; complete source bytes and ledger remain in ignored `data/events/`. See [`../events.md`](../events.md) for usage and limitations. Consensus requires an actual preannouncement snapshot; it is never fabricated or imported as though today's retrieval happened years earlier.

## Selection experiment

The existing legacy model and information timing were kept fixed. On each signal close, the first ten eligible ranked stocks form a shortlist. Every five-stock subset is enumerated, with ticker-based deterministic tie-breaking. All selected companies receive 20% weights. This isolates changing membership from the weighting experiments already rejected.

Four policies were fixed in advance:

- **Incumbent:** original top five by score.
- **Sector cap:** highest total score with at most two companies per sector in the shortlist. If necessary, relax to the smallest feasible cap and report the relaxation.
- **Variance guarded:** minimum equal-weight covariance risk subject to mean score within 0.05 of the incumbent.
- **Joint:** apply the sector cap, then minimize covariance risk subject to mean score within 0.05 of the best capped portfolio. This is the only adoption candidate.

Scores remain ordinal, so the 0.05 budget is not five percent expected return. The shortlist size, cap, covariance history, and score budget were not tuned. Covariance used Ledoit–Wolf shrinkage on exactly 252 split-aware nominal daily returns of the shortlist through the signal close. Sector labels are static labels supplied in the universe, not a historical sector-membership archive. This limitation is explicit.

The full sample used 30 stocks and the snapshot through 2026-10-21, with 126,780 price observations and 1,824 earnings events. The evaluation spans 2018-01-02–2026-10-21 with 1/5/19-session holdings. Every policy shares the same annual-reset execution calendar and strictly matured monthly legacy fits. Complete episodes enter primary statistics; incomplete final intervals remain in daily accounting. Dividends and fees are excluded from grading returns; fixed decisions are also replayed at 10 bps per traded dollar.

## Main result

Mean return and standard deviation below are per complete holding episode, not annualized forecasts.

| Horizon | Policy | Mean return | Standard deviation | Mean of worst 5% of episodes |
|---|---|---:|---:|---:|
| 1 | Incumbent | 0.0711% | 1.4060% | -3.40% |
| 1 | Joint | 0.0608% | 1.2282% | -2.92% |
| 5 | Incumbent | 0.3043% | 2.9220% | -6.81% |
| 5 | Joint | 0.2617% | 2.5707% | -6.27% |
| 19 | Incumbent | 0.7788% | 5.3055% | -11.36% |
| 19 | Joint | 0.7252% | 4.7752% | -10.89% |

There are 2,204 daily, 438 weekly, and 114 monthly paired episodes. Joint selection reduced observed standard deviation by 12.6%, 12.0%, and 10.0%, respectively. That is a larger reduction than the prior balanced-weight experiment, but average returns also declined. Arithmetic mean sacrifices scaled by 252/h were 2.59, 2.15, and 0.71 percentage points/year. The daily and weekly sacrifices exceeded the fixed two-point tolerance; this scaling is not a CAGR bound or a calibrated user preference.

Joint selection kept about 3.5 of the original five names on an average decision and replaced about 1.5 names. Its maximum sector allocation never exceeded 40% in the study, versus up to 100% for the original daily portfolio over the full interval. No sector-cap relaxation was required in the actual full run. The mean score sacrifice was about 0.038–0.040, an ordinal amount rather than an estimate of return lost.

Lower aggregate volatility did not make every adverse outcome smaller. The worst monthly episode deteriorated from -18.09% to -19.83%, while the average of the worst monthly episodes improved. Daily/weekly losing-episode frequencies increased from 44.60%/42.69% to 45.05%/44.98%; monthly frequency declined from 43.86% to 42.11%. Daily-path maximum drawdown improved from -35.98% to -33.05%, weekly from -30.89% to -30.16%, but monthly worsened from -33.49% to -36.16%.

[Plot of all four policies (PDF)](selection_2026_10_31/selection_tradeoffs.pdf) and [PNG](selection_2026_10_31/selection_tradeoffs.png) show the historical mean/standard-deviation comparison.

## Acceptance test and rollback

The declared utility model is CRRA with reference gamma=3, carried over from the previous risk study as an analysis convention rather than an elicited user preference. The full preference grid remains gamma=[0,1,3,5,10]. Primary inference uses paired whole-episode circular blocks stratified by outer year, 5,000 draws, seed 7008, with 20 requested sessions rounded up to whole episodes. At 19 sessions this is a 38-session block. Ten/forty-session equivalents are descriptive sensitivities. Intervals are conditional pointwise intervals; the comparison remains retrospective on already inspected history.

| Horizon | Joint minus incumbent certainty equivalent, pp/episode | Paired 95% interval, pp/episode | Primary Holm p-value |
|---|---:|---:|---:|
| 1 | -0.0032 | [-0.0274, 0.0214] | 1.0000 |
| 5 | -0.0135 | [-0.1305, 0.0974] | 1.0000 |
| 19 | +0.0243 | [-0.3528, 0.4095] | 1.0000 |

No primary horizon passes the corrected utility test. All pass the observed volatility and mean-tail safeguards; only monthly also passes the mean-return-sacrifice limit. None qualifies for adoption. The experimental selection/CLI commit is therefore reverted without substituting a more attractive ablation after the fact. The unrelated event-source infrastructure remains because it is validated independently and changes no predictions or holdings.

The other policies have useful point estimates, retained in full rather than selected:

- Sector-cap-only mean returns were 0.0736%, 0.3161%, and 0.7023%. Its daily/weekly standard deviations were slightly below the incumbent's. It changes only about 0.2 original names per decision, making it a smaller perturbation than the joint rule.
- Variance-guarded monthly selection had mean 0.8467% and standard deviation 4.8271%, compared with incumbent 0.7788% and 5.3055%. This observed mean–variance improvement is statistically unresolved.
- None of the 45 alternative/preference/horizon comparisons passes the separate Holm correction; all adjusted p-values are 1. The best unadjusted comparison is still above 0.09.

These point estimates motivate questions for prospective validation, not a new live winner. The existing historical data has now influenced multiple studies, so the aggregate researcher search remains a limitation beyond each experiment's declared multiplicity family.

## What was learned

Joint membership selection has more influence on covariance risk than changing weights within a fixed five-stock set. In this implementation it also discards more ranking information and increases turnover. Minimizing covariance within a score budget is therefore a concrete tradeoff, not a source of new alpha. A cheap sector cap perturbs the ranking less than full covariance selection, but its apparent advantages did not achieve reliable evidence either.

Information quality remains the unresolved bottleneck. The event ledger now makes honest collection possible, but no announcement-based feature has been validated. A next study needs an actual source history or prospective collection, explicit accounting units, announcement timestamps, and any consensus snapshot collected before release. It should isolate one information hypothesis before combining it with a new stock selector. Increasing the score budget, changing the shortlist, or choosing the most attractive ablation from these same outcomes would be additional retrospective development, not confirmation.

## Validation and reproducibility

The tested experimental revision passed all 40 tests with warnings treated as errors and passed Ruff. The event foundation contributes six behavioral cases covering no backdating, append revisions, stable old as-of views, idempotence, timezone/date precision, checksums, identity, clock behavior, and CLI inspection. Five selection cases check exact subset objectives, row-order invariance and ties, explicit cap fallback, missing covariance history, purged time boundaries, full-investment/fees, future prices/filings, truncation, utility, pairing, and acceptance/rejection.

The original sample backtest reproduced its ten numeric artifacts. Rollback commit `38bfab1` removed only the experimental selection integration. The retained event infrastructure and original trading code pass all 35 active tests with warnings treated as errors and pass Ruff. Full and sample baseline runs reproduce all twenty corresponding numeric artifacts; active trading source and configurations match the starting revision. The independent audit reruns successfully without the rejected modules. `selection_2026_10_31/rollback_verification.json` records those checks. The real-data sample selection study separately validated software and could not trigger adoption. The independent audit in [`audit_selection.py`](audit_selection.py) imports no experimental selector or inference code and checked:

- 2,277 dated shortlist covariance windows using direct Ledoit–Wolf sample-moment formulas.
- 11,088 exact subset choices through independent full enumeration.
- 321 strict fitted-model boundaries.
- 13,860 incumbent holdings matching the previous study's unchanged baseline.
- 11,024 complete nominal episodes and 53,112 daily share/fee observations.
- 120 preference summaries, 51 inference comparisons including sensitivities, and all three adoption decisions.

Maximum covariance error was 4.34e-19, maximum episode error 6.18e-16, and maximum relative NAV error 2.44e-14. These checks establish implementation consistency on the snapshot, not independent truth of every vendor record or predictive validity.

Compact metrics, annual results, all failed comparisons, figures, and exact source/config/data hashes are in [`selection_2026_10_31/`](selection_2026_10_31/). Larger shortlists, holdings, risk metadata, and daily returns remain ignored in `results/selection_study/`. Audit those local outputs with:

```sh
uv run --frozen python docs/research/audit_selection.py results/selection_study
```

To reproduce the experiment, export revision `a87b453` into a separate directory, supply the normalized snapshot matching `selection_2026_10_31/provenance.json`, and run:

```sh
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest -q -W error
uv run --frozen stock-ensemble --config samples/selection.json selection-study
uv run --frozen stock-ensemble --config config.selection.json selection-study
```

The active revision intentionally omits the rejected selection-study integration. `config.selection.json` and `samples/selection.json` remain archived protocol settings; a legacy command does not activate this study. The active `stock-events` commands remain usable, with instructions in [`../events.md`](../events.md).
