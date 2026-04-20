# Agent 4: Results-Agent Review

## CSV Source Check
PASS. Tables are generated from `results_paperlock/tables/*.csv`. No simulation rerun is used.

## Percentage Check
PASS. `scripts/verify_numbers.py` verifies the known paper-lock percentages, including P3C-LR Core versus Reactive-3C, MUCCO-like, and RateMax.

## Confidence Interval Check
PASS. Main table uses mean ± 95% CI from `table_main_results_paperlock.csv`.

## Significance Check
PASS. Corrected p-values are taken from `table_significance_paperlock.csv`. The manuscript states that only Holm-corrected significant differences are treated as reliable. It does not claim improvement for energy where P3C-LR Core is worse.

## Oracle Check
PASS. OracleLink Upper Bound and OracleSelector Upper Bound are labelled upper bounds and excluded from practical ranking.

## Remaining Items
PASS. No fabricated result claim found.
