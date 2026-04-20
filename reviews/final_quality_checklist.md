# Final Quality Checklist

| Item | Status | Evidence |
| --- | --- | --- |
| PDF compiles successfully | PASS | `bash build.sh` completed and wrote `output/manuscript.pdf`. |
| No missing figures | PASS | `scripts/copy_assets.py` copied all 12 required figure groups from `results_paperlock/figures/`. |
| No undefined citations | PASS | Final `manuscript.log` contains no undefined citation warnings. |
| No undefined references | PASS | Final `manuscript.log` contains no undefined reference warnings. |
| No LaTeX errors | PASS | Build completed with exit code 0. |
| No `??` references in PDF | PASS | `pdftotext` check found no `??`. |
| No `[?]` citations | PASS | `pdftotext` and `citation_check.py` found no `[?]`. |
| All result numbers match CSV | PASS | `scripts/verify_numbers.py` passed. |
| All claims using statistical reliability have corrected p < 0.05 | PASS | Claims are tied to Holm-corrected p-values in `table_significance_paperlock.csv`. |
| Oracle methods are upper bounds | PASS | Manuscript and tables label OracleLink and OracleSelector as upper bounds and exclude them from practical ranking. |
| P3C-SR is not called negative | PASS | P3C-SR is described as the stability-regularized internal variant. |
| ET-P3C is exploratory/unsuccessful ablation | PASS | ET-P3C is described as exploratory and without reliable additional gain. |
| Limitations section exists | PASS | Section IX includes simulation, data, weather, PHY, security, and tradeoff limitations. |
| GitHub project URL exists | PASS | Repository URL is set to `https://github.com/Devchandrasen/p3c-orch`. |
| Simple Indian English tone | PASS | Writing review found simple, professional wording and no promotional claim. |
| Author metadata ready | NEEDS MANUAL CHECK | Placeholder author names and affiliation comments remain. |
| GitHub URL ready | PASS | URL updated to `https://github.com/Devchandrasen/p3c-orch`. |
| Citation metadata fully verified | NEEDS MANUAL CHECK | `camp2002mobility` is marked TODO for manual DOI/pages verification. |
| IEEE IoTJ formatting/page limits | NEEDS MANUAL CHECK | PDF is 9 pages; final journal limits should be checked by authors. |
