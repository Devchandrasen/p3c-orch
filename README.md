# Predictive 3C Orchestration for UAV-Swarm Low-Altitude IoT under Environmental Link Uncertainty

This folder contains the IEEE-style LaTeX manuscript for the P3C-Orch paper. The manuscript uses only the final paper-lock outputs from `../results_paperlock/`.

## Build

```bash
cd paper_project
bash build.sh
```

The compiled PDF is written to:

```text
paper_project/output/manuscript.pdf
```

## Reproducibility

The paper reports the paper-lock blind evaluation on seeds `4001--4080`. Earlier result folders are treated only as development archives. No tuning was run on the paper-lock seeds. The implementation will be made available at:

```text
https://github.com/Devchandrasen/p3c-orch
```

GitHub link set for the project repository.

## Tables and Figures

Figures are copied from `../results_paperlock/figures/`. Tables are generated from `../results_paperlock/tables/*.csv` using:

```bash
python scripts/make_tables.py
```

The raw simulation logs are not included in `paper.zip`; the manuscript uses the saved CSV summaries and final figures.

## Citation Verification

The bibliography is in `refs/references.bib`. The file `refs/citation_audit.md` lists the source used for each citation. Entries marked with TODO should be checked manually before submission.

## Manual TODOs Before Submission

- Replace author names, emails, ORCID IDs, and affiliation details.
- GitHub URL is set to `https://github.com/Devchandrasen/p3c-orch`.
- Verify any BibTeX entry marked with `% TODO`.
- Check final journal formatting and page limits.
