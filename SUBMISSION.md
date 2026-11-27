# Submission record

## Submission

| | |
|---|---|
| Journal | The Journal of Economic Inequality (Springer) |
| Article type | Regular article (not a special issue) |
| Title | Inequality measurement with machine-learning poverty maps |
| Author | Quang-Vinh Dang, British University Vietnam (corresponding) |
| Submitted | 27 November 2026 |
| Manuscript ID | _to be added when Springer confirms_ |
| Submitted version | git commit `2593f7d` on `main` |

The following files were uploaded:

- `submission/JEI_latex_source.zip`: the flat LaTeX source, 14 files. It contains `main.tex`, `main.bbl`, `references.bib`, `sn-jnl.cls`, `sn-mathphys-ay.bst`, `tab_*.tex` and `Fig1-5.pdf`.
- `paper/main.pdf`: the manuscript, 22 pages.
- `cover_letter/cover_letter.pdf`: the cover letter, 1 page.

The manuscript has 6 sections, 5 figures and 6 tables. The abstract has 224 words. The keywords are inequality measurement, poverty mapping, machine learning, satellite data, Lorenz dominance and inequality of opportunity. The JEL codes are D31, D63, I32, C55 and O15.

## Promised in the manuscript

The data availability statement says that the replication code is "available from the author on request during review and will be deposited in a public repository (Zenodo) with a persistent identifier before publication".

- [ ] Deposit `replication/` on Zenodo (code, `README.md`, `output/`). Do not include `data/`: the data are downloaded by the scripts.
- [ ] Put the Zenodo DOI into the Data availability and Code availability statements at the revision stage.
- [ ] Send the replication package to the editor or the referees if they ask for it during review.

## Known weaknesses that referees may raise

1. **No LMIC household consumption data.** The design-based validation uses United States income, so it covers neither low- and middle-income countries nor consumption. The planned fix is the Malawi IHS5 or Nigeria GHS-Panel W4 (World Bank Microdata Library, free registration). For Malawi the files are `ihs5_consumption_aggregate.dta` and `householdgeovariables_ihs5.dta`. For Nigeria they are `totcons_final` and `nga_householdgeovars_y4`. Do not commit the raw LSMS files.
2. **Fewer replications than designed.** The ACS main exercise has 10 replications against 20 planned, and the bootstrap coverage has 5 against 10 planned. The coverage figures are indicative only. Rerun both with more replications on a dedicated machine.
3. **ACS quantile forest (C3)** uses 19 quantile levels, from 0.05 to 0.95, instead of 99. This thins the tails, and the size of the effect is not measured.
4. **Satellite application.**
   - C1, C2 and C4 use the outer cross-fitting folds instead of nested folds. This causes mild leakage of unquantified size.
   - Survey weights are not used.
   - The outcome is between-cluster inequality of an asset index.
5. **Inference.** Bootstrap intervals undercover at the state level: 25% and 49% against a nominal 90%. A model for area-specific residual dispersion, such as a random effect in the conditional variance, is left to future work.
6. **Rankings.** No correction recovers the ranking of areas by inequality when the features do not predict within-area dispersion.

## Where things are

- Run notes, deviations and sanity checks: `replication/output/e1_checks.txt`, `e2_notes.txt` and `e3_notes.txt`.
- How to rebuild every table and figure: `replication/README.md`.
- Pre-submission internal review: every finding was addressed in the commit "Revise manuscript after internal review".
- Design document: `DESIGN.md`.
- Candidate ideas: `RESEARCH_IDEAS.md`.

## Log

- 2026-11-27: submitted to JEI as a regular article.
