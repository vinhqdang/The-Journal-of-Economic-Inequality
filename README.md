# The Journal of Economic Inequality — manuscript

Target journal: [The Journal of Economic Inequality](https://link.springer.com/journal/10888) (Springer, ISSN 1569-1721 / 1573-8701).

## Layout

- `paper/` — the manuscript. Upload the files in this folder **flat** (the journal forbids subfolders in LaTeX submissions).
  - `main.tex` — manuscript (Springer Nature `sn-jnl` class, author–year style `sn-mathphys-ay`)
  - `references.bib` — bibliography
  - `sn-jnl.cls`, `sn-mathphys-ay.bst` — template files
- `template/` — untouched Springer Nature LaTeX template, for reference.

## Build

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Journal requirements checklist

- [ ] Title concise and informative; running title set
- [ ] Authors, affiliations (institution, city, country), corresponding author e-mail, ORCID
- [ ] Abstract 150–250 words, no undefined abbreviations, no citations
- [ ] 4–6 keywords
- [ ] JEL codes
- [ ] Headings: decimal numbering, at most three levels
- [ ] Abbreviations defined at first use
- [ ] Footnotes (not endnotes); no footnote consisting only of a citation
- [ ] Citations by name and year: (Atkinson 1970); Becker and Seligman (1996)
- [ ] Reference list alphabetical, only published/accepted works, full DOI links, ISO 4 journal abbreviations
- [ ] Tables: Arabic numbering, caption above, footnotes with superscript lower-case letters or asterisks
- [ ] Figures: `Fig1.eps` naming, Arabic numbering, no title inside the image, lettering 8–12 pt sans serif, patterns in addition to colours
- [ ] Appendix figures/tables continue main numbering; online appendix referred to as "Online Resource 1" (ESM_1.pdf)
- [ ] "Statements and Declarations": funding, competing interests, data availability, author contributions
- [ ] Replication package (data, code, list of sources, README) ready for deposit in a public repository (openICPSR, Zenodo, Dataverse, ...)
- [ ] LLM use, if any beyond copy editing, documented in the methods section
