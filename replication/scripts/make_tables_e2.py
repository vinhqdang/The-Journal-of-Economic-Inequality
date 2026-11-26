"""Build the LaTeX table for the ACS design-based validation (paper Table 3)."""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"
PAPER = Path(__file__).resolve().parents[2] / "paper"

s = pd.read_csv(OUT / "e2_summary.csv").rename(columns={"index": "idx"})
s = s[s.model == "lgbm"]


def fmt(v):
    if pd.isna(v):
        return "n.a."
    t = f"{v:.1f}"
    return t.replace("-", "$-$")


def get(regime, est, level, idx, col="mean_rel_bias_pct"):
    r = s[(s.regime == regime) & (s.estimator == est) & (s.level == level) & (s.idx == idx)]
    return r[col].iloc[0] if len(r) else float("nan")


lines = []
for regime, label in [("unit", "Unit-level"), ("context", "Area-context")]:
    if lines:
        lines.append(r"\midrule")
    for k, est in enumerate(["N", "C1", "C2", "C3", "C4"]):
        cells = [fmt(get(regime, est, "national", i)) for i in ["gini", "mld", "fgt0"]]
        cells += [fmt(get(regime, est, "puma", i)) for i in ["gini", "mld", "fgt0"]]
        cells.append(fmt(get(regime, est, "puma", "gini", "rel_rmse_pct")))
        rho = get(regime, est, "puma", "gini", "spearman_gini_puma")
        cells.append("n.a." if pd.isna(rho) else f"{rho:.2f}".replace("-", "$-$"))
        lines.append(f"{label if k == 0 else ''} & {est} & " + " & ".join(cells) + r" \\")

tab = r"""\begin{table}[ht]
\caption{Design-based validation with the American Community Survey: relative bias (\%) of inequality and poverty estimates for non-sampled households}\label{tab:acs}%
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{National} & \multicolumn{3}{c}{PUMA} & \multicolumn{2}{c}{PUMA Gini}\\
\cmidrule{3-5}\cmidrule{6-8}\cmidrule{9-10}
Predictors & Est. & Gini & MLD & FGT0 & Gini & MLD & FGT0 & RMSE & $\rho$\\
\midrule
""" + "\n".join(lines) + r"""
\botrule
\end{tabular*}
\footnotetext{Gradient boosting on $\log Y$; ten replications of a two-stage sample (20\% of PUMAs in each state, 30 households per sampled PUMA, 14{,}790 training households). Targets are all households in the 1{,}969 non-sampled PUMAs. National and PUMA columns report the mean relative bias; RMSE is the relative root mean squared error of PUMA Ginis (\%); $\rho$ is the Spearman correlation between estimated and true PUMA Ginis. n.a.: the estimate is constant within PUMAs, so all PUMA Ginis are zero. Out-of-fold $R^2$ of $\log Y$: 0.50 (unit-level) and 0.04 (area-context).}
\end{table}
"""
(PAPER / "tab_acs.tex").write_text(tab)
print(tab)
