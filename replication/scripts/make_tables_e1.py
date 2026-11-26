"""Build LaTeX tables for the theory benchmark and Monte Carlo (paper Tables 1-2)."""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"
PAPER = Path(__file__).resolve().parents[2] / "paper"

# Table 1: lognormal benchmark
p = pd.read_csv(OUT / "e1_prop3_table.csv")
rows = []
for s, g in p.groupby("sigma"):
    gt = g.gini_true.iloc[0]
    cells = " & ".join(f"{v:.3f}" for v in g.sort_values("R2").gini_pred)
    rows.append(f"{s:.1f} & {gt:.3f} & {cells} \\\\")
t1 = r"""\begin{table}[ht]
\caption{Gini coefficient of conditional-mean predictions in the lognormal benchmark}\label{tab:lognormal}%
\begin{tabular}{@{}cccccc@{}}
\toprule
 & & \multicolumn{4}{c}{$G(m(X))$ for $R^2_\ell=$}\\
\cmidrule{3-6}
$\sigma$ & $G(Y)$ & 0.2 & 0.4 & 0.6 & 0.8\\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular}
\footnotetext{$\sigma$ is the standard deviation of $\log Y$ and $R^2_\ell$ the population $R^2$ of $\log Y$ on $X$. Values from Eq.~(\ref{eq:lognormal}); Monte Carlo values with $2\times 10^6$ draws differ by less than 0.001.}
\end{table}
"""
(PAPER / "tab_lognormal.tex").write_text(t1)

# Table 2: Monte Carlo relative bias, lgbm, n=10000, heteroskedastic
s = pd.read_csv(OUT / "e1_summary.csv").rename(columns={"index": "idx"})
idx = ["gini", "mld", "theil", "fgt0", "fgt1", "fgt2"]
names = {"oracle": "Oracle $m(X)$", "N": "N (plug-in)", "C1": "C1 (recalibration)",
         "C2": "C2 (residual simulation)", "C3": "C3 (quantile forest)", "C4": "C4 (binned bootstrap)"}
lines = []
for r2 in [0.2, 0.4, 0.6, 0.8]:
    fit = s[(s.estimator == "fit") & (s.idx == "oof_r2_log") & (s.model == "lgbm") &
            (s.n_train == 10000) & (s.hetero == True) & (s.r2 == r2)].mean_est.iloc[0]
    lines.append(r"\midrule" if lines else "")
    lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{$R^2_\ell={r2}$ (out-of-fold $R^2$ of the model: {fit:.2f})}}}}\\")
    for est in ["oracle", "N", "C1", "C2", "C3", "C4"]:
        if est == "oracle":
            d = s[(s.estimator == "oracle") & (s.n_train == 10000) & (s.hetero == True) & (s.r2 == r2)]
        else:
            d = s[(s.estimator == est) & (s.model == "lgbm") & (s.n_train == 10000) & (s.hetero == True) & (s.r2 == r2)]
        vals = [d[d.idx == i].rel_bias_pct.mean() for i in idx]
        lines.append(names[est] + " & " + " & ".join(f"${v:.1f}$" for v in vals) + r" \\")
t2 = r"""\begin{table}[ht]
\caption{Monte Carlo relative bias (\%) of inequality and poverty estimates}\label{tab:mc}%
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Estimator & Gini & MLD & Theil & FGT0 & FGT1 & FGT2\\
""" + "\n".join(l for l in lines if l) + r"""
\botrule
\end{tabular}
\footnotetext{Heteroskedastic design, gradient boosting (LightGBM) fitted on $n=10{,}000$ training observations, target population of 100{,}000, 8 replications per cell, $R=50$ draws for C2--C4. The poverty line is 60\% of the true median. The oracle uses the true conditional mean $m(X)$.}
\end{table}
"""
(PAPER / "tab_mc.tex").write_text(t2)
print(t2)
