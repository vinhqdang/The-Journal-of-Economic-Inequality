"""Build the LaTeX table for the satellite application (paper Table 4)."""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"
PAPER = Path(__file__).resolve().parents[2] / "paper"
s = pd.read_csv(OUT / "e3_summary.csv")


def f(v, d=1):
    return f"{v:.{d}f}".replace("-", "$-$")


rows = []
for scheme, lab in [("in_country", "In-country"), ("out_country", "Out-of-country")]:
    if rows:
        rows.append(r"\midrule")
    first = True
    for model, mlab, ests in [("ridge", "Ridge", ["N", "C1", "C2", "C4"]),
                              ("lgbm", "Boosting", ["N", "C1", "C2", "C4"]),
                              ("qrf", "QRF", ["C3"])]:
        for e in ests:
            r = s[(s.scheme == scheme) & (s.model == model) & (s.estimator == e)].iloc[0]
            cells = [f(r.rel_bias_abs_gini_pct), f(r.rel_bias_sd_pct), f(r.rel_bias_p90_p10_pct),
                     f(r.mae_share_below_pp), f(100 * r.share_lt_abs_gini, 0), f(r.spearman_abs_gini, 2)]
            rows.append(f"{lab if first else ''} & {mlab} & {e} & " + " & ".join(cells) + r" \\")
            first = False
            mlab = ""

fit = s[s.estimator == "N"].set_index(["scheme", "model"])
note = "; ".join(
    f"{lab}: $R^2$ {fit.loc[(sc, 'ridge'), 'median_oof_r2']:.2f} (ridge), {fit.loc[(sc, 'lgbm'), 'median_oof_r2']:.2f} (boosting)"
    for sc, lab in [("in_country", "in-country"), ("out_country", "out-of-country")])

tab = r"""\begin{table}[ht]
\caption{Between-cluster inequality of predicted wealth in 43 African country-years}\label{tab:sat}%
\setlength{\tabcolsep}{4pt}%
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lllrrrrrr@{}}
\toprule
 & & & \multicolumn{3}{c}{Relative bias (\%)} & MAE & Share & \\
\cmidrule{4-6}
Scheme & Model & Est. & Abs.\ Gini & SD & P90--P10 & share (pp) & low (\%) & $\rho$\\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\footnotetext{Outcome: cluster mean of the asset-wealth index of \citet{yeh2020}, 19{,}669 DHS clusters. Features: histograms of Landsat bands and night lights. In-country: five spatial blocks within each country; out-of-country: leave one country out. Relative bias is averaged over country-years. MAE share: mean absolute error, in percentage points, of the share of clusters below the pooled 40th percentile. Share low: share of country-years in which the estimated absolute Gini is below the true one. $\rho$: Spearman correlation of estimated and true absolute Gini across country-years. C3 uses a quantile regression forest (QRF) and does not depend on the point model. Median out-of-fold """ + note + r""".}
\end{table}
"""
(PAPER / "tab_sat.tex").write_text(tab)
print(tab)
