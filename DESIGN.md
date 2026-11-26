# Research design: inequality measurement with machine-learning poverty maps

Working title: **Predicted welfare is too equal: inequality and poverty measurement with machine-learning poverty maps**

## 1. Research questions

- **RQ1.** How large is the bias in Gini, mean log deviation (MLD), Theil, Atkinson and FGT poverty measures computed from ML-predicted welfare, relative to the truth? How does it depend on predictive R², the aggregation level and the index used?
- **RQ2.** Under what conditions is the bias signed and bounded, and what is its exact size?
- **RQ3.** Do distribution-recovering estimators remove the bias, and do they give valid uncertainty? Mean-only recalibration should not be enough.
- **RQ4.** How much do regional inequality rankings and poverty statistics change in real satellite-based poverty maps?

## 2. Theory (Section 3 of the paper)

Notation:
- Y > 0 is welfare, X are predictors, and m(X) = E[Y | X].
- m̂ is the fitted ML predictor.
- μ = E[Y] = E[m(X)].

**P1 (Lorenz dominance).** Y = m(X) + e with E[e | m(X)] = 0. So Y is a mean-preserving spread of m(X) (Rothschild–Stiglitz), and L_{m}(p) ≥ L_Y(p) for every p. Hence I(m(X)) ≤ I(Y) for every Lorenz-consistent index (Gini, GE(α), Atkinson). For FGT_α with α ≥ 1 the poverty function is convex, so P_α(m) ≤ P_α(Y). For the headcount (α = 0) the sign is ambiguous: it is underestimated when the poverty line lies below the bulk of the conditional means, and can be overestimated otherwise.

**P2 (exact bias = within-cell inequality).** Partition by X. Then:
- MLD: I_0(Y) = I_0(m(X)) + E[I_0(Y | X)], using population weights.
- Theil: T(Y) = T(m(X)) + E[(m(X)/μ) · T(Y | X)], using income-share weights.
- Variance of logs, with m_ℓ = E[log Y | X]: Var(log Y) = Var(m_ℓ) + E[Var(log Y | X)]. Equivalently, Var(m_ℓ) = R²_ℓ · Var(log Y).

The "inequality of the poverty map" is therefore the ex-ante inequality of opportunity lower bound (Ferreira and Gignoux 2011), with satellite features playing the role of circumstances.

**P3 (lognormal closed form).** If log Y | X ~ N(m_ℓ(X), σ²(1 − R²)) and m_ℓ ~ N(·, σ² R²), the Gini of the prediction exp(m_ℓ + const) is G = 2Φ(σ√R² / √2) − 1, while the true Gini is G = 2Φ(σ/√2) − 1. Report a table of the bias by (σ, R²).

**P4 (ML shrinkage).** If m̂ = μ + λ(m − μ) with λ ∈ (0, 1), which is typical of regularised or ensemble fits, then the absolute Gini scales by λ. Since the mean is preserved, the relative Gini also scales exactly: G(m̂) = λ G(m) ≤ G(m) ≤ G(Y). Mean recalibration (linear or Tweedie "unshrinking") can at most restore G(m), never G(Y).

## 3. Estimators compared

| Code | Estimator |
|------|-----------|
| N  | Naive plug-in: I(m̂(X)) |
| C1 | Mean recalibration: regress out-of-fold y on m̂, then use the calibrated m̂ (linear calibration correction) |
| C2 | Homoskedastic residual simulation (ELL-style): y* = exp(m̂_ℓ(X) + ε*), with ε* drawn from pooled out-of-fold log residuals; R = 100 replicates, average the index |
| C3 | Heteroskedastic distributional: quantile regression forest (or LightGBM quantile grid) gives a conditional distribution F̂(y \| X); draw y* ~ F̂(· \| X); R = 100 |
| C4 | Conditional residual bootstrap: residuals resampled within deciles of m̂ |

Uncertainty: cluster (area) bootstrap over the training survey. Report coverage of 90% intervals in the design-based exercise.

## 4. Empirical exercises

### E1. Monte Carlo
- DGP: log Y = Xβ + g(X) + σ(X)ε, with heteroskedasticity on and off.
- Grid: R²_ℓ ∈ {0.2, 0.4, 0.6, 0.8}, n_train ∈ {2 000, 10 000}.
- Models: OLS (log), Lasso, random forest, LightGBM.
- Population N = 200 000.
- Outputs: bias and RMSE of Gini, MLD, Theil, Atkinson(1), FGT0/1/2 at z = 0.6 × median, for N and C1–C4. Also verify P2 and P3 numerically.

### E2. Design-based validation with US ACS PUMS 2023 (public, no login)
- Population: ACS 2023 1-year household records, income HINCP > 0, with household-head person characteristics. Treated as a census.
- Areas: PUMAs, nested in states.
- Sampling: two-stage survey mimicking an LSMS. Sample 20% of PUMAs, then 30 households per sampled PUMA. 100 replications.
- Two predictor regimes:
  - (a) **Unit-level:** household covariates (size, number of children, head's age, sex, education and race, tenure, rooms, vehicles, workers, area).
  - (b) **Area-context only:** PUMA-level aggregates of the census covariates. This mimics satellite or geospatial features available for every location but not linked to households.
- Targets:
  - Inequality and poverty within each non-sampled PUMA and within each state.
  - National inequality.
  - Spearman rank correlation of PUMA Gini rankings.
- Report bias, RMSE and coverage for N, C1–C4.

### E3. Satellite application: DHS clusters in Africa (Yeh et al. 2020, Nat. Commun.)
- Data:
  - `dhs_clusters.csv` gives the cluster asset-wealth index for 19 669 clusters in 23 countries.
  - `dhs_image_hists.npz` gives per-cluster histograms of Landsat bands and night lights. These are the features.
- Model: predict cluster wealth with cross-validation that holds out a country (out-of-country) and with spatial blocks within a country (in-country).
- The asset index can be negative, so use translation-invariant measures:
  - absolute Gini (half the mean absolute difference),
  - variance and standard deviation,
  - the P90–P10 gap,
  - the share of clusters below a fixed cutoff (the pooled 40th percentile).
- Compare survey, naive-predicted and corrected (C2–C4) between-cluster inequality by country-year, and the country rankings.
- Report the empirical shrinkage factor λ (slope of prediction on truth) by country.

## 5. Contribution relative to the closest literature

- **Corral, Henderson and Segovia (2025, JDE):** ML poverty maps are biased for poverty rates in Mexico. We generalise to inequality, give a sign and an exact decomposition of the bias, and propose distributional corrections.
- **Elbers, Lanjouw and Lanjouw (2003, Econometrica):** ELL simulates residuals. We show that satellite-ML maps skip this step, extend it to heteroskedastic ML distributional models, and quantify what is lost.
- **AAAI 2026 Tweedie / linear calibration debiasing:** this targets treatment effects. By P4, mean unshrinking cannot recover within-cell inequality.
- **IOp literature (Ferreira and Gignoux 2011; Brunori, Hufe and Mahler 2023):** we make the formal link — poverty-map inequality is an ex-ante lower bound.

## 6. Repository layout

- `paper/` holds the flat LaTeX submission.
- `replication/` holds the code: `src/`, `scripts/`, `output/` (tables and figures, which are copied into `paper/` as Fig*.pdf and tab*.tex) and `README.md`. Raw data are downloaded by script, not committed.
