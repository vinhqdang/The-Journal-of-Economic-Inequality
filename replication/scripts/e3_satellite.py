#!/usr/bin/env python3
"""E3: satellite application. Inequality of ML-predicted DHS cluster wealth.

Outcome y = cluster mean asset index (wealth). Unit = cluster, unweighted.
Two cross-fitting schemes give out-of-fold (OOF) predictions for every cluster:
  in_country   : hold out fold_spatial block k in every country (pooled model)
  out_country  : leave one country out (fold_country)
Point models: ridge (standardise, clip +/-5, RidgeCV) and lgbm.
Between-cluster distribution estimators, by country-year:
  N   plug-in OOF prediction
  C1  linear recalibration y = a + b * yhat, (a, b) from training-fold OOF
  C2  additive homoskedastic residual simulation, R = 200
  C3  quantile regression forest, 99 predicted quantiles, R = 200 (model 'qrf')
  C4  residual bootstrap within deciles of yhat, R = 200
Indices (translation invariant, ineq.all_absolute): abs_gini, sd, p90_p10,
share_below (cutoff = pooled 40th percentile of true wealth).

Outputs (replication/output/):
  e3_results_raw.csv, e3_summary.csv, e3_country.csv, e3_country_all.csv,
  e3_notes.txt, e3_table_main.tex, Fig_e3_country.pdf
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REP, "src"))
import ineq  # noqa: E402  (not modified; used for all_absolute and wquantile)

DATA = os.path.join(REP, "data", "dhs", "dhs_features.parquet")
OUT = os.path.join(REP, "output")

R = 200                       # simulation replicates for C2, C3, C4
CLIP = 5.0                    # clip of standardised features for linear models
ALPHAS = np.logspace(-1, 3, 9)
QUANTILES = np.arange(1, 100) / 100.0
MIN_CLUSTERS = 50
SEED = 20260101
POINT_MODELS = ("ridge", "lgbm")
INDEX_KEYS = ("abs_gini", "sd", "p90_p10", "share_below")
SCHEMES = ("in_country", "out_country")

TIMES = {}


def tic(label):
    TIMES[label] = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start {label}", flush=True)


def toc(label):
    dt = time.time() - TIMES[label]
    TIMES[label] = dt
    print(f"[{time.strftime('%H:%M:%S')}] done  {label} in {dt:.1f}s", flush=True)
    return dt


# ---------------------------------------------------------------- models
def make_point_model(name, seed):
    if name == "ridge":
        return make_pipeline(
            StandardScaler(),
            FunctionTransformer(lambda z: np.clip(z, -CLIP, CLIP)),
            RidgeCV(alphas=ALPHAS),
        )
    if name == "lgbm":
        # subsample needs subsample_freq > 0 to take effect in LightGBM
        return LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, verbose=-1, random_state=seed, n_jobs=4,
        )
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xte, seed):
    m = make_point_model(name, seed)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), dtype=float)


def qrf_quantiles(Xtr, ytr, Xte, seed, chunk=1000):
    qrf = RandomForestQuantileRegressor(
        n_estimators=300, min_samples_leaf=20, max_features=0.3,
        n_jobs=4, random_state=seed,
    )
    qrf.fit(Xtr, ytr)
    out = np.empty((len(Xte), len(QUANTILES)))
    for s in range(0, len(Xte), chunk):
        out[s:s + chunk] = qrf.predict(Xte[s:s + chunk], quantiles=list(QUANTILES))
    return out


# ---------------------------------------------------------------- cross-fitting
def cross_fit(scheme, X, y, fold, seed):
    """Two passes per point model.

    Pass 1: outer out-of-fold (OOF) predictions yhat for every cluster. The
    prediction for fold k is the model trained on all other folds.
    Pass 2: for held-out fold k, the calibration (C1), residual pool (C2) and
    decile residual pools (C4) use the OOF predictions and outcomes of the
    training clusters (fold != k). This is the fallback allowed in the spec:
    nested inner folds were too slow on the shared machine, and the training
    OOF predictions were produced by models that saw fold k. This is a
    documented, mild leakage (see e3_notes.txt).

    Returns dict (model, est) -> array (n,) for N, C1, and (R, n) for C2, C4;
    ('qrf', 'C3') -> (R, n). Each cluster is predicted from the fold holding it out.
    """
    n = len(y)
    folds = np.unique(fold)
    res = {}
    for m in POINT_MODELS:
        res[(m, "N")] = np.full(n, np.nan)
        res[(m, "C1")] = np.full(n, np.nan)
        res[(m, "C2")] = np.full((R, n), np.nan)
        res[(m, "C4")] = np.full((R, n), np.nan)
    res[("qrf", "C3")] = np.full((R, n), np.nan)
    diag = {f"{m}_{p}": [] for m in POINT_MODELS for p in ("slope", "intercept")}

    # Pass 1: OOF point predictions
    oof = {m: np.full(n, np.nan) for m in POINT_MODELS}
    for m_i, m in enumerate(POINT_MODELS):
        t0 = time.time()
        for k_i, k in enumerate(folds):
            tr = fold != k
            te = ~tr
            oof[m][te] = fit_predict(m, X[tr], y[tr], X[te], seed + 100 * k_i + 10 * m_i)
        print(f"    {scheme} {m} OOF over {len(folds)} folds: {time.time() - t0:.1f}s", flush=True)
        res[(m, "N")][:] = oof[m]

    # Pass 2: calibration, C2, C4 for each held-out fold
    for k_i, k in enumerate(folds):
        tr = fold != k
        te = ~tr
        nte = int(te.sum())
        rng = np.random.default_rng(seed + 17 * k_i)
        for m in POINT_MODELS:
            yhat_tr, y_tr = oof[m][tr], y[tr]
            yhat_te = oof[m][te]

            b, a = np.polyfit(yhat_tr, y_tr, 1)
            res[(m, "C1")][te] = a + b * yhat_te
            diag[f"{m}_slope"].append(b)
            diag[f"{m}_intercept"].append(a)

            resid = y_tr - yhat_tr
            res[(m, "C2")][:, te] = yhat_te[None, :] + rng.choice(resid, size=(R, nte))

            edges = np.quantile(yhat_tr, np.arange(1, 10) / 10.0)
            b_tr = np.searchsorted(edges, yhat_tr, side="right")
            b_te = np.searchsorted(edges, yhat_te, side="right")
            c4 = np.empty((R, nte))
            for d in range(10):
                cols = np.where(b_te == d)[0]
                if len(cols) == 0:
                    continue
                pool = resid[b_tr == d]
                if len(pool) == 0:
                    pool = resid
                c4[:, cols] = yhat_te[cols][None, :] + rng.choice(pool, size=(R, len(cols)))
            res[(m, "C4")][:, te] = c4

    # C3: quantile regression forest per fold, one predicted quantile drawn per cluster
    for k_i, k in enumerate(folds):
        t0 = time.time()
        tr = fold != k
        te = ~tr
        nte = int(te.sum())
        rng = np.random.default_rng(seed + 31 * k_i + 5)
        Q = qrf_quantiles(X[tr], y[tr], X[te], seed + 7 * k_i)
        j = rng.integers(0, len(QUANTILES), size=(R, nte))
        res[("qrf", "C3")][:, te] = Q[np.arange(nte)[None, :], j]
        print(f"    {scheme} QRF fold {k} ({nte} clusters): {time.time() - t0:.1f}s", flush=True)
    return res, diag


# ---------------------------------------------------------------- evaluation
def index_point(v, idx, cutoff):
    return ineq.all_absolute(v[idx], cutoff)


def index_sim(M, idx, cutoff):
    vals = [ineq.all_absolute(M[r, idx], cutoff) for r in range(M.shape[0])]
    return {k: float(np.mean([d[k] for d in vals])) for k in INDEX_KEYS}


def group_stats(yhat, y, idx):
    yy, ph = y[idx], yhat[idx]
    vy = np.var(yy)
    lam = float(np.cov(ph, yy, bias=True)[0, 1] / vy)
    sse = float(np.sum((yy - ph) ** 2))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    return {
        "oof_r2": 1.0 - sse / sst,
        "lambda": lam,
        "sd_ratio": float(np.std(ph) / np.std(yy)),
    }


def main():
    t_all = time.time()
    tic("load")
    df = pd.read_parquet(DATA)
    fcols = [c for c in df.columns if c.startswith("f_")]
    X = df[fcols].to_numpy(dtype=float)
    y = df["wealth"].to_numpy(dtype=float)
    country = df["country"].astype(str).to_numpy()
    year = df["year"].to_numpy()
    fold_sp = df["fold_spatial"].to_numpy()
    fold_co = df["fold_country"].astype(str).to_numpy()
    n = len(y)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    cutoff = ineq.wquantile(y, 0.40)  # pooled 40th percentile of true wealth
    print(f"n={n} features={len(fcols)} cutoff(p40)={cutoff:.4f}", flush=True)
    toc("load")

    gkeys = df.groupby(["country", "year"]).indices
    groups = [(c, int(yr), idx) for (c, yr), idx in sorted(gkeys.items())
              if len(idx) >= MIN_CLUSTERS]
    dropped = [(c, int(yr), len(idx)) for (c, yr), idx in sorted(gkeys.items())
               if len(idx) < MIN_CLUSTERS]
    print(f"country-years kept: {len(groups)}; dropped (<{MIN_CLUSTERS}): {dropped}", flush=True)

    # Truth per country-year
    truth = {}
    for c, yr, idx in groups:
        truth[(c, yr)] = ineq.all_absolute(y[idx], cutoff)

    raw_rows = []
    summ_rows = []
    country_rows = []
    timing = {}
    diag_all = {}

    fold_specs = {"in_country": fold_sp, "out_country": fold_co}
    for si, scheme in enumerate(SCHEMES):
        fold = fold_specs[scheme]
        tic(f"cross_fit_{scheme}")
        res, diag = cross_fit(scheme, X, y, fold, SEED + 1000 * si)
        timing[scheme] = toc(f"cross_fit_{scheme}")
        diag_all[scheme] = diag

        tic(f"evaluate_{scheme}")
        # per-model country-year diagnostics (N predictions)
        lam_by_model = {}
        for m in POINT_MODELS:
            per_group_stats = []
            for c, yr, idx in groups:
                st = group_stats(res[(m, "N")], y, idx)
                st.update({"scheme": scheme, "model": m, "country": c, "year": yr,
                           "n": len(idx),
                           "abs_gini_true": truth[(c, yr)]["abs_gini"],
                           "abs_gini_N": index_point(res[(m, "N")], idx, cutoff)["abs_gini"]})
                per_group_stats.append(st)
            country_rows.extend(per_group_stats)
            lam_by_model[m] = per_group_stats

        for (m, est), arr in res.items():
            for c, yr, idx in groups:
                if arr.ndim == 1:
                    est_d = index_point(arr, idx, cutoff)
                else:
                    est_d = index_sim(arr, idx, cutoff)
                row = {"scheme": scheme, "model": m, "estimator": est,
                       "country": c, "year": yr, "n": len(idx)}
                for k in INDEX_KEYS:
                    row[f"{k}_true"] = truth[(c, yr)][k]
                    row[f"{k}_est"] = est_d[k]
                raw_rows.append(row)
        toc(f"evaluate_{scheme}")

        # Summaries for this scheme
        for (m, est) in sorted(res.keys(), key=lambda t: (POINT_MODELS + ("qrf",)).index(t[0])):
            sub = pd.DataFrame([r for r in raw_rows
                                if r["scheme"] == scheme and r["model"] == m and r["estimator"] == est])
            def rb(k):
                return float(np.mean((sub[f"{k}_est"] - sub[f"{k}_true"]) / sub[f"{k}_true"]) * 100)
            s = {
                "scheme": scheme, "model": m, "estimator": est,
                "n_country_years": len(sub),
                "rel_bias_abs_gini_pct": rb("abs_gini"),
                "rel_bias_sd_pct": rb("sd"),
                "rel_bias_p90_p10_pct": rb("p90_p10"),
                "mae_share_below_pp": float(np.mean(np.abs(sub["share_below_est"] - sub["share_below_true"])) * 100),
                "spearman_abs_gini": float(spearmanr(sub["abs_gini_est"], sub["abs_gini_true"]).statistic),
                "share_lt_abs_gini": float(np.mean(sub["abs_gini_est"] < sub["abs_gini_true"])),
                "share_lt_sd": float(np.mean(sub["sd_est"] < sub["sd_true"])),
                "share_lt_p90_p10": float(np.mean(sub["p90_p10_est"] < sub["p90_p10_true"])),
            }
            if m in POINT_MODELS:
                st = pd.DataFrame(lam_by_model[m])
                s["median_lambda"] = float(st["lambda"].median())
                s["median_oof_r2"] = float(st["oof_r2"].median())
                s["median_sd_ratio"] = float(st["sd_ratio"].median())
            else:
                s["median_lambda"] = np.nan
                s["median_oof_r2"] = np.nan
                s["median_sd_ratio"] = np.nan
            summ_rows.append(s)

    # ------------------------------------------------------------ write outputs
    tic("write")
    os.makedirs(OUT, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(os.path.join(OUT, "e3_results_raw.csv"), index=False, float_format="%.6g")

    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(OUT, "e3_summary.csv"), index=False, float_format="%.4f")

    cty = pd.DataFrame(country_rows)
    cty["ratio_N_over_true_abs_gini"] = cty["abs_gini_N"] / cty["abs_gini_true"]
    cty = cty[["scheme", "model", "country", "year", "n", "oof_r2", "lambda", "sd_ratio",
               "abs_gini_true", "abs_gini_N", "ratio_N_over_true_abs_gini"]]
    cty.to_csv(os.path.join(OUT, "e3_country_all.csv"), index=False, float_format="%.5g")

    main_cty = cty[(cty.scheme == "in_country") & (cty.model == "lgbm")].copy()
    c4_main = raw[(raw.scheme == "in_country") & (raw.model == "lgbm") & (raw.estimator == "C4")]
    main_cty = main_cty.merge(
        c4_main[["country", "year", "abs_gini_est"]].rename(columns={"abs_gini_est": "abs_gini_C4"}),
        on=["country", "year"], how="left")
    main_cty = main_cty.rename(columns={"lambda": "lambda_shrink"})
    main_cty = main_cty[["country", "year", "n", "oof_r2", "lambda_shrink", "sd_ratio",
                         "abs_gini_true", "abs_gini_N", "abs_gini_C4"]]
    main_cty.to_csv(os.path.join(OUT, "e3_country.csv"), index=False, float_format="%.5g")
    toc("write")

    write_table(summ)
    make_figure(raw)
    total = time.time() - t_all
    write_notes(summ, cty, timing, total, cutoff, groups, dropped, len(fcols), diag_all)
    print(f"TOTAL {total:.1f}s", flush=True)


# ---------------------------------------------------------------- table and figure
LABEL = {"N": "N", "C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4"}
SCHEME_LABEL = {"in_country": "In-country", "out_country": "Out-of-country"}
MODEL_LABEL = {"ridge": "Ridge", "lgbm": "LightGBM", "qrf": "QRF"}


def write_table(summ):
    lines = [
        "% E3 main table rows (booktabs). Columns:",
        "% scheme & model & estimator & rel. bias abs.\\ Gini (%) & rel. bias SD (%) & "
        "rel. bias P90--P10 (%) & MAE share below 40th pct (pp) & Spearman rho (abs.\\ Gini) \\\\",
        "% C3 uses a quantile regression forest fitted per fold; it does not depend on the "
        "point model, so it is labelled QRF.",
    ]
    first = True
    for scheme in SCHEMES:
        if not first:
            lines.append("\\midrule")
        first = False
        sub = summ[summ.scheme == scheme]
        for m in ("ridge", "lgbm", "qrf"):
            for est in ("N", "C1", "C2", "C3", "C4"):
                r = sub[(sub.model == m) & (sub.estimator == est)]
                if r.empty:
                    continue
                r = r.iloc[0]
                sch = SCHEME_LABEL[scheme]
                mod = MODEL_LABEL[m]
                lines.append(
                    f"{sch} & {mod} & {LABEL[est]} & {r.rel_bias_abs_gini_pct:.1f} & "
                    f"{r.rel_bias_sd_pct:.1f} & {r.rel_bias_p90_p10_pct:.1f} & "
                    f"{r.mae_share_below_pp:.2f} & {r.spearman_abs_gini:.3f} \\\\"
                )
    with open(os.path.join(OUT, "e3_table_main.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def make_figure(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 3.2), sharex=True, sharey=True)
    panels = [("in_country", "(a) In-country"), ("out_country", "(b) Out-of-country")]
    all_vals = []
    for ax, (scheme, title) in zip(axes, panels):
        sub_n = raw[(raw.scheme == scheme) & (raw.model == "lgbm") & (raw.estimator == "N")]
        sub_c = raw[(raw.scheme == scheme) & (raw.model == "lgbm") & (raw.estimator == "C4")]
        t = sub_n["abs_gini_true"].to_numpy()
        ax.scatter(t, sub_n["abs_gini_est"], s=16, marker="o", facecolor="0.55",
                   edgecolor="0.25", linewidth=0.4, label="N (plug-in)", zorder=3)
        ax.scatter(t, sub_c["abs_gini_est"], s=18, marker="^", facecolor="none",
                   edgecolor="black", linewidth=0.7, label="C4 (decile residual bootstrap)",
                   zorder=4)
        all_vals.append(t)
        all_vals.append(sub_n["abs_gini_est"].to_numpy())
        all_vals.append(sub_c["abs_gini_est"].to_numpy())
        ax.text(0.03, 0.97, title, transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.set_xlabel("True absolute Gini")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    vals = np.concatenate(all_vals)
    lo, hi = float(np.min(vals)) - 0.02, float(np.max(vals)) + 0.02
    for ax in axes:
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, linestyle="-", zorder=1,
                label="45-degree line")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Estimated absolute Gini")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "Fig_e3_country.pdf"), metadata={})
    plt.close(fig)


def write_notes(summ, cty, timing, total, cutoff, groups, dropped, nfeat, diag_all):
    def med_diag(scheme, key):
        return float(np.median(diag_all[scheme][key]))

    lines = []
    add = lines.append
    add("E3 satellite application: notes (generated by replication/scripts/e3_satellite.py)")
    add("=" * 80)
    add("")
    add("DATA AND UNIT")
    add("  Input: replication/data/dhs/dhs_features.parquet (19,669 DHS clusters, 23 countries,")
    add("  survey years 2009-2016, 636 f_* feature columns in the file).")
    add("  Outcome y = wealth (cluster mean asset index, wealthpooled). Real-valued, may be negative.")
    add("  Unit of analysis = cluster. Every cluster has equal weight (unweighted).")
    add(f"  Feature count used: {nfeat} f_* columns (constant histogram bins were removed in preparation).")
    add("  Country-years: grouped by (country, year). Groups with fewer than 50 clusters are dropped;")
    add(f"  dropped groups: {dropped if dropped else 'none'}. Country-years analysed: {len(groups)}.")
    add("")
    add("CROSS-FITTING SCHEMES")
    add("  in_country : fold = fold_spatial (5 within-country k-means blocks). Holding out block k")
    add("               removes that block in every country; the pooled model is trained on the other")
    add("               four blocks across all countries.")
    add("  out_country: fold = fold_country (leave one country out, 23 folds).")
    add("  Every cluster receives its prediction from the fold that holds it out.")
    add("")
    add("POINT MODELS")
    add("  ridge: StandardScaler fitted on the training fold, then clip to [-5, 5], then RidgeCV with")
    add("         alphas = logspace(-1, 3, 9) (generalised cross-validation on the training data).")
    add("  lgbm : LGBMRegressor, 600 trees, learning_rate 0.03, num_leaves 31, min_child_samples 40,")
    add("         subsample 0.8 (subsample_freq = 1 set, otherwise LightGBM ignores subsample),")
    add("         colsample_bytree 0.5, verbose -1, n_jobs 4. No feature standardisation (trees).")
    add("")
    add("OUTER OOF AND CALIBRATION POOL (used by C1, C2, C4)")
    add("  Pass 1: for each point model, the OOF prediction of fold k is the model trained on all other")
    add("  folds (5 fits for in_country, 23 for out_country per model).")
    add("  Pass 2: for held-out fold k, the calibration, residual pool and decile edges use the OOF")
    add("  predictions and outcomes of the training clusters (fold != k).")
    add("  DEVIATION: the spec's main option for C1 (nested 3-fold inside each training fold) was not")
    add("  used. On the shared 4-core machine a single LightGBM fit took several seconds to minutes, so")
    add("  nested folds (about 4x the number of fits) would have exceeded the runtime goal. The spec's")
    add("  fallback is used for all of C1, C2 and C4.")
    add("  Leakage: the training clusters' OOF predictions in pass 2 come from models that were trained")
    add("  on fold k's outcomes. This can make the calibration and residual pools slightly optimistic")
    add("  for the held-out fold. The effect is expected to be small; it is not quantified here.")
    add("")
    add("ESTIMATORS (all indices averaged over R = 200 draws where simulated)")
    add("  N : plug-in OOF prediction yhat.")
    add("  C1: y_cal = a + b * yhat. (a, b) by OLS of y on the OOF prediction over the training")
    add("      clusters of that outer fold. Applied to the held-out yhat.")
    add(f"      Median calibration slope b: in_country ridge {med_diag('in_country', 'ridge_slope'):.3f}, "
        f"lgbm {med_diag('in_country', 'lgbm_slope'):.3f}; out_country ridge "
        f"{med_diag('out_country', 'ridge_slope'):.3f}, lgbm {med_diag('out_country', 'lgbm_slope'):.3f}.")
    add("  C2: y* = yhat + e*, e* drawn with replacement from the pooled OOF residuals of the training")
    add("      clusters (y - yhat_oof). Homoskedastic.")
    add("  C3: quantile regression forest (quantile_forest.RandomForestQuantileRegressor, 300 trees,")
    add("      min_samples_leaf 20, max_features 0.3, n_jobs 4) fitted per outer fold on the training")
    add("      clusters. For each held-out cluster and each replicate, one of the 99 predicted")
    add("      quantiles (0.01, ..., 0.99) is drawn uniformly. C3 does not depend on the ridge/lgbm")
    add("      point model, so it is reported once with model label 'qrf'.")
    add("  C4: residual bootstrap within deciles of yhat. Deciles are the 10% quantiles of the OOF")
    add("      prediction over the training clusters. Each held-out cluster takes residuals")
    add("      from the training residuals in its decile bin, drawn with replacement.")
    add("")
    add("INDICES (ineq.all_absolute, unweighted, translation invariant)")
    add("  abs_gini = half the mean absolute difference (ineq.abs_gini); sd; p90_p10 = weighted 0.9 and")
    add("  0.1 quantiles (ineq.wquantile, interpolated); share_below = share of clusters with y below")
    add(f"  a fixed cutoff = pooled 40th percentile of true wealth across all clusters = {cutoff:.4f}")
    add("  (ineq.wquantile, not the np.quantile linear rule).")
    add("")
    add("SUMMARY DEFINITIONS")
    add("  Relative bias (%) = mean over country-years of 100 * (estimate - truth) / truth.")
    add("  MAE share_below = mean over country-years of |estimate - truth| in percentage points.")
    add("  Spearman = rank correlation across country-years of estimated vs true abs_gini.")
    add("  share_lt_X = share of country-years in which the estimate is below the truth.")
    add("  lambda = Cov(yhat, y) / Var(y) within country-year, using the N prediction (population")
    add("  moments, ddof 0). OOF R^2 = 1 - SSE/SST within country-year, N prediction, the scheme's")
    add("  out-of-fold predictions. sd_ratio = sd(yhat) / sd(y).")
    add("  Proposition 4 note: mean-preserving linear shrinkage yhat = mu + lambda (y - mu) gives")
    add("  abs_gini(yhat)/abs_gini(y) = lambda exactly. When yhat is close to linear in y, the ratio")
    add("  abs_gini_N / abs_gini_true per country-year should be close to lambda. This is checked in")
    add("  e3_country_all.csv (ratio_N_over_true_abs_gini against lambda). The relation is approximate:")
    add("  yhat is not exactly linear in y, and the mean of yhat need not equal the mean of y.")
    add("")
    add("OUTPUT FILES")
    add("  e3_results_raw.csv   one row per scheme x model x estimator x country-year, with true and")
    add("                       estimated abs_gini, sd, p90_p10, share_below.")
    add("  e3_summary.csv       per scheme x model x estimator summaries; median lambda, OOF R^2 and")
    add("                       sd ratio per scheme x point model (blank for qrf).")
    add("  e3_country.csv       lgbm in-country, per country-year: n, OOF R^2, lambda, true and")
    add("                       estimated abs_gini (N, C4).")
    add("  e3_country_all.csv   OOF R^2, lambda, sd ratio and N ratio for all scheme x point-model")
    add("                       combinations, per country-year.")
    add("  e3_table_main.tex    booktabs rows of the main table (no tabular environment).")
    add("  Fig_e3_country.pdf   in-country (a) and out-of-country (b), lgbm, N and C4.")
    add("")
    add("RUNTIME (wall clock, shared 4-core machine; another heavy job ran concurrently)")
    for k, v in timing.items():
        add(f"  {k}: {v:.1f} s")
    add(f"  total: {total:.1f} s ({total / 60:.1f} min)")
    add("")
    add("KNOWN LIMITATIONS")
    add("  - Survey weights and the unit of the DHS sampling design are not used (cluster unweighted).")
    add("  - Country-year sample sizes are small for some surveys (down to 140 clusters); estimates")
    add("    in those cells are noisier and affect the rank correlations.")
    add("  - Out-of-country predictions come from models trained on 22 countries.")
    add("  - C1, C2 and C4 use the outer OOF predictions of the training clusters (see the deviation")
    add("    above), which carries the mild leakage described there.")
    add("  - The C3 quantile forest is fitted once per fold and shared by both point models.")
    add("")
    add("SUMMARY (from e3_summary.csv)")
    add(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    with open(os.path.join(OUT, "e3_notes.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
