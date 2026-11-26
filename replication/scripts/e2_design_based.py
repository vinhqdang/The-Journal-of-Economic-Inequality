#!/usr/bin/env python3
"""E2 design-based validation with US ACS 2023 1-year household records.

Population: all occupied households with HINCP > 0 in
replication/data/acs/acs2023_households.parquet (built by e2_prepare_acs.py),
treated as a census with weights wgtp. Poverty line z = 0.6 x weighted national
median of y, fixed from the population.

Each replication r:
  1. Sample 20% of PUMAs within each state (at least one per state).
  2. Within each sampled PUMA draw 30 households with probability proportional
     to wgtp, without replacement. These are the training data.
  3. Target population = all households in the non-sampled PUMAs.
  4. Fit log-welfare predictors (lgbm, ols) on two feature regimes
     (unit-level u_* and area-context-only c_*), form predicted welfare for the
     target households, and compute N, C1-C4 estimates of Gini, MLD, Theil,
     FGT0 and FGT1 at PUMA, state and national level. Truth uses actual y.

Coverage (separate stage): PUMA-cluster bootstrap of the training sample
(lgbm only, estimators N and C4, national and state Gini and MLD), percentile
90% intervals.

Outputs (replication/output/):
  e2_results_raw.csv   long format, one row per (rep, regime, model, estimator, level, unit, index)
  e2_summary.csv       relative bias, relative RMSE, Spearman, share below, by cell
  e2_coverage.csv      bootstrap 90% interval coverage (lgbm, N and C4)
  e2_notes.txt         design actually used, runtimes, deviations, sanity checks
  e2_table_main.tex    booktabs rows (lgbm, PUMA level)
  Fig_e2_area_gini.pdf estimated vs true PUMA Gini for N and C4, one replication

Intermediate per-replication results are written to replication/data/e2_checkpoints
after each replication, and existing checkpoints are reused on restart.

Usage: python3 e2_design_based.py [--stage all|main|coverage|report]
                                  [--reps 20] [--cov-reps 10] [--B 30]
                                  [--cov-deadline-min 0] [--ckpt-dir DIR] [--out-dir DIR]
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REP_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REP_ROOT, "src"))
import ineq  # noqa: E402
import estimators  # noqa: E402
from estimators import predict_welfare_draws, ESTIMATORS  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from quantile_forest import RandomForestQuantileRegressor  # noqa: E402

LGB_THREADS = 2             # see the thread note in the notes file
C3_GRID = np.arange(1, 20) / 20.0   # 19 quantile levels 0.05, ..., 0.95

DATA_PATH = os.path.join(REP_ROOT, "data", "acs", "acs2023_households.parquet")
DEFAULT_OUT = os.path.join(REP_ROOT, "output")
DEFAULT_CKPT = os.path.join(REP_ROOT, "data", "e2_checkpoints")

SEED_ROOT = 20260601
SAMPLE_FRAC = 0.20          # share of PUMAs sampled within each state
N_PER_PUMA = 30             # households per sampled PUMA
POV_SHARE = 0.6             # poverty line = 0.6 x weighted national median
R_MAIN = 10                 # simulation draws for C2-C4 in the main exercise
R_COV = 3                   # simulation draws for C4 inside the bootstrap
N_FOLDS = 5                 # cross-fitting folds (as in estimators.py)
MODELS = ("lgbm", "ols")
REGIMES = ("unit", "context")
LEVELS = ("puma", "state", "national")
INDICES = ("gini", "mld", "theil", "fgt0", "fgt1")
COV_ESTIMATORS = ("N", "C4")
COV_INDICES = ("gini", "mld")
NOMINAL = 0.90

T0 = time.time()
LOG_PATH = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")


# ----------------------------------------------------------------- data
class Data:
    pass


def load_data():
    df = pd.read_parquet(DATA_PATH)
    ucols = [c for c in df.columns if c.startswith("u_")]
    ccols = [c for c in df.columns if c.startswith("c_")]
    D = Data()
    D.y = df["y"].to_numpy(dtype=float)
    D.w = df["wgtp"].to_numpy(dtype=float)
    D.st = df["st"].to_numpy(dtype=np.int64)
    D.area = df["area"].to_numpy(dtype=np.int64)
    D.Xu = df[ucols].to_numpy(dtype=float)
    D.Xc = df[ccols].to_numpy(dtype=float)
    D.ucols, D.ccols = ucols, ccols
    assert np.all(D.area // 100000 == D.st), "area must equal st*100000 + puma"
    assert np.all(D.y > 0) and np.all(D.w > 0)
    D.z = POV_SHARE * ineq.wquantile(D.y, 0.5, D.w)

    # PUMA index structure: rows of each PUMA, and PUMAs of each state
    order = np.argsort(D.area, kind="stable")
    codes = np.unique(D.area)
    starts = np.searchsorted(D.area[order], codes, side="left")
    ends = np.searchsorted(D.area[order], codes, side="right")
    D.area_codes = codes
    D.area_rows = {int(a): order[s:e] for a, s, e in zip(codes, starts, ends)}
    D.state_areas = {int(s): codes[codes // 100000 == s] for s in np.unique(D.st)}
    return D


def seg_indices(v, w, seg, nseg, z):
    """Weighted relative indices for nseg groups in one pass.

    v   : positive welfare (n,)
    w   : weights (n,), normalised within group here
    seg : integer group label in 0..nseg-1 (every label must occur)
    Returns (nseg, 5) array with columns gini, mld, theil, fgt0, fgt1.
    Same formulas as ineq.gini / mld / theil / fgt (checked in self_check).
    """
    W = np.bincount(seg, weights=w, minlength=nseg)
    wn = w / W[seg]
    o = np.lexsort((v, seg))                 # sort by group, then by welfare
    vs, ws, ss = v[o], wn[o], seg[o]
    cw = np.cumsum(ws)
    F = cw - ws / 2.0 - ss                   # each finished group sums to 1
    mu = np.bincount(ss, weights=ws * vs, minlength=nseg)
    Fbar = np.bincount(ss, weights=ws * F, minlength=nseg)
    gini = 2.0 * np.bincount(ss, weights=ws * vs * (F - Fbar[ss]), minlength=nseg) / mu
    r = vs / mu[ss]
    lr = np.log(r)
    mld = -np.bincount(ss, weights=ws * lr, minlength=nseg)
    theil = np.bincount(ss, weights=ws * r * lr, minlength=nseg)
    fgt0 = np.bincount(ss, weights=ws * (vs < z), minlength=nseg)
    g = np.clip((z - vs) / z, 0.0, None)
    fgt1 = np.bincount(ss, weights=ws * g, minlength=nseg)
    return np.column_stack([gini, mld, theil, fgt0, fgt1])


def self_check(D):
    """Checks the segmented index code against ineq.py, and the reference numbers
    from e2_data_summary.txt."""
    rng = np.random.default_rng(7)
    idx = rng.choice(len(D.y), 60000, replace=False)
    seg = rng.integers(0, 5, size=len(idx))
    v, w = D.y[idx], D.w[idx]
    got = seg_indices(v, w, seg, 5, D.z)
    worst = 0.0
    for g in range(5):
        m = seg == g
        ref = np.array([ineq.gini(v[m], w[m]), ineq.mld(v[m], w[m]), ineq.theil(v[m], w[m]),
                        ineq.fgt(v[m], D.z, 0, w[m]), ineq.fgt(v[m], D.z, 1, w[m])])
        worst = max(worst, float(np.max(np.abs(got[g] - ref) / np.maximum(np.abs(ref), 1e-12))))
    assert worst < 1e-8, f"segmented indices disagree with ineq.py (rel diff {worst:.2e})"
    nat_gini = ineq.gini(D.y, D.w)
    med = ineq.wquantile(D.y, 0.5, D.w)
    log(f"self-check: segmented indices match ineq.py (max rel diff {worst:.1e})")
    log(f"population: N={len(D.y):,} households, sum wgtp={D.w.sum():,.0f}, "
        f"PUMAs={len(D.area_codes)}, states={len(D.state_areas)}")
    log(f"population: weighted median y={med:,.2f} (summary file: 79,522.40), "
        f"z={D.z:,.2f}, national Gini={nat_gini:.4f} (summary file: 0.4739)")
    return {"self_check_max_rel_diff": worst, "median_y": med, "z": D.z,
            "national_gini": nat_gini}


# ------------------------------------------------------------- design
def draw_sample(D, rng):
    """Stage 1: 20% of PUMAs per state (at least 1). Stage 2: 30 households per
    sampled PUMA, probability proportional to wgtp, without replacement."""
    sampled = []
    for s in sorted(D.state_areas):
        codes = D.state_areas[s]
        n_s = max(1, int(np.floor(SAMPLE_FRAC * len(codes) + 0.5)))
        sampled.append(rng.choice(codes, size=n_s, replace=False))
    sampled = np.sort(np.concatenate(sampled))
    train = []
    for a in sampled:
        rows = D.area_rows[int(a)]
        p = D.w[rows] / D.w[rows].sum()
        train.append(rng.choice(rows, size=N_PER_PUMA, replace=False, p=p))
    return sampled, np.concatenate(train)


def target_positions(D, sampled):
    """Households in non-sampled PUMAs, sorted by area (so PUMAs and states are
    contiguous blocks)."""
    tgt = np.flatnonzero(~np.isin(D.area, sampled))
    return tgt[np.argsort(D.area[tgt], kind="stable")]


def group_labels(D, tgt):
    a_u, seg_a = np.unique(D.area[tgt], return_inverse=True)
    s_u, seg_s = np.unique(D.st[tgt], return_inverse=True)
    seg_n = np.zeros(len(tgt), dtype=np.int64)
    return {
        "puma": (seg_a.ravel(), len(a_u), a_u),
        "state": (seg_s.ravel(), len(s_u), s_u),
        "national": (seg_n, 1, np.array([0])),
    }


def c3_log_quantiles(X_tr, ylog_tr, X_te, seed, regime, chunk=5000):
    """Conditional log-quantiles on the grid C3_GRID from a quantile regression
    forest with the settings of estimators.fit_qrf_quantiles (300 trees,
    min_samples_leaf=20). Returns an (n_te, 19) array.
    In the context regime the target rows take one value per PUMA, so the
    forest is evaluated on distinct rows and expanded (identical result)."""
    qrf = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=20,
                                        n_jobs=LGB_THREADS, random_state=seed + 7919)
    qrf.fit(X_tr, ylog_tr)
    if regime == "context":
        Xq, inv = np.unique(X_te, axis=0, return_inverse=True)
        inv = inv.ravel()
    else:
        Xq, inv = X_te, None
    out = np.empty((len(Xq), len(C3_GRID)))
    for s in range(0, len(Xq), chunk):
        out[s:s + chunk] = qrf.predict(Xq[s:s + chunk], quantiles=list(C3_GRID))
    return out if inv is None else out[inv]


def c3_draws(Q, rng, R):
    """R draws of welfare from the conditional quantile function implied by the
    19 grid points: linear interpolation in probability between 0.05 and 0.95,
    constant beyond the end points (so the tails below 5% and above 95% are
    not represented). Returns an (R, n) array."""
    n, G = Q.shape
    rows = np.arange(n)
    step = C3_GRID[1] - C3_GRID[0]
    out = np.empty((R, n))
    for d in range(R):
        u = rng.random(n)
        pos = np.clip((u - C3_GRID[0]) / step, 0.0, G - 1.0)
        j = np.minimum(np.floor(pos).astype(int), G - 2)
        frac = pos - j
        lo, hi = Q[rows, j], Q[rows, j + 1]
        out[d] = np.exp(lo + frac * (hi - lo))
    return out


def cov_draws_N_C4(X_tr, y_tr, X_te, inv, rng, seed, fold_id, R):
    """N and C4 for one bootstrap sample, as in estimators.predict_welfare_draws
    (same steps and random-number use). X_te holds the distinct target rows
    when inv is not None, and inv maps households to those rows, so each
    distinct row is predicted once; the draws are still per household."""
    ylog_tr = np.log(y_tr)
    oof_log, te_u = estimators.oof_and_test_predictions_folds(
        X_tr, ylog_tr, X_te, "lgbm", seed, fold_id)
    te_log = te_u if inv is None else te_u[inv]
    resid = ylog_tr - oof_log
    smear = float(np.mean(np.exp(resid)))
    n_te = len(te_log)
    out = {"N": (np.exp(te_log) * smear)[None, :]}
    cuts = np.quantile(oof_log, np.linspace(0.1, 0.9, 9))
    b_tr = np.digitize(oof_log, cuts)
    b_te = np.digitize(te_log, cuts)
    order = np.argsort(b_tr, kind="stable")
    pool = resid[order]
    counts = np.bincount(b_tr, minlength=10)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    draws = []
    for _ in range(R):
        u = rng.random(n_te)
        idx = starts[b_te] + np.minimum((u * counts[b_te]).astype(int), counts[b_te] - 1)
        draws.append(np.exp(te_log + pool[idx]))
    out["C4"] = np.stack(draws)
    return out


def mean_indices(V, w, seg, G, z):
    acc = 0.0
    for d in range(V.shape[0]):
        acc = acc + seg_indices(V[d], w, seg, G, z)
    return acc / V.shape[0]


def ckpt_path(ckdir, kind, r):
    return os.path.join(ckdir, f"{kind}_rep{r:02d}.pkl")


# ------------------------------------------------------------ main run
def run_main_rep(D, r):
    t_start = time.time()
    seed = SEED_ROOT + 10000 * r
    rng_s = np.random.default_rng([SEED_ROOT, r, 1])
    sampled, train = draw_sample(D, rng_s)
    tgt = target_positions(D, sampled)
    labels = group_labels(D, tgt)
    y_tr, y_te, w_te = D.y[train], D.y[tgt], D.w[tgt]
    ylog_tr = np.log(y_tr)

    truth = {lev: seg_indices(y_te, w_te, labels[lev][0], labels[lev][1], D.z)
             for lev in LEVELS}
    meta = {"rep": r, "n_sampled_puma": int(len(sampled)), "n_train": int(len(train)),
            "n_target_hh": int(len(tgt)), "n_target_puma": int(len(np.unique(D.area[tgt]))),
            "n_target_state": int(len(labels["state"][2])), "fit": {}, "timing_s": {},
            "sanity": {}}
    frames = []
    for ri, regime in enumerate(REGIMES):
        X = D.Xu if regime == "unit" else D.Xc
        X_tr, X_te = X[train], X[tgt]
        t = time.time()
        Q3 = c3_log_quantiles(X_tr, ylog_tr, X_te, seed, regime)
        V3 = c3_draws(Q3, np.random.default_rng([SEED_ROOT, r, 5, ri]), R_MAIN)  # C3 ignores the model
        del Q3
        meta["timing_s"][f"qrf_{regime}"] = round(time.time() - t, 1)
        for mi, model in enumerate(MODELS):
            t = time.time()
            rng_sim = np.random.default_rng([SEED_ROOT, r, 2, ri, mi])
            dr = predict_welfare_draws(X_tr, y_tr, X_te, model, R_MAIN, rng_sim,
                                       seed=seed, which=("N", "C1", "C2", "C4"))
            dr["C3"] = V3
            meta["fit"][f"{regime}_{model}"] = dr["fit"]
            for est in ESTIMATORS:
                V = dr[est]
                for lev in LEVELS:
                    seg, G, units = labels[lev]
                    est_idx = mean_indices(V, w_te, seg, G, D.z)
                    frames.append(pd.DataFrame({
                        "rep": r, "regime": regime, "model": model,
                        "estimator": est, "level": lev,
                        "unit": np.repeat(units, len(INDICES)),
                        "index": np.tile(INDICES, G),
                        "truth": truth[lev].ravel(),
                        "estimate": est_idx.ravel()}))
                    if lev == "puma" and regime == "context" and model == "lgbm" \
                            and est in ("N", "C1"):
                        meta["sanity"][f"max_abs_within_puma_gini_{est}_context"] = \
                            float(np.max(np.abs(est_idx[:, 0])))
            meta["timing_s"][f"{regime}_{model}"] = round(time.time() - t, 1)
            log(f"rep {r}: {regime}/{model} done in {meta['timing_s'][f'{regime}_{model}']} s")
    rows = pd.concat(frames, ignore_index=True)
    meta["timing_s"]["total"] = round(time.time() - t_start, 1)
    with open(ckpt_path(D.ckdir, "main", r), "wb") as fh:
        pickle.dump({"rows": rows, "meta": meta}, fh)
    log(f"rep {r}: saved ({len(rows):,} rows), total {meta['timing_s']['total']} s")


# ------------------------------------------------------- coverage run
def run_cov_rep(D, r, B):
    t_start = time.time()
    seed = SEED_ROOT + 10000 * r
    rng_s = np.random.default_rng([SEED_ROOT, r, 1])
    sampled, train = draw_sample(D, rng_s)          # same sample as the main run
    tgt = target_positions(D, sampled)
    labels = group_labels(D, tgt)
    y_tr, y_te, w_te = D.y[train], D.y[tgt], D.w[tgt]
    z = D.z
    truth_n = seg_indices(y_te, w_te, labels["national"][0], 1, z)[0, :2]
    seg_s, Gs, s_u = labels["state"]
    truth_s = seg_indices(y_te, w_te, seg_s, Gs, z)[:, :2]

    rng_c = np.random.default_rng([SEED_ROOT, r, 3])
    # household-level fold labels: copies of a household in a bootstrap sample
    # always share a fold, so a duplicate never predicts itself out of fold
    fold_train = rng_c.permutation(np.arange(len(train)) % N_FOLDS)
    train_area = D.area[train]
    pos_by_puma = [np.flatnonzero(train_area == a) for a in sampled]
    K = len(sampled)

    rows = []
    for ri, regime in enumerate(REGIMES):
        X = D.Xu if regime == "unit" else D.Xc
        X_tr, X_te = X[train], X[tgt]
        if regime == "context":        # one prediction per distinct PUMA-level row
            Xq, inv = np.unique(X_te, axis=0, return_inverse=True)
            inv = inv.ravel()
        else:
            Xq, inv = X_te, None
        boot_n = np.empty((B, len(COV_ESTIMATORS), len(COV_INDICES)))
        boot_s = np.empty((B, len(COV_ESTIMATORS), Gs, len(COV_INDICES)))
        for b in range(B):
            pick = rng_c.integers(0, K, size=K)
            pos = np.concatenate([pos_by_puma[k] for k in pick])
            rng_sim = np.random.default_rng([SEED_ROOT, r, 4, ri, b])
            dr = cov_draws_N_C4(X_tr[pos], y_tr[pos], Xq, inv, rng_sim,
                                seed + 100 + b, fold_train[pos], R_COV)
            for ei, est in enumerate(COV_ESTIMATORS):
                V = dr[est]
                boot_n[b, ei] = mean_indices(V, w_te, labels["national"][0], 1, z)[0, :2]
                boot_s[b, ei] = mean_indices(V, w_te, seg_s, Gs, z)[:, :2]
        for ei, est in enumerate(COV_ESTIMATORS):
            lo, hi = np.percentile(boot_n[:, ei, :], [5, 95], axis=0)
            mid = boot_n[:, ei, :].mean(axis=0)
            for ii, idx in enumerate(COV_INDICES):
                t_v = truth_n[ii]
                rows.append(dict(rep=r, regime=regime, model="lgbm", estimator=est,
                                 level="national", unit=0, index=idx, truth=t_v,
                                 lo=lo[ii], hi=hi[ii], boot_mean=mid[ii],
                                 covered=bool(lo[ii] <= t_v <= hi[ii])))
            lo_s, hi_s = np.percentile(boot_s[:, ei], [5, 95], axis=0)
            mid_s = boot_s[:, ei].mean(axis=0)
            for g in range(Gs):
                for ii, idx in enumerate(COV_INDICES):
                    t_v = truth_s[g, ii]
                    rows.append(dict(rep=r, regime=regime, model="lgbm", estimator=est,
                                     level="state", unit=int(s_u[g]), index=idx, truth=t_v,
                                     lo=lo_s[g, ii], hi=hi_s[g, ii], boot_mean=mid_s[g, ii],
                                     covered=bool(lo_s[g, ii] <= t_v <= hi_s[g, ii])))
        log(f"rep {r}: coverage {regime} done ({B} bootstrap fits)")
    out = pd.DataFrame(rows)
    elapsed = round(time.time() - t_start, 1)
    with open(ckpt_path(D.ckdir, "cov", r), "wb") as fh:
        pickle.dump({"rows": out, "elapsed_s": elapsed, "B": B}, fh)
    log(f"rep {r}: coverage saved, {elapsed} s")


# ------------------------------------------------------------- reports
def load_checkpoints(ckdir, kind):
    files = sorted(f for f in os.listdir(ckdir) if f.startswith(kind + "_rep") and f.endswith(".pkl"))
    out = []
    for f in files:
        with open(os.path.join(ckdir, f), "rb") as fh:
            out.append(pickle.load(fh))
    return out


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.a."
    s = f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def write_reports(D, outdir, ckdir, meta_extra, cov_done_reps):
    mains = load_checkpoints(ckdir, "main")
    if not mains:
        raise SystemExit("no main-run checkpoints found")
    raw = pd.concat([m["rows"] for m in mains], ignore_index=True)
    metas = [m["meta"] for m in mains]
    reps_done = sorted(raw["rep"].unique().tolist())
    log(f"reporting: {len(reps_done)} main replications")

    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, "e2_results_raw.csv")
    raw.to_csv(raw_path, index=False, float_format="%.6g")
    log(f"wrote {raw_path} ({len(raw):,} rows)")

    # ---- summary
    raw = raw.copy()
    raw["rel_err"] = np.where(raw["truth"] > 0, (raw["estimate"] - raw["truth"]) / raw["truth"], np.nan)
    keys = ["regime", "model", "estimator", "level", "index"]
    g = raw.groupby(keys)
    summ = g.agg(n_reps=("rep", "nunique"), n_units=("rel_err", "size"),
                 mean_truth=("truth", "mean"), mean_estimate=("estimate", "mean"),
                 rel_err_mean=("rel_err", "mean"),
                 rel_err_sq=("rel_err", lambda x: np.nanmean(np.square(x)))).reset_index()
    summ["mean_rel_bias_pct"] = 100 * summ["rel_err_mean"]
    summ["rel_rmse_pct"] = 100 * np.sqrt(summ["rel_err_sq"])
    summ["n_units_per_rep"] = summ["n_units"] / summ["n_reps"]
    summ = summ.drop(columns=["rel_err_mean", "rel_err_sq"])

    pg = raw[(raw["level"] == "puma") & (raw["index"] == "gini")]
    sp_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for keyv, sub in pg.groupby(["rep", "regime", "model", "estimator"]):
            est_v, tru_v = sub["estimate"].to_numpy(), sub["truth"].to_numpy()
            # constant predictions within PUMA (N, C1 in the context regime): rank correlation undefined
            degenerate = np.ptp(est_v) < 1e-10 * max(1.0, float(np.abs(tru_v).max()))
            rho = np.nan if degenerate else spearmanr(est_v, tru_v).statistic
            below = float(np.mean(est_v < tru_v))
            sp_rows.append(dict(zip(["rep", "regime", "model", "estimator"], keyv),
                                spearman=float(rho), share_below=below))
    sp = pd.DataFrame(sp_rows)
    sp_agg = sp.groupby(["regime", "model", "estimator"]).agg(
        spearman_gini_puma=("spearman", "mean"),
        spearman_nan_reps=("spearman", lambda x: int(np.isnan(x).sum())),
        share_gini_below_puma=("share_below", "mean")).reset_index()
    summ = summ.merge(sp_agg, on=["regime", "model", "estimator"], how="left")
    summ.loc[~((summ["level"] == "puma") & (summ["index"] == "gini")),
             ["spearman_gini_puma", "spearman_nan_reps", "share_gini_below_puma"]] = np.nan
    sum_path = os.path.join(outdir, "e2_summary.csv")
    summ.to_csv(sum_path, index=False, float_format="%.6g")
    log(f"wrote {sum_path}")

    # ---- main LaTeX table (lgbm, PUMA level)
    lines = []
    est_label = {"N": "N", "C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4"}
    reg_label = {"unit": "Unit-level", "context": "Area-context"}
    for ri, regime in enumerate(REGIMES):
        if ri > 0:
            lines.append("\\midrule")
        for est in ESTIMATORS:
            cell = summ[(summ.regime == regime) & (summ.model == "lgbm") &
                        (summ.estimator == est) & (summ.level == "puma")]
            vals = {r_["index"]: r_["mean_rel_bias_pct"] for _, r_ in cell.iterrows()}
            sp_v = cell.loc[cell["index"] == "gini", "spearman_gini_puma"]
            sp_v = float(sp_v.iloc[0]) if len(sp_v) else np.nan
            cells = [fmt(vals.get(i, np.nan), 1) for i in ("gini", "mld", "theil", "fgt0", "fgt1")]
            sp_s = "n.a." if np.isnan(sp_v) else f"{sp_v:.2f}".replace("-", "$-$")
            label = reg_label[regime] if est == ESTIMATORS[0] else ""
            lines.append(" & ".join([label, est_label[est]] + cells + [sp_s]) + " \\\\")
    tex_path = os.path.join(outdir, "e2_table_main.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {tex_path}")

    # ---- coverage
    cov_metas = load_checkpoints(ckdir, "cov")
    cov_path = os.path.join(outdir, "e2_coverage.csv")
    if cov_metas:
        cov = pd.concat([c["rows"] for c in cov_metas], ignore_index=True)
        cov["width_rel_pct"] = 100 * (cov["hi"] - cov["lo"]) / cov["truth"].abs()
        cg = cov.groupby(["regime", "model", "estimator", "level", "index"]).agg(
            n_reps=("rep", "nunique"), n_units=("covered", "size"),
            coverage=("covered", "mean"),
            mean_width_rel_pct=("width_rel_pct", "mean")).reset_index()
        cg["nominal"] = NOMINAL
        cg["B"] = cov_metas[0]["B"]
        cg.to_csv(cov_path, index=False, float_format="%.4f")
        log(f"wrote {cov_path} ({len(cov_metas)} coverage replications)")
    else:
        cg = None
        log("no coverage checkpoints; e2_coverage.csv not written")

    # ---- figure: representative replication, PUMA-level Gini, lgbm
    rep_fig = reps_done[0]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 3.1), sharex=True, sharey=True)
    panel_names = {"unit": "(a) Unit-level predictors", "context": "(b) Area-context predictors"}
    style = {"N": dict(marker="o", s=7, facecolors="0.6", edgecolors="0.2", linewidths=0.3),
             "C4": dict(marker="o", s=7, facecolors="none", edgecolors="k", linewidths=0.5)}
    lim_max = 0.0
    for ax, regime in zip(axes, REGIMES):
        base = raw[(raw.rep == rep_fig) & (raw.regime == regime) & (raw.model == "lgbm") &
                   (raw.level == "puma") & (raw["index"] == "gini")]
        t_v = base[base.estimator == "N"].sort_values("unit")["truth"].to_numpy()
        lim_max = max(lim_max, float(t_v.max()))
        for est in ("N", "C4"):
            e_v = base[base.estimator == est].sort_values("unit")["estimate"].to_numpy()
            lim_max = max(lim_max, float(e_v.max()))
            ax.scatter(t_v, e_v, label=("N (naive)" if est == "N" else "C4 (conditional bootstrap)"),
                       **style[est])
        ax.text(0.03, 0.97, panel_names[regime], transform=ax.transAxes, va="top", ha="left")
        ax.set_xlabel("True PUMA Gini")
    axes[0].set_ylabel("Estimated PUMA Gini")
    hi_lim = lim_max * 1.05
    for ax in axes:
        ax.plot([0, hi_lim], [0, hi_lim], color="k", lw=0.7, ls="--", zorder=0)
        ax.set_xlim(0, hi_lim)
        ax.set_ylim(0, hi_lim)
        ax.set_aspect("equal", adjustable="box")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig_path = os.path.join(outdir, "Fig_e2_area_gini.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {fig_path} (replication {rep_fig})")

    # ---- sanity numbers
    ctx_n = raw[(raw.regime == "context") & (raw.estimator == "N") & (raw.level == "puma") &
                (raw["index"] == "gini")]
    ctx_c1 = raw[(raw.regime == "context") & (raw.estimator == "C1") & (raw.level == "puma") &
                 (raw["index"] == "gini")]
    max_abs_N = float(ctx_n["estimate"].abs().max())
    max_abs_C1 = float(ctx_c1["estimate"].abs().max())
    max_abs_N_mld = float(raw[(raw.regime == "context") & (raw.estimator == "N") &
                              (raw.level == "puma") & (raw["index"] == "mld")]["estimate"].abs().max())
    return {"raw_path": raw_path, "sum_path": sum_path, "tex_path": tex_path, "cov_path": cov_path,
            "fig_path": fig_path, "reps_done": reps_done, "summary": summ, "coverage": cg,
            "metas": metas, "max_abs_N_gini_context": max_abs_N,
            "max_abs_C1_gini_context": max_abs_C1, "max_abs_N_mld_context": max_abs_N_mld,
            "cov_reps_done": sorted({int(x) for c in cov_metas for x in c["rows"]["rep"].unique()}),
            "cov_elapsed": [c["elapsed_s"] for c in cov_metas]}


def write_notes(path, info, args, sc, n_main_total_s, n_cov_total_s):
    s = info["summary"]
    metas = info["metas"]
    lines = []
    add = lines.append
    add("E2 design-based validation: notes (generated by e2_design_based.py)")
    add("=" * 72)
    add("Data: replication/data/acs/acs2023_households.parquet (ACS 2023 1-year PUMS,")
    add("  occupied units with HINCP > 0), 1,326,716 households, 2,462 PUMAs, 51 states/DC.")
    add("  Treated as a census; weights wgtp; y = adjusted HINCP (2023 dollars).")
    add(f"  Population weighted median of y = {sc['median_y']:,.2f}; z = 0.6 x median = {sc['z']:,.2f}.")
    add(f"  National weighted Gini of y = {sc['national_gini']:.4f}.")
    add("")
    add("Design actually used")
    add(f"  Replications (main exercise): {len(info['reps_done'])} (seeds SEED_ROOT + 10000 r, SEED_ROOT = 20260601).")
    add(f"  Reduced from the target design (main 20, coverage 10) for run time: main {len(info['reps_done'])}, "
        f"coverage {len(info['cov_reps_done'])}.")
    add(f"  Sampling stage 1: {int(SAMPLE_FRAC*100)}% of PUMAs within each state, rounded, at least 1 per state,")
    add("    drawn uniformly without replacement.")
    add(f"  Sampling stage 2: {N_PER_PUMA} households per sampled PUMA, probability proportional to wgtp,")
    add("    without replacement (numpy choice with p). Training fits are unweighted.")
    add("  Target population: all households in non-sampled PUMAs (about 80% of the population).")
    add("  Predictor regimes: unit = all u_* columns (34); context = all c_* columns (13),")
    add("    constant within PUMA.")
    add("  Models: lgbm (estimators.make_model 'lgbm', main) and ols (LinearRegression), on log y.")
    add("  Estimators: N, C1 (linear recalibration), C2 (pooled residual simulation), C3 (quantile")
    add("    regression forest, 19 quantile levels 0.05-0.95, see deviations), C4 (decile residual")
    add(f"    bootstrap). R = {R_MAIN} draws for C2, C3 and C4 in the main exercise.")
    add("    Why R = 10 and not 100: cost. Each draw is one pass over about 1.06M target households at")
    add("    three aggregation levels; R = 100 would have multiplied the run time by about 10 on this")
    add("    machine. The simulation variability of the averaged indices was not measured separately.")
    add("    Coverage uses R = 3 (task specification).")
    add("  Indices: Gini, MLD, Theil, FGT0, FGT1 (z fixed), weighted by wgtp.")
    add("  Truth: indices of actual y in the same target population of the same replication.")
    add("  Metrics: relative bias = mean over (units, reps) of (est - true)/true, in %;")
    add("    relative RMSE = sqrt(mean over (units, reps) of ((est - true)/true)^2), in %.")
    add("    Area level units are non-sampled PUMAs; state and national levels pool over states/reps.")
    add("    Spearman: rank correlation of estimated and true PUMA Gini across PUMAs, per rep, then")
    add("    averaged over reps. share_gini_below_puma: share of (PUMA, rep) pairs with est < true.")
    add("    Relative errors with a zero truth (none expected) are excluded.")
    add("")
    add("Cross-fitting: estimators.py uses 5-fold random KFold on the training rows (unchanged).")
    add("Estimates use predict_welfare_draws (added to estimators.py); it follows estimate_indices")
    add("  step by step, with the same random-number order (existing functions unchanged).")
    add("")
    add("Sample sizes")
    for m in metas[:3]:
        add(f"  rep {m['rep']}: sampled PUMAs {m['n_sampled_puma']}, training households {m['n_train']}, "
            f"target households {m['n_target_hh']:,}, target PUMAs {m['n_target_puma']}, states {m['n_target_state']}")
    add("")
    add("Runtimes (wall clock, shared 4-core machine with another job)")
    for m in metas:
        t = m["timing_s"]
        add(f"  rep {m['rep']}: total {t['total']} s; QRF unit {t.get('qrf_unit')} s, "
            f"QRF context {t.get('qrf_context')} s; lgbm unit {t.get('unit_lgbm')} s, "
            f"ols unit {t.get('unit_ols')} s, lgbm context {t.get('context_lgbm')} s, "
            f"ols context {t.get('context_ols')} s")
    add(f"  main exercise total (sum of replications): {n_main_total_s/60:.1f} min")
    add("")
    add("Coverage (bootstrap)")
    if info["coverage"] is not None:
        add(f"  Replications: {info['cov_reps_done']} (B = {info['coverage']['B'].iloc[0]}, R = {R_COV}).")
        add("  PUMA-cluster bootstrap: sampled PUMAs resampled with replacement; training households of")
        add("  a duplicated PUMA are duplicated; cross-fitting folds are assigned per household so")
        add("  that copies of the same household share a fold. Percentile 5-95 interval.")
        add("  The bootstrap resamples the training sample only: the target population of each rep is")
        add("  held fixed, so intervals are conditional on the set of non-sampled PUMAs.")
        add(f"  Coverage run time: {n_cov_total_s/60:.1f} min total.")
    else:
        add("  Not run.")
    add("")
    add("Sanity checks")
    add(f"  Context regime, N: max |within-PUMA Gini| over all PUMAs and reps = {info['max_abs_N_gini_context']:.3e}")
    add(f"  Context regime, N: max |within-PUMA MLD| = {info['max_abs_N_mld_context']:.3e}")
    add(f"  Context regime, C1: max |within-PUMA Gini| = {info['max_abs_C1_gini_context']:.3e}")
    add("  (N and C1 are affine in one PUMA-constant prediction, so their within-PUMA inequality is zero.)")
    add(f"  Self-check of segmented index code against ineq.py: max relative difference {sc['self_check_max_rel_diff']:.1e}.")
    add("  predict_welfare_draws (default options, 99-quantile C3) vs estimate_indices (same seed, 3,000")
    add("    training and 20,000 target households, unit features, model ols): indices of N, C1-C4 agree")
    add("    exactly (differences 0 at double precision). The driver itself uses the 19-quantile C3.")
    add("    The lgbm version of that check was started but stopped for time; the lgbm code path is identical.")
    add("")
    add("Deviations and limitations")
    add("  - C3 uses 19 quantile levels (0.05, 0.10, ..., 0.95) instead of the 99 levels of")
    add("    estimators.fit_qrf_quantiles, with linear interpolation in probability between grid points")
    add("    and constant values beyond 0.05 and 0.95. Reason: predicting 99 quantiles for about 1.06M")
    add("    target households took several minutes per replication on the shared machine. The tails")
    add("    below the 5th and above the 95th percentile are therefore not represented. C3 inequality is")
    add("    likely biased downward relative to a full 99-point forest for this reason; the size of that")
    add("    effect was not measured.")
    add("  - LightGBM runs with 2 threads; estimators.make_model specifies 4. The script sets the thread")
    add("    count at run time, because a second job shared the 4 cores and 4-thread fits took 10-90 s")
    add("    instead of under 1 s. The model is the same; floating-point results may differ in the last digits.")
    add("  - The quantile forest also uses 2 threads, with the settings of fit_qrf_quantiles.")
    add("  - predict_welfare_draws (added to estimators.py) returns predicted welfare draws instead of")
    add("    aggregate indices, because area-level indices need each household's predicted welfare.")
    add("    estimate_indices itself is not called in the experiment.")
    add("  - Population is the 1.33M occupied, positive-income households of the 2023 1-year file")
    add("    (not the full 2023 PUMS with group quarters). Households are the unit; no person weights.")
    add("  - Context features are PUMA means of census variables, as in the design, not satellite data.")
    add("  - Area-level Spearman is computed on PUMAs; NaN if the estimate is constant (N, C1 in context).")
    add("  - Target PUMAs are fixed per replication by the sample; truth varies across replications.")
    add("")
    add("Files")
    add(f"  {info['raw_path']}")
    add(f"  {info['sum_path']}")
    add(f"  {info['tex_path']}")
    add(f"  {info['cov_path'] if info['coverage'] is not None else '(coverage not written)'}")
    add(f"  {info['fig_path']}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {path}")


def main():
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "main", "coverage", "report"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cov-reps", type=int, default=10)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--cov-deadline-min", type=float, default=0.0,
                    help="stop starting coverage replications after this many minutes (0 = no limit)")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--lgb-threads", type=int, default=LGB_THREADS)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    LOG_PATH = os.path.join(args.ckpt_dir, "e2_run.log")
    log(f"start: stage={args.stage} reps={args.reps} cov_reps={args.cov_reps} B={args.B} "
        f"lgb_threads={args.lgb_threads}")

    # estimators.make_model fixes LightGBM at n_jobs=4. On this shared machine that
    # oversubscribes the cores and fits took 10-90 s instead of about 0.6 s, so the
    # thread count is set to args.lgb_threads at run time. The model is unchanged.
    _make_model_orig = estimators.make_model

    def _make_model_threads(name, seed):
        m = _make_model_orig(name, seed)
        if isinstance(m, LGBMRegressor):
            m.set_params(n_jobs=args.lgb_threads)
        return m

    estimators.make_model = _make_model_threads
    D = load_data()
    D.ckdir = args.ckpt_dir
    sc = self_check(D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.stage in ("all", "main"):
            for r in range(args.reps):
                if os.path.exists(ckpt_path(args.ckpt_dir, "main", r)):
                    log(f"rep {r}: checkpoint exists, skipped")
                    continue
                run_main_rep(D, r)
        if args.stage in ("all", "coverage"):
            deadline = T0 + args.cov_deadline_min * 60 if args.cov_deadline_min > 0 else None
            for r in range(args.cov_reps):
                if os.path.exists(ckpt_path(args.ckpt_dir, "cov", r)):
                    log(f"coverage rep {r}: checkpoint exists, skipped")
                    continue
                if deadline is not None and time.time() > deadline:
                    log(f"coverage: time limit reached before rep {r}; stopping")
                    break
                run_cov_rep(D, r, args.B)
        if args.stage in ("all", "report"):
            mains = load_checkpoints(args.ckpt_dir, "main")
            covs = load_checkpoints(args.ckpt_dir, "cov")
            n_main = sum(m["meta"]["timing_s"]["total"] for m in mains)
            n_cov = sum(c["elapsed_s"] for c in covs)
            info = write_reports(D, args.out_dir, args.ckpt_dir, None, None)
            write_notes(os.path.join(args.out_dir, "e2_notes.txt"), info, args, sc, n_main, n_cov)
    log("finished")


if __name__ == "__main__":
    main()
