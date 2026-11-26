#!/usr/bin/env python3
"""E1 Monte Carlo: bias of inequality and poverty indices computed from
ML-predicted welfare, with estimators N, C1-C4 (see replication/src/estimators.py).

Outputs (replication/output/):
  e1_results_raw.csv   long format, one row per (cell, rep, model, estimator, index)
  e1_summary.csv       bias, relative bias (%), RMSE by cell x model x estimator x index
  e1_prop3_table.csv   lognormal Gini: true vs predicted (Proposition 3)
  e1_checks.txt        numerical checks of Propositions 1-3, DGP calibration, run settings
  Fig_e1_bias.pdf      relative bias of Gini and FGT0 vs R^2 (lgbm, n=10000, hetero)

Usage: python3 e1_montecarlo.py [--reps 10] [--R 50] [--smoke]
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REP_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REP_ROOT, "src"))
import ineq  # noqa: E402
from estimators import (estimate_indices, fit_qrf_quantiles,  # noqa: E402
                        MODELS, ESTIMATORS)

P = 10                      # number of predictors
RHO = 0.3                   # pairwise correlation of predictors
SIGMA2 = 0.64               # total variance of log welfare
MU0 = 10.0                  # intercept of log welfare
BETA = np.array([1.0, -0.8, 0.6, 0.5, -0.4, 0.3, 0.3, -0.2, 0.2, 0.1])
Z_SHARE = 0.6               # poverty line = 0.6 x median of true welfare
SEED_ROOT = 20260501
E_HET = 0.18                # E[exp(0.6 X5)] = exp(0.18) for X5 ~ N(0,1)

R2_GRID = [0.2, 0.4, 0.6, 0.8]
N_GRID = [2000, 10000]
HET_GRID = [False, True]
SIGMA_GRID = [0.5, 0.7, 0.9, 1.1]

INDEX_NAMES = ("gini", "mld", "theil", "atk1", "varlog", "fgt0", "fgt1", "fgt2")


# ----------------------------------------------------------------- DGP
def draw_X(n, rng):
    """P standard normal predictors with pairwise correlation RHO (common factor)."""
    c = rng.standard_normal((n, 1))
    Z = rng.standard_normal((n, P))
    return np.sqrt(RHO) * c + np.sqrt(1.0 - RHO) * Z


def f_raw(X):
    """Unscaled conditional mean of log welfare: linear terms plus nonlinearity.
    Column indices are 0-based: X1 = X[:,0], ..., X5 = X[:,4]."""
    return (X @ BETA + 0.5 * X[:, 0] * X[:, 1] + np.sin(X[:, 2])
            + X[:, 3] ** 2 / 2.0)


def reference_var_fraw(n=1_000_000, seed=424242):
    """Variance of f_raw from a fixed reference sample; used to scale f so that
    Var(f) = R2 * SIGMA2 in the population (up to sampling error)."""
    rng = np.random.default_rng(seed)
    return float(np.var(f_raw(draw_X(n, rng))))


def draw_sample(n, r2, het, vf, rng):
    X = draw_X(n, rng)
    f = np.sqrt(r2 * SIGMA2 / vf) * f_raw(X)
    if het:
        s0 = np.sqrt((1.0 - r2) * SIGMA2 / np.exp(E_HET))
        s = s0 * np.exp(0.3 * X[:, 4])
    else:
        s = np.full(n, np.sqrt((1.0 - r2) * SIGMA2))
    eps = rng.standard_normal(n)
    ell = MU0 + f                        # E[log Y | X]
    logy = ell + s * eps
    m = np.exp(ell + s ** 2 / 2.0)       # E[Y | X] for lognormal Y|X
    return {"X": X, "logy": logy, "y": np.exp(logy), "ell": ell, "s": s,
            "m": m, "f": f}


# ----------------------------------------------------------------- runner
def run_rep(ci, cell, rep, cfg, vf, rows):
    r2, n, het = cell
    ss = np.random.SeedSequence([SEED_ROOT, ci, rep])
    rng_tr, rng_pop = [np.random.default_rng(c) for c in ss.spawn(2)]
    seed_int = 1000 * ci + rep + 1

    pop = draw_sample(cfg["n_pop"], r2, het, vf, rng_pop)
    z = Z_SHARE * float(np.median(pop["y"]))
    tr = draw_sample(n, r2, het, vf, rng_tr)

    def add(model, rep_, est, idx, val):
        rows.append((r2, n, het, model, rep_, est, idx, float(val)))

    truth = ineq.all_relative(pop["y"], z)
    oracle = ineq.all_relative(pop["m"], z)
    for k in INDEX_NAMES:
        add("none", rep, "true", k, truth[k])
        add("none", rep, "oracle", k, oracle[k])
    add("none", rep, "dgp", "var_logy", np.var(pop["logy"]))
    add("none", rep, "dgp", "share_var_f", np.var(pop["f"]) / np.var(pop["logy"]))
    add("none", rep, "dgp", "mean_s2", np.mean(pop["s"] ** 2))
    add("none", rep, "dgp", "within_mld", np.mean(pop["s"] ** 2 / 2.0))
    add("none", rep, "dgp", "log_meanY_over_meanm",
        np.log(pop["y"].mean() / pop["m"].mean()))

    Q = fit_qrf_quantiles(tr["X"], tr["logy"], pop["X"], seed_int + 7919)
    for mi, model in enumerate(MODELS):
        rng_m = np.random.default_rng(
            np.random.SeedSequence([SEED_ROOT, ci, rep, mi + 1]))
        res = estimate_indices(tr["X"], tr["y"], pop["X"], None, z, model,
                               cfg["R"], rng_m, seed=seed_int, c3_quantiles=Q)
        for est in ESTIMATORS:
            for k in INDEX_NAMES:
                add(model, rep, est, k, res[est][k])
        for k, v in res["fit"].items():
            add(model, rep, "fit", k, v)


def cell_list(cfg):
    return list(itertools.product(cfg["r2_grid"], cfg["n_grid"], cfg["het_grid"]))


COLS = ["r2", "n_train", "hetero", "model", "rep", "estimator", "index", "value"]


def run_all(cfg, out_dir, scratch):
    ckpt = os.path.join(scratch, "e1_checkpoint.csv")
    done = set()
    if cfg["resume"] and os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        done = set(map(tuple, prev.loc[prev.estimator == "true",
                                       ["r2", "n_train", "hetero", "rep"]]
                       .astype({"hetero": bool}).values.tolist()))
        print(f"resuming: {len(done)} completed reps found in checkpoint", flush=True)
    vf = reference_var_fraw()
    print(f"reference Var(f_raw) = {vf:.6f}", flush=True)
    cells = cell_list(cfg)
    total = len(cells) * cfg["reps"]
    t0 = time.time()
    k = 0
    for ci, cell in enumerate(cells):
        for rep in range(cfg["reps"]):
            key = (cell[0], cell[1], bool(cell[2]), rep)
            k += 1
            if key in done:
                continue
            t1 = time.time()
            rows = []
            run_rep(ci, cell, rep, cfg, vf, rows)
            df = pd.DataFrame(rows, columns=COLS)
            df.to_csv(ckpt, mode="a", index=False,
                      header=not os.path.exists(ckpt))
            el = time.time() - t0
            eta = el / k * (total - k)
            print(f"[{k}/{total}] r2={cell[0]} n={cell[1]} hetero={cell[2]} "
                  f"rep={rep}  rep_time={time.time() - t1:.1f}s  "
                  f"elapsed={el / 60:.1f}min  eta={eta / 60:.1f}min", flush=True)
    raw = pd.read_csv(ckpt)
    raw["hetero"] = raw["hetero"].astype(bool)
    return raw, time.time() - t0


# ----------------------------------------------------------------- outputs
def summarise(raw):
    keys = ["r2", "n_train", "hetero", "rep", "index"]
    tr = (raw[raw.estimator == "true"][keys + ["value"]]
          .rename(columns={"value": "true"}))
    est = raw[raw.estimator.isin(list(ESTIMATORS) + ["oracle"])]
    m = est.merge(tr, on=keys, how="left")
    m["err"] = m.value - m.true
    m["rel"] = m.err / m.true
    g = m.groupby(["r2", "n_train", "hetero", "model", "estimator", "index"])
    s = g.agg(n_reps=("rep", "nunique"), mean_true=("true", "mean"),
              mean_est=("value", "mean"), sd_est=("value", "std"),
              mean_bias=("err", "mean"),
              rel_bias_pct=("rel", lambda x: 100 * x.mean()),
              rmse=("err", lambda x: float(np.sqrt(np.mean(x ** 2)))),
              rel_rmse_pct=("rel", lambda x: 100 * float(np.sqrt(np.mean(x ** 2))))
              ).reset_index()
    fit = raw[raw.estimator == "fit"]
    f = (fit.groupby(["r2", "n_train", "hetero", "model", "index"])
         .agg(n_reps=("rep", "nunique"), mean_est=("value", "mean"),
              sd_est=("value", "std")).reset_index())
    f["estimator"] = "fit"
    f = f.rename(columns={"index": "index"})
    return s, f


def prop3_table(sigmas=SIGMA_GRID, r2s=R2_GRID, n_sim=2_000_000, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for sg in sigmas:
        for r2 in r2s:
            g_true = 2 * norm.cdf(sg / np.sqrt(2)) - 1
            g_pred = 2 * norm.cdf(sg * np.sqrt(r2) / np.sqrt(2)) - 1
            # Monte Carlo check of the closed forms
            e = rng.standard_normal(n_sim)
            ysim = np.exp(sg * e)
            ypred = np.exp(sg * np.sqrt(r2) * e)
            rows.append({"sigma": sg, "R2": r2,
                         "gini_true": g_true, "gini_pred": g_pred,
                         "ratio_pred_over_true": g_pred / g_true,
                         "bias_pct": 100 * (g_pred - g_true) / g_true,
                         "mc_gini_true": ineq.gini(ysim),
                         "mc_gini_pred": ineq.gini(ypred)})
    return pd.DataFrame(rows)


def fig_bias(summ, out_path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    })
    sub = summ[(summ.model == "lgbm") & (summ.n_train == 10000)
               & (summ.hetero == True)]  # noqa: E712
    styles = {"N": ("o", "-"), "C1": ("s", "--"), "C2": ("^", ":"),
              "C3": ("D", "-."), "C4": ("v", (0, (5, 1, 1, 1)))}
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 2.9), sharex=True)
    panels = [("gini", "(a) Gini"), ("fgt0", "(b) FGT(0)")]
    handles = []
    for ax, (idx, lab) in zip(axes, panels):
        for est in ["N", "C1", "C2", "C3", "C4"]:
            d = sub[(sub.estimator == est) & (sub["index"] == idx)].sort_values("r2")
            mk, ls = styles[est]
            h, = ax.plot(d.r2, d.rel_bias_pct, color="black", marker=mk,
                         linestyle=ls, markersize=4, linewidth=1.0,
                         markerfacecolor="white", label=est)
            if ax is axes[0]:
                handles.append(h)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=(0, (1, 2)))
        ax.set_xlabel(r"Target $R^2$ of log welfare")
        ax.set_xticks(R2_GRID)
        ax.text(0.03, 0.95, lab, transform=ax.transAxes, va="top", ha="left")
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Relative bias (%)")
    axes[1].set_ylabel("Relative bias (%)")
    fig.legend(handles, ["N", "C1", "C2", "C3", "C4"], loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def checks_text(raw, summ, prop3, cfg, runtime):
    L = []
    w = L.append
    w("E1 Monte Carlo: numerical checks and run settings")
    w("=" * 60)
    w(f"Runtime: {runtime / 60:.1f} minutes")
    w(f"Population (test) size N_pop = {cfg['n_pop']}; reps per cell = {cfg['reps']}; "
      f"R (simulation replicates for C2-C4) = {cfg['R']}")
    w(f"Grid: R2 in {cfg['r2_grid']}; n_train in {cfg['n_grid']}; hetero in {cfg['het_grid']}; "
      f"models {list(MODELS)}")
    w("Poverty line z = 0.6 x median of the true population welfare (per replication).")
    w("Seeds: SeedSequence([20260501, cell, rep]); model draws use SeedSequence([20260501, cell, rep, model]).")
    w("Reference variance of the unscaled f used for scaling: from a fixed 1,000,000-row sample (seed 424242).")
    w("")

    # DGP calibration
    w("1. DGP calibration (population, averaged over replications)")
    dg = raw[raw.estimator == "dgp"].copy()
    dg = dg.merge(raw[raw.estimator == "true"][["r2", "n_train", "hetero", "rep"]].drop_duplicates(),
                  on=["r2", "n_train", "hetero", "rep"])
    cal = (dg[dg["index"].isin(["share_var_f", "var_logy", "mean_s2"])]
           .groupby(["r2", "hetero", "index"])["value"].mean().unstack("index"))
    w("Target R2 | hetero | Var(f)/Var(logY) | Var(logY) | E[s^2]  (targets: R2, 0.64, (1-R2)*0.64)")
    for (r2, het), row in cal.iterrows():
        tgt = (1 - r2) * SIGMA2
        w(f"  {r2:.1f} | {str(het):5s} | {row['share_var_f']:.4f} | {row['var_logy']:.4f} | "
          f"{row['mean_s2']:.4f} (target {tgt:.4f})")
    w("")

    # Proposition 1
    w("2. Proposition 1: oracle index m(X) = E[Y|X] vs true index Y")
    w("Claim: I(m(X)) <= I(Y) for Lorenz-consistent indices (Gini, MLD, Theil, Atkinson(1), FGT1, FGT2).")
    w("FGT0 (headcount) sign is ambiguous, reported for information only.")
    tru = raw[raw.estimator == "true"][["r2", "n_train", "hetero", "rep", "index", "value"]].rename(columns={"value": "true"})
    orc = raw[raw.estimator == "oracle"][["r2", "n_train", "hetero", "rep", "index", "value"]].rename(columns={"value": "oracle"})
    p1 = orc.merge(tru, on=["r2", "n_train", "hetero", "rep", "index"])
    p1["ok"] = p1.oracle <= p1.true * (1 + 1e-12) + 1e-12
    p1["gap_pct"] = 100 * (p1.oracle - p1.true) / p1.true
    p1_ok_all = True
    for idx in INDEX_NAMES:
        d = p1[p1["index"] == idx]
        frac = d.ok.mean()
        lab = "info" if idx == "fgt0" else "claim"
        if lab == "claim" and frac < 1.0:
            p1_ok_all = False
        w(f"  {idx:7s} [{lab}] share of reps with oracle <= true: {frac:.3f} "
          f"({int(d.ok.sum())}/{len(d)}); oracle/true-1 in % : mean {d.gap_pct.mean():+.2f}, "
          f"range [{d.gap_pct.min():+.2f}, {d.gap_pct.max():+.2f}]")
    w(f"  Proposition 1 check (all Lorenz-consistent indices, all reps): "
      f"{'PASSED' if p1_ok_all else 'FAILED'}")
    w("")

    # Proposition 2
    w("3. Proposition 2: MLD decomposition I0(Y) = I0(m(X)) + E[I0(Y|X)]")
    w("(a) Algebraic identity, with E[log Y | X] = 10 + f(X) and log m - ell = s^2/2:")
    w("    log(mean m) - mean(ell) - [log(mean m) - mean(log m)] must equal mean(s^2/2) exactly.")
    # Algebraic check on one fresh population per cell
    ncell = 0
    maxres = 0.0
    rng = np.random.default_rng(99)
    vf_ref = reference_var_fraw()
    for r2 in cfg["r2_grid"]:
        for het in cfg["het_grid"]:
            d = draw_sample(cfg["n_pop"], r2, het, vf_ref, rng)
            lhs = np.log(d["m"].mean()) - np.mean(d["ell"])
            rhs = (np.log(d["m"].mean()) - np.mean(np.log(d["m"]))) + np.mean(d["s"] ** 2 / 2.0)
            maxres = max(maxres, abs(lhs - rhs))
            ncell += 1
    w(f"    max |residual| over {ncell} DGP cells: {maxres:.3e}  "
      f"-> {'PASSED (machine precision)' if maxres < 1e-10 else 'FAILED'}")
    w("(b) Replication-level: true MLD - MLD(oracle) vs E[within-cell MLD] = mean(s^2/2).")
    w("    Realized gap = [log(mean Y) - log(mean m)] + [mean(log m) - mean(log Y)]; the second")
    w("    term equals mean(s^2/2) - mean(s*eps), so the residual is sampling noise of order 1/sqrt(N).")
    mld_t = raw[(raw.estimator == "true") & (raw["index"] == "mld")][["r2", "n_train", "hetero", "rep", "value"]].rename(columns={"value": "t"})
    mld_o = raw[(raw.estimator == "oracle") & (raw["index"] == "mld")][["r2", "n_train", "hetero", "rep", "value"]].rename(columns={"value": "o"})
    wd = raw[(raw.estimator == "dgp") & (raw["index"] == "within_mld")][["r2", "n_train", "hetero", "rep", "value"]].rename(columns={"value": "within"})
    p2 = mld_t.merge(mld_o, on=["r2", "n_train", "hetero", "rep"]).merge(wd, on=["r2", "n_train", "hetero", "rep"])
    p2["gap"] = p2.t - p2.o
    p2["resid"] = p2.gap - p2.within
    p2["rel_resid_pct"] = 100 * p2.resid / p2.within
    for (r2, het), g in p2.groupby(["r2", "hetero"]):
        w(f"    R2={r2:.1f} hetero={str(het):5s}: E[within] = {g.within.mean():.4f}, "
          f"mean gap = {g.gap.mean():.4f}, mean residual = {g.resid.mean():+.4f} "
          f"({g.rel_resid_pct.mean():+.2f}% of within), max |residual|% = {g.rel_resid_pct.abs().max():.2f}")
    w(f"    Mean over all cells: E[within] = {p2.within.mean():.4f}, mean gap = {p2.gap.mean():.4f}, "
      f"mean residual = {p2.resid.mean():+.5f}")
    w("    Proposition 2 check: the gap equals E[within-cell MLD] up to sampling noise "
      f"(residual < 5% of within-cell MLD in {100 * (p2.rel_resid_pct.abs() < 5).mean():.1f}% of reps)")
    w("")

    # Proposition 3
    w("4. Proposition 3: lognormal Gini, true vs predicted (table in e1_prop3_table.csv)")
    mx = (prop3.gini_true - prop3.mc_gini_true).abs().max()
    mxp = (prop3.gini_pred - prop3.mc_gini_pred).abs().max()
    w(f"    Max |closed form - Monte Carlo| (n=2,000,000): true Gini {mx:.2e}, predicted Gini {mxp:.2e}")
    w(f"    Predicted/true Gini ratio range: {prop3.ratio_pred_over_true.min():.3f} to "
      f"{prop3.ratio_pred_over_true.max():.3f}")
    w("")

    # Estimator summary at R2 = 0.4 and 0.8
    w("5. Relative bias (%) of Gini and MLD, lgbm model, n_train = 10000")
    for het in [True, False]:
        for r2 in [0.4, 0.8]:
            d = summ[(summ.model == "lgbm") & (summ.n_train == 10000) & (summ.hetero == het)
                     & (summ.r2 == r2)]
            if len(d) == 0:
                w(f"    R2={r2} hetero={het}: not in grid")
                continue
            s_txt = []
            for idx in ["gini", "mld"]:
                vals = {}
                for e in ["N", "C1", "C2", "C3", "C4", "oracle"]:
                    q = d[(d.estimator == e) & (d["index"] == idx)]
                    if len(q):
                        vals[e] = q.rel_bias_pct.values[0]
                s_txt.append(idx + ": " + ", ".join(f"{k}={v:+.2f}" for k, v in vals.items()))
            w(f"    R2={r2} hetero={het}: " + " | ".join(s_txt))
    w("")
    w("6. Out-of-fold R2 of log welfare (mean over reps)")
    fit = raw[(raw.estimator == "fit") & (raw["index"] == "oof_r2_log")]
    for mdl in MODELS:
        d = fit[fit.model == mdl].groupby(["r2", "n_train"]).value.mean()
        w(f"    {mdl:6s}: " + ", ".join(f"R2={a} n={b}: {v:.3f}" for (a, b), v in d.items()))
    w("")
    w("Notes")
    w("- Population size and replication count are set by the run settings above (see Runtime).")
    w("- The true poverty line is recomputed from each replication's population median.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny configuration for testing the pipeline")
    ap.add_argument("--out-dir", default=os.path.join(REP_ROOT, "output"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    scratch = os.environ.get(
        "E1_SCRATCH",
        "/tmp/claude-0/-home-user-The-Journal-of-Economic-Inequality/"
        "aeb92f95-1fd2-5ae6-9eae-422e88f0fb11/scratchpad")
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = dict(n_pop=N_POP_DEFAULT, reps=args.reps, R=args.R,
               r2_grid=R2_GRID, n_grid=N_GRID, het_grid=HET_GRID,
               resume=not args.no_resume)
    if args.smoke:
        cfg.update(n_pop=20000, reps=1, R=5, r2_grid=[0.4], n_grid=[2000],
                   het_grid=[False, True])
        scratch = os.path.join(scratch, "smoke")
        os.makedirs(scratch, exist_ok=True)
        cfg["resume"] = False
    else:
        cfg["n_pop"] = N_POP_DEFAULT
    if cfg["resume"] is False and os.path.exists(os.path.join(scratch, "e1_checkpoint.csv")):
        os.remove(os.path.join(scratch, "e1_checkpoint.csv"))

    print(f"config: {cfg}", flush=True)
    raw, runtime = run_all(cfg, args.out_dir, scratch)
    print(f"all replications done in {runtime / 60:.1f} min", flush=True)

    out = args.out_dir
    raw.to_csv(os.path.join(out, "e1_results_raw.csv"), index=False)
    summ, fit = summarise(raw)
    summ = pd.concat([summ, fit], ignore_index=True)
    summ.to_csv(os.path.join(out, "e1_summary.csv"), index=False)
    prop3 = prop3_table()
    prop3.to_csv(os.path.join(out, "e1_prop3_table.csv"), index=False)
    txt = checks_text(raw, summ, prop3, cfg, runtime)
    with open(os.path.join(out, "e1_checks.txt"), "w") as fh:
        fh.write(txt)
    fig_bias(summ, os.path.join(out, "Fig_e1_bias.pdf"))
    print(txt, flush=True)
    print("outputs written to", out, flush=True)


N_POP_DEFAULT = 100000

if __name__ == "__main__":
    main()
