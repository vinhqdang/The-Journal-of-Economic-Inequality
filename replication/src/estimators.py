"""Welfare-prediction estimators for inequality and poverty measures.

Given a training sample (X_tr, y_tr > 0) and a target sample X_te, the
functions here fit a log-welfare predictor and return, for each estimator,
the inequality/poverty indices computed on the predicted welfare:

  N   naive plug-in: exp(m_hat(X)) * Duan smearing factor
  C1  mean recalibration: linear regression of y on out-of-fold N-predictions
  C2  homoskedastic residual simulation (pooled OOF log residuals)
  C3  heteroskedastic distributional: quantile regression forest draws
  C4  conditional residual bootstrap within deciles of OOF m_hat

Simulation estimators (C2-C4) average each index over R replicates.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ineq  # noqa: E402

MODELS = ("ols", "lasso", "rf", "lgbm")
ESTIMATORS = ("N", "C1", "C2", "C3", "C4")
INDICES = ("gini", "mld", "theil", "atk1", "varlog", "fgt0", "fgt1", "fgt2")
QUANTILES = np.arange(1, 100) / 100.0  # 0.01, ..., 0.99


def make_model(name, seed):
    if name == "ols":
        return LinearRegression()
    if name == "lasso":
        return LassoCV(cv=5, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                     n_jobs=4, random_state=seed)
    if name == "lgbm":
        return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             min_child_samples=50, verbose=-1, random_state=seed,
                             n_jobs=4)
    raise ValueError(f"unknown model {name}")


def oof_and_test_predictions(X_tr, ylog_tr, X_te, model, seed, n_folds=5):
    """5-fold cross-fitted predictions of log y on the training data, plus
    predictions on X_te from a model refit on the full training data."""
    oof = np.empty(len(ylog_tr))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, ho) in enumerate(kf.split(X_tr)):
        m = make_model(model, seed + k + 1)
        m.fit(X_tr[tr], ylog_tr[tr])
        oof[ho] = m.predict(X_tr[ho])
    final = make_model(model, seed)
    final.fit(X_tr, ylog_tr)
    return oof, np.asarray(final.predict(X_te), dtype=float)


def fit_qrf_quantiles(X_tr, ylog_tr, X_te, seed, chunk=5000):
    """Quantile regression forest on log y (300 trees, min_samples_leaf=20).
    Returns an (n_te, 99) matrix of conditional log-quantiles 0.01..0.99."""
    qrf = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=20,
                                        n_jobs=4, random_state=seed)
    qrf.fit(X_tr, ylog_tr)
    out = np.empty((len(X_te), len(QUANTILES)))
    for s in range(0, len(X_te), chunk):
        q = qrf.predict(X_te[s:s + chunk], quantiles=list(QUANTILES))
        out[s:s + chunk] = q
    return out


def oof_r2(y, pred):
    y = np.asarray(y, dtype=float)
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _avg_indices(list_of_dicts):
    keys = list_of_dicts[0].keys()
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}


def estimate_indices(X_tr, y_tr, X_te, w_te, z, model, R, rng,
                     seed=0, c3_quantiles=None):
    """Estimate inequality and poverty indices of welfare in the target sample.

    Parameters
    ----------
    X_tr, y_tr : training features and welfare (y_tr > 0)
    X_te       : target features (the population whose indices are estimated)
    w_te       : optional weights for the target (None = equal weights)
    z          : poverty line (levels)
    model      : one of MODELS, the point predictor of log y
    R          : number of simulation replicates for C2, C3, C4
    rng        : numpy Generator used for all simulation draws
    seed       : integer seed for the fitted models and cross-fitting folds
    c3_quantiles : optional precomputed (n_te, 99) matrix from
                   fit_qrf_quantiles; if None it is fitted here. C3 does not
                   depend on `model`, so callers may reuse one fit across models.

    Returns
    -------
    dict with keys 'N', 'C1', 'C2', 'C3', 'C4' (each maps index name -> value)
    and 'fit' (maps 'oof_r2_log' and 'smear' -> value).
    """
    y_tr = np.asarray(y_tr, dtype=float)
    if np.any(y_tr <= 0):
        raise ValueError("y_tr must be strictly positive")
    ylog_tr = np.log(y_tr)
    X_te = np.asarray(X_te, dtype=float)
    n_te = len(X_te)

    oof_log, te_log = oof_and_test_predictions(X_tr, ylog_tr, X_te, model, seed)
    resid = ylog_tr - oof_log                       # OOF log residuals
    smear = float(np.mean(np.exp(resid)))           # Duan smearing factor

    out = {}
    fit_info = {"oof_r2_log": oof_r2(ylog_tr, oof_log), "smear": smear}
    out["fit"] = fit_info

    # N: predicted conditional mean as poverty maps report it
    yhat_te = np.exp(te_log) * smear
    out["N"] = ineq.all_relative(yhat_te, z, w_te)

    # C1: linear recalibration of levels on out-of-fold N-predictions
    yhat_oof = np.exp(oof_log) * smear
    xm, ym = yhat_oof.mean(), y_tr.mean()
    b = np.sum((yhat_oof - xm) * (y_tr - ym)) / np.sum((yhat_oof - xm) ** 2)
    a = ym - b * xm
    floor = 0.01 * np.min(y_tr[y_tr > 0])
    yc_te = np.maximum(a + b * yhat_te, floor)
    out["C1"] = ineq.all_relative(yc_te, z, w_te)
    fit_info["C1_slope"] = float(b)
    fit_info["C1_intercept"] = float(a)

    # C2: homoskedastic residual simulation from pooled OOF residuals
    sims = []
    for _ in range(R):
        e = rng.choice(resid, size=n_te, replace=True)
        sims.append(ineq.all_relative(np.exp(te_log + e), z, w_te))
    out["C2"] = _avg_indices(sims)

    # C3: quantile regression forest, draw a uniform quantile per unit
    if c3_quantiles is None:
        c3_quantiles = fit_qrf_quantiles(X_tr, ylog_tr, X_te, seed + 7919)
    sims = []
    rows = np.arange(n_te)
    for _ in range(R):
        k = rng.integers(0, len(QUANTILES), size=n_te)
        sims.append(ineq.all_relative(np.exp(c3_quantiles[rows, k]), z, w_te))
    out["C3"] = _avg_indices(sims)

    # C4: residuals resampled within deciles of OOF m_hat (cutpoints from train)
    cuts = np.quantile(oof_log, np.linspace(0.1, 0.9, 9))
    b_tr = np.digitize(oof_log, cuts)               # 0..9
    b_te = np.digitize(te_log, cuts)
    order = np.argsort(b_tr, kind="stable")
    pool = resid[order]                              # residuals sorted by bin
    counts = np.bincount(b_tr, minlength=10)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    if np.any(counts == 0):
        raise ValueError("empty decile bin in training OOF predictions")
    sims = []
    for _ in range(R):
        u = rng.random(n_te)
        idx = starts[b_te] + np.minimum((u * counts[b_te]).astype(int),
                                        counts[b_te] - 1)
        sims.append(ineq.all_relative(np.exp(te_log + pool[idx]), z, w_te))
    out["C4"] = _avg_indices(sims)

    return out
