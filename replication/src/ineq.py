"""Inequality and poverty indices (weighted). All functions take 1-D arrays."""
import numpy as np


def _w(y, w):
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    return y, w / w.sum()


def mean(y, w=None):
    y, w = _w(y, w)
    return float(np.sum(w * y))


def gini(y, w=None):
    """Relative Gini via the weighted covariance formula (y > 0)."""
    y, w = _w(y, w)
    o = np.argsort(y)
    y, w = y[o], w[o]
    cw = np.cumsum(w)
    F = cw - w / 2.0
    mu = np.sum(w * y)
    return float(2.0 * np.sum(w * y * (F - np.sum(w * F))) / mu)


def abs_gini(y, w=None):
    """Absolute Gini = mean * relative Gini = half the mean absolute difference.
    Translation invariant, valid for real-valued y."""
    y, w = _w(y, w)
    o = np.argsort(y)
    y, w = y[o], w[o]
    F = np.cumsum(w) - w / 2.0
    return float(2.0 * np.sum(w * y * (F - np.sum(w * F))))


def ge(y, alpha, w=None):
    """Generalised entropy GE(alpha); alpha=0 MLD, alpha=1 Theil."""
    y, w = _w(y, w)
    mu = np.sum(w * y)
    r = y / mu
    if alpha == 0:
        return float(-np.sum(w * np.log(r)))
    if alpha == 1:
        return float(np.sum(w * r * np.log(r)))
    return float((np.sum(w * r ** alpha) - 1.0) / (alpha * (alpha - 1.0)))


def mld(y, w=None):
    return ge(y, 0, w)


def theil(y, w=None):
    return ge(y, 1, w)


def atkinson(y, eps=1.0, w=None):
    y, w = _w(y, w)
    mu = np.sum(w * y)
    if eps == 1.0:
        return float(1.0 - np.exp(np.sum(w * np.log(y))) / mu)
    return float(1.0 - np.sum(w * y ** (1.0 - eps)) ** (1.0 / (1.0 - eps)) / mu)


def var_log(y, w=None):
    y, w = _w(y, w)
    ly = np.log(y)
    m = np.sum(w * ly)
    return float(np.sum(w * (ly - m) ** 2))


def fgt(y, z, alpha=0, w=None):
    y, w = _w(y, w)
    g = np.clip((z - y) / z, 0.0, None)
    if alpha == 0:
        return float(np.sum(w * (y < z)))
    return float(np.sum(w * g ** alpha))


def wquantile(y, q, w=None):
    y, w = _w(y, w)
    o = np.argsort(y)
    cw = np.cumsum(w[o])
    return float(np.interp(q, cw - w[o] / 2.0, y[o]))


def all_relative(y, z, w=None):
    """Dictionary of relative indices for positive welfare and poverty line z."""
    return {
        "gini": gini(y, w), "mld": mld(y, w), "theil": theil(y, w),
        "atk1": atkinson(y, 1.0, w), "varlog": var_log(y, w),
        "fgt0": fgt(y, z, 0, w), "fgt1": fgt(y, z, 1, w), "fgt2": fgt(y, z, 2, w),
    }


def all_absolute(y, cutoff, w=None):
    """Translation-invariant measures for real-valued indices (e.g. asset index)."""
    y, w = _w(y, w)
    mu = np.sum(w * y)
    return {
        "abs_gini": abs_gini(y, w), "sd": float(np.sqrt(np.sum(w * (y - mu) ** 2))),
        "p90_p10": wquantile(y, 0.9, w) - wquantile(y, 0.1, w),
        "share_below": float(np.sum(w * (y < cutoff))),
    }
