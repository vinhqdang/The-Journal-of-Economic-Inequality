#!/usr/bin/env python3
"""Prepare DHS cluster data (Yeh et al. 2020) for the E3 satellite application.

Inputs, downloaded to replication/data/dhs/ if missing (raw GitHub URLs):
  dhs_clusters.csv       cluster outcome (wealthpooled), location, year, households
  dhs_image_hists.npz    per-cluster pixel histograms (features); loaded with allow_pickle=False
  dhs_wealth_index.csv   used only to cross-check the urban/rural coding

Outputs:
  replication/data/dhs/dhs_features.parquet
  replication/output/e3_data_summary.txt

Conventions and assumptions (the npz and CSV are not fully documented upstream):
  * image_hists has shape (N, 8, 102). Each channel is a histogram over the
    224 x 224 = 50176 pixels of one patch. Channels 0-6 are treated as Landsat
    bands (named lsat0..lsat6, order not documented). Channel 7 is treated as
    night lights ("ntl"); its mean bin index explains nls_mean well (R^2 ~ 0.84),
    which supports this.
  * Bin edges are not documented. Histogram moments use the bin index as the bin
    centre. Under uniform bins this is an affine rescaling of physical units, and
    standardised ridge is invariant to that.
  * Rows of the npz are a permutation of the CSV rows. They are matched by
    (lat, lon) and then verified on year and wealth.
  * urban_rural: 1 = urban, 0 = rural (checked against URBAN_RURA in
    dhs_wealth_index.csv).
  * Folds: fold_country = country (out-of-country). fold_spatial = k-means (k=5)
    on lat/lon fitted separately within each country (in-country spatial blocks).
    The labels are within-country, so the pooled CV uses fold k as the k-th block
    of every country.
  * The repository's dhs_incountry_folds.pkl is a pickle and is not loaded.
"""

import pathlib
import urllib.request

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "replication" / "data" / "dhs"
OUT = ROOT / "replication" / "output"
BASE = "https://raw.githubusercontent.com/sustainlab-group/africa_poverty/master/data/"
FILES = ["dhs_clusters.csv", "dhs_image_hists.npz", "dhs_wealth_index.csv"]
BAND_NAMES = [f"lsat{c}" for c in range(7)] + ["ntl"]
N_SPATIAL = 5
RIDGE_ALPHA = 10.0
ALPHA_SENSITIVITY = [1.0, 10.0, 100.0]
CLIP_Z = 5.0  # clip standardised features; rare histogram bins otherwise produce extreme z-scores


def fetch(name):
    dest = DATA / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(BASE + name, timeout=180) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.replace(dest)
    return dest


def ridge_pipeline(alpha, clip=CLIP_Z):
    steps = [StandardScaler()]
    if clip is not None:
        steps.append(FunctionTransformer(lambda z: np.clip(z, -clip, clip)))
    return make_pipeline(*steps, Ridge(alpha=alpha))


def cv_spatial(X, y, folds, alpha, clip=CLIP_Z):
    """Pooled out-of-fold predictions for spatial folds, plus per-fold R^2."""
    pred = np.empty_like(y)
    per_fold = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        model = ridge_pipeline(alpha, clip).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        per_fold.append(r2_score(y[te], pred[te]))
    return r2_score(y, pred), float(np.mean(per_fold)), per_fold


def cv_country(X, y, countries, alpha, clip=CLIP_Z):
    pred = np.empty_like(y)
    per_fold = []
    for c in np.unique(countries):
        tr, te = countries != c, countries == c
        model = ridge_pipeline(alpha, clip).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        per_fold.append(r2_score(y[te], pred[te]))
    return r2_score(y, pred), float(np.mean(per_fold))


def main():
    paths = {name: fetch(name) for name in FILES}

    clusters = pd.read_csv(paths["dhs_clusters.csv"]).reset_index(drop=True)
    npz = np.load(paths["dhs_image_hists.npz"], allow_pickle=False)
    npz_keys = list(npz.files)
    hists = npz["image_hists"]
    labels = npz["labels"]
    locs = npz["locs"]
    years = npz["years"]
    nls_mean = npz["nls_mean"]

    n = len(clusters)
    assert hists.ndim == 3 and hists.shape[1:] == (8, 102), hists.shape
    assert hists.shape[0] == labels.shape[0] == locs.shape[0] == years.shape[0] == n
    pix = hists.sum(axis=2)
    assert (pix == pix[0, 0]).all(), "histogram channels do not sum to a common pixel count"
    n_pixels = int(pix[0, 0])

    # Match npz rows to CSV rows by location (npz rows are a permutation of the CSV).
    nn = NearestNeighbors(n_neighbors=1).fit(locs.astype(np.float64))
    dist, idx = nn.kneighbors(clusters[["lat", "lon"]].values)
    dist, idx = dist[:, 0], idx[:, 0]
    assert dist.max() < 1e-4, f"unmatched location, max distance {dist.max()}"
    assert len(np.unique(idx)) == n, "location match is not one-to-one"
    assert (years[idx] == clusters["year"].values).all(), "year mismatch after matching"
    wealth_diff = np.abs(labels[idx].astype(np.float64) - clusters["wealthpooled"].values).max()
    assert wealth_diff < 1e-5, f"label mismatch after matching: {wealth_diff}"
    match_dist_max = float(dist.max())

    # Cross-check urban/rural coding against the wealth index file, keyed on location and year.
    wi = pd.read_csv(paths["dhs_wealth_index.csv"]).dropna(subset=["URBAN_RURA", "LATNUM", "LONGNUM"])
    # Keep only location-years that are clusters in dhs_clusters.csv. The file also has
    # empty clusters (households == 0) whose location-years carry conflicting codes.
    in_clusters = pd.MultiIndex.from_arrays([clusters["lat"].round(6), clusters["lon"].round(6), clusters["year"]])
    wi_key = pd.MultiIndex.from_arrays([wi["LATNUM"].round(6), wi["LONGNUM"].round(6), wi["year"]])
    wi = wi[wi_key.isin(in_clusters)]
    key = ["LATNUM", "LONGNUM", "year"]
    loc_nuniq = wi.groupby(key)["URBAN_RURA"].nunique()
    assert (loc_nuniq <= 1).all(), "conflicting urban/rural codes at a location-year"
    loc_urb = wi.groupby(key)["URBAN_RURA"].first().reset_index()
    cl = clusters.assign(_lat=clusters["lat"].round(6), _lon=clusters["lon"].round(6))
    loc_urb = loc_urb.assign(_lat=loc_urb["LATNUM"].round(6), _lon=loc_urb["LONGNUM"].round(6))
    merged = cl[["_lat", "_lon", "year", "urban_rural"]].merge(
        loc_urb[["_lat", "_lon", "year", "URBAN_RURA"]], on=["_lat", "_lon", "year"], how="left")
    matched = merged["URBAN_RURA"].notna()
    urb_agree = float(((merged.loc[matched, "URBAN_RURA"] == "U").astype(int)
                       == merged.loc[matched, "urban_rural"]).mean())
    assert urb_agree == 1.0, f"urban_rural coding disagrees with URBAN_RURA ({urb_agree})"
    urb_matched = int(matched.sum())

    # Features from histograms, in CSV row order.
    shares = hists[idx].astype(np.float64) / pix[idx][:, :, None]
    centres = np.arange(hists.shape[2], dtype=np.float64)
    feat_cols = {}
    per_band_counts = {}
    for c, band in enumerate(BAND_NAMES):
        sh = shares[:, c, :]
        mean = sh @ centres
        var = sh @ (centres ** 2) - mean ** 2
        feat_cols[f"f_{band}_mean"] = mean
        feat_cols[f"f_{band}_sd"] = np.sqrt(np.clip(var, 0.0, None))
        keep = np.nonzero(sh.max(axis=0) > sh.min(axis=0))[0]
        for k in keep:
            feat_cols[f"f_{band}_b{k:03d}"] = sh[:, k]
        per_band_counts[band] = len(keep)
    feat_cols["f_year"] = clusters["year"].values.astype(np.float64)
    feat_names = list(feat_cols)
    features = pd.DataFrame(feat_cols)
    assert np.isfinite(features.values).all()

    # Folds.
    fold_spatial = np.full(n, -1, dtype=np.int8)
    spatial_sizes = []
    for ctry, g in clusters.groupby("country"):
        assert len(g) >= N_SPATIAL, f"{ctry} has fewer than {N_SPATIAL} clusters"
        km = KMeans(n_clusters=N_SPATIAL, n_init=10, random_state=0).fit(g[["lat", "lon"]].values)
        assert len(np.unique(km.labels_)) == N_SPATIAL
        fold_spatial[g.index.values] = km.labels_.astype(np.int8)
        spatial_sizes.append({"country": ctry, **{f"s{k}": int((km.labels_ == k).sum()) for k in range(N_SPATIAL)}})
    assert (fold_spatial >= 0).all()
    spatial_sizes = pd.DataFrame(spatial_sizes)

    out = pd.DataFrame({
        "country": clusters["country"].astype("string"),
        "year": clusters["year"].astype("int16"),
        "lat": clusters["lat"].astype(np.float64),
        "lon": clusters["lon"].astype(np.float64),
        "urban_rural": clusters["urban_rural"].astype("int8"),
        "households": clusters["households"].astype("int32"),
        "wealth": clusters["wealthpooled"].astype(np.float64),
        "fold_country": clusters["country"].astype("string"),
        "fold_spatial": pd.Series(fold_spatial, dtype="int8"),
    })
    out = pd.concat([out, features.astype(np.float64)], axis=1)
    out_path = DATA / "dhs_features.parquet"
    out.to_parquet(out_path, index=False)
    check = pd.read_parquet(out_path)
    assert check.shape == out.shape

    # Summary.
    X = features.values
    y = out["wealth"].values
    folds_s = out["fold_spatial"].values.astype(int)
    countries = out["country"].values.astype(str)
    r2_sp, r2_sp_mean, r2_sp_folds = cv_spatial(X, y, folds_s, RIDGE_ALPHA)
    r2_c, r2_c_mean = cv_country(X, y, countries, RIDGE_ALPHA)
    sens = {a: cv_spatial(X, y, folds_s, a)[0] for a in ALPHA_SENSITIVITY}
    r2_sp_noclip = cv_spatial(X, y, folds_s, RIDGE_ALPHA, clip=None)[0]
    r2_c_noclip = cv_country(X, y, countries, RIDGE_ALPHA, clip=None)[0]

    L = []
    add = L.append
    add("E3 DHS data summary (Yeh et al. 2020 DHS clusters, Africa)")
    add("=" * 72)
    add("")
    add("Inputs and checks")
    add(f"  dhs_image_hists.npz keys: {npz_keys}")
    add(f"  image_hists shape: {hists.shape} dtype {hists.dtype}; labels {labels.shape}; locs {locs.shape}; years {years.shape}; nls_mean {nls_mean.shape}")
    add(f"  pixels per channel per cluster: {n_pixels} (224 x 224)")
    add(f"  channel mapping (assumed): {', '.join(f'{i}={b}' for i, b in enumerate(BAND_NAMES))}")
    add("  bin edges: undocumented; histogram moments use bin index as centre")
    add(f"  npz-to-CSV matching: max location distance {match_dist_max:.2e} deg; year and wealth verified")
    add(f"  urban_rural coding (1=urban) agrees with dhs_wealth_index.csv URBAN_RURA: {urb_agree:.4f} on {urb_matched} of {n} clusters matched by location-year")
    add("")
    add("Sample")
    add(f"  clusters: {n}")
    add(f"  countries: {clusters['country'].nunique()}")
    add(f"  survey years (min..max): {clusters['year'].min()}..{clusters['year'].max()}")
    add(f"  urban clusters: {int(clusters['urban_rural'].sum())} ({clusters['urban_rural'].mean():.3f})")
    add(f"  households per cluster: mean {clusters['households'].mean():.2f}, min {clusters['households'].min()}, max {clusters['households'].max()}")
    add("")
    add("Country-year surveys (clusters per country-year; svyid not in the data, so this is a proxy)")
    cy = clusters.groupby(["country", "year"]).size().rename("clusters").reset_index()
    add(cy.to_string(index=False))
    add("")
    add("Clusters per country")
    add(clusters.groupby("country").size().rename("clusters").to_string())
    add("")
    add("Wealth (wealthpooled) summary")
    add(out["wealth"].describe().round(4).to_string())
    add("")
    add("Wealth by country")
    wc = out.groupby("country")["wealth"].agg(["count", "mean", "std", "min", "median", "max"]).round(3)
    add(wc.to_string())
    add("")
    add("Features")
    add(f"  total feature columns: {len(feat_names)}")
    add(f"  night-lights moments: f_ntl_mean, f_ntl_sd; Landsat moments: f_lsat{{0..6}}_mean, _sd")
    add(f"  histogram share columns kept per band (non-constant bins): {per_band_counts}")
    add(f"  plus f_year")
    add("  lat/lon are not features; they are kept as columns for spatial blocking")
    add("")
    add("Folds")
    add(f"  fold_country: {out['fold_country'].nunique()} folds (one per country); sizes:")
    add(out["fold_country"].value_counts().sort_index().to_string())
    add("  fold_spatial: 5 within-country k-means blocks; sizes per country and block:")
    add(spatial_sizes.to_string(index=False))
    add(f"  pooled sizes per fold_spatial: {np.bincount(folds_s, minlength=N_SPATIAL).tolist()}")
    add("")
    add("Sanity check: ridge regression of wealth on f_* (standardised within training folds)")
    add(f"  alpha = {RIDGE_ALPHA} (fixed, not tuned); standardised features clipped to +/-{CLIP_Z:g}")
    add(f"  5-fold spatial CV (in-country blocks): pooled OOF R^2 = {r2_sp:.3f}; mean per-fold R^2 = {r2_sp_mean:.3f}")
    add(f"    per-fold R^2: {[round(v, 3) for v in r2_sp_folds]}")
    add(f"  leave-one-country-out (out-of-country, extra): pooled OOF R^2 = {r2_c:.3f}; mean per-country R^2 = {r2_c_mean:.3f}")
    add(f"  without clipping: spatial pooled R^2 = {r2_sp_noclip:.3f}; leave-one-country-out pooled R^2 = {r2_c_noclip:.3f}")
    add("    (rare histogram bins are nearly constant in training and give extreme test z-scores)")
    add("  spatial CV pooled R^2 sensitivity to alpha (reported only, not used for selection):")
    for a, v in sens.items():
        add(f"    alpha={a:g}: {v:.3f}")
    add("")

    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e3_data_summary.txt").write_text(text)
    print(text)
    print(f"wrote {out_path}")
    print(f"wrote {OUT / 'e3_data_summary.txt'}")


if __name__ == "__main__":
    main()
