#!/usr/bin/env python3
"""Prepare ACS 2023 1-year PUMS household data for design-based validation (E2).

Downloads the national household and person PUMS CSV archives (if missing),
builds the occupied-household analysis file with unit-level (u_*) and
area-context (c_*) features, and writes:

  replication/data/acs/acs2023_households.parquet   (analysis file)
  replication/output/e2_data_summary.txt            (summary and feature list)

Archives are read in place through zipfile; nothing is extracted to disk.
Run with: python3 -I replication/scripts/e2_prepare_acs.py
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "replication" / "data" / "acs"
OUT_DIR = ROOT / "replication" / "output"
HH_NAME = "csv_hus.zip"   # household records; members psam_husa.csv, psam_husb.csv
PP_NAME = "csv_pus.zip"   # person records;   members psam_pusa.csv, psam_pusb.csv
PARQUET_PATH = DATA_DIR / "acs2023_households.parquet"
SUMMARY_PATH = OUT_DIR / "e2_data_summary.txt"

HH_COLS = ["SERIALNO", "PUMA", "STATE", "WGTP", "NP", "TYPEHUGQ", "ADJINC", "HINCP",
           "TEN", "RMSP", "BDSP", "VEH", "WIF", "NOC", "HHL", "BLD", "YRBLT",
           "ACCESSINET", "HUPAC"]
PP_COLS = ["SERIALNO", "RELSHIPP", "AGEP", "SEX", "SCHL", "RAC1P", "HISP", "MAR",
           "WKHP", "ESR", "DIS", "NATIVITY"]
HH_DTYPES = {"SERIALNO": str, "PUMA": str, "STATE": str}
PP_DTYPES = {"SERIALNO": str}

# Unit-level feature descriptions (u_ prefix)
UNIT_FEATURES = {
    "u_np": "Number of persons in household (NP)",
    "u_n_under18": "Number of persons aged < 18 in household (person file count)",
    "u_n_65plus": "Number of persons aged >= 65 in household (person file count)",
    "u_noc": "Number of own children in household (NOC)",
    "u_owner": "Owner-occupied dummy (TEN in {1,2})",
    "u_tenure": "Tenure code (TEN: 1 owned w/ mortgage, 2 owned free, 3 rented, 4 no rent)",
    "u_rooms": "Number of rooms (RMSP; 2023 name for RMSPP)",
    "u_bedrooms": "Number of bedrooms (BDSP)",
    "u_vehicles": "Number of vehicles (VEH; 6 = six or more)",
    "u_workers": "Workers in family (WIF; 3 = three or more)",
    "u_hhl": "Household language (HHL)",
    "u_bld": "Building type (BLD; 2 = one-family detached)",
    "u_yrblt": "Year structure built (YRBLT), -1 if missing",
    "u_yrblt_miss": "Missing-indicator for YRBLT",
    "u_accessinet": "Internet access (ACCESSINET: 1 paid, 2 unpaid, 3 none), -1 if missing",
    "u_accessinet_miss": "Missing-indicator for ACCESSINET",
    "u_hupac": "Presence/age of children (HUPAC; 4 = no children)",
    "u_head_missing": "1 if no reference person (RELSHIPP==20) found for household",
    "u_head_age": "Head age in years (AGEP)",
    "u_head_female": "Head female dummy (SEX==2)",
    "u_head_schl": "Head educational attainment (SCHL, 1-24), -1 if missing",
    "u_head_schl_miss": "Missing-indicator for head SCHL",
    "u_head_ba": "Head holds BA or higher (SCHL >= 21)",
    "u_head_white": "Head race White alone (RAC1P==1)",
    "u_head_black": "Head race Black alone (RAC1P==2)",
    "u_head_asian": "Head race Asian alone (RAC1P==6)",
    "u_head_aian": "Head race AIAN/AN alone (RAC1P in 3,4,5)",
    "u_head_hispanic": "Head Hispanic dummy (HISP > 1)",
    "u_head_married": "Head married, spouse present (MAR==1)",
    "u_head_wkhp": "Head usual hours worked per week (WKHP), -1 if missing",
    "u_head_wkhp_miss": "Missing-indicator for head WKHP (blank = did not work in past 12 months)",
    "u_head_employed": "Head employed dummy (ESR in 1,2,4,5)",
    "u_head_disability": "Head disability dummy (DIS==1)",
    "u_head_foreign_born": "Head foreign-born dummy (NATIVITY==2)",
}

# Area-context feature descriptions (c_ prefix), PUMA-level weighted means
CONTEXT_FEATURES = {
    "c_share_owner": "Share of occupied households that own their home (TEN in 1,2)",
    "c_mean_rooms": "Mean number of rooms per household (RMSP)",
    "c_mean_vehicles": "Mean number of vehicles per household (VEH)",
    "c_share_head_ba": "Share of household heads with BA or higher",
    "c_share_head_employed": "Share of household heads employed",
    "c_mean_hh_size": "Mean household size (NP)",
    "c_share_hh_children": "Share of households with own children (NOC > 0)",
    "c_share_hh_65plus": "Share of households with a person aged 65+",
    "c_share_internet": "Share of households with internet access (ACCESSINET in 1,2; denominator excludes missing)",
    "c_share_head_hispanic": "Share of household heads Hispanic",
    "c_share_head_black": "Share of household heads Black alone",
    "c_share_sf_detached": "Share of housing units one-family detached (BLD==2)",
    "c_mean_head_age": "Mean age of household heads",
}


def fetch(name):
    """Download a PUMS archive into DATA_DIR if it is not already present."""
    dest = DATA_DIR / name
    if dest.exists() and zipfile.is_zipfile(dest):
        return dest
    tmp = dest.with_suffix(".part")
    print(f"Downloading {BASE_URL + name} ...", flush=True)
    urllib.request.urlretrieve(BASE_URL + name, tmp)
    os.replace(tmp, dest)
    return dest


def read_members(zip_path, prefix, cols, dtypes):
    """Read the CSV members of a zip whose names start with prefix, keeping only cols."""
    dtypes = {k: v for k, v in dtypes.items() if k in cols}
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(n for n in zf.namelist()
                         if n.startswith(prefix) and n.lower().endswith(".csv"))
        if not members:
            raise RuntimeError(f"No {prefix}*.csv members in {zip_path}")
        for m in members:
            print(f"  reading {m} from {zip_path.name}", flush=True)
            with zf.open(m) as fh:
                frames.append(pd.read_csv(fh, usecols=cols, dtype=dtypes, low_memory=False))
    out = pd.concat(frames, ignore_index=True)
    print(f"  {len(out):,} rows", flush=True)
    return out


def weighted_gini(y, w):
    """Weighted Gini via midpoint ranks: G = 2*sum(w*y*p)/sum(w*y) - 1, p = (cumw - w/2)/W."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    o = np.argsort(y, kind="mergesort")
    y, w = y[o], w[o]
    W = w.sum()
    p = (np.cumsum(w) - w / 2.0) / W
    return 2.0 * np.sum(w * y * p) / np.sum(w * y) - 1.0


def weighted_median(y, w):
    """Smallest y whose cumulative weight reaches half of the total weight."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    o = np.argsort(y, kind="mergesort")
    y, w = y[o], w[o]
    cw = np.cumsum(w)
    return y[np.searchsorted(cw, cw[-1] / 2.0)]


def check_gini():
    """Sanity checks of the Gini implementation against closed-form cases."""
    assert abs(weighted_gini([1, 1, 1], [1, 1, 1])) < 1e-12
    assert abs(weighted_gini([0, 0, 1], [1, 1, 1]) - 2.0 / 3.0) < 1e-12
    # Weight-2 point equals two unit-weight points
    assert abs(weighted_gini([0, 1, 3], [2, 1, 1]) - weighted_gini([0, 0, 1, 3], [1, 1, 1, 1])) < 1e-12


def group_wmean(values, weights, groups):
    """Weighted mean of a Series by group, ignoring NaN values (denominator excludes them)."""
    ok = values.notna()
    num = (values.where(ok, 0.0) * weights).groupby(groups).sum()
    den = (weights * ok).groupby(groups).sum()
    return num / den


def main():
    check_gini()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hh_zip = fetch(HH_NAME)
    pp_zip = fetch(PP_NAME)

    print("Loading household records", flush=True)
    hh = read_members(hh_zip, "psam_hus", HH_COLS, HH_DTYPES)
    n_hh_all = len(hh)
    assert hh["SERIALNO"].is_unique, "household SERIALNO not unique"

    print("Loading person records", flush=True)
    pp = read_members(pp_zip, "psam_pus", PP_COLS, PP_DTYPES)

    # Occupied housing units: TYPEHUGQ==1 (housing unit) and NP>0 (not vacant).
    occ = hh[(hh["TYPEHUGQ"] == 1) & (hh["NP"] > 0)].copy()
    n_occ = len(occ)
    occ["st"] = occ["STATE"].astype(int)
    occ["puma"] = occ["PUMA"].astype(int)
    occ["area"] = occ["st"] * 100000 + occ["puma"]
    occ = occ.set_index("SERIALNO")

    # Person-level counts within household
    age = pp["AGEP"]
    cnt_u18 = pp[age < 18].groupby("SERIALNO").size()
    cnt_65 = pp[age >= 65].groupby("SERIALNO").size()

    # Reference person (head): RELSHIPP == 20
    head = pp[pp["RELSHIPP"] == 20].drop_duplicates("SERIALNO").set_index("SERIALNO")
    head = head[PP_COLS[2:]]

    df = occ.join(head, how="left")
    df["n_u18"] = cnt_u18.reindex(df.index).fillna(0)
    df["n_65"] = cnt_65.reindex(df.index).fillna(0)
    head_missing = df["AGEP"].isna()
    print(f"Occupied housing units: {n_occ:,}; without reference person: {int(head_missing.sum()):,}",
          flush=True)

    # Analysis features for all occupied units (context is built on this universe)
    def hd(cond):
        """Head dummy as float, -1 where the head is missing."""
        return np.where(head_missing, -1.0, cond.astype(float))

    f = pd.DataFrame(index=df.index)
    f["u_np"] = df["NP"]
    f["u_n_under18"] = df["n_u18"]
    f["u_n_65plus"] = df["n_65"]
    f["u_noc"] = df["NOC"].fillna(-1)
    f["u_owner"] = np.where(df["TEN"].isna(), -1.0, df["TEN"].isin([1, 2]).astype(float))
    f["u_tenure"] = df["TEN"].fillna(-1)
    f["u_rooms"] = df["RMSP"].fillna(-1)
    f["u_bedrooms"] = df["BDSP"].fillna(-1)
    f["u_vehicles"] = df["VEH"].fillna(-1)
    f["u_workers"] = df["WIF"].fillna(-1)
    f["u_hhl"] = df["HHL"].fillna(-1)
    f["u_bld"] = df["BLD"].fillna(-1)
    f["u_yrblt"] = df["YRBLT"].fillna(-1)
    f["u_yrblt_miss"] = df["YRBLT"].isna().astype(float)
    f["u_accessinet"] = df["ACCESSINET"].fillna(-1)
    f["u_accessinet_miss"] = df["ACCESSINET"].isna().astype(float)
    f["u_hupac"] = df["HUPAC"].fillna(-1)
    f["u_head_missing"] = head_missing.astype(float)
    f["u_head_age"] = np.where(head_missing, -1.0, df["AGEP"])
    f["u_head_female"] = hd(df["SEX"] == 2)
    f["u_head_schl"] = df["SCHL"].fillna(-1)
    f["u_head_schl_miss"] = (df["SCHL"].isna() & ~head_missing).astype(float)
    f["u_head_ba"] = hd(df["SCHL"] >= 21)
    f["u_head_white"] = hd(df["RAC1P"] == 1)
    f["u_head_black"] = hd(df["RAC1P"] == 2)
    f["u_head_asian"] = hd(df["RAC1P"] == 6)
    f["u_head_aian"] = hd(df["RAC1P"].isin([3, 4, 5]))
    f["u_head_hispanic"] = hd(df["HISP"] > 1)
    f["u_head_married"] = hd(df["MAR"] == 1)
    f["u_head_wkhp"] = df["WKHP"].fillna(-1)
    f["u_head_wkhp_miss"] = (df["WKHP"].isna() & ~head_missing).astype(float)
    f["u_head_employed"] = hd(df["ESR"].isin([1, 2, 4, 5]))
    f["u_head_disability"] = hd(df["DIS"] == 1)
    f["u_head_foreign_born"] = hd(df["NATIVITY"] == 2)

    # Area-context features: PUMA weighted means over all occupied units (no income)
    w = df["WGTP"].astype(float)
    area = df["area"]
    ctx_src = pd.DataFrame(index=df.index)
    ctx_src["c_share_owner"] = np.where(df["TEN"].isna(), np.nan, df["TEN"].isin([1, 2]).astype(float))
    ctx_src["c_mean_rooms"] = df["RMSP"]
    ctx_src["c_mean_vehicles"] = df["VEH"].where(df["VEH"] >= 0)
    ctx_src["c_share_head_ba"] = np.where(head_missing | df["SCHL"].isna(), np.nan,
                                          (df["SCHL"] >= 21).astype(float))
    ctx_src["c_share_head_employed"] = np.where(head_missing | df["ESR"].isna(), np.nan,
                                                df["ESR"].isin([1, 2, 4, 5]).astype(float))
    ctx_src["c_mean_hh_size"] = df["NP"]
    ctx_src["c_share_hh_children"] = np.where(df["NOC"].isna(), np.nan, (df["NOC"] > 0).astype(float))
    ctx_src["c_share_hh_65plus"] = (df["n_65"] > 0).astype(float)
    ctx_src["c_share_internet"] = np.where(df["ACCESSINET"].isin([1, 2, 3]),
                                           df["ACCESSINET"].isin([1, 2]).astype(float), np.nan)
    ctx_src["c_share_head_hispanic"] = np.where(head_missing | df["HISP"].isna(), np.nan,
                                                (df["HISP"] > 1).astype(float))
    ctx_src["c_share_head_black"] = np.where(head_missing | df["RAC1P"].isna(), np.nan,
                                             (df["RAC1P"] == 2).astype(float))
    ctx_src["c_share_sf_detached"] = np.where(df["BLD"].isna(), np.nan, (df["BLD"] == 2).astype(float))
    ctx_src["c_mean_head_age"] = np.where(head_missing, np.nan, df["AGEP"])
    ctx = pd.DataFrame({c: group_wmean(ctx_src[c], w, area) for c in CONTEXT_FEATURES})
    ctx.index.name = "area"
    n_ctx_areas = ctx.shape[0]
    assert ctx.notna().all().all(), "context feature with missing PUMA values"

    # Analysis sample: HINCP > 0
    sample = df[df["HINCP"] > 0].copy()
    out = pd.DataFrame({
        "serialno": sample.index.astype(str),
        "st": sample["st"].astype("int32").values,
        "puma": sample["puma"].astype("int32").values,
        "area": sample["area"].astype("int64").values,
        "wgtp": sample["WGTP"].astype("float64").values,
        "y": (sample["HINCP"] * sample["ADJINC"] / 1e6).astype("float64").values,
    })
    f_s = f.loc[sample.index].reset_index(drop=True)
    out = pd.concat([out, f_s.astype("float32")], axis=1)
    ctx_s = ctx.reindex(sample["area"].values).reset_index(drop=True).astype("float32")
    out = pd.concat([out, ctx_s], axis=1)

    assert out["y"].notna().all() and (out["y"] > 0).all()
    assert (out["wgtp"] > 0).all()
    assert not any("income" in c.lower() or c.startswith("c_") and "inc" in c.lower() for c in out.columns)

    out.to_parquet(PARQUET_PATH, index=False)
    print(f"Wrote {PARQUET_PATH} ({len(out):,} rows, {out.shape[1]} columns)", flush=True)

    # Summary statistics (national, weighted by WGTP)
    yv, wv = out["y"].values, out["wgtp"].values
    gini = weighted_gini(yv, wv)
    median = weighted_median(yv, wv)
    mean = np.average(yv, weights=wv)
    n_puma = out["area"].nunique()
    n_state = out["st"].nunique()

    lines = []
    lines.append("E2 data summary: ACS 2023 1-year PUMS, occupied housing units with HINCP > 0")
    lines.append("")
    lines.append("Sources: " + BASE_URL + HH_NAME + " (psam_husa/b.csv), " + BASE_URL + PP_NAME + " (psam_pusa/b.csv)")
    lines.append(f"Household records in file:                {n_hh_all:,}")
    lines.append(f"Occupied housing units (TYPEHUGQ==1, NP>0): {n_occ:,}")
    lines.append(f"Analysis households (HINCP > 0):          {len(out):,}")
    lines.append(f"Weighted households (sum WGTP):           {wv.sum():,.0f}")
    lines.append(f"Number of PUMAs (area ids):               {n_puma:,}")
    lines.append(f"Number of states (50 states + DC; PR not in national files): {n_state:,}")
    lines.append(f"PUMAs with context features:              {n_ctx_areas:,}")
    lines.append("")
    lines.append("Income y = HINCP * ADJINC / 1e6 (2023 dollars)")
    lines.append(f"National weighted Gini of y:              {gini:.4f}")
    lines.append(f"National weighted median of y:            {median:,.2f}")
    lines.append(f"National weighted mean of y:              {mean:,.2f}")
    lines.append("")
    lines.append("Unit-level features (u_*):")
    for k, v in UNIT_FEATURES.items():
        lines.append(f"  {k:<20} {v}")
    lines.append("")
    lines.append("Area-context features (c_*), PUMA-level WGTP-weighted means over all occupied units:")
    for k, v in CONTEXT_FEATURES.items():
        lines.append(f"  {k:<22} {v}")
    lines.append("")
    lines.append("Other columns: serialno (household id), st (state FIPS), puma (PUMA code),")
    lines.append("  area = st*100000 + puma, wgtp (household weight), y (adjusted HINCP).")
    lines.append("")
    lines.append("Notes:")
    lines.append("  Head = person with RELSHIPP==20, merged on SERIALNO; missing values filled with -1 plus indicators.")
    lines.append("  Occupied = TYPEHUGQ==1 and NP>0. Context features use no income variables.")
    lines.append("  Gini: weighted midpoint-rank formula, checked against closed-form cases.")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
