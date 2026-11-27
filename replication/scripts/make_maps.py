#!/usr/bin/env python3
"""Publication maps for the economics paper.

Map 1 (output/Fig_map_acs.pdf): PUMA-level Gini of household income, contiguous US.
    True Gini, plug-in prediction (N) and corrected C4 estimate, rep 0, lgbm, unit regime.
Map 2 (output/Fig_map_africa.pdf): (a) DHS cluster wealth, (b) relative bias (%) of the
    plug-in absolute Gini, in-country validation, lgbm, averaged over survey years.

Run from anywhere:  python3 replication/scripts/make_maps.py
Boundary files are read from replication/data/geo/ and downloaded there if missing.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]          # replication/
OUT = ROOT / "output"
GEO = ROOT / "data" / "geo"
DHS_CSV = ROOT / "data" / "dhs" / "dhs_clusters.csv"

PUMA_FILE = GEO / "cb_2020_us_puma20_500k.zip"
STATE_FILE = GEO / "cb_2022_us_state_500k.zip"
NE_FILE = GEO / "ne_50m_admin_0_countries.geojson"
URLS = {
    # 2022 PUMA20 cartographic file is not on the server (404); the 2020 vintage is used.
    PUMA_FILE: "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_puma20_500k.zip",
    STATE_FILE: "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip",
    NE_FILE: "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
             "geojson/ne_50m_admin_0_countries.geojson",
}

EQUAL_AREA_US = "EPSG:5070"
AFRICA_AEA = "+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=17 +datum=WGS84 +units=m +no_defs"
AFRICA_BOX = box(-20, -36, 53, 38)                  # lon/lat crop for map 2
NON_CONTIG_FIPS = ["02", "15", "60", "66", "69", "72", "78"]  # AK, HI, territories
FIG_W = 174 / 25.4                                  # 174 mm journal width in inches

# DHS country keys -> Natural Earth ADMIN names
DHS_TO_NE = {
    "angola": "Angola", "benin": "Benin", "burkina_faso": "Burkina Faso",
    "cameroon": "Cameroon", "cote_d_ivoire": "Ivory Coast",
    "democratic_republic_of_congo": "Democratic Republic of the Congo",
    "ethiopia": "Ethiopia", "ghana": "Ghana", "guinea": "Guinea", "kenya": "Kenya",
    "lesotho": "Lesotho", "malawi": "Malawi", "mali": "Mali",
    "mozambique": "Mozambique", "nigeria": "Nigeria", "rwanda": "Rwanda",
    "senegal": "Senegal", "sierra_leone": "Sierra Leone",
    "tanzania": "United Republic of Tanzania", "togo": "Togo", "uganda": "Uganda",
    "zambia": "Zambia", "zimbabwe": "Zimbabwe",
}

GREY_TRAINING = "#d9d9d9"   # PUMAs with no row in rep 0 (training sample)
BOUNDARY = "#c4c4c4"
STATE_EDGE = "#4d4d4d"


def ensure_geo_files() -> None:
    GEO.mkdir(parents=True, exist_ok=True)
    for path, url in URLS.items():
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "hatch.linewidth": 0.4,
    })


def panel_label(ax, text: str) -> None:
    # placed just above the map, top-left, so it never sits on top of geography
    ax.text(0.0, 1.01, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def colours_for(values: pd.Series, norm: Normalize, cmap) -> list:
    clipped = np.clip(values.to_numpy(dtype=float), norm.vmin, norm.vmax)
    return [cmap(norm(v)) for v in clipped]


# ----------------------------------------------------------------------------- map 1
def make_map_acs() -> None:
    cols = ["rep", "regime", "model", "estimator", "level", "unit", "index", "truth", "estimate"]
    e2 = pd.read_csv(OUT / "e2_results_raw.csv", usecols=cols)
    sel = e2[(e2["rep"] == 0) & (e2["model"] == "lgbm") & (e2["level"] == "puma")
             & (e2["index"] == "gini") & (e2["regime"] == "unit")]

    def by_unit(est: str) -> pd.DataFrame:
        s = sel[sel["estimator"] == est]
        assert not s["unit"].duplicated().any(), f"duplicate units for {est}"
        return s.set_index("unit")

    n_rows, c4_rows = by_unit("N"), by_unit("C4")
    assert set(n_rows.index) == set(c4_rows.index), "N and C4 unit sets differ"
    max_truth_diff = np.max(np.abs(n_rows["truth"] - c4_rows["truth"].reindex(n_rows.index)))
    print(f"[map1] target units (rep 0): {len(n_rows)}; max |truth_N - truth_C4| = {max_truth_diff:.2e}")
    target = set(n_rows.index)

    pumas = gpd.read_file(f"zip://{PUMA_FILE}")
    pumas["unit"] = pumas["GEOID20"].astype(int)
    poly_units = set(pumas["unit"])
    print(f"[map1] PUMA polygons: {len(pumas)}, unique IDs: {len(poly_units)}")
    matched_all = len(target & poly_units)
    print(f"[map1] match rate, all {len(target)} target units: {matched_all / len(target):.2%} "
          f"({matched_all} matched)")

    pumas = pumas[~pumas["STATEFP20"].isin(NON_CONTIG_FIPS)].copy()
    contig_target = target & set(pumas["unit"])
    print(f"[map1] contiguous US polygons: {len(pumas)}; target units in contiguous US: "
          f"{len(contig_target)} of {len([u for u in target if u // 100000 not in (2, 15, 72)])} "
          f"(match rate {len(contig_target) / len([u for u in target if u // 100000 not in (2, 15, 72)]):.2%})")

    pumas["truth"] = pumas["unit"].map(n_rows["truth"])
    pumas["plugin"] = pumas["unit"].map(n_rows["estimate"])
    pumas["corrected"] = pumas["unit"].map(c4_rows["estimate"])
    pumas = pumas.to_crs(EQUAL_AREA_US)

    states = gpd.read_file(f"zip://{STATE_FILE}")
    states = states[~states["STATEFP"].isin(NON_CONTIG_FIPS)].to_crs(EQUAL_AREA_US)
    states["geometry"] = states.geometry.simplify(300, preserve_topology=True)

    grey_mask = pumas["truth"].isna()
    n_grey = int(grey_mask.sum())
    print(f"[map1] contiguous PUMAs drawn grey (no row in rep 0, i.e. training sample): {n_grey}")

    vmin, vmax = 0.30, 0.60
    norm = Normalize(vmin, vmax)
    cmap = matplotlib.colormaps["viridis"]

    # simplify PUMA outlines (metres, EPSG:5070) to keep the PDF small; topology is kept
    pumas["geometry"] = pumas.geometry.simplify(500, preserve_topology=True)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 70 / 25.4))
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.88, bottom=0.22)
    panels = [("truth", "(a) True"), ("plugin", "(b) Plug-in"), ("corrected", "(c) Corrected (C4)")]
    for ax, (col, label) in zip(axes, panels):
        have = pumas.loc[~grey_mask]
        # PUMA fills are rasterized so the PDF stays small; boundaries and states stay vector
        pumas.loc[~grey_mask].plot(
            ax=ax, color=colours_for(have[col], norm, cmap), edgecolor=BOUNDARY,
            linewidth=0.1, rasterized=True)
        pumas.loc[grey_mask].plot(
            ax=ax, color=GREY_TRAINING, edgecolor=BOUNDARY, linewidth=0.1, rasterized=True)
        states.plot(ax=ax, facecolor="none", edgecolor=STATE_EDGE, linewidth=0.45)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, loc="left", fontsize=8, pad=2)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.06, pad=0.02,
                        aspect=60, extend="both", ticks=np.arange(0.30, 0.601, 0.05))
    cbar.set_label("PUMA Gini coefficient of household income (colour clipped to 0.30-0.60)")
    cbar.outline.set_linewidth(0.4)

    path = OUT / "Fig_map_acs.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"[map1] wrote {path}")
    print(f"[map1] colour limits: {vmin} to {vmax} (clipped, colourbar extended both ends)")
    print(f"[map1] PUMAs drawn grey: {n_grey}")


# ----------------------------------------------------------------------------- map 2
def make_map_africa() -> None:
    dhs = pd.read_csv(DHS_CSV)
    dhs = dhs.dropna(subset=["lat", "lon", "wealthpooled"])
    lo, hi = np.percentile(dhs["wealthpooled"], [2, 98])
    print(f"[map2] DHS clusters: {len(dhs)} in {dhs['country'].nunique()} countries; "
          f"wealth colour limits (2nd-98th pct): {lo:.3f} to {hi:.3f}")

    # bias per country-year, then averaged over years
    e3 = pd.read_csv(OUT / "e3_results_raw.csv")
    f = e3[(e3["scheme"] == "in_country") & (e3["model"] == "lgbm") & (e3["estimator"] == "N")].copy()
    assert not f.duplicated(["country", "year"]).any()
    f["bias_pct"] = 100 * (f["abs_gini_est"] - f["abs_gini_true"]) / f["abs_gini_true"]
    per_country = f.groupby("country")["bias_pct"].agg(["mean", "count"])
    unmatched_dhs = sorted(set(per_country.index) - set(DHS_TO_NE))
    assert not unmatched_dhs, f"no Natural Earth name for {unmatched_dhs}"
    ne_bias = {DHS_TO_NE[c]: per_country.loc[c, "mean"] for c in per_country.index}

    ne = gpd.read_file(NE_FILE)
    africa = ne[ne["CONTINENT"] == "Africa"][["ADMIN", "geometry"]].copy()
    africa = gpd.clip(africa, AFRICA_BOX).to_crs(AFRICA_AEA)
    ne_names = set(africa["ADMIN"])
    missing_ne = [n for n in ne_bias if n not in ne_names]
    print(f"[map2] African countries in Natural Earth: {len(africa)}; "
          f"DHS countries matched: {len(ne_bias) - len(missing_ne)} of {len(ne_bias)}; "
          f"unmatched: {missing_ne if missing_ne else 'none'}")
    africa["bias"] = africa["ADMIN"].map(ne_bias)

    points = gpd.GeoDataFrame(
        dhs, geometry=[Point(xy) for xy in zip(dhs["lon"], dhs["lat"])], crs="EPSG:4326"
    ).to_crs(AFRICA_AEA)
    xmin, ymin, xmax, ymax = africa.total_bounds
    pad_x, pad_y = 0.01 * (xmax - xmin), 0.01 * (ymax - ymin)

    bg_fill, bg_edge = "#f2f2f2", "#b5b5b5"
    wealth_norm = Normalize(lo, hi)
    wealth_cmap = matplotlib.colormaps["viridis"]
    bias_lim = 30.0
    bias_norm = Normalize(-bias_lim, bias_lim)
    bias_cmap = matplotlib.colormaps["RdBu_r"]

    ax_w = FIG_W / 2 * 0.95
    ax_aspect = (xmax - xmin) / (ymax - ymin)        # width / height of the Africa crop
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, ax_w / ax_aspect + 1.0),
                             gridspec_kw={"wspace": 0.04})
    ax_a, ax_b = axes

    africa.plot(ax=ax_a, color=bg_fill, edgecolor=bg_edge, linewidth=0.25)
    ax_a.scatter(points.geometry.x, points.geometry.y,
                 c=dhs["wealthpooled"].clip(lo, hi), cmap=wealth_cmap, norm=wealth_norm,
                 s=1.5, linewidths=0)
    panel_label(ax_a, "(a) DHS cluster wealth")

    in_s = africa[africa["bias"].notna()]
    africa.plot(ax=ax_b, color=bg_fill, edgecolor=bg_edge, linewidth=0.25)
    in_s.plot(ax=ax_b, color=colours_for(in_s["bias"], bias_norm, bias_cmap),
              edgecolor=bg_edge, linewidth=0.25)
    panel_label(ax_b, "(b) Plug-in Gini bias (%)")

    for ax in axes:
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal")
        ax.set_axis_off()

    sm_a = ScalarMappable(norm=wealth_norm, cmap=wealth_cmap)
    sm_a.set_array([])
    cb_a = fig.colorbar(sm_a, ax=ax_a, orientation="horizontal", fraction=0.05, pad=0.02,
                        aspect=25, extend="both")
    cb_a.set_label("Cluster wealth index (wealthpooled)")
    sm_b = ScalarMappable(norm=bias_norm, cmap=bias_cmap)
    sm_b.set_array([])
    cb_b = fig.colorbar(sm_b, ax=ax_b, orientation="horizontal", fraction=0.05, pad=0.02,
                        aspect=25, extend="both", ticks=[-30, -15, 0, 15, 30])
    cb_b.set_label("Relative bias (%), mean over survey years")
    for cb in (cb_a, cb_b):
        cb.outline.set_linewidth(0.4)

    path = OUT / "Fig_map_africa.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[map2] wrote {path}")
    print(f"[map2] bias colour limits: -{bias_lim:g} to +{bias_lim:g} (RdBu_r, centred at 0)")
    print("[map2] per-country mean relative bias (%), lgbm, in-country, plug-in N:")
    for country, row in per_country.sort_values("mean").iterrows():
        flag = "" if abs(row["mean"]) <= bias_lim else "  (outside colour limits)"
        print(f"    {country:32s} {row['mean']:+7.2f}  ({int(row['count'])} survey-year(s)){flag}")
    not_in_sample = sorted(set(africa["ADMIN"]) - set(ne_bias))
    print(f"[map2] African countries shown as not in sample: {len(not_in_sample)}")


def main() -> None:
    ensure_geo_files()
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    make_map_acs()
    make_map_africa()


if __name__ == "__main__":
    main()
