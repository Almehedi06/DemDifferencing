"""Change statistics inside each mapped inventory polygon, Eagle Creek 2018-2021.

Takes the dDEM and the inventory as final. For every polygon of one epoch: the
inventory attributes plus three change layers inside it, each summarised the same
way (mean, median, 5/25/75/95th percentile, min, max, std, share of cells below zero):

  dz  elevation change, m            21_18.tif as prepared (prep/dz_2m.tif)
  dg  change in G = a S^(7/6), m     source_zone_pipeline_mosaic/DeltaG.tif
  dc  change in plan curvature       curvature_post - curvature_pre (same folder)

dz also gets erosion, deposition and net volume. Depths the mapper recorded in
feet are converted to metres. A second table gives the same statistics per
inventory class and per process group, pooling every cell of the class.
No threshold is applied.

    python experiments/bin/inventory_dod_stats.py --epoch 2018_2021
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask

FT = 0.3048
D = Path("/mnt/c/Users/amehedi/Downloads")
INVENTORY = D / "eaglecreek_inventory.gpkg"
TERRAIN = D / "eagle_creek/source_zone_pipeline_mosaic"
DDEM = D / "eagle_creek/prep/dz_2m.tif"
DELTA_G = TERRAIN / "DeltaG.tif"
CURV_PRE, CURV_POST = TERRAIN / "curvature_pre.tif", TERRAIN / "curvature_post.tif"
OUT = D / "eagle_creek/inventory_vs_ddem_2018_2021"
LAYERS = ("dz", "dg", "dc")


def values(src, geom):
    """Cell values of one raster inside one polygon; nodata as NaN, same shape for every layer."""
    try:
        a, _ = rasterio.mask.mask(src, [geom], crop=True, filled=False)
    except ValueError:                    # outside the raster
        return np.array([])
    return a[0].astype("float64").filled(np.nan)


def describe(v, prefix):
    """The statistics every layer gets, for one polygon or one class."""
    v = v[np.isfinite(v)]
    if not v.size:
        return {f"{prefix}_n": 0}
    return {f"{prefix}_n": v.size, f"{prefix}_mean": v.mean(), f"{prefix}_median": np.median(v),
            f"{prefix}_p05": np.percentile(v, 5), f"{prefix}_p25": np.percentile(v, 25),
            f"{prefix}_p75": np.percentile(v, 75), f"{prefix}_p95": np.percentile(v, 95),
            f"{prefix}_min": v.min(), f"{prefix}_max": v.max(), f"{prefix}_std": v.std(),
            f"{prefix}_frac_negative": np.mean(v < 0)}


def dz_volumes(dz, cell):
    dz = dz[np.isfinite(dz)]
    return {"net_volume_m3": dz.sum() * cell, "erosion_volume_m3": dz[dz < 0].sum() * cell,
            "deposition_volume_m3": dz[dz > 0].sum() * cell}


def class_summary(out, cells, cell, by):
    """One row per class: polygon counts and areas, pooled cells of each layer, mapped depths, agreement."""
    rows = []
    for name, grp in out.groupby(by, sort=True):
        signed = grp[grp.mapped_sign != "mixed"]
        agree = ((signed.mapped_sign == "erosion") & (signed.net_volume_m3 < 0)) | \
                ((signed.mapped_sign == "deposition") & (signed.net_volume_m3 > 0))
        r = {"group_by": by, "class": name, "n_polygons": len(grp), "total_area_m2": grp.area_m2.sum(),
             "median_area_m2": grp.area_m2.median()}
        for layer in LAYERS:
            r.update(describe(np.concatenate([cells[layer][i] for i in grp.index]), layer))
            if layer == "dz":
                r.update(dz_volumes(np.concatenate([cells["dz"][i] for i in grp.index]), cell))
        r.update(polygon_dz_median_median=grp.dz_median.median(), polygon_net_volume_median_m3=grp.net_volume_m3.median(),
                 polygon_dg_median_median=grp.dg_median.median(), polygon_dc_median_median=grp.dc_median.median(),
                 n_mapped_erosion_depth=int(grp.mapped_erosion_depth_m.notna().sum()),
                 mapped_erosion_depth_median_m=grp.mapped_erosion_depth_m.median(),
                 n_mapped_deposition_depth=int(grp.mapped_deposition_depth_m.notna().sum()),
                 mapped_deposition_depth_median_m=grp.mapped_deposition_depth_m.median(),
                 sign_agreement_pct=100 * agree.mean() if len(signed) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows).rename(columns={"dz_n": "n_cells", "dz_frac_negative": "frac_cells_eroded"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", default="2018_2021")
    args = ap.parse_args(argv)

    inv = gpd.read_file(INVENTORY, layer="inventory")
    inv["Primary"] = inv["Primary"].str.strip()
    inv = inv[inv.epoch == args.epoch].copy()

    srcs = {k: rasterio.open(p) for k, p in
            {"dz": DDEM, "dg": DELTA_G, "c_pre": CURV_PRE, "c_post": CURV_POST}.items()}
    ref = srcs["dz"]
    assert ref.crs.linear_units.lower().startswith("met"), ref.crs.linear_units
    for k, s in srcs.items():
        assert s.nodata is not None, f"{k} must declare nodata"
        assert s.shape == ref.shape and s.transform.almost_equals(ref.transform) and s.crs == ref.crs, \
            f"{k} is not on the dDEM grid"
    cell = abs(ref.res[0] * ref.res[1])
    inv = inv.to_crs(ref.crs)
    inv["area_m2"] = inv.area

    rows, cells = [], {k: {} for k in LAYERS}
    for i, geom in inv.geometry.items():
        v = {k: values(s, geom) for k, s in srcs.items()}
        layer = {"dz": v["dz"], "dg": v["dg"],
                 "dc": v["c_post"] - v["c_pre"] if v["c_pre"].size else np.array([])}   # NaN where either epoch is nodata
        r = {}
        for k, arr in layer.items():
            a = arr.ravel()
            a = a[np.isfinite(a)]
            cells[k][i] = a
            r.update(describe(a, k))
        r["coverage_pct"] = 100 * r["dz_n"] * cell / inv.at[i, "area_m2"]
        r.update(dz_volumes(cells["dz"][i], cell))
        rows.append(r)
    for s in srcs.values():
        s.close()
    stats = pd.DataFrame(rows, index=inv.index).rename(columns={"dz_n": "n_cells", "dz_frac_negative": "frac_cells_eroded"})

    keep = ["OBJECTID", "Primary", "Unique_ID", "Basin_Stre", "Notes", "epoch", "area_m2",
            "Min_Erosion_Depth_ft", "Max_Deposition_Depth_ft"]
    out = inv[keep + ["geometry"]].join(stats)
    out["mapped_erosion_depth_m"] = out.pop("Min_Erosion_Depth_ft") * FT
    out["mapped_deposition_depth_m"] = out.pop("Max_Deposition_Depth_ft") * FT
    out["process"] = out["Primary"].str[:2].map(
        lambda c: "landslide" if c in ("01", "02", "03", "04", "05", "06") else
        "df_pre_initiation" if c in ("07", "08") else "df_initiation" if c in ("09", "10", "11", "12", "13", "14", "15") else
        "df_transport" if c in ("16", "17", "18", "19", "20") else "df_deposition")
    out["mapped_sign"] = np.select([out["Primary"].str.contains(r"evacuated|erosion|Erosion|rills", regex=True),
                                    out["Primary"].str.contains(r"deposition|Deposition|Levee|jam", regex=True)],
                                   ["erosion", "deposition"], "mixed")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([class_summary(out, cells, cell, "Primary"), class_summary(out, cells, cell, "process")])
    summary.round(4).to_csv(OUT / f"class_summary_{args.epoch}.csv", index=False)
    out.round(4).to_file(OUT / f"polygons_ddem_stats_{args.epoch}.gpkg", driver="GPKG")
    out.drop(columns="geometry").round(4).to_csv(OUT / f"polygons_ddem_stats_{args.epoch}.csv", index=False)
    (OUT / "inputs.json").write_text(json.dumps({
        "dz": str(DDEM), "dg": str(DELTA_G), "dc": f"{CURV_POST} - {CURV_PRE}",
        "inventory": str(INVENTORY), "epoch": args.epoch, "cell_m2": cell}, indent=2))
    print(f"{len(out)} polygons in {args.epoch}; {int((out.coverage_pct >= 50).sum())} with >= 50% dDEM coverage")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
