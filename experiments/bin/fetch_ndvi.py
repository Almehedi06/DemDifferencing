"""Summer NDVI before, during and after the 2024 Pioneer Fire, on the project grids.

Two sources, both free and both with a near-infrared band:

* Sentinel-2 L2A (Element84 Earth Search, 10 m): a median composite of clear scenes
  for each summer window, on the Landlab 10 m lidar-footprint grid extended by
  BUFFER_M on every side (cells stay aligned; crop by window to match). Single-band
  NDVI; how many clear scenes back each composite is in the manifest. Every scene
  June-October is also summarised in a time series, which is how the fire-year
  window was chosen.
* NAIP 2023 (Microsoft Planetary Computer, 0.6 m): pre-fire NDVI averaged onto the
  1 m dDEM grid, extended by BUFFER_M the same way. NAIP is uncalibrated 8-bit, so
  its NDVI is relative: good for telling canopy from ground, not for comparing
  values with Sentinel-2.

Writes a fresh directory; reads nothing local except the two reference grids.

    conda activate ml_debris
    python experiments/bin/fetch_ndvi.py --output-dir .../Downloads/pioneer/ddem_correction/ndvi

Data only; the maps are drawn by make_ddem_correction_figures.py. Each summer is
also written as a Landlab-ready ESRI ASCII grid (vegetation__ndvi_summer<year>.asc)
matching the inputs in lidar_footprint_10m/. To write only those from NDVI already
downloaded:

    python experiments/bin/fetch_ndvi.py --output-dir .../ndvi --landlab-only

Change in NDVI between two summers already downloaded, post minus pre (negative =
vegetation lost), on the Sentinel-2 grid and on the Landlab grid (GeoTIFF and ASCII):

    python experiments/bin/fetch_ndvi.py --output-dir .../ndvi --delta 2023 2025

Things that are easy to get wrong
---------------------------------
* Reflectance is DN / 10000. Earth Search already removed the +1000 offset from
  baseline 04.00+ scenes (``earthsearch:boa_offset_applied``); its metadata still
  lists offset -0.1, and applying it makes dark pixels negative. The script refuses
  any scene where the offset has not been applied.
* Clear pixels are SCL classes 4-7 (vegetation, bare soil, water, unclassified).
  Cloud, cloud shadow, cirrus, snow, defective and dark/topographic-shadow pixels
  are dropped.
* Scene screening. Every scene inside a window is saved in scenes/<year>/, then
  screened before compositing. A scene is dropped when
  1. less than MIN_CLEAR of the land is clear (SCL), or
  2. its median aerosol optical thickness (L2A AOT band) exceeds MAX_AOT: smoke.
     Smoke passes the SCL cloud mask but lowers NDVI over the whole scene. Here the
     smoky scenes sit at AOT 0.26-0.27 and every other scene at 0.21 or less; or
  3. its median NDVI is more than OUTLIER_Z robust standard deviations
     (1.4826 x MAD) from the window's median, a backstop for anything else.
  Kept and dropped scenes, with the reason, are in s2_ndvi_timeseries.csv and the
  manifest.
* The fire burned through this area in mid-to-late July 2024: median NDVI over
  the grid was 0.57 on 10 July, 0.42 on 20 July and 0.24 on 4 August, then
  steady near 0.35.
  A July-September 2024 median would mix unburned and burned scenes, so the
  fire-year window starts after the drop (see WINDOWS).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")      # remote COG reads occasionally reset
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

PIONEER = Path("/mnt/c/Users/amehedi/Downloads/pioneer")
GRID_10M = PIONEER / "output/lidar_footprint_10m/burn__severity.tif"   # Landlab 10 m lidar-footprint grid
GRID_1M = PIONEER / "pre.tif"                                         # 1 m dDEM grid
# Landlab inputs: grid, header and model domain (cells with elevation) come from here
LANDLAB_REF = PIONEER / "lidar_footprint_10m/topographic__elevation.asc"
LANDLAB_NODATA = -9999.0
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLEAR_SCL = (4, 5, 6, 7)
BUFFER_M = 500          # context around the survey footprint, for maps; a multiple of both resolutions
MAX_CLOUD = 60
MIN_CLEAR = 0.5         # least share of clear land for a scene to enter a composite
MAX_AOT = 0.25          # median aerosol optical thickness above which a scene is smoke-affected
OUTLIER_Z = 3.0         # robust z beyond which a scene's median NDVI marks it as contaminated
# label: (start, end, short title, meaning). The fire-year window starts once the
# time series has settled after the burn and ends before October's low sun.
WINDOWS = {
    "2023": ("2023-07-01", "2023-09-30", "pre-fire", "summer before the fire"),
    "2024": ("2024-08-15", "2024-09-30", "just after the burn", "fire year, after the fire burned this area"),
    "2025": ("2025-07-01", "2025-09-30", "post-fire", "first summer after the fire, before the Dec 2025 debris flows"),
    "2026": ("2026-07-01", "2026-09-30", "post-debris-flow", "summer after the debris flows (UAV and NPS lidar season)"),
}


def grid(path, buffer_m=BUFFER_M):
    """The reference raster's grid, grown by buffer_m on every side with cells kept aligned."""
    from rasterio.transform import Affine
    with rasterio.open(path) as src:
        t = src.transform
        nx, ny = buffer_m / t.a, buffer_m / -t.e
        if nx != int(nx) or ny != int(ny):
            raise ValueError(f"buffer {buffer_m} m is not a whole number of {t.a} m cells")
        return {"crs": src.crs, "transform": Affine(t.a, 0, t.c - buffer_m, 0, t.e, t.f + buffer_m),
                "width": src.width + 2 * int(nx), "height": src.height + 2 * int(ny)}


def bbox_lonlat(g):
    from rasterio.warp import transform_bounds
    left, top = g["transform"] * (0, 0)
    right, bottom = g["transform"] * (g["width"], g["height"])
    return list(transform_bounds(g["crs"], "EPSG:4326", left, bottom, right, top))


def read_on(href, g, resampling):
    """Read band 1 of a remote raster warped onto grid g."""
    with rasterio.open(href) as src, WarpedVRT(src, crs=g["crs"], transform=g["transform"],
                                                width=g["width"], height=g["height"],
                                                resampling=resampling) as vrt:
        return vrt.read(1)


def scene_ndvi(item, g):
    if item.properties.get("earthsearch:boa_offset_applied") is not True:
        raise ValueError(f"{item.id}: reflectance offset not applied; DN/10000 would be wrong")
    red = read_on(item.assets["red"].href, g, Resampling.bilinear).astype("float32") / 10000
    nir = read_on(item.assets["nir"].href, g, Resampling.bilinear).astype("float32") / 10000
    scl = read_on(item.assets["scl"].href, g, Resampling.nearest)
    clear = np.isin(scl, CLEAR_SCL) & (red > 0) & (nir > 0)
    ndvi = np.where(clear, (nir - red) / np.maximum(nir + red, 1e-6), np.nan).astype("float32")
    aot = read_on(item.assets["aot"].href, g, Resampling.nearest).astype("float32")
    aot_median = float(np.median(aot[aot > 0])) / 1000 if (aot > 0).any() else np.nan
    return ndvi, scl, aot_median


def sentinel2(g, out_dir):
    from pystac_client import Client

    client = Client.open(EARTH_SEARCH)
    bbox = bbox_lonlat(g)
    items = []
    for year in sorted({w[0][:4] for w in WINDOWS.values()}):
        found = client.search(collections=["sentinel-2-l2a"], bbox=bbox,
                              datetime=f"{year}-06-01/{year}-10-31",
                              query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items()
        items += list(found)
    if not items:
        raise SystemExit("No Sentinel-2 scenes found")
    items.sort(key=lambda i: i.datetime)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: scene_ndvi(i, g), items))      # (ndvi, scl, aot) per scene

    rows = []
    for item, (ndvi, scl, aot) in zip(items, results):
        land = np.isfinite(ndvi) & (scl != 6)
        rows.append({"date": item.datetime.date().isoformat(), "scene": item.id,
                     "clear_land_fraction": round(float(land.mean()), 3), "aot_median": round(aot, 3),
                     "median_ndvi_land": round(float(np.median(ndvi[land])), 4) if land.any() else np.nan,
                     "window": "", "robust_z": "", "used": "", "drop_reason": ""})

    profile = {"driver": "GTiff", "crs": g["crs"], "transform": g["transform"], "width": g["width"],
               "height": g["height"], "count": 1, "dtype": "float32", "nodata": np.nan,
               "compress": "deflate"}
    composites = {}
    for label, (start, end, short, meaning) in WINDOWS.items():
        inside = [k for k, r in enumerate(rows) if start <= r["date"] <= end]
        if not inside:
            raise SystemExit(f"No scenes in the {label} window {start}..{end}")
        clear_enough = [k for k in inside if rows[k]["clear_land_fraction"] >= MIN_CLEAR
                        and rows[k]["aot_median"] <= MAX_AOT]
        medians = np.array([rows[k]["median_ndvi_land"] for k in clear_enough])
        centre = np.median(medians)
        spread = 1.4826 * np.median(np.abs(medians - centre))
        scene_dir = out_dir / "scenes" / label
        scene_dir.mkdir(parents=True, exist_ok=True)
        for k in inside:
            r = rows[k]
            z = (r["median_ndvi_land"] - centre) / spread if spread > 0 else 0.0
            reason = (f"clear land {r['clear_land_fraction']:.0%} < {MIN_CLEAR:.0%}"
                      if r["clear_land_fraction"] < MIN_CLEAR else
                      f"smoke: AOT {r['aot_median']:.2f} > {MAX_AOT}" if r["aot_median"] > MAX_AOT else
                      f"median NDVI {r['median_ndvi_land']:.3f}, robust z {z:+.1f}" if abs(z) > OUTLIER_Z else "")
            r.update(window=label, robust_z=round(float(z), 2), used="no" if reason else "yes", drop_reason=reason)
            with rasterio.open(scene_dir / f"{r['date']}_{r['scene']}.tif", "w", **profile) as dst:
                dst.write(results[k][0], 1)
                dst.update_tags(scene=r["scene"], date=r["date"], used=r["used"], drop_reason=reason,
                                median_ndvi_land=str(r["median_ndvi_land"]),
                                clear_land_fraction=str(r["clear_land_fraction"]))
        use = [results[k][0] for k in inside if rows[k]["used"] == "yes"]
        if not use:
            raise SystemExit(f"Every scene in the {label} window was dropped")
        stack = np.stack(use)
        n_clear = np.isfinite(stack).sum(0).astype("float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # pixels never clear stay NaN
            median = np.nanmedian(stack, axis=0).astype("float32")
        if np.isfinite(median).mean() < 0.95:
            raise SystemExit(f"{label}: only {np.isfinite(median).mean():.0%} of the grid has a clear pixel")
        path = out_dir / f"s2_ndvi_summer{label}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(median, 1)
            dst.set_band_description(1, "NDVI, median of clear scenes")
            dst.update_tags(source="Sentinel-2 L2A, Element84 Earth Search", window=f"{start}..{end}",
                            meaning=meaning, scenes_used=str(len(use)), scenes_in_window=str(len(inside)),
                            clear_scl=str(CLEAR_SCL),
                            screening=f"clear >= {MIN_CLEAR}, AOT <= {MAX_AOT}, |robust z| <= {OUTLIER_Z}")
        composites[label] = {"window": f"{start}..{end}", "short": short, "meaning": meaning, "scenes": len(use),
                             "scenes_in_window": len(inside),
                             "dropped": {rows[k]["date"]: rows[k]["drop_reason"] for k in inside
                                         if rows[k]["used"] == "no"},
                             "median_ndvi": float(np.nanmedian(median)), "path": path.name,
                             "clear_scenes_per_pixel_median": float(np.median(n_clear)),
                             "share_of_pixels_with_fewer_than_3_clear_scenes": float(np.mean(n_clear < 3))}

    with open(out_dir / "s2_ndvi_timeseries.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return composites, rows


def naip_2023(g, out_dir):
    import planetary_computer
    from pystac_client import Client

    client = Client.open(PLANETARY, modifier=planetary_computer.sign_inplace)
    items = sorted(client.search(collections=["naip"], bbox=bbox_lonlat(g),
                                 datetime="2023-06-01/2023-10-31").items(), key=lambda i: i.datetime)
    if not items:
        raise SystemExit("No NAIP 2023 imagery over the grid")
    ndvi = np.full((g["height"], g["width"]), np.nan, "float32")
    for item in items:                                            # earlier date wins where tiles overlap
        with rasterio.open(item.assets["image"].href) as src, WarpedVRT(
                src, crs=g["crs"], transform=g["transform"], width=g["width"], height=g["height"],
                resampling=Resampling.average) as vrt:
            red, nir = (vrt.read(b).astype("float32") for b in (1, 4))
            covered = vrt.read_masks(1) > 0
        new = covered & np.isnan(ndvi) & (red + nir > 0)
        ndvi[new] = (nir[new] - red[new]) / (nir[new] + red[new])
    path = out_dir / "naip_ndvi_2023.tif"
    with rasterio.open(path, "w", driver="GTiff", crs=g["crs"], transform=g["transform"],
                       width=g["width"], height=g["height"], count=1, dtype="float32",
                       nodata=np.nan, compress="deflate", predictor=3) as dst:
        dst.write(ndvi, 1)
        dst.update_tags(source="NAIP 2023, Planetary Computer", dates=", ".join(
            sorted({i.datetime.date().isoformat() for i in items})),
            note="Uncalibrated 8-bit DN: relative NDVI, not comparable with Sentinel-2 values")
    return {"dates": sorted({i.datetime.date().isoformat() for i in items}), "tiles": len(items),
            "coverage": float(np.isfinite(ndvi).mean()), "path": path.name}, ndvi


def landlab_header_and_domain():
    """Header lines, transform, shape and model domain of the Landlab inputs (LANDLAB_REF)."""
    header = []
    with open(LANDLAB_REF) as f:
        for line in f:
            if line.split()[0].lower() not in ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize",
                                               "nodata_value"):
                break
            header.append(line.rstrip("\n"))
    if not any(h.lower().startswith("nodata_value") for h in header):
        header.append(f"NODATA_value  {LANDLAB_NODATA}")
    with rasterio.open(LANDLAB_REF) as ref:
        return header, ref.transform, ref.shape, ref.read(1) != LANDLAB_NODATA


def crop_to_landlab(array, transform, t, shape, name):
    """Cut the Landlab window out of a raster on the extended Sentinel-2 grid (same cells, no resampling)."""
    col, row = (t.c - transform.c) / transform.a, (transform.f - t.f) / -transform.e
    if transform.a != t.a or transform.e != t.e or col != int(col) or row != int(row):
        raise ValueError(f"{name} does not share the Landlab cells")
    out = array[int(row):int(row) + shape[0], int(col):int(col) + shape[1]]
    if out.shape != shape:
        raise ValueError(f"{name} does not cover the Landlab grid")
    return out


def delta_ndvi(out_dir, pre, post):
    """NDVI change, post minus pre, written three ways with one definition:

    s2_delta_ndvi_<pre>_to_<post>.tif            extended Sentinel-2 grid (for maps)
    vegetation__delta_ndvi_<pre>_to_<post>.tif   Landlab grid, EPSG:6339, nodata -9999
    vegetation__delta_ndvi_<pre>_to_<post>.asc   Landlab grid, same header and domain as the inputs

    A cell missing in either summer is missing in the change, never filled.
    """
    for label in (pre, post):
        if label not in WINDOWS:
            raise ValueError(f"unknown summer {label}; choose from {list(WINDOWS)}")
    with rasterio.open(out_dir / f"s2_ndvi_summer{pre}.tif") as a, rasterio.open(out_dir / f"s2_ndvi_summer{post}.tif") as b:
        if a.shape != b.shape or a.transform != b.transform or a.crs != b.crs:
            raise ValueError(f"summers {pre} and {post} are not on one grid")
        d = (b.read(1).astype("float32") - a.read(1).astype("float32"))       # NaN in either stays NaN
        profile, s2_transform = a.profile.copy(), a.transform
    tags = {"meaning": f"NDVI change, post minus pre: summer {post} minus summer {pre}; negative = vegetation lost",
            "pre": f"{pre} ({WINDOWS[pre][0]} to {WINDOWS[pre][1]}, {WINDOWS[pre][3]})",
            "post": f"{post} ({WINDOWS[post][0]} to {WINDOWS[post][1]}, {WINDOWS[post][3]})",
            "source": "Sentinel-2 L2A median summer composites, s2_ndvi_summer<year>.tif"}
    stem = f"delta_ndvi_{pre}_to_{post}"
    native = out_dir / f"s2_{stem}.tif"
    profile.update(dtype="float32", nodata=float("nan"), compress="deflate")
    with rasterio.open(native, "w", **profile) as dst:
        dst.write(d, 1)
        dst.update_tags(**tags)

    header, t, shape, domain = landlab_header_and_domain()
    cut = crop_to_landlab(d, s2_transform, t, shape, native.name)
    out = np.where(domain & np.isfinite(cut), cut, LANDLAB_NODATA).astype("float32")
    tif = out_dir / f"vegetation__{stem}.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32",
                       crs=profile["crs"], transform=t, nodata=LANDLAB_NODATA, compress="deflate") as dst:
        dst.write(out, 1)
        dst.update_tags(**tags)
    asc = out_dir / f"vegetation__{stem}.asc"
    with open(asc, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, out, fmt="%.4f")
    v = cut[domain & np.isfinite(cut)]
    result = {"meaning": tags["meaning"], "pre": tags["pre"], "post": tags["post"],
              "files": [native.name, tif.name, asc.name], "domain_cells": int(domain.sum()),
              "domain_cells_without_change": int((domain & ~np.isfinite(cut)).sum()),
              "median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
              "share_negative": float(np.mean(v < 0))}
    manifest = out_dir / "ndvi_manifest.json"
    record = json.loads(manifest.read_text()) if manifest.exists() else {}
    record.setdefault("delta", {})[stem] = result
    manifest.write_text(json.dumps(record, indent=2))
    return result


def landlab_ascii(out_dir):
    """Each summer NDVI as an ESRI ASCII grid identical in grid, header and domain to
    the Landlab inputs (LANDLAB_REF): vegetation__ndvi_summer<year>.asc.

    The Sentinel-2 grid shares the Landlab cells, so this is a crop, not a resample.
    Cells outside the domain are LANDLAB_NODATA, like every other input. Domain cells
    with no clear scene are also LANDLAB_NODATA and are counted in the report, never
    filled.
    """
    header = []
    with open(LANDLAB_REF) as f:
        for line in f:
            if line.split()[0].lower() not in ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize",
                                               "nodata_value"):
                break
            header.append(line.rstrip("\n"))
    with rasterio.open(LANDLAB_REF) as ref:
        t, shape = ref.transform, ref.shape
        domain = ref.read(1) != LANDLAB_NODATA
    if not any(h.lower().startswith("nodata_value") for h in header):
        header.append(f"NODATA_value  {LANDLAB_NODATA}")

    report = {}
    for label in WINDOWS:
        path = out_dir / f"s2_ndvi_summer{label}.tif"
        with rasterio.open(path) as src:
            col, row = (t.c - src.transform.c) / src.transform.a, (src.transform.f - t.f) / -src.transform.e
            if src.transform.a != t.a or src.transform.e != t.e or col != int(col) or row != int(row):
                raise ValueError(f"{path.name} does not share the Landlab cells")
            col, row = int(col), int(row)
            ndvi = src.read(1)[row:row + shape[0], col:col + shape[1]]
        if ndvi.shape != shape:
            raise ValueError(f"{path.name} does not cover the Landlab grid")
        missing = domain & ~np.isfinite(ndvi)
        out = np.where(domain & np.isfinite(ndvi), ndvi, LANDLAB_NODATA)
        asc = out_dir / f"vegetation__ndvi_summer{label}.asc"
        with open(asc, "w") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, out, fmt="%.4f")
        report[asc.name] = {"domain_cells": int(domain.sum()), "domain_cells_without_ndvi": int(missing.sum())}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--landlab-only", action="store_true",
                    help="Only write the Landlab ASCII grids from NDVI already in --output-dir")
    ap.add_argument("--delta", nargs=2, metavar=("PRE", "POST"),
                    help="Only write the NDVI change POST minus PRE (e.g. 2023 2025) from NDVI already in --output-dir")
    args = ap.parse_args(argv)
    if args.delta:
        r = delta_ndvi(args.output_dir, *args.delta)
        print(f"{', '.join(r['files'])}: median {r['median']:+.3f}, p05 {r['p05']:+.3f}, p95 {r['p95']:+.3f}, "
              f"{r['share_negative']:.0%} of domain cells lost NDVI; "
              f"{r['domain_cells_without_change']} of {r['domain_cells']} domain cells without a value")
        return
    if args.landlab_only:
        for name, r in landlab_ascii(args.output_dir).items():
            print(f"{name}: {r['domain_cells']} domain cells, {r['domain_cells_without_ndvi']} without NDVI")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    composites, rows = sentinel2(grid(GRID_10M), args.output_dir)
    naip, _ = naip_2023(grid(GRID_1M), args.output_dir)
    (args.output_dir / "ndvi_manifest.json").write_text(json.dumps(
        {"sentinel2_10m_grid": f"{GRID_10M}, extended {BUFFER_M} m", "naip_1m_grid": f"{GRID_1M}, extended {BUFFER_M} m",
         "screening": {"min_clear_land_fraction": MIN_CLEAR, "max_aot": MAX_AOT, "max_abs_robust_z": OUTLIER_Z},
         "sentinel2": composites, "naip": naip, "timeseries_scenes": len(rows)}, indent=2))
    for label, c in composites.items():
        print(f"S2 {label}: {c['scenes']:2d} of {c['scenes_in_window']} scenes used {c['window']}, "
              f"median NDVI {c['median_ndvi']:.3f}; dropped {c['dropped']}")
    print(f"NAIP 2023: {naip['tiles']} tiles {naip['dates']}, coverage {naip['coverage']:.1%}")
    for name, r in landlab_ascii(args.output_dir).items():
        print(f"{name}: {r['domain_cells']} domain cells, {r['domain_cells_without_ndvi']} without NDVI")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
