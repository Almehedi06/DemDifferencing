"""M4: retain erosion below one negative threshold on the continuous corrected DoD.

Reads a corrected DoD and writes a new directory. Modifies no pipeline and no input.

Why one negative threshold and nothing else
-------------------------------------------
The post DEM is UAV structure-from-motion (Agisoft Metashape, per the USGS data
release). SfM reconstructs only what the cameras see, so under closed canopy it
returns treetops and no ground. That puts a large, strictly POSITIVE tail in the
DoD wherever vegetation survived. The negative side carries no such contamination:
across every land-cover class the 1st percentile sits between -0.94 and -1.03 m.

So keeping only the negative side removes the vegetation problem without excluding
any terrain, which is why this needs no land-cover mask, no domain restriction and
no per-class calibration. Earlier attempts that filtered on |DoD| (M2) or excluded
forest by land-cover class (M3) both discarded real events; see experiments/methods/README.md.

The threshold is calibrated against the 89 mapped debris-flow initiation points in
the data release, not against terrain merely assumed to be stable. -0.50 m is the
largest threshold that still detects every landslide-initiated debris flow (47/47);
at -0.60 m that becomes 46/47.

Runoff- and firehose-initiated flows stay partly undetected at any usable threshold.
That is a property of the data, not of this threshold: those mechanisms do not
deplete half a metre at a point. Report detection by mechanism, never as one number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

NODATA = -9999.0


def screen_erosion(dod, threshold):
    """Retain DoD where it is more negative than -threshold; zero elsewhere.

    Zero means "no erosion resolved here", never "no change occurred". NoData in
    the input stays NoData, so the output keeps the full valid DoD footprint.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be a positive, finite depth in metres")
    valid = np.isfinite(dod)
    out = np.full(dod.shape, np.nan, dtype="float32")
    out[valid] = np.where(dod[valid] < -threshold, dod[valid], 0.0)
    return out, valid


def validate(dod, transform, points_path, threshold, radius=5):
    """Detection rate by initiation mechanism, using the deepest cell within
    `radius` pixels. Mapped points mark where failure began, which is not
    necessarily the deepest cell, and remote mapping carries its own position
    error -- so a point-exact test understates detection."""
    import geopandas as gpd
    pts = gpd.read_file(points_path)
    inverse = ~transform
    counts = {}
    for _, feature in pts.iterrows():
        col, row = inverse * (feature.geometry.x, feature.geometry.y)
        row, col = int(round(row)), int(round(col))
        window = dod[max(0, row - radius):row + radius + 1,
                     max(0, col - radius):col + radius + 1]
        window = window[np.isfinite(window)]
        deepest = float(window.min()) if window.size else np.nan
        mechanism = str(feature.get("FailureTyp", "unknown"))
        total, hit = counts.get(mechanism, (0, 0))
        counts[mechanism] = (total + 1, hit + int(np.isfinite(deepest) and deepest < -threshold))
    return {m: {"points": t, "detected": h, "detection_rate": h / t}
            for m, (t, h) in sorted(counts.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dod", type=Path, required=True, help="Continuous corrected DoD")
    ap.add_argument("--threshold-m", type=float, default=0.50,
                    help="Erosion depth in metres; calibrated value is 0.50")
    ap.add_argument("--initiation-points", type=Path,
                    help="Optional shapefile of mapped initiation points, for validation")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")

    with rasterio.open(args.dod) as src:
        if src.crs is None:
            ap.error(f"Missing CRS: {args.dod}")
        dod = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile, transform = src.profile.copy(), src.transform
    resolution = abs(float(transform.a))

    filtered, valid = screen_erosion(dod, args.threshold_m)
    retained = valid & (dod < -args.threshold_m)
    if not retained.any():
        raise SystemExit(f"No cell erodes below -{args.threshold_m} m; nothing written")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "dod_erosion.tif"
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(filtered), filtered, NODATA).astype("float32"), 1)
        dst.update_tags(
            method="M4 single negative threshold on the continuous corrected DoD",
            threshold_m=args.threshold_m, operation="corrected post minus pre",
            retained="Cells eroding deeper than the threshold, at their original magnitude",
            zero_meaning="No erosion resolved; NOT proof that nothing changed",
            positive_side="Discarded: the post DEM is SfM and its positive tail is vegetation",
            status="Provisional; calibrated on mapped initiation points, not on verified ground")

    # Verify what landed on disk rather than trusting the array in memory.
    with rasterio.open(out_path) as src:
        back = src.read(1, masked=True).astype("float64").filled(np.nan)
    if not np.array_equal(np.isfinite(back), valid):
        raise ValueError("Saved footprint does not match the valid input footprint")
    if not np.allclose(back[retained], dod[retained], atol=3e-4):
        raise ValueError("Saved output altered retained erosion magnitudes")
    if not np.all(back[valid & ~retained] == 0):
        raise ValueError("Saved output failed to zero cells above the threshold")

    report = {
        "method": "M4 single negative threshold",
        "dod": str(args.dod.resolve()), "threshold_m": args.threshold_m,
        "valid_pixels": int(valid.sum()), "retained_pixels": int(retained.sum()),
        "retained_fraction_of_valid": float(retained.sum() / valid.sum()),
        "retained_area_km2": float(retained.sum() * resolution ** 2 / 1e6),
        "median_retained_depth_m": float(np.median(dod[retained])),
        "deepest_m": float(dod[retained].min()),
        "limitations": [
            "Zero means no erosion resolved, not that the surface did not change.",
            "The positive side is discarded, so deposition is not mapped by this product.",
            "The post DEM is SfM: under surviving canopy there is no ground surface at "
            "all, so erosion there is undetectable and is reported as zero.",
            "Detection is mechanism-dependent; quote it by mechanism, not as one rate.",
            "Vertical datum is declared NAVD88 but no geoid model is documented, and the "
            "measured offset against 3DEP resembles a geoid separation. A constant offset "
            "cancels in differencing; an uncorrected tilt would not.",
            "The 3DEP acquisition date is unknown, so the change interval is unverified.",
        ],
    }
    if args.initiation_points:
        report["validation"] = validate(dod, transform, args.initiation_points, args.threshold_m)
    (args.output_dir / "m4_report.json").write_text(json.dumps(report, indent=2))

    print(f"Threshold: -{args.threshold_m:.2f} m")
    print(f"Retained: {report['retained_pixels']:,} px "
          f"({100 * report['retained_fraction_of_valid']:.2f}% of valid), "
          f"{report['retained_area_km2']:.3f} km2")
    print(f"Median retained depth {report['median_retained_depth_m']:.2f} m, "
          f"deepest {report['deepest_m']:.2f} m")
    for mechanism, v in report.get("validation", {}).items():
        print(f"  {mechanism:<10} {v['detected']:>3}/{v['points']:<3} "
              f"({100 * v['detection_rate']:.0f}%)")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
