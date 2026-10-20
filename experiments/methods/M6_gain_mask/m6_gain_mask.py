"""M6: keep elevation gains only where debris-flow deposition is possible.

Reads the M5 DoD and writes a new directory: the masked DoD, the NDVI-test DoD, the
mask reasons and a report. Maps are drawn by
experiments/bin/make_ddem_correction_figures.py. Modifies no pipeline and no input.

Why
---
The post DEM is UAV structure-from-motion with no vegetation removed, so trees,
standing dead trees and shrubs read as gains. On the +/-1 m colour scale that is
the hillside-wide blue in the headwaters. Two rules remove it:

1. Hillslope gains are set to zero. Gains are kept only in channels (G > 200,
   the driving index of the pre DEM) or where the slope is below 15 degrees,
   which is roughly where debris flows begin to deposit. Steep slopes outside
   channels do not receive deposits.
2. Green gains are set to zero everywhere, channels included. Greenness is the
   green leaf index (2G-R-B)/(2G+R+B) of the same-day 10 cm UAV orthophoto
   averaged to 1 m. Fresh deposits six months after the event are bare, so a
   green gain is vegetation even inside a channel.

Losses are never touched, so erosion is exactly the M5 product. Zero means
"no gain resolved here", not "no change": deposition under trees cannot be seen
in this data.

NDVI test (Erkan's rule), written alongside for comparison
----------------------------------------------------------
Gains are set to zero where post-fire summer NDVI is green (Sentinel-2 summer
2026, above --ndvi-threshold) outside channels (G > 200); channels keep every
gain so channel deposition is recorded. Applied to gains only, so erosion under
green cells is kept. The UAV camera is RGB only, so true NDVI exists only at
Sentinel-2's 10 m, too coarse to separate a channel from the trees beside it,
which is why the final mask uses the 1 m orthophoto greenness instead.

dNDVI test, written alongside for the record
--------------------------------------------
The hillslope rule as in M6, with the "not green" rule replaced by "vegetation
changed across the debris-flow event": gains are set to zero where Sentinel-2
NDVI changed by less than --dndvi-threshold between summer 2025 (before the
debris flows) and summer 2026 (after). The threshold sits near the year-to-year
noise of unchanged ground (NMAD 0.035). Caveats: fresh deposits on already bare
ground do not change NDVI, and standing dead trees changed NDVI during the fire,
not across this event.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

NODATA = -9999.0
KEPT, HILLSLOPE_GAIN, GREEN_GAIN, NO_DATA = 0, 1, 2, 255


def read(path, ref=None):
    """Band 1 as float32 with NaN for nodata, and the grid it sits on. With ``ref``,
    the raster must sit on that grid."""
    with rasterio.open(path) as src:
        grid = SimpleNamespace(crs=src.crs, transform=src.transform, shape=src.shape,
                               height=src.height, width=src.width, profile=src.profile.copy())
        if ref is not None and (grid.crs != ref.crs or grid.transform != ref.transform or grid.shape != ref.shape):
            raise ValueError(f"{path} is not on the DoD grid")
        return src.read(1, masked=True).astype("float32").filled(np.nan), grid


def greenness(ortho_path, ref):
    """Green leaf index of the orthophoto, averaged to the DoD grid. NaN where the
    orthophoto does not fully cover a cell."""
    rgba = np.zeros((4, ref.height, ref.width), "float32")
    with rasterio.open(ortho_path) as src:
        if src.count < 4:
            raise ValueError(f"{ortho_path}: expected RGB + alpha")
        for b in range(4):
            reproject(rasterio.band(src, b + 1), rgba[b], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.average,
                      num_threads=4)
    r, g, b, alpha = rgba
    gli = (2 * g - r - b) / np.maximum(2 * g + r + b, 1)
    return np.where(alpha > 250, gli, np.nan)


def ndvi_on_grid(path, ref):
    """NDVI resampled (nearest) onto the DoD grid; NaN where it has no value."""
    out = np.full((ref.height, ref.width), np.nan, "float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=ref.transform, dst_crs=ref.crs,
                  dst_nodata=np.nan, resampling=Resampling.nearest)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dod", type=Path, required=True, help="M5 detilted DoD")
    ap.add_argument("--driving-index", type=Path, required=True, help="G of the pre DEM (G_pre_dinf.tif)")
    ap.add_argument("--slope-percent", type=Path, required=True, help="Slope of the pre DEM in percent")
    ap.add_argument("--ortho", type=Path, required=True, help="UAV orthophoto, RGB + alpha")
    ap.add_argument("--channel-g", type=float, default=200.0)
    ap.add_argument("--max-deposition-slope-deg", type=float, default=15.0)
    ap.add_argument("--green-index", type=float, default=0.04,
                    help="Green leaf index above which a cell is vegetation")
    ap.add_argument("--ndvi", type=Path, required=True,
                    help="Post-fire summer NDVI for the NDVI test (s2_ndvi_summer2026.tif)")
    ap.add_argument("--ndvi-threshold", type=float, default=0.5)
    ap.add_argument("--ndvi-before", type=Path, required=True,
                    help="Summer NDVI before the debris flows, for the dNDVI test (s2_ndvi_summer2025.tif)")
    ap.add_argument("--dndvi-threshold", type=float, default=0.05)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")

    dod, ref = read(args.dod)
    g, _ = read(args.driving_index, ref)
    slope_pct, _ = read(args.slope_percent, ref)
    gli = greenness(args.ortho, ref)
    ndvi = ndvi_on_grid(args.ndvi, ref)
    dndvi = ndvi - ndvi_on_grid(args.ndvi_before, ref)

    valid = np.isfinite(dod)
    no_terrain = valid & ~(np.isfinite(g) & np.isfinite(slope_pct))
    channel = g > args.channel_g
    gentle = slope_pct < 100 * np.tan(np.radians(args.max_deposition_slope_deg))
    gain = valid & (dod > 0)
    hillslope_gain = gain & ~channel & ~gentle                 # includes cells with no terrain values
    green_gain = gain & ~hillslope_gain & (gli > args.green_index)

    out = np.where(hillslope_gain | green_gain, 0.0, dod).astype("float32")
    reason = np.full(dod.shape, NO_DATA, "uint8")
    reason[valid] = KEPT
    reason[hillslope_gain] = HILLSLOPE_GAIN
    reason[green_gain] = GREEN_GAIN

    ndvi_green_gain = gain & ~channel & (ndvi > args.ndvi_threshold)   # NDVI test: Erkan's rule
    ndvi_test = np.where(ndvi_green_gain, 0.0, dod).astype("float32")
    unchanged_gain = gain & ~hillslope_gain & (np.abs(dndvi) < args.dndvi_threshold)   # dNDVI test
    dndvi_test = np.where(hillslope_gain | unchanged_gain, 0.0, dod).astype("float32")

    loss = valid & (dod < 0)
    if not all(np.array_equal(a[loss], dod[loss]) for a in (out, ndvi_test, dndvi_test)):
        raise ValueError("A loss was altered; only gains may be masked")
    if not all(m.any() for m in (hillslope_gain, green_gain, ndvi_green_gain, unchanged_gain)):
        raise ValueError("A rule removed no cells; check the inputs and thresholds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = ref.profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", count=1, compress="deflate")
    dod_path = args.output_dir / "dod_gain_masked.tif"
    with rasterio.open(dod_path, "w", **{**profile, "dtype": "float32", "nodata": NODATA}) as dst:
        dst.write(np.where(valid, out, NODATA).astype("float32"), 1)
        dst.update_tags(method="M6: M5 DoD with gains kept only where deposition is possible",
                        source=str(args.dod.resolve()),
                        zero_meaning="No gain resolved (hillslope or vegetation); NOT proof of no change",
                        losses="Unchanged from M5")
    with rasterio.open(args.output_dir / "dod_ndvi_test.tif", "w", **{**profile, "dtype": "float32", "nodata": NODATA}) as dst:
        dst.write(np.where(valid, ndvi_test, NODATA).astype("float32"), 1)
        dst.update_tags(method="NDVI test: M5 gains set to zero where summer-2026 NDVI > "
                               f"{args.ndvi_threshold} outside channels (G > {args.channel_g})",
                        ndvi=str(args.ndvi.resolve()), losses="Unchanged from M5")
    with rasterio.open(args.output_dir / "dod_dndvi_test.tif", "w", **{**profile, "dtype": "float32", "nodata": NODATA}) as dst:
        dst.write(np.where(valid, dndvi_test, NODATA).astype("float32"), 1)
        dst.update_tags(method="dNDVI test: M6 hillslope rule, plus gains set to zero where |NDVI 2026 - NDVI 2025| < "
                               f"{args.dndvi_threshold}", losses="Unchanged from M5")
    with rasterio.open(args.output_dir / "mask_reason.tif", "w",
                       **{**profile, "dtype": "uint8", "nodata": NO_DATA}) as dst:
        dst.write(reason, 1)
        dst.update_tags(codes=f"{KEPT}=kept, {HILLSLOPE_GAIN}=gain on hillslope (outside channel, "
                              f"slope >= {args.max_deposition_slope_deg} deg), {GREEN_GAIN}=green gain, "
                              f"{NO_DATA}=no data")

    with rasterio.open(dod_path) as src:                       # verify what landed on disk
        back = src.read(1, masked=True).astype("float32").filled(np.nan)
    if not np.array_equal(np.isfinite(back), valid) or not np.allclose(back[valid], out[valid], atol=1e-6):
        raise ValueError("Saved DoD differs from the computed one")

    km2 = lambda m: float(m.sum() / 1e6 * abs(ref.transform.a * ref.transform.e))
    report = {
        "method": "M6 gain mask", "dod": str(args.dod.resolve()),
        "thresholds": {"channel_g": args.channel_g, "max_deposition_slope_deg": args.max_deposition_slope_deg,
                       "green_index": args.green_index},
        "area_km2": {"valid": km2(valid), "channel": km2(valid & channel), "gentle": km2(valid & gentle),
                     "gains_before": km2(gain), "hillslope_gains_zeroed": km2(hillslope_gain),
                     "green_gains_zeroed": km2(green_gain), "gains_kept": km2(gain & ~hillslope_gain & ~green_gain),
                     "gains_kept_above_0.5m": km2(valid & (out > 0.5))},
        "share_above_0.5m": {"before": float(np.mean(dod[valid] > 0.5)), "after": float(np.mean(out[valid] > 0.5))},
        "ndvi_test": {"ndvi": str(args.ndvi.resolve()), "ndvi_threshold": args.ndvi_threshold,
                      "gains_zeroed_km2": km2(ndvi_green_gain),
                      "gains_kept_km2": km2(gain & ~ndvi_green_gain),
                      "share_above_0.5m": float(np.mean(ndvi_test[valid] > 0.5)),
                      "cells_without_ndvi": int((valid & ~np.isfinite(ndvi)).sum())},
        "dndvi_test": {"ndvi_before": str(args.ndvi_before.resolve()), "dndvi_threshold": args.dndvi_threshold,
                       "unchanged_gains_zeroed_km2": km2(unchanged_gain),
                       "gains_kept_km2": km2(gain & ~hillslope_gain & ~unchanged_gain),
                       "share_above_0.5m": float(np.mean(dndvi_test[valid] > 0.5)),
                       "cells_without_dndvi": int((valid & ~np.isfinite(dndvi)).sum())},
        "cells_without_terrain_values": int(no_terrain.sum()),
        "limitations": [
            "Zero means no gain resolved, not no change; deposition under trees is not visible in SfM data.",
            "Real deposition on steep slopes outside channels is removed with the vegetation.",
            "Non-green gains in channels (dead trees, shaded trees) remain.",
            "Losses are the M5 product unchanged, including its limits.",
        ],
    }
    (args.output_dir / "m6_report.json").write_text(json.dumps(report, indent=2))
    a = report["area_km2"]
    print(f"Gains: {a['gains_before']:.2f} km2 -> kept {a['gains_kept']:.2f} km2 "
          f"(hillslope {a['hillslope_gains_zeroed']:.2f}, green {a['green_gains_zeroed']:.2f} zeroed)")
    print(f"Cells above +0.5 m: {report['share_above_0.5m']['before']:.1%} -> {report['share_above_0.5m']['after']:.1%} "
          f"(NDVI test: {report['ndvi_test']['share_above_0.5m']:.1%}, dNDVI test: "
          f"{report['dndvi_test']['share_above_0.5m']:.1%})")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
