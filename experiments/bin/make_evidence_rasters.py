"""Rasterise mapped initiation points to the model grid as 3x3 evidence blocks.

Each mapped point becomes its centre cell plus the 8 neighbours -- at 10 m that
is a 30 x 30 m footprint, matching the positional tolerance a remotely mapped
point deserves. Output is written on the *exact* grid of the reference model
raster, so scoring is cell-to-cell with no resampling.

Outputs (all on the reference grid, uint8, nodata 255)
------------------------------------------------------
evidence_LS.tif / evidence_Runoff.tif / evidence_Firehose.tif
    1 = evidence block for that mechanism, 0 = everything else in the domain.
evidence_absence.tif
    1 = 3x3 block around each pseudo-absence point.
evidence_combined.tif
    1=LS, 2=Runoff, 3=Firehose, 8=absence, 0=no evidence. For QGIS display.
    Where mechanisms overlap, the lowest code wins; the count of contested
    cells is reported rather than silently resolved.

Overlap
-------
Points closer than ~30 m share cells, so the number of evidence cells is less
than 9 x n_points. That is expected and is reported per mechanism -- do not
assume 9x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

MECHANISMS = ["LS", "Runoff", "Firehose"]
CODES = {"LS": 1, "Runoff": 2, "Firehose": 3, "absence": 8}


def point_mask(points, transform, shape):
    """Boolean mask with one True per occupied cell."""
    mask = np.zeros(shape, bool)
    rows, cols = rasterio.transform.rowcol(transform, points.geometry.x, points.geometry.y)
    for r, c in zip(np.atleast_1d(rows), np.atleast_1d(cols)):
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            mask[r, c] = True
    return mask


def expand(mask, cells):
    """Grow each True cell to a cells x cells block (3 -> centre + 8 neighbours)."""
    if cells <= 1:
        return mask
    return binary_dilation(mask, structure=np.ones((cells, cells), bool))


def write_u8(path, arr, profile, **tags):
    p = {k: v for k, v in profile.items()
         if k not in ("blockxsize", "blockysize", "tiled", "interleave")}
    p.update(driver="GTiff", dtype="uint8", count=1, nodata=255, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("uint8"), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    ap.add_argument("--reference", type=Path, default=D / "pioneer/probability_fields/P_Landslide.tif",
                    help="Grid to rasterise onto; output matches it exactly")
    ap.add_argument("--initiation-points", type=Path,
                    default=D / "pioneer/Initiation_Points_final/Initiation_Points_final.shp")
    ap.add_argument("--absence-points", type=Path,
                    default=D / "pioneer/negative_points/random90_unbiased.shp")
    ap.add_argument("--window-cells", type=int, default=3,
                    help="Block size in cells; 3 = centre + 8 neighbours")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/evidence_rasters")
    args = ap.parse_args(argv)
    if args.window_cells % 2 == 0:
        ap.error("--window-cells must be odd so the block is centred on the point")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.reference) as src:
        profile, transform, shape, crs = src.profile.copy(), src.transform, src.shape, src.crs
        res = abs(src.transform.a)

    ip = gpd.read_file(args.initiation_points).to_crs(crs)
    absence = gpd.read_file(args.absence_points).to_crs(crs)

    report = {"reference_grid": str(args.reference), "cell_size_m": res,
              "window_cells": args.window_cells,
              "window_m": args.window_cells * res, "grid_cells": int(shape[0] * shape[1]),
              "mechanisms": {}}

    blocks = {}
    for mech in MECHANISMS:
        sel = ip[ip.FailureTyp == mech]
        blk = expand(point_mask(sel, transform, shape), args.window_cells)
        blocks[mech] = blk
        write_u8(args.output_dir / f"evidence_{mech}.tif", blk, profile,
                 mechanism=mech, n_points=len(sel), n_cells=int(blk.sum()),
                 window=f"{args.window_cells}x{args.window_cells} cells "
                        f"= {args.window_cells * res:.0f} m",
                 meaning="1 = mapped initiation evidence, 0 = no evidence")
        report["mechanisms"][mech] = {
            "n_points": int(len(sel)), "n_cells": int(blk.sum()),
            "cells_per_point": round(blk.sum() / max(1, len(sel)), 2),
            "overlap_loss_vs_9x": int(len(sel) * args.window_cells ** 2 - blk.sum()),
        }

    abs_blk = expand(point_mask(absence, transform, shape), args.window_cells)
    write_u8(args.output_dir / "evidence_absence.tif", abs_blk, profile,
             n_points=len(absence), n_cells=int(abs_blk.sum()),
             meaning="1 = pseudo-absence block")
    report["absence"] = {"n_points": int(len(absence)), "n_cells": int(abs_blk.sum())}

    # Combined display raster. Lowest code wins where mechanisms overlap; count
    # contested cells rather than pretending the assignment was unambiguous.
    combined = np.zeros(shape, "uint8")
    combined[abs_blk] = CODES["absence"]
    for mech in reversed(MECHANISMS):
        combined[blocks[mech]] = CODES[mech]
    pair_overlap = int(sum((blocks[a] & blocks[b]).sum()
                           for i, a in enumerate(MECHANISMS) for b in MECHANISMS[i + 1:]))
    any_ev = blocks["LS"] | blocks["Runoff"] | blocks["Firehose"]
    report["contested_cells_between_mechanisms"] = pair_overlap
    report["absence_cells_touching_evidence"] = int((abs_blk & any_ev).sum())
    report["total_evidence_cells"] = int(any_ev.sum())
    write_u8(args.output_dir / "evidence_combined.tif", combined, profile,
             codes=json.dumps(CODES), contested_cells=pair_overlap,
             meaning="1=LS 2=Runoff 3=Firehose 8=absence 0=none; lowest code wins on overlap")

    (args.output_dir / "evidence_report.json").write_text(json.dumps(report, indent=2))

    print(f"grid: {shape[1]}x{shape[0]} @ {res:.0f} m  "
          f"({args.window_cells}x{args.window_cells} = {args.window_cells * res:.0f} m blocks)\n")
    for mech, v in report["mechanisms"].items():
        print(f"  {mech:<9} {v['n_points']:>3} pts -> {v['n_cells']:>5} cells "
              f"({v['cells_per_point']:.1f}/pt, {v['overlap_loss_vs_9x']} lost to overlap)")
    print(f"  {'absence':<9} {report['absence']['n_points']:>3} pts -> "
          f"{report['absence']['n_cells']:>5} cells")
    if pair_overlap:
        print(f"\n  {pair_overlap} cells contested between mechanisms (lowest code wins)")
    if report["absence_cells_touching_evidence"]:
        print(f"  {report['absence_cells_touching_evidence']} absence cells touch evidence "
              f"-- excluded from negatives at scoring time")
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
