"""M5: remove the large-scale tilt from the corrected DoD.

Reads a corrected DoD and writes a new directory. Modifies no pipeline and no input.

Why
---
M2 aligns the pair with one vertical shift and one translation (Nuth-Kaab). That
centres the histogram over the whole scene but cannot remove a tilt, and the UAV
SfM survey used no ground control. In 250 m blocks the histogram peak of the M2
DoD runs from about -0.4 m in the southwest to +0.5 m in the northeast
headwaters, about 1.1 m across the scene. On a +/-1 m colour scale that, plus
canopy, reads as hillside-wide deposition in the headwaters.

Why the plane is fitted on block histogram peaks, not on pixels
---------------------------------------------------------------
Surviving canopy adds a long positive tail to a block's DoD values but leaves the
histogram peak -- the ground that did not change -- where it is. A plane fitted
on pixels fits the canopy too; that is why xDEM's deramp made M1.5 worse. The
peak of each block is a canopy-robust estimate of local zero change, and a robust
plane through the peaks is the tilt.

The intercept is fitted too, so after correction the typical change in every
block is zero. The report states how far that moves the overall level.

Validation is a checkerboard split: the plane is fitted on half the blocks and
scored on the other half, then refitted on all blocks for the output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import rasterio

NODATA = -9999.0
BINS = np.arange(-3.0, 3.0001, 0.05)       # metres; the peak is sought within +/-3 m
MIN_PIXELS = 5000                          # valid pixels a block needs for a peak


def block_peaks(dod, transform, block_px):
    """Histogram peak of each block, with the block-centre coordinates.

    Returns x, y (map units), peak (m) and the block's (row, col) index, for
    blocks holding at least MIN_PIXELS valid pixels.
    """
    mids = (BINS[:-1] + BINS[1:]) / 2
    out = []
    for i in range(dod.shape[0] // block_px):
        for j in range(dod.shape[1] // block_px):
            v = dod[i * block_px:(i + 1) * block_px, j * block_px:(j + 1) * block_px]
            v = v[np.isfinite(v)]
            if v.size < MIN_PIXELS:
                continue
            counts = np.convolve(np.histogram(v, BINS)[0], np.ones(3) / 3, "same")
            x, y = transform * ((j + 0.5) * block_px, (i + 0.5) * block_px)
            out.append((x, y, mids[np.argmax(counts)], i, j))
    if len(out) < 10:
        raise ValueError(f"Only {len(out)} blocks have {MIN_PIXELS} valid pixels; too few to fit a plane")
    x, y, peak, i, j = map(np.array, zip(*out))
    return x, y, peak, i.astype(int), j.astype(int)


def fit_plane(x, y, z, iterations=20):
    """Plane z = c0 + c1*x + c2*y by iteratively reweighted least squares (L1-like),
    so the few edge blocks with odd peaks do not steer it."""
    a = np.c_[np.ones_like(x), x, y]
    w = np.ones_like(z)
    for _ in range(iterations):
        c = np.linalg.lstsq(a * w[:, None], z * w, rcond=None)[0]
        w = 1 / np.sqrt(np.maximum(np.abs(z - a @ c), 0.05))
    return c


def plane_at(c, x, y):
    return c[0] + c[1] * x + c[2] * y


def nmad(a):
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def m4_validate():
    """M4's detection test, imported rather than copied so there is one definition."""
    path = Path(__file__).resolve().parents[1] / "M4_erosion_threshold" / "m4_erosion_threshold.py"
    spec = importlib.util.spec_from_file_location("m4_erosion_threshold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate


def figure(before, after, peaks_before, peaks_after, shape_blocks, extent, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grid(p):
        g = np.full(shape_blocks, np.nan)
        g[p[1], p[2]] = p[0]
        return g

    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    panels = [(before[::8, ::8], "dDEM before (M2)"), (after[::8, ::8], "dDEM after tilt removal (M5)"),
              (grid(peaks_before), "250 m block histogram peak, before"),
              (grid(peaks_after), "250 m block histogram peak, after")]
    for a, (arr, title) in zip(ax.flat, panels):
        im = a.imshow(arr, cmap="RdBu", vmin=-1, vmax=1, extent=extent, interpolation="nearest")
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6, label="post minus pre (m); blue = higher after")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def histogram_summary(values):
    """Peak (1 cm bins) and NMAD of all valid DoD values. The peak, not the median,
    is the balance test: canopy drags the median up but leaves the peak alone."""
    counts, edges = np.histogram(values, np.arange(-3.0, 3.0001, 0.01))
    k = np.argmax(counts)
    return {"peak_m": float((edges[k] + edges[k + 1]) / 2), "median_m": float(np.median(values)),
            "nmad_m": nmad(values)}


def histogram_figure(values, summary, path):
    """Histogram of the output DoD. The x-axis stops at +/-2 m; the share of pixels
    beyond it (mostly canopy above +2 m) is stated on the figure, not hidden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.arange(-2.0, 2.0001, 0.02)
    mids = (edges[:-1] + edges[1:]) / 2
    share = 100 * np.histogram(values, edges)[0] / values.size
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.axvline(0, color="#8a8a85", lw=1, ls="--", zorder=1)
    ax.plot(mids, share, color="#2a78d6", lw=2, zorder=2)
    k = np.argmax(share)
    ax.annotate(f"peak {summary['peak_m']:+.2f} m", (mids[k], share[k]), xytext=(12, -4),
                textcoords="offset points", fontsize=9, color="#3d3d3a")
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, share.max() * 1.12)
    ax.set_xlabel("post minus pre (m)")
    ax.set_ylabel("% of pixels per 2 cm")
    ax.set_title("dDEM histogram, all pixels", loc="left")
    ax.text(0.02, 0.97, f"NMAD {summary['nmad_m']:.2f} m\n{values.size / 1e6:.1f} M pixels at 1 m\n"
            f"not shown: {100 * np.mean(values > 2):.1f}% above +2 m,\n"
            f"{100 * np.mean(values < -2):.1f}% below -2 m", transform=ax.transAxes, va="top",
            fontsize=8.5, color="#6b6b66", linespacing=1.5)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dod", type=Path, required=True, help="Continuous corrected DoD (M2)")
    ap.add_argument("--block-m", type=float, default=250.0, help="Block size in metres")
    ap.add_argument("--initiation-points", type=Path,
                    help="Optional shapefile of mapped initiation points: M4 detection before and after")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")

    with rasterio.open(args.dod) as src:
        if src.crs is None:
            ap.error(f"Missing CRS: {args.dod}")
        dod = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile, transform, bounds = src.profile.copy(), src.transform, src.bounds
    block_px = int(round(args.block_m / abs(transform.a)))
    valid = np.isfinite(dod)

    x, y, peak, bi, bj = block_peaks(dod, transform, block_px)
    x0, y0 = x.mean(), y.mean()                      # centre coordinates for a well-conditioned fit
    xs, ys = x - x0, y - y0

    # Checkerboard: fit on one half, score the other, both ways round.
    held_before, held_after = [], []
    for part in (0, 1):
        fit, test = (bi + bj) % 2 == part, (bi + bj) % 2 != part
        c = fit_plane(xs[fit], ys[fit], peak[fit])
        held_before.append(peak[test])
        held_after.append(peak[test] - plane_at(c, xs[test], ys[test]))
    held_before, held_after = np.concatenate(held_before), np.concatenate(held_after)

    c = fit_plane(xs, ys, peak)
    rows, cols = np.indices(dod.shape)
    px, py = transform * (cols + 0.5, rows + 0.5)
    plane = plane_at(c, px - x0, py - y0)
    detilted = np.where(valid, dod - plane, np.nan)
    _, _, peak_after, ai, aj = block_peaks(detilted, transform, block_px)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "dod_detilted.tif"
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(valid, detilted, NODATA).astype("float32"), 1)
        dst.update_tags(
            method="M5: M2 DoD minus a plane fitted on 250 m block histogram peaks",
            source=str(args.dod.resolve()), operation="corrected post minus pre, tilt removed",
            plane_m_per_km=f"east {c[1] * 1000:+.4f}, north {c[2] * 1000:+.4f}",
            status="Provisional; canopy is still in the post DEM (SfM), so the positive side is not deposition")

    # Verify what landed on disk rather than trusting the array in memory.
    with rasterio.open(out_path) as src:
        back = src.read(1, masked=True).astype("float64").filled(np.nan)
    if not np.array_equal(np.isfinite(back), valid):
        raise ValueError("Saved footprint does not match the valid input footprint")
    if not np.allclose(back[valid], detilted[valid], atol=3e-4):
        raise ValueError("Saved output differs from the computed detilted DoD")

    plane_valid = plane[valid]
    hist = {"before": histogram_summary(dod[valid]), "after": histogram_summary(detilted[valid])}
    report = {
        "method": "M5 tilt removal on block histogram peaks",
        "dod": str(args.dod.resolve()), "block_m": args.block_m, "blocks": int(peak.size),
        "histogram_all_pixels": hist,
        "plane": {"east_m_per_km": float(c[1] * 1000), "north_m_per_km": float(c[2] * 1000),
                  "removed_min_m": float(plane_valid.min()), "removed_max_m": float(plane_valid.max()),
                  "removed_median_m": float(np.median(plane_valid))},
        "block_peaks_before": {"median_m": float(np.median(peak)), "nmad_m": nmad(peak),
                               "p10_m": float(np.percentile(peak, 10)), "p90_m": float(np.percentile(peak, 90))},
        "block_peaks_after": {"median_m": float(np.median(peak_after)), "nmad_m": nmad(peak_after),
                              "p10_m": float(np.percentile(peak_after, 10)),
                              "p90_m": float(np.percentile(peak_after, 90))},
        "held_out_blocks": {"n": int(held_before.size),
                            "median_abs_peak_before_m": float(np.median(np.abs(held_before))),
                            "median_abs_peak_after_m": float(np.median(np.abs(held_after))),
                            "nmad_before_m": nmad(held_before), "nmad_after_m": nmad(held_after)},
        "limitations": [
            "Only a plane is removed. Smaller-scale warps or offsets between flight blocks remain.",
            "The post DEM is SfM with canopy; the positive side is still not deposition under trees.",
            "The intercept assumes the typical 250 m block did not change; a real uniform lowering "
            "across the whole scene would be removed with it.",
        ],
    }
    if args.initiation_points:
        validate = m4_validate()
        report["m4_detection_at_0.50m"] = {
            "before": validate(dod, transform, args.initiation_points, 0.50),
            "after": validate(detilted, transform, args.initiation_points, 0.50)}
    (args.output_dir / "m5_report.json").write_text(json.dumps(report, indent=2))

    shape_blocks = (dod.shape[0] // block_px, dod.shape[1] // block_px)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    figure(dod, detilted, (peak, bi, bj), (peak_after, ai, aj), shape_blocks, extent,
           args.output_dir / "m5_detilt.png")
    histogram_figure(detilted[valid], hist["after"], args.output_dir / "m5_histogram.png")

    p, h = report["plane"], report["held_out_blocks"]
    print(f"Histogram peak {hist['before']['peak_m']:+.3f} -> {hist['after']['peak_m']:+.3f} m, "
          f"NMAD {hist['before']['nmad_m']:.3f} -> {hist['after']['nmad_m']:.3f} m")
    print(f"Plane: east {p['east_m_per_km']:+.3f} m/km, north {p['north_m_per_km']:+.3f} m/km; "
          f"removed {p['removed_min_m']:+.2f} to {p['removed_max_m']:+.2f} m")
    print(f"Held-out block peaks: median |peak| {h['median_abs_peak_before_m']:.2f} -> "
          f"{h['median_abs_peak_after_m']:.2f} m, NMAD {h['nmad_before_m']:.2f} -> {h['nmad_after_m']:.2f} m")
    for when in ("before", "after"):
        for mechanism, v in report.get("m4_detection_at_0.50m", {}).get(when, {}).items():
            print(f"  M4 {when:<6} {mechanism:<10} {v['detected']:>3}/{v['points']}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
