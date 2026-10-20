"""Publication-quality figures for the Pioneer/Stehekin hazard-layer comparison.

Standalone. Reads only; writes PNG + PDF at 300 dpi to a new directory.

Figures
-------
1  Regional baseline: the three Landlab probability layers over the full
   870 km2 model domain, shared colour scale, UAV survey footprint marked.
2  Local detail: dDEM erosion over a hillshade of the pre-event DEM, with
   the 89 mapped initiation points by mechanism.
3  Performance: prediction-rate curves by mechanism, plus the confusion
   metrics at the operating threshold.

Design notes
------------
No regional DEM exists at the 30 m model extent, so Figure 1 uses clean
cartography on a neutral background rather than a hillshade; Figure 2 does
use a real hillshade, derived from the 1 m pre-event DEM it has available.
Scale bars and north arrows are drawn explicitly (matplotlib has neither).
Colour scale is YlOrRd throughout for continuity with the source model's own
renders, and zero is drawn as background rather than as pale yellow so the
90%+ of each grid that is exactly zero does not dominate the figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def scale_bar(ax, length_m, label, loc=(0.06, 0.06), height=0.012, color="black"):
    """Alternating black/white scale bar in axes coordinates."""
    x0, y0 = loc
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    frac = length_m / xr
    n = 4
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * frac / n, y0), frac / n, height,
                               transform=ax.transAxes, zorder=12,
                               facecolor=color if i % 2 == 0 else "white",
                               edgecolor=color, linewidth=0.6))
    ax.text(x0 + frac / 2, y0 + height + 0.012, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, zorder=12, color=color)


def north_arrow(ax, loc=(0.94, 0.86), color="black"):
    """Arrow head at the top (north); annotate draws from xytext to xy."""
    x, y = loc
    ax.annotate("", xy=(x, y + 0.055), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3), zorder=12)
    ax.text(x, y + 0.072, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color, zorder=12)


def panel_label(ax, text):
    ax.text(0.025, 0.975, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", zorder=13,
            bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                      edgecolor="0.4", linewidth=0.6))


def read(path, decimate=1):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True, out_shape=(1, src.height // decimate, src.width // decimate)
                       if decimate > 1 else None).filled(np.nan)
        b = src.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


def figure_regional(pf_dir, out_dir, survey_bounds):
    layers = [("P_Landslide", "Landslide"), ("P_Combined_H", "Combined hillslope"),
              ("P_Channel_ini", "Channel initiation")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.4), constrained_layout=True)
    im = None
    for ax, (stem, title), tag in zip(axes, layers, "abc"):
        arr, ext = read(pf_dir / f"{stem}_regional_baseline.tif")
        ax.set_facecolor("#e8e6e1")
        shown = np.where(arr > 0.001, arr, np.nan)          # zero drawn as background
        im = ax.imshow(shown, extent=ext, cmap="YlOrRd", vmin=0, vmax=1,
                       interpolation="nearest", zorder=2)
        ax.add_patch(Rectangle((survey_bounds[0], survey_bounds[2]),
                               survey_bounds[1] - survey_bounds[0],
                               survey_bounds[3] - survey_bounds[2],
                               fill=False, edgecolor="#1a53ff", linewidth=1.4, zorder=6))
        ax.set_title(title, pad=6)
        panel_label(ax, f"({tag})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    scale_bar(axes[0], 5000, "5 km")
    north_arrow(axes[0])
    axes[2].text(0.97, 0.05, "blue box: UAV survey\nfootprint (13 km²)",
                 transform=axes[2].transAxes, ha="right", va="bottom", fontsize=7.5,
                 zorder=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="0.6", linewidth=0.5))
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.45, aspect=32, pad=0.02)
    cb.set_label("Modelled probability of failure", fontsize=9)
    cb.outline.set_linewidth(0.6)
    for ext_name in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_regional_baseline.{ext_name}")
    plt.close(fig)
    return "fig1_regional_baseline"


def figure_local(pioneer, out_dir):
    dem, ext = read(pioneer / "pre.tif", decimate=3)
    ero, _ = read(pioneer / "m4_erosion/dod_erosion.tif", decimate=3)
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)), vert_exag=1.4,
                      dx=3.0, dy=3.0)

    fig, ax = plt.subplots(figsize=(8.2, 7.4), constrained_layout=True)
    ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1.15, zorder=1, interpolation="bilinear")
    depth = np.where(np.isfinite(ero) & (ero < 0), -ero, np.nan)
    ero_cmap = LinearSegmentedColormap.from_list("ero", ["#ffe9b0", "#e8632a", "#8c1010"])
    im = ax.imshow(depth, extent=ext, cmap=ero_cmap, vmin=0.5, vmax=3.0,
                   zorder=2, interpolation="nearest")

    pts = gpd.read_file(pioneer / "Initiation_Points_final/Initiation_Points_final.shp")
    style = {"LS": ("#d40000", "o", "Landslide"),
             "Runoff": ("#00a83c", "^", "Runoff"),
             "Firehose": ("#1a53ff", "s", "Firehose")}
    for mech, (colour, marker, label) in style.items():
        sel = pts[pts.FailureTyp == mech]
        ax.scatter(sel.geometry.x, sel.geometry.y, s=26, c=colour, marker=marker,
                   edgecolor="white", linewidth=0.7, zorder=5,
                   label=f"{label} (n={len(sel)})")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    scale_bar(ax, 1000, "1 km")
    north_arrow(ax)
    leg = ax.legend(loc="upper left", framealpha=0.92, borderpad=0.6,
                    title="Mapped initiation", title_fontsize=8)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(12)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, aspect=24, pad=0.015, extend="max")
    cb.set_label("Erosion depth (m), dDEM", fontsize=9)
    cb.outline.set_linewidth(0.6)
    for ext_name in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_local_erosion_points.{ext_name}")
    plt.close(fig)
    return "fig2_local_erosion_points"


def compute_curves(pioneer, pf_dir, window_m=30.0):
    """Recompute full prediction-rate curves rather than interpolating the five
    sampled fractions in the CSV, so the published curve is measured throughout.
    Reuses the scoring functions from the validation script so the numbers here
    cannot drift from the numbers in the tables."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from validate_hazard_layers import (area_fraction_needed, capture_curve,
                                        crop_to_bounds, read_probability_layer, window_max)

    pts = gpd.read_file(pioneer / "Initiation_Points_final/Initiation_Points_final.shp")
    ls_pts = pts[pts.FailureTyp == "LS"]
    layers = []
    for stem in ["P_Landslide", "P_Combined_H", "P_Channel_ini"]:
        value, transform, crs, _ = read_probability_layer(pf_dir / f"{stem}.tif")
        layers.append((stem, value, transform, crs))
    with rasterio.open(pioneer / "main_pipeline_run/corrections/pioneer/"
                                 "final_dem_products/dod_corrected.tif") as src:
        layers.append(("dod_erosion", -src.read(1, masked=True).filled(np.nan),
                       src.transform, src.crs))

    common_bounds = rasterio.transform.array_bounds(*layers[0][1].shape, layers[0][2])
    fine = np.linspace(0.002, 1.0, 400)
    out = {}
    for name, value, transform, crs in layers:
        score = window_max(value, transform, ls_pts.to_crs(crs), window_m)[0]
        domain = crop_to_bounds(value, transform, common_bounds)
        needed = area_fraction_needed(domain, score)
        out[name] = (fine, capture_curve(needed, fine))
    return out


def figure_performance(validation_dir, out_dir, curves):
    import csv
    rows = list(csv.DictReader(open(validation_dir / "validation_metrics.csv")))
    ls_rows = [r for r in rows if r["mechanism"] == "LS"]
    names = {"P_Landslide": "Landslide", "P_Combined_H": "Combined H",
             "P_Channel_ini": "Channel ini", "dod_erosion": "dDEM erosion"}
    colours = {"P_Landslide": "#8c8c8c", "P_Combined_H": "#e8a33d",
               "P_Channel_ini": "#3d8ce8", "dod_erosion": "#c1272d"}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), constrained_layout=True)
    metrics = ["recall", "F1", "CSI"]
    x = np.arange(len(metrics)); width = 0.2
    for i, r in enumerate(ls_rows):
        vals = [float(r[m]) for m in metrics]
        axes[0].bar(x + (i - 1.5) * width, vals, width, label=names[r["layer"]],
                    color=colours[r["layer"]], edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Recall", "F1", "CSI"])
    axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("Score")
    axes[0].set_title("Landslide initiation (n=47)", pad=6)
    axes[0].legend(frameon=True, framealpha=0.95)
    axes[0].grid(axis="y", linewidth=0.4, alpha=0.5); axes[0].set_axisbelow(True)
    panel_label(axes[0], "(a)")

    axes[1].plot([0, 1], [0, 1], "--", color="0.45", linewidth=0.9, label="No skill")
    for layer, curve in curves.items():
        frac, capture = curve
        axes[1].plot(frac, capture, linewidth=1.6, color=colours[layer], label=names[layer])
    axes[1].legend(frameon=True, framealpha=0.95, loc="lower right")
    axes[1].set_xlabel("Fraction of area flagged"); axes[1].set_ylabel("Fraction of points captured")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_title("Prediction-rate curve, landslides", pad=6)
    axes[1].grid(linewidth=0.4, alpha=0.5); axes[1].set_axisbelow(True)
    panel_label(axes[1], "(b)")
    for ext_name in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_performance.{ext_name}")
    plt.close(fig)
    return "fig3_performance"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    ap.add_argument("--pioneer", type=Path, default=D / "pioneer")
    ap.add_argument("--probability-fields", type=Path,
                    default=D / "regional_baseline/regional_baseline")
    ap.add_argument("--validation-dir", type=Path,
                    default=D / "pioneer/hazard_layer_validation_window30m")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures")
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.pioneer / "pre.tif") as src:
        sb = rasterio.warp.transform_bounds(src.crs, "EPSG:32610", *src.bounds)
    survey_bounds = (sb[0], sb[2], sb[1], sb[3])   # left, right, bottom, top

    curves = compute_curves(args.pioneer, args.pioneer.parent / "probability_fields")
    made = [figure_regional(args.probability_fields, args.output_dir, survey_bounds),
            figure_local(args.pioneer, args.output_dir),
            figure_performance(args.validation_dir, args.output_dir, curves)]
    for m in made:
        print(f"  {m}.png / .pdf")
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
