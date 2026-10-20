"""All figures for the dDEM correction test (tilt, NDVI rule, deposition-zone mask).

Writes to Downloads/pioneer/ddem_correction/figures/:

* ndvi_maps.png  - six maps: Sentinel-2 summer NDVI 2023-2026, NAIP 2023 at 1 m, and
  the 2025 minus 2023 change, with the survey footprint and initiation points by type.
* slide_ndvi.png - 16:9 slide: the four summers, the NDVI lost to the fire, and the
  scene-by-scene time series with the fire, debris-flow and survey dates.
* slide_ddem.png - 16:9 slide: the xDEM-corrected dDEM, the NDVI rule (Erkan's test)
  and the deposition-zone rule (M6), with the share of area raised > 0.5 m.
* ndvi_scenes_<year>.png - every Sentinel-2 scene inside each summer window, with
  its median NDVI and clear share; scenes dropped by the screening are marked red.

Reads the outputs of fetch_ndvi.py, M5 and M6; computes nothing new.

    conda activate ml_debris
    python experiments/bin/make_ddem_correction_figures.py
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PIONEER = Path("/mnt/c/Users/amehedi/Downloads/pioneer")
WORK = PIONEER / "ddem_correction"
NDVI = WORK / "ndvi"
DOD = {"m5": WORK / "m5_detilt/dod_detilted.tif",
       "ndvi_test": WORK / "m6_gain_mask/dod_ndvi_test.tif",
       "m6": WORK / "m6_gain_mask/dod_gain_masked.tif"}
M6_REPORT = WORK / "m6_gain_mask/m6_report.json"
POINTS = PIONEER / "obs_data/Initiation_Points_final.shp"
FOOTPRINT = PIONEER / "pioneer_lidar_footprint.shp"           # UAV survey footprint, 13.3 km2
OUT = WORK / "figures"
SLIDE = (13.333, 6.0)          # inches: 16:9 slide width, leaving room for a slide title
DPI = 200
YEARS = ("2023", "2024", "2025", "2026")
POINT_STYLE = {"Firehose": ("*", 9, "#ff2a2a", "Firehose"), "LS": ("D", 4.5, "#7b1020", "Landslide"),
               "Runoff": ("D", 4.5, "#6fa8dc", "Runoff")}
# (date, label, side of the line the label sits on)
EVENTS = [(date(2024, 7, 20), "Pioneer Fire burns the area (Jul 2024)", "right"),
          (date(2025, 12, 10), "Debris flows (10 Dec 2025)", "left"),
          (date(2026, 6, 16), "UAV survey (15–17 Jun 2026)", "left")]


def load(path):
    with rasterio.open(path) as src:
        b = src.bounds
        return src.read(1, masked=True).filled(np.nan), (b.left, b.right, b.bottom, b.top)


def draw_points(ax, pts, scale=1.0):
    for kind, (marker, size, colour, label) in POINT_STYLE.items():
        sel = pts[pts["FailureTyp"] == kind]
        ax.plot(sel.geometry.x, sel.geometry.y, marker, ms=size * scale, mfc=colour, mec="black", mew=0.4,
                ls="none", label=label)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def scale_bar(ax, extent):
    x0, y0 = extent[0] + 200, extent[2] + 200
    ax.plot([x0, x0 + 1000], [y0, y0], color="black", lw=2.5)
    ax.text(x0 + 500, y0 + 110, "1 km", ha="center", fontsize=9)


def north_arrow(ax, extent):
    ax.annotate("N", xy=(extent[1] - 250, extent[3] - 150), xytext=(extent[1] - 250, extent[3] - 700),
                ha="center", fontsize=12, arrowprops=dict(arrowstyle="-|>", color="black"))


def map_row(fig, n, left, right, bottom, gap, extent):
    """n equal map axes in a row on a slide-sized figure, sized to the map's aspect."""
    w = (right - left - gap * (n - 1)) / n
    h = w * SLIDE[0] / SLIDE[1] * (extent[3] - extent[2]) / (extent[1] - extent[0])
    return [fig.add_axes([left + k * (w + gap), bottom, w, h]) for k in range(n)], w


def scene_sheets(manifest, footprint):
    """One sheet per summer window: every scene's NDVI, dropped scenes marked."""
    rows = list(csv.DictReader(open(NDVI / "s2_ndvi_timeseries.csv")))
    for year in YEARS:
        scenes = [r for r in rows if r["window"] == year]
        ncol = 5
        nrow = -(-len(scenes) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.3 * nrow + 0.6), squeeze=False,
                                 constrained_layout=True)
        for ax, r in zip(axes.flat, scenes):
            arr, extent = load(NDVI / "scenes" / year / f"{r['date']}_{r['scene']}.tif")
            im = ax.imshow(arr, cmap="Greens", vmin=0, vmax=0.9, extent=extent, interpolation="nearest")
            footprint.boundary.plot(ax=ax, color="black", lw=0.5)
            dropped = r["used"] == "no"
            colour = "#c0392b" if dropped else "black"
            ax.set_title(f"{r['date']}   median {float(r['median_ndvi_land']):.2f}   "
                         f"clear {float(r['clear_land_fraction']):.0%}"
                         + (f"\nDROPPED: {r['drop_reason']}" if dropped else "\nused"),
                         fontsize=9, loc="left", color=colour)
            if dropped:
                for side in ax.spines.values():
                    side.set_edgecolor(colour)
                    side.set_linewidth(2.5)
            bare(ax)
        for ax in list(axes.flat)[len(scenes):]:
            ax.axis("off")
        c = manifest[year]
        fig.colorbar(im, ax=axes, shrink=0.5, label="NDVI (Sentinel-2, 10 m)")
        fig.suptitle(f"Summer {year} ({c['short']}): {len(scenes)} scenes in {c['window'].replace('..', ' to ')}, "
                     f"{c['scenes']} used in the composite", fontsize=13, x=0.01, ha="left")
        fig.savefig(OUT / f"ndvi_scenes_{year}.png", dpi=120)
        plt.close(fig)


def ndvi_maps(manifest, pts, footprint):
    s2 = {y: load(NDVI / f"s2_ndvi_summer{y}.tif") for y in YEARS}
    naip, naip_ext = load(NDVI / "naip_ndvi_2023.tif")
    extent = s2["2023"][1]

    def window(year):
        start, end = manifest[year]["window"].split("..")
        c = manifest[year]
        return (f"{start[8:]}.{start[5:7]} – {end[8:]}.{end[5:7]}.{year}, "
                f"{c['scenes']} of {c['scenes_in_window']} scenes used")

    panels = [
        (s2["2023"][0], extent, "a) 2023 — before the fire", window("2023"), "ndvi"),
        (s2["2024"][0], extent, "b) 2024 — just after the burn", window("2024") + "; white: no clear pixel (charred ground or shadow)", "ndvi"),
        (s2["2025"][0], extent, "c) 2025 — after the fire, before the debris flows", window("2025"), "ndvi"),
        (s2["2026"][0], extent, "d) 2026 — after the debris flows", window("2026"), "ndvi"),
        (naip[::2, ::2], naip_ext, "e) 2023 — before the fire, NAIP 1 m", "2 Aug and 9 Sep 2023, relative NDVI", "naip"),
        (s2["2025"][0] - s2["2023"][0], extent, "f) change 2025 minus 2023 — the fire",
         "Sentinel-2, brown = NDVI lost", "change"),
    ]
    styles = {"ndvi": ("Greens", 0, 0.9, "NDVI (Sentinel-2, 10 m), panels a-d"),
              "naip": ("Greens", 0, 0.5, "NDVI (NAIP, relative)"),
              "change": ("BrBG", -0.5, 0.5, "NDVI change")}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), constrained_layout=True)
    images = {}
    for ax, (arr, ext, title, sub, kind) in zip(axes.flat, panels):
        cmap, vmin, vmax, _ = styles[kind]
        images[kind] = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
        footprint.boundary.plot(ax=ax, color="black", lw=0.8)
        draw_points(ax, pts, scale=1.1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{title}\n", loc="left", fontsize=12)
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=9, color="#555550")
        bare(ax)
    scale_bar(axes[0, 0], extent)
    north_arrow(axes[0, 0], extent)
    axes[0, 0].legend(title="Mapped initiation", loc="upper left", fontsize=8, title_fontsize=9, framealpha=0.9)
    for ax, kind in zip(axes[1], ("ndvi", "naip", "change")):
        fig.colorbar(images[kind], ax=ax, shrink=0.7, orientation="horizontal", label=styles[kind][3])
    fig.suptitle("Summer NDVI at Stehekin: Pioneer Fire (July 2024) and debris flows (December 2025)  ·  "
                 "black outline: UAV survey footprint", fontsize=13, x=0.01, ha="left")
    fig.savefig(OUT / "ndvi_maps.png", dpi=DPI)
    plt.close(fig)


def slide_ndvi(manifest, pts, footprint):
    maps = {y: load(NDVI / f"s2_ndvi_summer{y}.tif")[0] for y in YEARS}
    extent = load(NDVI / "s2_ndvi_summer2023.tif")[1]
    titles = ["2023\nbefore the fire", "2024\njust after the burn", "2025\nbefore the debris flows",
              "2026\nafter the debris flows", "NDVI change 2025 − 2023\n(dots: initiation points)"]

    fig = plt.figure(figsize=SLIDE)
    axes, w = map_row(fig, 5, 0.06, 0.99, 0.47, 0.012, extent)
    for ax, year, title in zip(axes, maps, titles):
        im = ax.imshow(maps[year], cmap="Greens", vmin=0, vmax=0.9, extent=extent, interpolation="nearest")
        footprint.boundary.plot(ax=ax, color="black", lw=0.7)
        ax.set_title(title, fontsize=12, loc="left")
        bare(ax)
    imc = axes[4].imshow(maps["2025"] - maps["2023"], cmap="BrBG", vmin=-0.5, vmax=0.5, extent=extent,
                         interpolation="nearest")
    footprint.boundary.plot(ax=axes[4], color="black", lw=0.7)
    axes[4].plot(pts.geometry.x, pts.geometry.y, "o", ms=3.2, mfc="white", mec="black", mew=0.5, ls="none")
    axes[4].set_title(titles[4], fontsize=12, loc="left")
    bare(axes[4])
    scale_bar(axes[0], extent)
    fig.colorbar(im, cax=fig.add_axes([0.06 + 0.5 * w, 0.43, 3 * w, 0.02]), orientation="horizontal").set_label(
        "NDVI (Sentinel-2, 10 m)", fontsize=10)
    fig.colorbar(imc, cax=fig.add_axes([0.06 + 4 * (w + 0.012) + 0.1 * w, 0.43, 0.8 * w, 0.02]),
                 orientation="horizontal", ticks=[-0.4, 0, 0.4]).set_label("brown = NDVI lost", fontsize=10)

    ts = fig.add_axes([0.06, 0.07, 0.93, 0.23])
    rows = [r for r in csv.DictReader(open(NDVI / "s2_ndvi_timeseries.csv")) if r["median_ndvi_land"] != "nan"]
    groups = [(lambda r: r["used"] == "yes", dict(marker="o", ms=3.5, color="#2a78d6", ls="none"),
               "used in a summer map"),
              (lambda r: r["used"] == "no", dict(marker="x", ms=6, mew=1.6, color="#c0392b", ls="none"),
               "dropped (cloud or smoke)"),
              (lambda r: r["window"] == "", dict(marker="o", ms=3, mfc="none", color="#9a9a94", ls="none"),
               "outside the summer windows")]
    for keep, style, label in groups:
        sel = [r for r in rows if keep(r)]
        ts.plot([date.fromisoformat(r["date"]) for r in sel], [float(r["median_ndvi_land"]) for r in sel],
                label=label, **style)
    ts.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=9, frameon=False,
              handletextpad=0.3, columnspacing=1.5, borderaxespad=0.1)
    ts.set_title("Sentinel-2 scenes, June–October", fontsize=9, loc="right", color="#555550")
    for year, c in manifest.items():
        start, end = (date.fromisoformat(d) for d in c["window"].split(".."))
        ts.axvspan(start, end, color="#e6e6e3", zorder=0)
        ts.text(start, 0.97, f" {year} map", transform=ts.get_xaxis_transform(), va="top", fontsize=9,
                color="#555550")
    for k, (when, label, side) in enumerate(EVENTS):
        ts.axvline(when, color="#e34948", lw=1.2)
        text = f" {label}" if side == "right" else f"{label} "
        ts.text(when, 0.04 + 0.13 * (k == 2), text, transform=ts.get_xaxis_transform(), fontsize=9,
                color="#b02a2a", ha="left" if side == "right" else "right")
    ts.set_ylabel("median NDVI\n(survey area + 500 m)", fontsize=10)
    ts.set_ylim(0.1, 0.65)
    ts.spines[["top", "right"]].set_visible(False)
    ts.grid(axis="y", color="#e6e6e3", lw=0.8)
    fig.savefig(OUT / "slide_ndvi.png", dpi=DPI)
    plt.close(fig)


def slide_ddem(pts):
    report = json.loads(M6_REPORT.read_text())
    share = {"m5": report["share_above_0.5m"]["before"], "ndvi_test": report["ndvi_test"]["share_above_0.5m"],
             "m6": report["share_above_0.5m"]["after"]}
    t = report["thresholds"]
    panels = [("m5", "a) dDEM, xDEM-corrected\n    before masking"),
              ("ndvi_test", f"b) NDVI rule: green (NDVI > {report['ndvi_test']['ndvi_threshold']})\n"
                            f"    outside channels set to 0"),
              ("m6", f"c) Gains kept only in channels (G > {t['channel_g']:.0f})\n"
                     f"    or on slopes < {t['max_deposition_slope_deg']:.0f}°, and not green")]

    extent = load(DOD["m5"])[1]
    fig = plt.figure(figsize=SLIDE)
    axes, _ = map_row(fig, 3, 0.01, 0.99, 0.22, 0.012, extent)
    for ax, (key, title) in zip(axes, panels):
        im = ax.imshow(load(DOD[key])[0][::2, ::2], cmap="RdBu", vmin=-1, vmax=1, extent=extent)
        draw_points(ax, pts)
        ax.set_title(title, fontsize=12, loc="left")
        ax.text(0.5, -0.02, f"area raised > 0.5 m: {share[key]:.1%}", transform=ax.transAxes,
                ha="center", va="top", fontsize=13, fontweight="bold", color="#2b2b28")
        bare(ax)
    scale_bar(axes[0], extent)
    north_arrow(axes[0], extent)
    fig.colorbar(im, cax=fig.add_axes([0.35, 0.075, 0.30, 0.03]), orientation="horizontal",
                 ticks=[-1, -0.5, 0, 0.5, 1]).set_label(
        "post minus pre (m):  red = lowered (erosion),  blue = raised", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Mapped initiation", loc="lower left", bbox_to_anchor=(0.01, 0.02),
               ncol=3, fontsize=9, title_fontsize=9, frameon=False)
    fig.savefig(OUT / "slide_ddem.png", dpi=DPI)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((NDVI / "ndvi_manifest.json").read_text())["sentinel2"]
    with rasterio.open(DOD["m5"]) as src:
        crs = src.crs
    pts = gpd.read_file(POINTS).to_crs(crs)
    footprint = gpd.read_file(FOOTPRINT).to_crs(crs)
    ndvi_maps(manifest, pts, footprint)
    scene_sheets(manifest, footprint)
    slide_ndvi(manifest, pts, footprint)
    slide_ddem(pts)
    print(f"Wrote ndvi_maps.png, ndvi_scenes_<year>.png, slide_ndvi.png and slide_ddem.png to {OUT}")


if __name__ == "__main__":
    main()
