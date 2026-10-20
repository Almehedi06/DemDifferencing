"""USGS post-fire debris-flow hazard, basin (watershed) scale, Pioneer Fire.

Source: the official USGS emergency assessment release pio2024 (basins,
perimeter, web-info), not hazard_results_pioneer.gpkg, which is a separate run.
Basins are coloured by combined hazard class at the design rainfall USGS uses
for its own map legend (web-info "Legend Rainfall"). No debris-flow
observations are drawn: this is the forecast only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.warp
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cascade_slide as C  # noqa: E402  fonts, save, Stehekin, tile helpers (C.M)

D = Path("/mnt/c/Users/amehedi/Downloads")
HAZARD = {1: ("Low", "#4DAF4A"), 2: ("Moderate", "#FFD23F"), 3: ("High", "#D7191C")}


def legend_rainfall(folder):
    text = (folder / "pio2024-web-info.txt").read_text()
    m = re.search(r"Legend Rainfall:\s*(\d+)", text)
    if not m:
        raise ValueError("pio2024-web-info.txt has no 'Legend Rainfall' entry")
    return int(m.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usgs", type=Path, default=D / "pioneer/usgs_hazard_pio2024")
    ap.add_argument("--rain", type=int, default=None,
                    help="15-min rainfall intensity, mm/h (default: USGS legend rainfall)")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures_final")
    args = ap.parse_args(argv)
    C.use_arial()

    rain = args.rain or legend_rainfall(args.usgs)
    basins = gpd.read_file(args.usgs / "pio2024-basins.shp").to_crs(3857)
    perim = gpd.read_file(args.usgs / "pio2024-perimeter.shp").to_crs(3857)
    field = f"H_{rain}mmh"
    if field not in basins:
        raise ValueError(f"no {field} in basins; available: {sorted(c for c in basins if c.startswith('H_') and not c.endswith('LG'))}")
    unknown = set(basins[field].dropna().astype(int)) - set(HAZARD)
    if unknown or basins[field].isna().any():
        raise ValueError(f"{field}: unexpected classes {unknown} or missing values")
    assessed = pd.Timestamp(perim.AssessDate.iloc[0])

    pb = basins.total_bounds
    ext = (pb[0] - 6000, pb[2] + 9000, pb[1] - 3000, pb[3] + 3500)
    ll = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326", ext[0], ext[2], ext[1], ext[3])
    img, img_ext = C.M.fetch_tiles(ll, 13, C.M.IMAGERY)

    aspect = (ext[1] - ext[0]) / (ext[3] - ext[2])
    fig = plt.figure(figsize=(6.4 * aspect, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=img_ext, zorder=1, interpolation="bilinear")
    for cls, (_, colour) in HAZARD.items():
        basins[basins[field].astype(int) == cls].plot(ax=ax, color=colour, alpha=0.72,
                                                      edgecolor="white", linewidth=0.35, zorder=2)
    perim.boundary.plot(ax=ax, color="black", linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)

    sx, sy = gpd.GeoSeries([Point(*C.STEHEKIN)], crs=4326).to_crs(3857).iloc[0].coords[0]
    ax.plot(sx, sy, "o", ms=6, mfc="white", mec="black", mew=1.0, zorder=8)
    ax.text(sx - 700, sy - 250, "Stehekin", ha="right", va="top", fontsize=10, fontweight="bold",
            color="white", zorder=8, path_effects=C.HALO)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.25")

    info = VPacker(children=[
        TextArea("USGS post-fire debris-flow hazard",
                 textprops=dict(fontsize=11.5, fontweight="bold", color=C.INK)),
        TextArea(f"Pioneer Fire · assessed {C.day(assessed)}\n"
                 f"Peak 15-min rainfall {rain} mm/h",
                 textprops=dict(fontsize=10, color=C.INK, linespacing=1.4))],
        align="left", pad=0, sep=5)
    box = AnchoredOffsetbox(loc="upper right", child=info, pad=0.6, borderpad=0.7, frameon=True)
    box.patch.set_boxstyle("round,pad=0,rounding_size=0.35")
    box.patch.set(facecolor="white", edgecolor="0.35", linewidth=0.8, alpha=0.95)
    box.set_zorder(14)
    ax.add_artist(box)

    handles = [Patch(facecolor=c, edgecolor="0.3", linewidth=0.5, alpha=0.72, label=lab)
               for _, (lab, c) in sorted(HAZARD.items(), reverse=True)]
    handles.append(plt.Line2D([], [], color="black", lw=1.1, ls=(0, (4, 2)), label="Fire perimeter"))
    leg = ax.legend(handles=handles, loc="lower left", title="Combined hazard, by basin",
                    title_fontsize=10.5, fontsize=9.5, framealpha=0.95, borderpad=0.6,
                    handlelength=1.6, edgecolor="0.35", bbox_to_anchor=(0.012, 0.012))
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(14)

    km = 5
    W, H = ext[1] - ext[0], ext[3] - ext[2]
    x1, y0, t = ext[1] - 0.045 * W, ext[2] + 0.075 * H, 0.011 * H
    lat = np.degrees(2 * np.arctan(np.exp(y0 / 6378137.0)) - np.pi / 2)   # latitude of the bar itself
    s2 = np.sin(np.radians(lat)) ** 2                              # Web Mercator scale on WGS84
    merc = km * 1000 * np.sqrt(1 - 0.00669437999 * s2) / np.cos(np.radians(lat))
    x0 = x1 - merc
    ax.add_patch(FancyBboxPatch((x0 - 0.03 * W, y0 - 0.025 * H), merc + 0.06 * W, 0.075 * H,
                                boxstyle="round,pad=0,rounding_size=300", fc="white", ec="0.35",
                                lw=0.8, alpha=0.95, zorder=11))
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * merc / 4, y0), merc / 4, t,
                               fc="black" if i % 2 == 0 else "white", ec="black", lw=0.8, zorder=12))
    for frac, lab in ((0, "0"), (0.5, f"{km / 2:g}"), (1, f"{km} km")):
        ax.text(x0 + frac * merc, y0 + t * 1.6, lab, ha="center", va="bottom", fontsize=9, zorder=13)
    nx = (x1 - merc / 2 - ext[0]) / W
    ax.add_patch(FancyBboxPatch((nx - 0.035, 0.145), 0.07, 0.1, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.012", fc="white", ec="0.35",
                                lw=0.8, alpha=0.95, zorder=11))
    ax.annotate("", xy=(nx, 0.205), xytext=(nx, 0.155), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=14), zorder=13)
    ax.text(nx, 0.207, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=12,
            fontweight="bold", zorder=13)
    ax.text(0.99, 0.008, "Hazard: USGS emergency assessment  ·  Imagery: Esri, Maxar, Earthstar Geographics",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="white", zorder=13,
            path_effects=[C.pe.withStroke(linewidth=1.8, foreground="black")])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    C.save(fig, args.output_dir, f"usgs_basin_hazard_{rain}mmh", pad=0.02)
    counts = basins[field].astype(int).map(lambda k: HAZARD[k][0]).value_counts()
    print("   ", dict(counts))


if __name__ == "__main__":
    main()
