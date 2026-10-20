"""Publication figures on a satellite basemap, composed directly in Python.

An alternative to screenshotting Google Earth: tiles are fetched, mosaicked and
plotted at a chosen DPI, so the figure is reproducible, correctly projected and
print-resolution, and the scale bar is computed rather than eyeballed.

Basemap is Esri World Imagery (same underlying aerial/satellite sources Google
Earth draws on for this area). Attribution is required and is drawn on the
figure; check the service's terms before submitting to a journal.

Projection note
---------------
Web Mercator (EPSG:3857) is the tile CRS, so everything is reprojected into it.
Mercator exaggerates distance by 1/cos(latitude) -- about 1.5x at 48.3 N -- so a
scale bar drawn naively in Mercator units would be ~50% too short. `scale_bar`
corrects for this; do not replace it with a plain axis-fraction bar.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import math
import urllib.request
import matplotlib.patheffects
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

TILE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ATTRIB = "Imagery: Esri, Maxar, Earthstar Geographics"
R = 6378137.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
    "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def lonlat_to_merc(lon, lat):
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def tile_index(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_merc(x, y, z):
    n = 2 ** z
    span = 2 * math.pi * R / n
    return (-math.pi * R + x * span, math.pi * R - (y + 1) * span,
            -math.pi * R + (x + 1) * span, math.pi * R - y * span)


def fetch_basemap(bounds_lonlat, z):
    """Mosaic Esri World Imagery tiles covering bounds; returns (rgb, extent_merc)."""
    w, s, e, n = bounds_lonlat
    x0, y0 = tile_index(w, n, z)
    x1, y1 = tile_index(e, s, z)
    xs, ys = range(x0, x1 + 1), range(y0, y1 + 1)
    mosaic = Image.new("RGB", (256 * len(list(xs)), 256 * len(list(ys))))

    def grab(args):
        i, j, x, y = args
        req = urllib.request.Request(TILE.format(z=z, x=x, y=y),
                                     headers={"User-Agent": "research-figure/1.0"})
        for _ in range(3):
            try:
                return i, j, Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read()))
            except Exception:
                continue
        return i, j, None

    jobs = [(i, j, x, y) for i, x in enumerate(xs) for j, y in enumerate(ys)]
    missing = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, j, img in ex.map(grab, jobs):
            if img is None:
                missing += 1
                continue
            mosaic.paste(img, (i * 256, j * 256))
    if missing:
        print(f"    warning: {missing}/{len(jobs)} tiles failed to load")
    left, _, _, top = tile_bounds_merc(x0, y0, z)
    _, bottom, right, _ = tile_bounds_merc(x1, y1, z)
    return np.asarray(mosaic), (left, right, bottom, top)


def scale_bar(ax, length_m, label, lat, loc=(0.045, 0.055), height_frac=0.012):
    """Scale bar corrected for Mercator distortion at this latitude."""
    merc_len = length_m / math.cos(math.radians(lat))
    x0f, y0f = loc
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + x0f * (xlim[1] - xlim[0])
    y0 = ylim[0] + y0f * (ylim[1] - ylim[0])
    h = height_frac * (ylim[1] - ylim[0])
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * merc_len / 4, y0), merc_len / 4, h,
                               facecolor="white" if i % 2 else "black",
                               edgecolor="black", linewidth=0.7, zorder=12))
    ax.text(x0 + merc_len / 2, y0 + h * 1.5, label, ha="center", va="bottom",
            fontsize=8, color="white", zorder=12,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="black")])


def north_arrow(ax, loc=(0.955, 0.87)):
    x, y = loc
    ax.annotate("", xy=(x, y + 0.055), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="white", linewidth=1.6), zorder=12)
    ax.text(x, y + 0.074, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="white", zorder=12,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="black")])


def warp_to_merc(path, band_op=None):
    with rasterio.open(path) as src:
        t, w, h = rasterio.warp.calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, "float32")
        rasterio.warp.reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=t, dst_crs="EPSG:3857", dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear)
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:3857", *src.bounds)
    if band_op is not None:
        dst = band_op(dst)
    return dst, (bounds[0], bounds[2], bounds[1], bounds[3])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    ap.add_argument("--pioneer", type=Path, default=D / "pioneer")
    ap.add_argument("--regional", type=Path, default=D / "regional_baseline/regional_baseline")
    ap.add_argument("--zoom-local", type=int, default=15)
    ap.add_argument("--zoom-regional", type=int, default=12)
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures")
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pts = gpd.read_file(args.pioneer / "Initiation_Points_final/Initiation_Points_final.shp")
    pts_m = pts.to_crs(3857)

    # ---------------- local: erosion + points on imagery ----------------
    ero, ero_ext = warp_to_merc(args.pioneer / "m4_erosion/dod_erosion.tif")
    ll = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326",
                                        ero_ext[0], ero_ext[2], ero_ext[1], ero_ext[3])
    print(f"  fetching local basemap (z={args.zoom_local})...")
    img, img_ext = fetch_basemap(ll, args.zoom_local)
    lat_mid = (ll[1] + ll[3]) / 2

    fig, ax = plt.subplots(figsize=(8.4, 7.6), constrained_layout=True)
    ax.imshow(img, extent=img_ext, zorder=1, interpolation="bilinear")
    depth = np.where(np.isfinite(ero) & (ero < 0), -ero, np.nan)
    cmap = LinearSegmentedColormap.from_list("ero", ["#ffe9a8", "#f2872f", "#a11616"])
    im = ax.imshow(depth, extent=ero_ext, cmap=cmap, vmin=0.5, vmax=3.0,
                   zorder=2, interpolation="nearest", alpha=0.92)
    style = {"LS": ("#ff2b2b", "o", "Landslide"), "Runoff": ("#2bff7a", "^", "Runoff"),
             "Firehose": ("#3d9bff", "s", "Firehose")}
    for mech, (colour, marker, label) in style.items():
        sel = pts_m[pts_m.FailureTyp == mech]
        ax.scatter(sel.geometry.x, sel.geometry.y, s=30, c=colour, marker=marker,
                   edgecolor="black", linewidth=0.8, zorder=5,
                   label=f"{label} (n={len(sel)})")
    ax.set_xlim(ero_ext[0], ero_ext[1]); ax.set_ylim(ero_ext[2], ero_ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    scale_bar(ax, 1000, "1 km", lat_mid)
    north_arrow(ax)
    leg = ax.legend(loc="upper left", framealpha=0.9, borderpad=0.55,
                    title="Mapped initiation", title_fontsize=8)
    leg.get_frame().set_linewidth(0.6); leg.set_zorder(12)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, aspect=24, pad=0.015, extend="max")
    cb.set_label("Erosion depth (m), dDEM", fontsize=9)
    ax.text(0.995, 0.006, ATTRIB, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="white", zorder=12,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.8, foreground="black")])
    for e in ("png", "pdf"):
        fig.savefig(args.output_dir / f"fig4_satellite_local.{e}")
    plt.close(fig)
    print("  fig4_satellite_local.png / .pdf")

    # ---------------- regional: P_Landslide on imagery ----------------
    prob, prob_ext = warp_to_merc(args.regional / "P_Landslide_regional_baseline.tif")
    llr = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326",
                                         prob_ext[0], prob_ext[2], prob_ext[1], prob_ext[3])
    print(f"  fetching regional basemap (z={args.zoom_regional})...")
    img_r, img_r_ext = fetch_basemap(llr, args.zoom_regional)

    fig, ax = plt.subplots(figsize=(7.4, 8.6), constrained_layout=True)
    ax.imshow(img_r, extent=img_r_ext, zorder=1, interpolation="bilinear")
    shown = np.where(prob > 0.001, prob, np.nan)
    im = ax.imshow(shown, extent=prob_ext, cmap="YlOrRd", vmin=0, vmax=1,
                   zorder=2, interpolation="nearest", alpha=0.88)
    sb = pts_m.total_bounds
    ax.add_patch(Rectangle((sb[0], sb[1]), sb[2] - sb[0], sb[3] - sb[1],
                           fill=False, edgecolor="#00e5ff", linewidth=1.6, zorder=6))
    ax.set_xlim(prob_ext[0], prob_ext[1]); ax.set_ylim(prob_ext[2], prob_ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    scale_bar(ax, 5000, "5 km", (llr[1] + llr[3]) / 2)
    north_arrow(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, aspect=26, pad=0.015)
    cb.set_label("Modelled probability of failure (landslide)", fontsize=9)
    ax.text(0.995, 0.006, ATTRIB, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="white", zorder=12,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.8, foreground="black")])
    ax.text(0.02, 0.975, "cyan box: UAV survey footprint", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color="white", zorder=12,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.8, foreground="black")])
    for e in ("png", "pdf"):
        fig.savefig(args.output_dir / f"fig5_satellite_regional.{e}")
    plt.close(fig)
    print("  fig5_satellite_regional.png / .pdf")
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
