"""Publication figures: satellite basemap, 3D oblique terrain, modelled layers.

Composed in Python rather than screenshotted from Google Earth, so the output is
reproducible, correctly projected and print-resolution.

Figures
-------
fig4  UAV survey window, the three modelled probability layers on satellite
      imagery, with the mapped initiation points.
fig5  Regional baseline, 3D oblique -- P_Landslide draped over terrain, the
      Google-Earth-style perspective.
fig6  UAV window, 3D oblique -- dDEM erosion draped over the 1 m terrain.

Data sources
------------
Imagery   Esri World Imagery tiles (attribution drawn on every figure).
Terrain   AWS terrarium elevation tiles, decoded as
          (R*256 + G + B/256) - 32768 metres. Used only where no local DEM
          covers the extent; fig6 uses the 1 m pre-event DEM instead.

Scale bars on 3D panels
-----------------------
A perspective view has no single scale, so the 3D figures carry a stated scene
width and vertical exaggeration instead of a bar. Do not add a scale bar to a
perspective panel -- it would be wrong at every depth but one. The 2D panels do
get a proper bar, corrected for Web Mercator's 1/cos(latitude) stretch (~1.5x
at 48.3 N); a naive bar would be about 50% too short.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import math
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from PIL import Image

IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery"
           "/MapServer/tile/{z}/{y}/{x}")
TERRAIN = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ATTRIB = "Imagery: Esri, Maxar, Earthstar Geographics"
R = 6378137.0
ERO_CMAP = LinearSegmentedColormap.from_list("ero", ["#ffe9a8", "#f2872f", "#a11616"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
    "axes.labelsize": 10, "axes.titlesize": 12, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def tile_index(lon, lat, z):
    n = 2 ** z
    return (int((lon + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))


def tile_bounds_merc(x, y, z):
    n = 2 ** z
    span = 2 * math.pi * R / n
    return (-math.pi * R + x * span, math.pi * R - (y + 1) * span,
            -math.pi * R + (x + 1) * span, math.pi * R - y * span)


def fetch_tiles(bounds_lonlat, z, template, decode=None):
    """Mosaic XYZ tiles. decode=None -> RGB uint8; else applied per tile."""
    w, s, e, n = bounds_lonlat
    x0, y0 = tile_index(w, n, z)
    x1, y1 = tile_index(e, s, z)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    out = np.zeros((256 * ny, 256 * nx, 3), "uint8") if decode is None \
        else np.full((256 * ny, 256 * nx), np.nan)

    def grab(job):
        i, j, x, y = job
        req = urllib.request.Request(template.format(z=z, x=x, y=y),
                                     headers={"User-Agent": "research-figure/1.0"})
        for _ in range(3):
            try:
                img = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read()))
                return i, j, np.asarray(img.convert("RGB"))
            except Exception:
                continue
        return i, j, None

    jobs = [(i, j, x0 + i, y0 + j) for i in range(nx) for j in range(ny)]
    missing = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, j, arr in ex.map(grab, jobs):
            if arr is None:
                missing += 1
                continue
            sl = (slice(j * 256, (j + 1) * 256), slice(i * 256, (i + 1) * 256))
            out[sl] = arr if decode is None else decode(arr.astype("float64"))
    if missing:
        print(f"    warning: {missing}/{len(jobs)} tiles missing")
    left, _, _, top = tile_bounds_merc(x0, y0, z)
    _, bottom, right, _ = tile_bounds_merc(x1, y1, z)
    return out, (left, right, bottom, top)


def terrarium(rgb):
    return (rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256) - 32768


def warp_to_merc(path):
    with rasterio.open(path) as src:
        t, w, h = rasterio.warp.calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, "float32")
        rasterio.warp.reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=t, dst_crs="EPSG:3857", dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear)
        b = rasterio.warp.transform_bounds(src.crs, "EPSG:3857", *src.bounds)
    return dst, (b[0], b[2], b[1], b[3])


def regrid(arr, src_ext, dst_ext, shape):
    """Nearest-neighbour resample arr onto a regular grid covering dst_ext."""
    ny, nx = shape
    xs = np.linspace(dst_ext[0], dst_ext[1], nx)
    ys = np.linspace(dst_ext[3], dst_ext[2], ny)
    sy, sx = arr.shape[:2]
    cx = ((xs - src_ext[0]) / (src_ext[1] - src_ext[0]) * (sx - 1)).round().astype(int)
    cy = ((src_ext[3] - ys) / (src_ext[3] - src_ext[2]) * (sy - 1)).round().astype(int)
    ok_x = (cx >= 0) & (cx < sx)
    ok_y = (cy >= 0) & (cy < sy)
    out = np.full((ny, nx) + arr.shape[2:], np.nan if arr.dtype.kind == "f" else 0, arr.dtype)
    yy = np.where(ok_y, cy, 0)[:, None]
    xx = np.where(ok_x, cx, 0)[None, :]
    out[...] = arr[yy, xx]
    out[~ok_y, :] = np.nan if arr.dtype.kind == "f" else 0
    out[:, ~ok_x] = np.nan if arr.dtype.kind == "f" else 0
    return out


def furniture_box(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, zorder=11,
                           facecolor="white", edgecolor="0.3", linewidth=0.8, alpha=0.93))


def scale_bar(ax, length_m, label, lat, loc=(0.035, 0.035)):
    """White-backed bar, corrected for Mercator stretch at this latitude."""
    merc = length_m / math.cos(math.radians(lat))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
    frac = merc / span_x
    furniture_box(ax, loc[0] - 0.014, loc[1] - 0.012, frac + 0.028, 0.082)
    x0 = xlim[0] + (loc[0]) * span_x
    y0 = ylim[0] + (loc[1] + 0.012) * span_y
    h = 0.018 * span_y
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * merc / 4, y0), merc / 4, h,
                               facecolor="white" if i % 2 else "black",
                               edgecolor="black", linewidth=0.9, zorder=12))
    ax.text(x0 + merc / 2, y0 + h * 1.35, label, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="black", zorder=13)


def north_arrow(ax, loc=(0.945, 0.845)):
    x, y = loc
    furniture_box(ax, x - 0.042, y - 0.022, 0.084, 0.135)
    ax.annotate("", xy=(x, y + 0.072), xytext=(x, y + 0.012), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2.0), zorder=13)
    ax.text(x, y + 0.078, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="black", zorder=13)


def attribution(ax, extra=""):
    ax.text(0.995, 0.008, (extra + "  " if extra else "") + ATTRIB, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="white", zorder=13,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])


def drape(base_rgb, layer, norm, cmap, alpha=0.9):
    """Alpha-composite a colour-mapped layer over an RGB basemap."""
    out = base_rgb.astype("float64") / 255.0
    m = np.isfinite(layer)
    if m.any():
        rgba = cmap(norm(np.where(m, layer, 0)))[..., :3]
        a = np.where(m, alpha, 0.0)[..., None]
        out = out * (1 - a) + rgba * a
    return np.clip(out, 0, 1)


def render_3d(ax, dem, colours, scene_w_m, vert_exag, view=(46, -60)):
    """Oblique perspective of terrain with pre-computed face colours."""
    ny, nx = dem.shape
    y, x = np.mgrid[0:ny, 0:nx].astype("float64")
    x *= scene_w_m / nx
    y *= scene_w_m / nx
    z = np.nan_to_num(dem, nan=np.nanmin(dem)) * vert_exag
    ls = LightSource(azdeg=315, altdeg=50)
    shade = ls.hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)),
                         dx=scene_w_m / nx, dy=scene_w_m / nx, vert_exag=2.0)
    lit = np.clip(colours * (0.55 + 0.65 * shade[..., None]), 0, 1)
    ax.plot_surface(x, y[::-1], z, facecolors=lit, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    ax.set_box_aspect((1, ny / nx, 0.30))
    ax.margins(0)
    # matplotlib pads 3D axes heavily; overfill the figure so terrain fills the frame
    ax.set_position([-0.10, -0.20, 1.16, 1.40])
    try:
        ax.set_proj_type("persp", focal_length=0.62)
    except TypeError:
        ax.set_proj_type("persp")


def save_cropped(fig, out_dir, stem, pad_in=0.12):
    """Save PNG+PDF trimmed to rendered content.

    A 3D axes reports a bounding box far larger than the terrain it actually
    draws, so bbox_inches="tight" leaves wide white margins. Render once, find
    the non-white extent, and reuse that box for both formats so the vector PDF
    and the raster PNG frame identically.
    """
    from matplotlib.transforms import Bbox
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    buf.seek(0)
    a = np.asarray(Image.open(buf).convert("L"))
    ink = np.where(a < 250)
    if ink[0].size == 0:
        for e in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{e}")
        return
    h, w = a.shape
    fw, fh = fig.get_size_inches()
    x0 = ink[1].min() / w * fw - pad_in
    x1 = ink[1].max() / w * fw + pad_in
    y0 = (1 - ink[0].max() / h) * fh - pad_in
    y1 = (1 - ink[0].min() / h) * fh + pad_in
    box = Bbox([[max(0, x0), max(0, y0)], [min(fw, x1), min(fh, y1)]])
    for e in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{e}", bbox_inches=box)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    ap.add_argument("--pioneer", type=Path, default=D / "pioneer")
    ap.add_argument("--local-layers", type=Path, default=D / "probability_fields")
    ap.add_argument("--regional", type=Path, default=D / "regional_baseline/regional_baseline")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures")
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pts = gpd.read_file(args.pioneer / "Initiation_Points_final/Initiation_Points_final.shp")
    pts_m = pts.to_crs(3857)
    style = {"LS": ("#ff2b2b", "o", "Landslide"), "Runoff": ("#2bff7a", "^", "Runoff"),
             "Firehose": ("#3d9bff", "s", "Firehose")}

    # ---------- fig4: UAV window, the three modelled layers on imagery ----------
    ref, ref_ext = warp_to_merc(args.local_layers / "P_Landslide.tif")
    ll = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326",
                                        ref_ext[0], ref_ext[2], ref_ext[1], ref_ext[3])
    print("  fetching UAV-window imagery (z=15)...")
    img, img_ext = fetch_tiles(ll, 15, IMAGERY)
    lat_mid = (ll[1] + ll[3]) / 2

    titles = [("P_Landslide", "Landslide"), ("P_Combined_H", "Combined hillslope"),
              ("P_Channel_ini", "Channel initiation")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4), constrained_layout=True)
    im = None
    for ax, (stem, title), tag in zip(axes, titles, "abc"):
        arr, ext = warp_to_merc(args.local_layers / f"{stem}.tif")
        ax.imshow(img, extent=img_ext, zorder=1, interpolation="bilinear")
        im = ax.imshow(np.where(arr > 0.001, arr, np.nan), extent=ext, cmap="YlOrRd",
                       vmin=0, vmax=1, zorder=2, interpolation="nearest", alpha=0.85)
        for mech, (colour, marker, label) in style.items():
            sel = pts_m[pts_m.FailureTyp == mech]
            ax.scatter(sel.geometry.x, sel.geometry.y, s=26, c=colour, marker=marker,
                       edgecolor="black", linewidth=0.7, zorder=5,
                       label=f"{label} (n={len(sel)})" if tag == "a" else None)
        ax.set_xlim(ref_ext[0], ref_ext[1]); ax.set_ylim(ref_ext[2], ref_ext[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, pad=7)
        furniture_box(ax, 0.018, 0.93, 0.062, 0.055)
        ax.text(0.049, 0.9565, f"({tag})", transform=ax.transAxes, ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=13)
        attribution(ax)
    scale_bar(axes[0], 1000, "1 km", lat_mid)
    north_arrow(axes[2])
    leg = axes[0].legend(loc="upper right", framealpha=0.95, borderpad=0.6,
                         title="Mapped initiation", title_fontsize=10)
    leg.get_frame().set_linewidth(0.8); leg.set_zorder(14)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.4, aspect=34, pad=0.015)
    cb.set_label("Modelled probability of failure", fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(args.output_dir / f"fig4_uav_window_modelled.{e}")
    plt.close(fig)
    print("  fig4_uav_window_modelled")

    # ---------- fig5: regional 3D oblique ----------
    prob, prob_ext = warp_to_merc(args.regional / "P_Landslide_regional_baseline.tif")
    llr = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326",
                                         prob_ext[0], prob_ext[2], prob_ext[1], prob_ext[3])
    print("  fetching regional imagery + terrain...")
    rimg, rimg_ext = fetch_tiles(llr, 12, IMAGERY)
    rdem, rdem_ext = fetch_tiles(llr, 11, TERRAIN, decode=terrarium)

    shape = (420, 350)
    base = regrid(rimg, rimg_ext, prob_ext, shape)
    dem = regrid(rdem, rdem_ext, prob_ext, shape)
    lay = regrid(np.where(prob > 0.001, prob, np.nan), prob_ext, prob_ext, shape)
    colours = drape(base, lay, Normalize(0, 1), matplotlib.colormaps["YlOrRd"], alpha=0.92)

    scene_w = prob_ext[1] - prob_ext[0]
    fig = plt.figure(figsize=(14.0, 7.2))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    render_3d(ax, dem, colours, scene_w, vert_exag=2.2, view=(46, -58))
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, 1))
    cax = fig.add_axes([0.905, 0.28, 0.015, 0.44])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Modelled probability of failure (landslide)", fontsize=11)
    fig.text(0.012, 0.045, f"Scene width ≈ {scene_w * math.cos(math.radians((llr[1]+llr[3])/2))/1000:.0f} km"
                           f"\nVertical exaggeration ×2.2",
             fontsize=10.5, va="bottom", zorder=13,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.3", linewidth=0.8))
    fig.text(0.988, 0.018, ATTRIB, ha="right", fontsize=7.5, color="0.25")
    fig.suptitle("Regional baseline landslide probability", fontsize=14, y=0.975)
    for e in ("png", "pdf"):
        fig.savefig(args.output_dir / f"fig5_regional_3d.{e}", bbox_inches=None)
    plt.close(fig)
    print("  fig5_regional_3d")

    # ---------- fig6: UAV window 3D oblique, dDEM erosion on the 1 m DEM ----------
    ero, ero_ext = warp_to_merc(args.pioneer / "m4_erosion/dod_erosion.tif")
    dem_l, dem_l_ext = warp_to_merc(args.pioneer / "pre.tif")
    lle = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326",
                                         ero_ext[0], ero_ext[2], ero_ext[1], ero_ext[3])
    print("  fetching UAV-window imagery for 3D (z=15)...")
    limg, limg_ext = fetch_tiles(lle, 15, IMAGERY)
    shape = (430, 470)
    base = regrid(limg, limg_ext, ero_ext, shape)
    demL = regrid(dem_l, dem_l_ext, ero_ext, shape)
    depth = regrid(np.where(np.isfinite(ero) & (ero < 0), -ero, np.nan), ero_ext, ero_ext, shape)
    colours = drape(base, depth, Normalize(0.5, 3.0), ERO_CMAP, alpha=0.95)

    scene_w = ero_ext[1] - ero_ext[0]
    fig = plt.figure(figsize=(14.0, 7.2))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    render_3d(ax, demL, colours, scene_w, vert_exag=1.8, view=(44, -60))
    sm = plt.cm.ScalarMappable(cmap=ERO_CMAP, norm=Normalize(0.5, 3.0))
    cax = fig.add_axes([0.905, 0.28, 0.015, 0.44])
    cb = fig.colorbar(sm, cax=cax, extend="max")
    cb.set_label("Erosion depth (m), dDEM", fontsize=11)
    fig.text(0.012, 0.045, f"Scene width ≈ {scene_w * math.cos(math.radians((lle[1]+lle[3])/2))/1000:.1f} km"
                           f"\nVertical exaggeration ×1.8",
             fontsize=10.5, va="bottom", zorder=13,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.3", linewidth=0.8))
    fig.text(0.988, 0.018, ATTRIB, ha="right", fontsize=7.5, color="0.25")
    fig.suptitle("UAV survey window: dDEM erosion", fontsize=14, y=0.975)
    for e in ("png", "pdf"):
        fig.savefig(args.output_dir / f"fig6_uav_window_3d.{e}", bbox_inches=None)
    plt.close(fig)
    print("  fig6_uav_window_3d")
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
