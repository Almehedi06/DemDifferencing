"""Final publication figure set: Pioneer/Stehekin hazard layers.

Supersedes make_publication_figures.py and make_3d_figures.py. Composed in
Python from tile services rather than screenshotted, so output is
reproducible, correctly projected and print-resolution (300 dpi, vector text
in the PDFs). Sized and styled for a Nature/Elsevier double-column figure.

Figures
-------
fig1  Regional, 2D: all three modelled layers on satellite imagery.
fig2  UAV window, 2D: all three modelled layers + mapped initiation points.
fig3  Regional, 3D oblique: all three modelled layers draped on terrain.
fig4  UAV window, 3D oblique: all three modelled layers draped on terrain.
fig5  UAV window, 2D: dDEM erosion + mapped initiation points.
fig6  UAV window, 3D oblique: dDEM erosion draped on the 1 m terrain.

Data sources
------------
Imagery   Esri World Imagery tiles; attribution drawn on every panel.
Terrain   AWS terrarium tiles, decoded (R*256 + G + B/256) - 32768 metres,
          except fig6 which uses the 1 m pre-event DEM.

Two conventions worth preserving
--------------------------------
1. Scale bars are corrected for Web Mercator's 1/cos(latitude) stretch
   (~1.5x at 48.3 N). A naive bar would read ~50% short.
2. 3D panels carry a stated scene width and vertical exaggeration, never a
   scale bar: a perspective view has no single scale, so a bar would be right
   at exactly one depth and wrong everywhere else.
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
# Panel order is deliberate: the two specific mechanisms first, the combined
# layer last. Names are written in full -- no abbreviations in published panels.
LAYERS = [("P_Landslide", "Landslide"),
          ("P_Channel_ini", "Channel initiation"),
          ("P_Combined_H", "Combined hillslope")]
POINT_STYLE = {"LS": ("#ff2b2b", "o", "Landslide"), "Runoff": ("#2bff7a", "^", "Runoff"),
               "Firehose": ("#3d9bff", "s", "Firehose")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.0,
    "axes.labelsize": 14, "axes.titlesize": 16, "legend.fontsize": 12.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300,
})


# ----------------------------------------------------------------- tiles ----
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
    w, s, e, n = bounds_lonlat
    x0, y0 = tile_index(w, n, z)
    x1, y1 = tile_index(e, s, z)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    out = (np.zeros((256 * ny, 256 * nx, 3), "uint8") if decode is None
           else np.full((256 * ny, 256 * nx), np.nan))

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
    ny, nx = shape
    xs = np.linspace(dst_ext[0], dst_ext[1], nx)
    ys = np.linspace(dst_ext[3], dst_ext[2], ny)
    sy, sx = arr.shape[:2]
    cx = ((xs - src_ext[0]) / (src_ext[1] - src_ext[0]) * (sx - 1)).round().astype(int)
    cy = ((src_ext[3] - ys) / (src_ext[3] - src_ext[2]) * (sy - 1)).round().astype(int)
    ok_x, ok_y = (cx >= 0) & (cx < sx), (cy >= 0) & (cy < sy)
    fill = np.nan if arr.dtype.kind == "f" else 0
    out = np.full((ny, nx) + arr.shape[2:], fill, arr.dtype)
    out[...] = arr[np.where(ok_y, cy, 0)[:, None], np.where(ok_x, cx, 0)[None, :]]
    out[~ok_y, :] = fill
    out[:, ~ok_x] = fill
    return out


# ------------------------------------------------------------- furniture ----
def box(ax, x, y, w, h, z=11):
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, zorder=z,
                           facecolor="white", edgecolor="0.25", linewidth=1.0, alpha=0.95))


def scale_bar(ax, length_m, label, lat, loc=(0.045, 0.045)):
    merc = length_m / math.cos(math.radians(lat))
    xl, yl = ax.get_xlim(), ax.get_ylim()
    sx, sy = xl[1] - xl[0], yl[1] - yl[0]
    box(ax, loc[0] - 0.022, loc[1] - 0.018, merc / sx + 0.044, 0.115)
    x0, y0 = xl[0] + loc[0] * sx, yl[0] + (loc[1] + 0.016) * sy
    h = 0.024 * sy
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * merc / 4, y0), merc / 4, h,
                               facecolor="white" if i % 2 else "black",
                               edgecolor="black", linewidth=1.0, zorder=12))
    ax.text(x0 + merc / 2, y0 + h * 1.4, label, ha="center", va="bottom",
            fontsize=13, fontweight="bold", zorder=13)


def north_arrow(ax, loc=(0.935, 0.83)):
    x, y = loc
    box(ax, x - 0.055, y - 0.028, 0.11, 0.165)
    ax.annotate("", xy=(x, y + 0.088), xytext=(x, y + 0.015), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2.4), zorder=13)
    ax.text(x, y + 0.092, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontweight="bold", zorder=13)


def panel_tag(ax, tag):
    box(ax, 0.015, 0.918, 0.072, 0.068, z=13)
    ax.text(0.051, 0.952, tag, transform=ax.transAxes, ha="center", va="center",
            fontsize=17, fontweight="bold", zorder=14)


def attribution(ax):
    ax.text(0.99, 0.012, ATTRIB, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="white", zorder=13,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])


def drape(base_rgb, layer, norm, cmap, alpha=0.92):
    out = base_rgb.astype("float64") / 255.0
    m = np.isfinite(layer)
    if m.any():
        rgba = cmap(norm(np.where(m, layer, 0)))[..., :3]
        a = np.where(m, alpha, 0.0)[..., None]
        out = out * (1 - a) + rgba * a
    return np.clip(out, 0, 1)


def render_3d(ax, dem, colours, scene_w_m, vert_exag, view=(46, -60), position=None):
    ny, nx = dem.shape
    y, x = np.mgrid[0:ny, 0:nx].astype("float64")
    x *= scene_w_m / nx
    y *= scene_w_m / nx
    filled = np.nan_to_num(dem, nan=np.nanmin(dem))
    z = np.where(np.isfinite(dem), dem, np.nan) * vert_exag   # NaN -> quad not drawn
    ls = LightSource(azdeg=315, altdeg=50)
    shade = ls.hillshade(filled, dx=scene_w_m / nx, dy=scene_w_m / nx, vert_exag=2.0)
    ax.plot_surface(x, y[::-1], z, facecolors=np.clip(colours * (0.55 + 0.65 * shade[..., None]), 0, 1),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    ax.set_box_aspect((1, ny / nx, 0.30))
    ax.margins(0)
    try:
        ax.set_proj_type("persp", focal_length=0.62)
    except TypeError:
        ax.set_proj_type("persp")
    if position:
        ax.set_position(position)


def save(fig, out_dir, stem):
    for e in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{e}", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  {stem}")


# ----------------------------------------------------------------- panels ---
def panels_2d(layer_dir, suffix, img, img_ext, ref_ext, lat, pts_m, out_dir, stem,
              bar_m, bar_label, show_points, floor=0.001, clip_footprint=None,
              boundary=None):
    """floor: values at or below this are drawn as basemap, not colour. Raising it
    above the nominal zero-floor is a *display* choice for readability -- it does
    not change any score, and the value is stated on the figure so the reader
    knows what was hidden.

    clip_footprint: path to a raster whose modelled domain (cells > 0) limits
    where colour is drawn. Needed when comparing runs with different domains --
    the 30 m regional model covers the whole frame while the 10 m local run was
    clipped to the UAV watersheds, so without this the two read as different map
    extents when the axes limits are in fact identical.
    """
    mask_fp = warp_to_merc(clip_footprint) if clip_footprint is not None else None
    bnd = gpd.read_file(boundary).to_crs(3857) if boundary is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 8.2), constrained_layout=True)
    im = None
    for ax, (key, title), tag in zip(axes, LAYERS, "abc"):
        arr, ext = warp_to_merc(layer_dir / f"{key}{suffix}.tif")
        shown = np.where(arr > floor, arr, np.nan)
        if mask_fp is not None:
            fp, fp_ext = mask_fp
            keep = regrid(np.where(np.isfinite(fp) & (fp > 0), 1.0, np.nan), fp_ext, ext, arr.shape)
            shown = np.where(np.isfinite(keep), shown, np.nan)
        ax.imshow(img, extent=img_ext, zorder=1, interpolation="bilinear")
        im = ax.imshow(shown, extent=ext, cmap="YlOrRd",
                       vmin=0, vmax=1, zorder=2, interpolation="nearest", alpha=0.88)
        if show_points:
            for mech, (colour, marker, label) in POINT_STYLE.items():
                sel = pts_m[pts_m.FailureTyp == mech]
                ax.scatter(sel.geometry.x, sel.geometry.y, s=46, c=colour, marker=marker,
                           edgecolor="black", linewidth=0.9, zorder=5,
                           label=f"{label} (n={len(sel)})" if tag == "a" else None)
        if bnd is not None:
            bnd.boundary.plot(ax=ax, color="#00e5ff", linewidth=1.8, zorder=7)
        ax.set_xlim(ref_ext[0], ref_ext[1]); ax.set_ylim(ref_ext[2], ref_ext[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, pad=10)
        panel_tag(ax, tag)
        attribution(ax)
    scale_bar(axes[0], bar_m, bar_label, lat)
    north_arrow(axes[2])
    if show_points:
        leg = axes[0].legend(loc="upper right", framealpha=0.96, borderpad=0.7,
                             title="Mapped initiation", title_fontsize=13)
        leg.get_frame().set_linewidth(1.0); leg.set_zorder(14)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.35, aspect=36, pad=0.015)
    label = "Modelled probability of failure"
    if floor > 0.001:
        label += f"   (cells \u2264 {floor:g} not shown)"
    cb.set_label(label, fontsize=15)
    cb.ax.tick_params(labelsize=13)
    save(fig, out_dir, stem)


def panels_3d(layer_dir, suffix, base, dem, ref_ext, shape, scene_w_m, vert_exag,
              out_dir, stem, scene_km, view=(46, -60)):
    fig = plt.figure(figsize=(21.0, 8.4))
    for i, ((key, title), tag) in enumerate(zip(LAYERS, "abc")):
        arr, ext = warp_to_merc(layer_dir / f"{key}{suffix}.tif")
        lay = regrid(np.where(arr > 0.001, arr, np.nan), ext, ref_ext, shape)
        colours = drape(base, lay, Normalize(0, 1), matplotlib.colormaps["YlOrRd"])
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", computed_zorder=False)
        render_3d(ax, dem, colours, scene_w_m, vert_exag, view=view,
                  position=[i / 3 - 0.038, 0.005, 1 / 3 + 0.076, 1.045])
        fig.text(i / 3 + 1 / 6, 0.975, title, ha="center", va="top", fontsize=16)
        fig.text(i / 3 + 0.022, 0.925, tag, ha="left", va="top", fontsize=17,
                 fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.34", facecolor="white",
                           edgecolor="0.25", linewidth=1.0))
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, 1))
    cax = fig.add_axes([0.31, 0.035, 0.38, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Modelled probability of failure", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.text(0.012, 0.030, f"Scene width ≈ {scene_km}\nVertical exaggeration ×{vert_exag}",
             fontsize=12.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="0.25", linewidth=1.0))
    fig.text(0.988, 0.012, ATTRIB, ha="right", fontsize=9, color="0.3")
    save(fig, out_dir, stem)



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    ap.add_argument("--pioneer", type=Path, default=D / "pioneer")
    ap.add_argument("--local-layers", type=Path, default=D / "pioneer/probability_fields")
    ap.add_argument("--regional", type=Path, default=D / "pioneer/regional_baseline/regional_baseline")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures_final")
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pts_m = gpd.read_file(args.pioneer / "Initiation_Points_final/Initiation_Points_final.shp").to_crs(3857)

    # ---- regional context ----
    ref_r, ext_r = warp_to_merc(args.regional / "P_Landslide_regional_baseline.tif")
    ll_r = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326", ext_r[0], ext_r[2], ext_r[1], ext_r[3])
    lat_r = (ll_r[1] + ll_r[3]) / 2
    print("  fetching regional imagery + terrain...")
    img_r, img_r_ext = fetch_tiles(ll_r, 12, IMAGERY)
    dem_r_raw, dem_r_ext = fetch_tiles(ll_r, 11, TERRAIN, decode=terrarium)
    km_r = f"{(ext_r[1] - ext_r[0]) * math.cos(math.radians(lat_r)) / 1000:.0f} km"

    panels_2d(args.regional, "_regional_baseline", img_r, img_r_ext, ext_r, lat_r,
              pts_m, args.output_dir, "fig1_regional_layers_2d", 5000, "5 km", False)

    shape_r = (420, 350)
    panels_3d(args.regional, "_regional_baseline",
              regrid(img_r, img_r_ext, ext_r, shape_r), regrid(dem_r_raw, dem_r_ext, ext_r, shape_r),
              ext_r, shape_r, ext_r[1] - ext_r[0], 2.2, args.output_dir,
              "fig3_regional_layers_3d", km_r)

    # ---- UAV window ----
    ref_l, ext_l = warp_to_merc(args.local_layers / "P_Landslide.tif")
    ll_l = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326", ext_l[0], ext_l[2], ext_l[1], ext_l[3])
    lat_l = (ll_l[1] + ll_l[3]) / 2
    print("  fetching UAV-window imagery + terrain...")
    img_l, img_l_ext = fetch_tiles(ll_l, 15, IMAGERY)
    dem_l, dem_l_ext = warp_to_merc(args.pioneer / "pre.tif")
    km_l = f"{(ext_l[1] - ext_l[0]) * math.cos(math.radians(lat_l)) / 1000:.1f} km"

    panels_2d(args.local_layers, "", img_l, img_l_ext, ext_l, lat_l,
              pts_m, args.output_dir, "fig2_uav_layers_2d", 1000, "1 km", True)

    shape_l = (430, 470)
    panels_3d(args.local_layers, "",
              regrid(img_l, img_l_ext, ext_l, shape_l), regrid(dem_l, dem_l_ext, ext_l, shape_l),
              ext_l, shape_l, ext_l[1] - ext_l[0], 1.8, args.output_dir,
              "fig4_uav_layers_3d", km_l, view=(44, -60))

    # ---- dDEM erosion, 2D and 3D ----
    ero, ero_ext = warp_to_merc(args.pioneer / "m4_erosion/dod_erosion.tif")
    fig, ax = plt.subplots(figsize=(10.5, 9.4), constrained_layout=True)
    ax.imshow(img_l, extent=img_l_ext, zorder=1, interpolation="bilinear")
    depth = np.where(np.isfinite(ero) & (ero < 0), -ero, np.nan)
    im = ax.imshow(depth, extent=ero_ext, cmap=ERO_CMAP, vmin=0.5, vmax=3.0,
                   zorder=2, interpolation="nearest", alpha=0.95)
    for mech, (colour, marker, label) in POINT_STYLE.items():
        sel = pts_m[pts_m.FailureTyp == mech]
        ax.scatter(sel.geometry.x, sel.geometry.y, s=48, c=colour, marker=marker,
                   edgecolor="black", linewidth=0.9, zorder=5, label=f"{label} (n={len(sel)})")
    ax.set_xlim(ero_ext[0], ero_ext[1]); ax.set_ylim(ero_ext[2], ero_ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    scale_bar(ax, 1000, "1 km", lat_l)
    north_arrow(ax)
    leg = ax.legend(loc="upper left", framealpha=0.96, borderpad=0.7,
                    title="Mapped initiation", title_fontsize=13)
    leg.get_frame().set_linewidth(1.0); leg.set_zorder(14)
    cb = fig.colorbar(im, ax=ax, shrink=0.58, aspect=26, pad=0.015, extend="max")
    cb.set_label("Erosion depth (m), dDEM", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    attribution(ax)
    save(fig, args.output_dir, "fig5_uav_erosion_2d")

    base_l = regrid(img_l, img_l_ext, ero_ext, shape_l)
    demL = regrid(dem_l, dem_l_ext, ero_ext, shape_l)
    colours = drape(base_l, regrid(depth, ero_ext, ero_ext, shape_l),
                    Normalize(0.5, 3.0), ERO_CMAP, alpha=0.95)
    fig = plt.figure(figsize=(14.5, 8.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    render_3d(ax, demL, colours, ero_ext[1] - ero_ext[0], 1.8, view=(44, -60),
              position=[-0.10, -0.16, 1.14, 1.30])
    sm = plt.cm.ScalarMappable(cmap=ERO_CMAP, norm=Normalize(0.5, 3.0))
    cax = fig.add_axes([0.895, 0.30, 0.017, 0.44])
    cb = fig.colorbar(sm, cax=cax, extend="max")
    cb.set_label("Erosion depth (m), dDEM", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.text(0.012, 0.045, f"Scene width ≈ {km_l}\nVertical exaggeration ×1.8",
             fontsize=12.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.25", linewidth=1.0))
    fig.text(0.988, 0.012, ATTRIB, ha="right", fontsize=9, color="0.3")
    save(fig, args.output_dir, "fig6_uav_erosion_3d")

    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
