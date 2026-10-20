"""Slide 14 (the cascade): locator map, burn severity map, storm forcing, timeline.

Each figure stands alone so the slide can be arranged freely. Numbers are read
from the source data, not typed in:

  fire size, ignition date    BAER burn-boundary attributes (BURNBNDAC, IG_DATE)
  severity class areas        the BAER SBS raster (20 m pixels, EPSG:26910)
  storm forcing               landlab_data_prep PRISM output, lidar-footprint AOI
  mapped initiations          the USGS initiation-point shapefile

The only typed-in facts are in EVENTS: storm and survey dates from the USGS data
release abstract, and the reported time the plant was destroyed (03:00 PST 11 Dec).

PRISM day convention (PRISM_datasets.pdf): "A PRISM day is defined as 1200
UTC-1200 UTC ... The dataset uses a day-ending naming convention". At Stehekin
(PST, UTC-8) the value labelled 11 Dec fell 04:00 10 Dec -> 04:00 11 Dec. Bars
are drawn over the hours they cover, so the plant loss at 03:00 11 Dec lands in
the last hour of the wettest PRISM day, where it belongs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
from matplotlib import font_manager
from matplotlib.colors import (BoundaryNorm, LightSource, LinearSegmentedColormap, ListedColormap,
                               Normalize, to_rgb)
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D, ScaledTranslation
from pyproj import Transformer
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_final_figures as M  # noqa: E402  tile fetch + Web Mercator helpers

D = Path("/mnt/c/Users/amehedi/Downloads")
P = D / "pioneer"
PERIM = P / "output/baer_wa4818312053120240608_10030843_burn_area.shp"
SBS_TIF = P / "output/pioneer_wa4818312053120240608_sbs.tif"
UAV_REF = P / "probability_fields/P_Landslide.tif"
POINTS = P / "Initiation_Points_final/Initiation_Points_final.shp"
FORCING = P / "output/lidar_footprint_10m/prism_forcing/forcing_daily_prism.csv"
FORCING_AOI = P / "pioneer_lidar_footprint.shp"
BASEMAP = P / "basemap"
STEHEKIN = (-120.65528, 48.30944)                 # lon, lat: 48°18′34″N 120°39′19″W (GNIS via Wikipedia)

EVENTS = {"storm": ("2025-12-08", "2025-12-12"), "plant_destroyed": "2025-12-11 03:00",
          "survey": ("2026-06-15", "2026-06-17")}
SBS_CLASSES = [(1, "Unburned / very low", "#4daf4a"), (2, "Low", "#ffeda0"),
               (3, "Moderate", "#fd8d3c"), (4, "High", "#bd0026")]
FIRE, STORM, SURVEY = "#D8491F", "#2F6DB5", "#138A8A"
INK, MUTED = "#1F2A37", "#5B6573"
ACRE_KM2 = 0.0040468564224
HALO = [pe.withStroke(linewidth=2.6, foreground="black")]
# hypsometric tints for the physical locator: (elevation m, colour)
HYPSO = LinearSegmentedColormap.from_list("hypso", [(e / 4400, c) for e, c in (
    (0, "#6FA56A"), (150, "#8DBB76"), (400, "#B9D28E"), (800, "#E6E3A4"), (1300, "#DCC08A"),
    (1900, "#BE9464"), (2600, "#A87C5A"), (3300, "#D8CFC6"), (4400, "#FFFFFF"))])
SEA, LAKE, RIVER = "#C6DDF0", "#A9CDEB", "#5B9BD5"


def use_arial():
    fonts = Path("/mnt/c/Windows/Fonts")
    for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        if (fonts / name).exists():
            font_manager.fontManager.addfont(str(fonts / name))
    mpl.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 300, "axes.linewidth": 0.8})


def save(fig, out, stem, svg=False, pad=0.04):
    for ext in ("png", "pdf") + (("svg",) if svg else ()):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", pad_inches=pad,
                    facecolor="white")
    plt.close(fig)
    print(f"  {stem}")


def day(ts):
    return f"{ts.day} {ts:%b %Y}"


def fire_facts():
    perim = gpd.read_file(PERIM)
    assert len(perim) == 1, f"expected one burn boundary, got {len(perim)}"
    acres = int(perim.BURNBNDAC.iloc[0])
    return perim, acres, acres * ACRE_KM2, pd.Timestamp(perim.IG_DATE.iloc[0])


# ------------------------------------------------------------------ icons ---
def flame_paths():
    """Outer and inner flame, base at (0, 0), height 1."""
    C, MOVE, CLOSE = MPath.CURVE4, MPath.MOVETO, MPath.CLOSEPOLY
    outer = [(0, 0), (0.30, 0.00), (0.36, 0.25), (0.30, 0.45), (0.26, 0.60), (0.12, 0.72),
             (0.06, 1.00), (0.02, 0.86), (-0.06, 0.74), (-0.10, 0.66), (-0.12, 0.72),
             (-0.16, 0.78), (-0.22, 0.82), (-0.34, 0.62), (-0.36, 0.30), (-0.30, 0.18),
             (-0.24, 0.02), (-0.12, 0.00), (0, 0), (0, 0)]
    inner = [(0, 0.07), (0.16, 0.07), (0.20, 0.20), (0.15, 0.32), (0.11, 0.42), (0.05, 0.47),
             (0.03, 0.60), (-0.02, 0.50), (-0.12, 0.44), (-0.15, 0.32), (-0.19, 0.20),
             (-0.15, 0.07), (0, 0.07), (0, 0.07)]
    return (MPath(outer, [MOVE] + [C] * 18 + [CLOSE]),
            MPath(inner, [MOVE] + [C] * 12 + [CLOSE]))


def icon_transform(ax, x, y, unit_in):
    """Icon units -> display: 1 unit = unit_in inches, origin at data point (x, y)."""
    return (Affine2D().scale(unit_in) + ax.figure.dpi_scale_trans
            + ScaledTranslation(x, y, ax.transData))


def draw_flame(ax, x, y, height_in, z=10, colour=FIRE, core="#FFC83D", halo="white"):
    tr = Affine2D().translate(0, -0.5) + icon_transform(ax, x, y, height_in)
    outer, inner = flame_paths()
    if halo:
        ax.add_patch(PathPatch(outer, transform=tr, fc=halo, ec=halo, lw=3.0, zorder=z))
    ax.add_patch(PathPatch(outer, transform=tr, fc=colour, ec="#8C2A0E" if halo else colour,
                           lw=0.6, zorder=z + 1))
    ax.add_patch(PathPatch(inner, transform=tr, fc=core, ec="none", zorder=z + 2))


def draw_cloud_rain(ax, x, y, r_in, z=10):
    tr = icon_transform(ax, x, y, r_in)
    for cx, cy, rr in ((-0.30, 0.08, 0.25), (0.02, 0.24, 0.33), (0.31, 0.06, 0.24)):
        ax.add_patch(Circle((cx, cy), rr, transform=tr, fc="white", ec="none", zorder=z))
    ax.add_patch(Rectangle((-0.30, -0.17), 0.61, 0.25, transform=tr, fc="white", ec="none",
                           zorder=z))
    for dx in (-0.24, 0.03, 0.30):
        ax.add_line(Line2D([dx, dx - 0.09], [-0.30, -0.60], transform=tr, color="white",
                           lw=2.0, solid_capstyle="round", zorder=z))


def draw_drone(ax, x, y, r_in, bg, z=10):
    tr = icon_transform(ax, x, y, r_in)
    for sx, sy in ((1, 1), (1, -1)):
        ax.add_line(Line2D([-0.42 * sx, 0.42 * sx], [-0.42 * sy, 0.42 * sy], transform=tr,
                           color="white", lw=2.2, zorder=z))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Circle((0.42 * sx, 0.42 * sy), 0.19, transform=tr, fc=bg,
                                ec="white", lw=1.8, zorder=z + 1))
    ax.add_patch(FancyBboxPatch((-0.17, -0.17), 0.34, 0.34, transform=tr,
                                boxstyle="round,pad=0,rounding_size=0.07", fc="white",
                                ec="none", zorder=z + 2))


# ------------------------------------------------------------ burn severity ---
def figure_burn_severity(out):
    perim, acres, km2, ign = fire_facts()
    perim_m = perim.to_crs(3857)

    with rasterio.open(SBS_TIF) as src:
        raw = src.read(1)
        px_km2 = abs(src.res[0] * src.res[1]) / 1e6
        t, w, h = rasterio.warp.calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        sbs = np.zeros((h, w), "uint8")
        rasterio.warp.reproject(rasterio.band(src, 1), sbs, src_transform=src.transform,
                                src_crs=src.crs, dst_transform=t, dst_crs="EPSG:3857",
                                resampling=rasterio.warp.Resampling.nearest)
    sbs_ext = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)       # exact grid, not transformed bounds
    sbs = np.where(np.isin(sbs, [1, 2, 3, 4]), sbs, np.nan)

    areas = {k: float((raw == k).sum() * px_km2) for k, _, _ in SBS_CLASSES}
    assert abs(sum(areas.values()) - km2) / km2 < 0.01, \
        f"severity classes sum to {sum(areas.values()):.1f} km², perimeter says {km2:.1f}"

    _, uav = M.warp_to_merc(UAV_REF)
    pb = perim_m.total_bounds
    ext = (pb[0] - 6000, pb[2] + 9000, pb[1] - 3000, pb[3] + 3500)
    ll = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326", ext[0], ext[2], ext[1], ext[3])
    img, img_ext = M.fetch_tiles(ll, 13, M.IMAGERY)

    aspect = (ext[1] - ext[0]) / (ext[3] - ext[2])
    fig = plt.figure(figsize=(6.4 * aspect, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=img_ext, zorder=1, interpolation="bilinear")
    ax.imshow(sbs, extent=sbs_ext, cmap=ListedColormap([c for _, _, c in SBS_CLASSES]),
              norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4), zorder=2,
              interpolation="nearest", alpha=0.85)
    perim_m.boundary.plot(ax=ax, color="white", linewidth=1.3, zorder=4)

    # UAV survey window, labelled directly
    ax.add_patch(Rectangle((uav[0], uav[2]), uav[1] - uav[0], uav[3] - uav[2], fill=False,
                           edgecolor="#00E5FF", linewidth=1.8, zorder=6))
    ax.text(uav[1] + 600, uav[3] - 200, "UAV survey\narea", ha="left", va="top", fontsize=9.5,
            fontweight="bold", color="#00E5FF", zorder=7, linespacing=1.05, path_effects=HALO)

    # Stehekin
    sx, sy = gpd.GeoSeries([Point(*STEHEKIN)], crs=4326).to_crs(3857).iloc[0].coords[0]
    ax.plot(sx, sy, "o", ms=6, mfc="white", mec="black", mew=1.0, zorder=8)
    ax.text(sx - 700, sy - 250, "Stehekin", ha="right", va="top", fontsize=10,
            fontweight="bold", color="white", zorder=8, path_effects=HALO)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.25")

    # fire facts box, top right
    facts = VPacker(children=[
        TextArea("Pioneer Fire (2024)", textprops=dict(fontsize=12, fontweight="bold", color=INK)),
        TextArea(f"{acres:,} ac  ·  {km2:.0f} km²\nIgnited {day(ign)}",
                 textprops=dict(fontsize=10.5, color=INK, linespacing=1.4))],
        align="left", pad=0, sep=5)
    box_ = AnchoredOffsetbox(loc="upper right", child=facts, pad=0.6, borderpad=0.7, frameon=True)
    box_.patch.set_boxstyle("round,pad=0,rounding_size=0.35")
    box_.patch.set(facecolor="white", edgecolor="0.35", linewidth=0.8, alpha=0.95)
    box_.set_zorder(14)
    ax.add_artist(box_)

    # severity legend with class areas, bottom left
    handles = [Patch(facecolor=c, edgecolor="0.3", linewidth=0.5,
                     label=f"{lab}  ({areas[k]:.0f} km²)") for k, lab, c in SBS_CLASSES]
    leg = ax.legend(handles=handles[::-1], loc="lower left", title="Soil burn severity",
                    title_fontsize=10.5, fontsize=9.5, framealpha=0.95, borderpad=0.6,
                    handlelength=1.4, handleheight=1.0, edgecolor="0.35",
                    bbox_to_anchor=(0.012, 0.012))
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(14)

    # scale bar + north arrow, right side
    km = 5
    x1, y0 = ext[1] - 0.045 * (ext[1] - ext[0]), ext[2] + 0.075 * (ext[3] - ext[2])
    lat = np.degrees(2 * np.arctan(np.exp(y0 / 6378137.0)) - np.pi / 2)   # latitude of the bar itself
    s2 = np.sin(np.radians(lat)) ** 2                              # Web Mercator scale on WGS84
    merc = km * 1000 * np.sqrt(1 - 0.00669437999 * s2) / np.cos(np.radians(lat))
    x0 = x1 - merc
    bh = 0.011 * (ext[3] - ext[2])
    ax.add_patch(FancyBboxPatch((x0 - 0.03 * (ext[1] - ext[0]), y0 - 0.025 * (ext[3] - ext[2])),
                                merc + 0.06 * (ext[1] - ext[0]), 0.075 * (ext[3] - ext[2]),
                                boxstyle="round,pad=0,rounding_size=300", fc="white",
                                ec="0.35", lw=0.8, alpha=0.95, zorder=11))
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * merc / 4, y0), merc / 4, bh,
                               fc="black" if i % 2 == 0 else "white", ec="black", lw=0.8,
                               zorder=12))
    for frac, lab in ((0, "0"), (0.5, f"{km / 2:g}"), (1, f"{km} km")):
        ax.text(x0 + frac * merc, y0 + bh * 1.6, lab, ha="center", va="bottom", fontsize=9,
                zorder=13)
    nx = (x1 - merc / 2 - ext[0]) / (ext[1] - ext[0])          # centred over the scale bar
    ax.add_patch(FancyBboxPatch((nx - 0.035, 0.145), 0.07, 0.1, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.012", fc="white",
                                ec="0.35", lw=0.8, alpha=0.95, zorder=11))
    ax.annotate("", xy=(nx, 0.205), xytext=(nx, 0.155), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=14),
                zorder=13)
    ax.text(nx, 0.207, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="black", zorder=13)

    ax.text(0.99, 0.008, "Imagery: Esri, Maxar, Earthstar Geographics  ·  Severity: BAER",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="white",
            zorder=13, path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
    save(fig, out, "burn_severity_map", pad=0.02)
    return areas


# ---------------------------------------------------------------- locator ---
def figure_locator(out):
    states = gpd.read_file(f"zip://{BASEMAP / 'cb_2023_us_state_500k.zip'}")
    wa = states[states.STUSPS == "WA"].to_crs(32148)            # NAD83 / Washington North
    site = gpd.GeoSeries([Point(*STEHEKIN)], crs=4326).to_crs(32148).iloc[0]

    b = wa.total_bounds
    px, py = 0.03 * (b[2] - b[0]), 0.05 * (b[3] - b[1])
    ext = (b[0] - px, b[2] + px, b[1] - py, b[3] + py)
    aspect = (ext[1] - ext[0]) / (ext[3] - ext[2])
    fig = plt.figure(figsize=(3.9 * aspect, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    wa.plot(ax=ax, color="#F6C9A6", edgecolor="#B8551F", linewidth=0.9, zorder=2)

    draw_flame(ax, site.x, site.y, 0.30, z=10)
    ax.annotate("Stehekin, WA", xy=(site.x, site.y), xytext=(13, 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=11, fontweight="bold", color=INK, zorder=12,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.annotate("Pioneer Fire, 2024", xy=(site.x, site.y), xytext=(13, 3),
                textcoords="offset points", ha="left", va="top", fontsize=9.5, color=MUTED,
                zorder=12, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.55")
        s.set_linewidth(0.8)
    save(fig, out, "locator_washington", svg=True, pad=0.02)


def spaced(text):
    return "   ".join(" ".join(word) for word in text.split())


def figure_locator_topo(out):
    """Physical map of Washington: hillshaded hypsometric tints, neighbours faded."""
    crs = "EPSG:32148"                                           # NAD83 / Washington North
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    states = gpd.read_file(f"zip://{BASEMAP / 'cb_2023_us_state_500k.zip'}").to_crs(crs)
    wa = states[states.STUSPS == "WA"]
    b = wa.total_bounds
    bw, bh = b[2] - b[0], b[3] - b[1]
    res = 300.0
    W = int((b[2] + 0.06 * bw - (b[0] - 0.09 * bw)) // res)
    H = int((b[3] + 0.14 * bh - (b[1] - 0.13 * bh)) // res)
    ext = (b[0] - 0.09 * bw, b[0] - 0.09 * bw + W * res, b[3] + 0.14 * bh - H * res, b[3] + 0.14 * bh)
    dst_t = rasterio.transform.from_origin(ext[0], ext[3], res, res)

    # elevation: AWS terrarium tiles (as in make_final_figures), reprojected to the map grid
    ll = rasterio.warp.transform_bounds(crs, "EPSG:4326", ext[0], ext[2], ext[1], ext[3],
                                        densify_pts=50)
    dem, dem_ext = M.fetch_tiles((ll[0] - 0.1, ll[1] - 0.1, ll[2] + 0.1, ll[3] + 0.1), 9,
                                 M.TERRAIN, decode=M.terrarium)
    src_t = rasterio.transform.from_bounds(dem_ext[0], dem_ext[2], dem_ext[1], dem_ext[3],
                                           dem.shape[1], dem.shape[0])
    elev = np.full((H, W), np.nan, "float32")
    rasterio.warp.reproject(dem.astype("float32"), elev, src_transform=src_t, src_crs="EPSG:3857",
                            dst_transform=dst_t, dst_crs=crs, src_nodata=np.nan,
                            dst_nodata=np.nan, resampling=rasterio.warp.Resampling.bilinear)
    assert np.isfinite(elev).all(), f"{(~np.isfinite(elev)).sum()} cells without elevation"

    water = elev <= 0
    rgb = LightSource(azdeg=315, altdeg=45).shade(
        np.where(water, 0.0, elev), cmap=HYPSO, norm=Normalize(0, 4400), blend_mode="soft",
        vert_exag=5, dx=res, dy=res)[..., :3]
    in_wa = rasterio.features.geometry_mask(wa.geometry, (H, W), dst_t, invert=True)
    grey = rgb.mean(axis=2, keepdims=True)
    rgb = np.where(in_wa[..., None], rgb, 0.45 * (0.55 * rgb + 0.45 * grey) + 0.55)
    rgb[water] = to_rgb(SEA)

    fig = plt.figure(figsize=(4.4 * W / H, 4.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, extent=ext, interpolation="bilinear", zorder=1)

    clip_ll = box(ll[0] - 0.5, ll[1] - 0.5, ll[2] + 0.5, ll[3] + 0.5)
    clip_xy = box(ext[0], ext[2], ext[1], ext[3])

    def layer(*names):
        parts = [gpd.read_file(f"zip://{BASEMAP / (n + '.zip')}").clip(clip_ll) for n in names]
        return pd.concat(parts, ignore_index=True).to_crs(crs).clip(clip_xy)

    lakes = layer("ne_10m_lakes", "ne_10m_lakes_north_america")
    lakes.plot(ax=ax, color=LAKE, edgecolor="#6FA6D0", linewidth=0.3, zorder=3)
    for names, lw in ((("ne_10m_rivers_north_america",), 0.5), (("ne_10m_rivers_lake_centerlines",), 0.9)):
        rivers = layer(*names)
        rivers.plot(ax=ax, color=RIVER, linewidth=lw, alpha=0.4, zorder=3)
        rivers.clip(wa.geometry.iloc[0]).plot(ax=ax, color=RIVER, linewidth=lw, zorder=4)

    states[states.STUSPS == "ID"].boundary.clip(clip_xy).plot(
        ax=ax, color="#7A7A7A", linewidth=0.6, linestyle=(0, (4, 2)), zorder=5)
    wa.boundary.plot(ax=ax, color="#3A3A3A", linewidth=1.0, zorder=6)

    halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    def put(lon, lat, text, **kw):
        x, y = tf.transform(lon, lat)
        return ax.text(x, y, text, **kw)

    region = dict(fontsize=11, color="#6B6B6B", ha="center", va="center", fontweight="bold",
                  zorder=7)
    put(-120.2, 49.25, spaced("CANADA"), **region)
    put(-120.2, 45.35, spaced("OREGON"), **region)
    put(-116.68, 47.0, spaced("IDAHO"), rotation=-90, **region)
    put(-119.2, 46.85, spaced("WASHINGTON"), fontsize=17, color="#3D3A35", alpha=0.85,
        ha="center", va="center", fontweight="bold", zorder=7)
    put(-121.35, 47.3, spaced("CASCADE RANGE"), fontsize=8.5, color="#5B4632", rotation=-84,
        ha="center", va="center", fontweight="bold", zorder=7, path_effects=halo)
    put(-125.0, 47.1, "Pacific\nOcean", fontsize=9, style="italic", color="#2E6DA4",
        ha="center", va="center", zorder=7)
    put(-119.93, 47.95, "Lake Chelan", fontsize=8, style="italic", color="#2E6DA4", ha="left",
        va="center", zorder=7, path_effects=halo)

    for name, lon, lat, dx, ha in (("Seattle", -122.3321, 47.6062, 5, "left"),
                                   ("Spokane", -117.4260, 47.6588, -5, "right")):
        x, y = tf.transform(lon, lat)
        ax.plot(x, y, "o", ms=4, mfc="white", mec="#222222", mew=0.9, zorder=8)
        ax.annotate(name, (x, y), xytext=(dx, 0), textcoords="offset points", ha=ha,
                    va="center", fontsize=9, color="#222222", zorder=8, path_effects=halo)
    x, y = tf.transform(-121.7604, 46.8529)
    ax.plot(x, y, "^", ms=6, color="#222222", zorder=8)
    ax.annotate("Mt. Rainier", (x, y), xytext=(-5, 0), textcoords="offset points", ha="right",
                va="center", fontsize=8.5, color="#222222", zorder=8, path_effects=halo)

    sx, sy = tf.transform(*STEHEKIN)
    draw_flame(ax, sx, sy, 0.30, z=10)
    ax.annotate("Stehekin", (sx, sy), xytext=(12, 5), textcoords="offset points", ha="left",
                va="bottom", fontsize=10.5, fontweight="bold", color=INK, zorder=12,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.annotate("Pioneer Fire, 2024", (sx, sy), xytext=(12, 3), textcoords="offset points",
                ha="left", va="top", fontsize=9, color=MUTED, zorder=12,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # scale bar and true-north arrow, bottom left
    span_x, span_y = ext[1] - ext[0], ext[3] - ext[2]
    x0, y0, L, t = ext[0] + 0.035 * span_x, ext[2] + 0.05 * span_y, 100e3, 0.012 * span_y
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * L / 4, y0), L / 4, t, fc="black" if i % 2 == 0 else "white",
                               ec="black", lw=0.7, zorder=9))
    for frac, lab in ((0, "0"), (0.5, "50"), (1, "100 km")):
        ax.text(x0 + frac * L, y0 + 1.7 * t, lab, ha="center", va="bottom", fontsize=8, zorder=9)
    ax_, ay_ = x0 + 0.1 * L, y0 + 0.11 * span_y
    lon_a, lat_a = inv.transform(ax_, ay_)
    nx_, ny_ = tf.transform(lon_a, lat_a + 0.3)
    u = np.array([nx_ - ax_, ny_ - ay_]) / np.hypot(nx_ - ax_, ny_ - ay_) * 0.065 * span_y
    ax.annotate("", xy=(ax_ + u[0], ay_ + u[1]), xytext=(ax_, ay_),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4, mutation_scale=11), zorder=9)
    ax.text(ax_ + u[0] * 1.08, ay_ + u[1] * 1.08, "N", ha="center", va="bottom", fontsize=10,
            fontweight="bold", zorder=9)

    cax = ax.inset_axes([0.70, 0.065, 0.26, 0.022])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 4400), cmap=HYPSO), cax=cax,
                      orientation="horizontal", ticks=[0, 1000, 2000, 3000, 4000])
    cb.ax.tick_params(labelsize=8, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cax.set_title("Elevation (m)", fontsize=8.5, pad=3)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.45")
        sp.set_linewidth(0.8)
    ax.text(0.995, 0.005, "Elevation: AWS Terrain Tiles  ·  Boundaries: US Census, Natural Earth",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="#555555", zorder=9)
    save(fig, out, "locator_washington_topo", pad=0.02)


# ---------------------------------------------------------------- forcing ---
def figure_forcing(out):
    df = pd.read_csv(FORCING, parse_dates=["datetime"])
    assert df.datetime.is_monotonic_increasing and df.datetime.diff().dropna().eq(pd.Timedelta("1D")).all()
    aoi_km2 = gpd.read_file(FORCING_AOI).to_crs(32610).area.sum() / 1e6
    start = df.datetime - pd.Timedelta(days=1) + pd.Timedelta(hours=4)   # PRISM day in PST
    mid = start + pd.Timedelta(hours=12)
    s0 = pd.Timestamp(EVENTS["storm"][0])
    s1 = pd.Timestamp(EVENTS["storm"][1]) + pd.Timedelta(days=1)
    in_storm = ((mid >= s0) & (mid < s1)).to_numpy()
    storm_mm = df.precip_mm[in_storm].sum()
    ipk = int(df.precip_mm.idxmax())
    lost = pd.Timestamp(EVENTS["plant_destroyed"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.9), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.75, 1], hspace=0.07))
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    ax1.bar(start, df.precip_mm, width=1.0, align="edge", zorder=2,
            color=np.where(in_storm, STORM, "#A9C1E3"), edgecolor="white", linewidth=0.7)
    ax1.set_ylim(0, 140)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel("Precipitation (mm day$^{-1}$)", fontsize=10)
    ax1.grid(axis="y", color="0.9", lw=0.7, zorder=0)
    ax1.text(start[ipk] - pd.Timedelta(hours=2), df.precip_mm[ipk], f"{df.precip_mm[ipk]:.0f} mm",
             ha="right", va="top", fontsize=9.5, fontweight="bold", color=STORM)
    # atmospheric-river bracket
    yb = 124
    ax1.plot([s0, s0, s1, s1], [yb - 4, yb, yb, yb - 4], color=INK, lw=1.0, zorder=3)
    ax1.text(s0, yb + 2.5,
             f"Atmospheric river, 8–12 Dec  ·  {storm_mm:.0f} mm in {in_storm.sum()} days",
             ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
    # plant destroyed
    ym = 114
    ax1.plot([lost, lost], [0, ym - 3], color=FIRE, lw=1.2, ls=(0, (4, 2)), zorder=4)
    ax1.plot(lost, ym, marker="v", ms=7, color=FIRE, zorder=4)
    ax1.annotate(f"Wastewater plant destroyed, {lost.day} {lost:%b}, {lost:%H:%M}", (lost, ym),
                 xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=9,
                 color=FIRE, fontweight="bold")
    ax2.axvline(lost, color=FIRE, lw=1.2, ls=(0, (4, 2)), zorder=4)
    ax1.set_title("December 2025 storm at Stehekin (PRISM daily)", loc="left", fontsize=11.5,
                  fontweight="bold", color=INK, pad=6)

    ax2.fill_between(mid, df.tmin_c, df.tmax_c, color="#F2B38F", alpha=0.35, lw=0, zorder=1)
    ax2.plot(mid, df.tmax_c, color="#C8432B", lw=1.6, marker="o", ms=3.2, label="Tmax", zorder=3)
    ax2.plot(mid, df.tmin_c, color="#3A6FB0", lw=1.6, marker="o", ms=3.2, label="Tmin", zorder=3)
    ax2.axhline(0, color="0.35", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax2.set_ylim(-8, 10)
    ax2.set_yticks([-5, 0, 5])
    ax2.set_ylabel("Temperature (°C)", fontsize=10)
    ax2.legend(loc="lower left", ncol=2, fontsize=9, frameon=False, handlelength=1.6,
               borderaxespad=0.2)

    days = pd.date_range(start.iloc[0].normalize(), start.iloc[-1].normalize() + pd.Timedelta(days=1))
    ax2.set_xticks(days)
    ax2.set_xticklabels([])
    ax2.set_xticks(days[:-1] + pd.Timedelta(hours=12), minor=True)
    ax2.set_xticklabels([str(d.day) for d in days[:-1]], minor=True, fontsize=9)
    ax2.tick_params(axis="x", which="minor", length=0, pad=4)
    ax2.tick_params(axis="x", which="major", length=3)
    ax2.set_xlim(days[0], start.iloc[-1] + pd.Timedelta(days=1))
    ax2.set_xlabel("December 2025 (PST)", fontsize=10)
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", labelsize=9)

    pts = gpd.read_file(POINTS)
    n_in = int(pts.within(gpd.read_file(FORCING_AOI).to_crs(pts.crs).union_all()).sum())
    held = f"all {len(pts)}" if n_in == len(pts) else f"{n_in} of {len(pts)}"
    fig.text(0.01, 0.0,
             f"PRISM 800 m, mean over the {aoi_km2:.0f} km² area holding {held} mapped "
             "initiation points.\nEach bar spans one PRISM day (04:00–04:00 PST).",
             ha="left", va="top", fontsize=8.3, color=MUTED)
    save(fig, out, "storm_forcing_prism", pad=0.06)
    return storm_mm, df.precip_mm[ipk]


# --------------------------------------------------------------- timeline ---
def figure_timeline(out):
    _, acres, km2, ign = fire_facts()
    n_pts = len(gpd.read_file(POINTS))
    s0, s1 = map(pd.Timestamp, EVENTS["storm"])
    v0, v1 = map(pd.Timestamp, EVENTS["survey"])
    storm_at = s0 + (s1 - s0) / 2
    survey_at = v0 + (v1 - v0) / 2
    lost = pd.Timestamp(EVENTS["plant_destroyed"])

    fig = plt.figure(figsize=(13.333, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    x0, x1 = ign - pd.Timedelta(days=40), survey_at + pd.Timedelta(days=55)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    ax.add_patch(FancyArrowPatch((mpl.dates.date2num(ign), 0),
                                 (mpl.dates.date2num(x1 - pd.Timedelta(days=6)), 0),
                                 arrowstyle="-|>,head_length=0.9,head_width=0.55",
                                 mutation_scale=14, color="#C3CAD3", lw=4.5, zorder=1,
                                 shrinkA=0, shrinkB=0, capstyle="round"))

    r = 0.27                                            # marker radius, inches
    for when, col in ((ign, FIRE), (storm_at, STORM), (survey_at, SURVEY)):
        tr = icon_transform(ax, mpl.dates.date2num(when), 0, r)
        ax.add_patch(Circle((0, 0), 1.18, transform=tr, fc="white", ec="none", zorder=3))
        ax.add_patch(Circle((0, 0), 1.0, transform=tr, fc=col, ec="none", zorder=4))
    draw_flame(ax, mpl.dates.date2num(ign), 0, r * 1.25, z=6, colour="white", core=FIRE, halo=None)
    draw_cloud_rain(ax, mpl.dates.date2num(storm_at), 0.012, r, z=6)
    draw_drone(ax, mpl.dates.date2num(survey_at), 0, r, SURVEY, z=6)

    def gap(a, b):
        months = round((b - a).days / 30.44)
        ax.text(a + (b - a) / 2, 0, f"{months} months", ha="center", va="center", fontsize=11,
                style="italic", color=MUTED, zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C3CAD3", lw=1.0))
    gap(ign, storm_at)
    gap(storm_at, survey_at)

    def block(x, y, ha, va, date, title, detail, col):
        lines = [(date, 15, "bold", col), (title, 13, "bold", INK), (detail, 11.5, "normal", MUTED)]
        order = lines if va == "top" else lines[::-1]
        yy = y
        for text, size, weight, colour in order:
            ax.text(x, yy, text, ha=ha, va=va, fontsize=size, fontweight=weight, color=colour)
            dy = size / 72 / 2.9 * 2 * 1.32
            yy = yy - dy if va == "top" else yy + dy

    rx = pd.Timedelta(days=r / 13.333 * (x1 - x0).days)
    block(ign - rx, -0.25, "left", "top", day(ign), "Pioneer Fire ignites",
          f"{acres:,} ac ({km2:.0f} km²) burned", FIRE)
    block(storm_at, 0.25, "center", "bottom", f"{s0.day}–{s1.day} {s1:%b %Y}",
          "Atmospheric river triggers debris flows",
          f"Stehekin wastewater plant destroyed {lost.day} {lost:%b}", STORM)
    block(survey_at + rx, -0.25, "right", "top", f"{v0.day}–{v1.day} {v1:%b %Y}", "UAV survey",
          f"{n_pts} debris-flow initiation points mapped", SURVEY)
    save(fig, out, "timeline", svg=True, pad=0.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="burn,locator,topo,forcing,timeline",
                    help="comma-separated subset of: burn, locator, topo, forcing, timeline")
    ap.add_argument("--output-dir", type=Path, default=P / "clash_ppt/s14_cascade")
    args = ap.parse_args(argv)
    use_arial()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    todo = set(args.only.split(","))
    unknown = todo - {"burn", "locator", "topo", "forcing", "timeline"}
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(sorted(unknown))}")
    if "burn" in todo:
        areas = figure_burn_severity(args.output_dir)
        print("    class areas km²:", {k: round(v, 1) for k, v in areas.items()})
    if "locator" in todo:
        figure_locator(args.output_dir)
    if "topo" in todo:
        figure_locator_topo(args.output_dir)
    if "forcing" in todo:
        storm_mm, peak = figure_forcing(args.output_dir)
        print(f"    storm total {storm_mm:.0f} mm, peak day {peak:.0f} mm")
    if "timeline" in todo:
        figure_timeline(args.output_dir)


if __name__ == "__main__":
    main()
