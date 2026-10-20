"""Slide 13 schematic: landlab_data_prep, from public data to a Landlab-ready stack.

Styled after Erkan's ClaSH "Data pipeline illustration" slide (same box colours,
chevrons, raw-input cards and modeling routes). The layer stack is not clip
art: it is the real Pioneer 30 m output of the pipeline (pipeline_v0.3.0).

Writes pipeline_schematic.{png,pdf,svg}. The SVG keeps text as text, so in
PowerPoint "Convert to Shape" makes every label editable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import font_manager
from matplotlib.colors import LightSource, LinearSegmentedColormap, LogNorm, Normalize, to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D

D = Path("/mnt/c/Users/amehedi/Downloads")
REPO_URL = "github.com/gaia-hazlab/landlab_data_prep"
COMMAND = "landlab-prep-pipeline --config fire.yaml"

# (fill, edge) pairs taken from Erkan's slide
BLUE = ("#EAF1FC", "#4A78C8")
YELLOW = ("#FFF5DF", "#E3A33B")
GREEN = ("#E7F5EB", "#3E9B5A")
PURPLE = ("#F1ECFA", "#8A6CC6")
RED = ("#FDECEA", "#D9534F")
EVIDENCE = ("#FFFFFF", "#8C96A3")
INK, MUTED, ARROW = "#1F2A37", "#4F5B6B", "#9DB2D6"

NLCD = {11: "#466B9F", 12: "#D1DEF8", 21: "#DEC5C5", 22: "#D99282", 23: "#EB0000",
        24: "#AB0000", 31: "#B3AC9F", 41: "#68AB5F", 42: "#1C5F2C", 43: "#B5C58F",
        52: "#CCB879", 71: "#DFDFC2", 81: "#DCD939", 82: "#AB6C28", 90: "#B8D9EB",
        95: "#6C9FB8"}
# BAER soil burn severity: unburned/very low, low, moderate, high
SBS = {1: "#1B7837", 2: "#7FFFD4", 3: "#FFE600", 4: "#E31A1C"}
TERRAIN = LinearSegmentedColormap.from_list(
    "terrain_soft", ["#2F6B3A", "#8FB565", "#E0D49A", "#A47B53", "#F4F4F4"])


def use_slide_fonts():
    fonts = Path("/mnt/c/Windows/Fonts")
    for name in ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "consola.ttf"):
        if (fonts / name).exists():
            font_manager.fontManager.addfont(str(fonts / name))
    mpl.rcParams.update({"font.family": "Calibri", "svg.fonttype": "none",
                         "pdf.fonttype": 42})


# --------------------------------------------------------------------------
# Real layers for the stack
# --------------------------------------------------------------------------
def read(stack_dir, name):
    with rasterio.open(stack_dir / f"{name}.tif") as src:
        a = src.read(1).astype(float)
        nd = src.nodata
    if nd is not None:
        a[a == nd] = np.nan
    a[a == -9999] = np.nan
    return a


def rgba_from(values, colors_fn, mask):
    rgba = np.zeros(values.shape + (4,))
    rgba[mask] = colors_fn(values[mask])
    rgba[..., 3] = mask
    return rgba


def classed(table):
    lut = {k: to_rgba(v) for k, v in table.items()}
    return lambda v: np.array([lut.get(int(round(x)), (0.8, 0.8, 0.8, 1)) for x in v])


def build_layers(stack_dir):
    """Top of the stack first. Each entry: (name, source, rgba)."""
    elev = read(stack_dir, "topographic__elevation")
    aoi = np.isfinite(elev)
    rows, cols = np.where(aoi)
    win = np.s_[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    aoi = aoi[win]

    def layer(name):
        a = read(stack_dir, name)[win]
        return a, aoi & np.isfinite(a)

    # Elevation: hillshaded, NaN filled only for the shading kernel
    z = elev[win]
    zf = np.where(aoi, z, np.nanmedian(z))
    lo, hi = np.nanpercentile(z, [2, 98])
    shaded = LightSource(azdeg=315, altdeg=40).shade(
        zf, cmap=TERRAIN, norm=Normalize(lo, hi), blend_mode="soft",
        vert_exag=1.5, dx=30, dy=30)
    elev_rgba = shaded.copy()
    elev_rgba[..., 3] = aoi

    burn, _ = layer("burn__severity")          # no nodata by design; AOI mask applies
    lc, lc_ok = layer("landcover")
    depth, depth_ok = layer("soil__thickness")
    ksat, ksat_ok = layer("soil__saturated_hydraulic_conductivity")
    coh, coh_ok = layer("soil__mode_total_cohesion")

    def cmap_fn(name, norm):
        cmap = mpl.colormaps[name]
        return lambda v: cmap(norm(v))

    return [
        ("Elevation", "USGS 3DEP", elev_rgba),
        ("Burn severity", "USGS BAER", rgba_from(burn, classed(SBS), aoi)),
        ("Land cover", "NLCD 2021", rgba_from(lc, classed(NLCD), lc_ok)),
        ("Soil depth", "SOLUS100",
         rgba_from(depth, cmap_fn("YlOrBr", Normalize(0, np.nanpercentile(depth, 98))), depth_ok)),
        ("Sat. conductivity", "derived from soils",
         rgba_from(ksat, cmap_fn("Blues", Normalize(*np.nanpercentile(ksat, [2, 98]))), ksat_ok)),
        ("Root cohesion", "derived from land cover",
         rgba_from(coh, cmap_fn("YlGn", LogNorm(np.nanmin(coh) / 3, np.nanmax(coh))), coh_ok)),
    ]


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, style, title, lines, title_pt=15, body_pt=11.5, lw=1.6,
        dashed=False, reserve=0.0):
    fill, edge = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fill, ec=edge, lw=lw, ls=(0, (4, 2)) if dashed else "-",
                                zorder=2))
    title_h, body_h = title_pt / 6 * 1.2, len(lines) * body_pt / 6 * 1.3
    block = title_h + 0.6 + body_h
    top = y + reserve + (h - reserve + block) / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_pt,
            weight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, top - title_h - 0.6, "\n".join(lines), ha="center",
            va="top", fontsize=body_pt, color=MUTED, linespacing=1.3, zorder=3)


def chevron(ax, xc, yc, w=2.4, h=6.0, t=1.2):
    pts = [(xc - w / 2, yc + h / 2), (xc - w / 2 + t, yc + h / 2), (xc + w / 2, yc),
           (xc - w / 2 + t, yc - h / 2), (xc - w / 2, yc - h / 2), (xc + w / 2 - t, yc)]
    ax.add_patch(Polygon(pts, closed=True, fc=ARROW, ec="none", zorder=1))


def arrow(ax, start, end, lw=2.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 color=ARROW, lw=lw, zorder=1, shrinkA=0, shrinkB=0))


def draw_stack(ax, layers, x0, y0, w, h, shear, dy, label_x):
    """Oblique sheets, bottom layer first. Negative shear keeps each layer's
    front strip the right-most thing at its height, so leaders never cross."""
    n = len(layers)
    for i, (name, source, rgba) in enumerate(reversed(layers)):
        yb = y0 + i * dy
        corners = [(x0, yb), (x0 + w, yb), (x0 + w - shear, yb + h), (x0 - shear, yb + h)]
        z = 10 + 3 * i
        shadow = [(cx + 0.45, cy - 0.45) for cx, cy in corners]
        ax.add_patch(Polygon(shadow, fc="#000000", alpha=0.10, ec="none", zorder=z))
        ax.add_patch(Polygon(corners, fc="#F7F9FC", ec="#9AA5B1", lw=0.9, zorder=z + 1))
        tr = Affine2D().from_values(w, 0, -shear, h, x0, yb) + ax.transData
        ax.imshow(rgba, extent=(0, 1, 0, 1), transform=tr, interpolation="bilinear",
                  zorder=z + 2)
        ly = yb + dy / 2
        edge_x = x0 + w - shear * (dy / 2) / h
        ax.plot([edge_x + 0.6, label_x - 0.8], [ly, ly], color="#9AA5B1", lw=0.8,
                zorder=40)
        ax.text(label_x, ly + 0.25, name, ha="left", va="bottom", fontsize=11.5,
                weight="bold", color=INK, zorder=40)
        ax.text(label_x, ly - 0.25, source, ha="left", va="top", fontsize=9,
                color=MUTED, zorder=40)
    return y0 + (n - 1) * dy + h


def make_figure(stack_dir, out_dir):
    use_slide_fonts()
    layers = build_layers(stack_dir)
    n_fields = len(list(stack_dir.glob("*.asc")))

    W, H = 160, 90                                   # 16:9, 1 unit = 6 pt
    fig = plt.figure(figsize=(13.333, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # --- top row: the pipeline ------------------------------------------------
    stages = [
        (BLUE, "Event config", ["one YAML file per fire", "AOI polygon · fire ID · dates",
                                "checked before anything runs"]),
        (BLUE, "Retrieve", ["terrain · burn severity · soils", "land cover · vegetation · climate",
                            "downloaded once, then cached"]),
        (YELLOW, "Harmonize", ["one grid in the AOI's UTM zone", "reproject + resample every layer",
                               "nodata made explicit (−9999)"]),
        (YELLOW, "Derive", ["Ksat · transmissivity · porosity", "field capacity · wilting point",
                            "friction angle · root cohesion"]),
        (GREEN, "Landlab-ready stack", ["fields on one RasterModelGrid", "ESRI ASCII + GeoTIFF",
                                        "missing data → closed nodes"]),
    ]
    bw, gap, x, y, bh = 28.6, 3.5, 1.5, 71.0, 16.5
    stage_x = []
    for i, (style, title, lines) in enumerate(stages):
        box(ax, x, y, bw, bh, style, title, lines)
        stage_x.append(x)
        if i < len(stages) - 1:
            chevron(ax, x + bw + gap / 2, y + bh / 2)
        x += bw + gap

    # --- raw inputs -------------------------------------------------------------
    ax.text(2, 66.6, "Raw inputs — public, CONUS-wide", ha="left", va="bottom", fontsize=13,
            weight="bold", color=BLUE[1])
    cards = [
        ("Terrain", ["USGS 3DEP 10 m (OpenTopography)", "or any local DEM (lidar, UAV)"], BLUE, False),
        ("Burn severity", ["USGS BAER soil burn severity", "dNBR (optional)"], BLUE, False),
        ("Soils", ["SOLUS100: depth, texture, density, pH", "STATSGO erodibility (K-factor)"], BLUE, False),
        ("Land cover & vegetation", ["NLCD 2021", "NASA HLS: NDVI · NBR · NDMI"], BLUE, False),
        ("Climate forcing", ["PRISM daily precipitation", "min / max temperature"], BLUE, False),
        ("Evidence, for testing", ["mapped initiation points", "pre/post DEM difference (dDEM)"],
         EVIDENCE, True),
    ]
    cw, cgap, ch = 30.3, 2.5, 12.0
    for k, (title, lines, style, dashed) in enumerate(cards):
        r, c = divmod(k, 3)
        cx, cy = 2 + c * (cw + cgap), 53.6 - r * (ch + 2.6)
        box(ax, cx, cy, cw, ch, ("#FFFFFF", style[1]), title, lines, title_pt=13,
            body_pt=10.5, lw=1.3, dashed=dashed)
    retrieve_cx = stage_x[1] + bw / 2
    arrow(ax, (retrieve_cx, 65.8), (retrieve_cx, 70.7))

    # --- the stack: real Pioneer output -----------------------------------------
    sx0, sw, sh, shear, sdy = 113.0, 28.0, 14.0, 10.0, 7.2
    top = draw_stack(ax, layers, x0=sx0, y0=14.0, w=sw, h=sh, shear=shear, dy=sdy,
                     label_x=143.5)
    arrow(ax, (stage_x[4] + 3.0, 70.7), (sx0 + sw - shear - 3.0, top + 0.5))
    ax.text(sx0 + sw / 2 - shear / 2, 12.3,
            f"Pioneer Fire, 30 m: {n_fields} Landlab fields, one grid",
            ha="center", va="top", fontsize=9.5, style="italic", color=MUTED)

    # --- downstream routes ------------------------------------------------------
    routes = [
        (PURPLE, "Route 1: Physics", ["Landlab process models", "runoff · flow routing",
                                      "landslide · channel initiation"]),
        (PURPLE, "Route 2: Observed ML", ["source detection", "segmentation · susceptibility",
                                          "terrain-change learning"]),
        (PURPLE, "Route 3: Surrogates", ["learn from Landlab runs", "emulate the physics",
                                         "fast scenario rollout"]),
        (RED, "Route 4: Digital twin", ["forcing + observations", "+ simulations + surrogates",
                                        "updates · decision support"]),
    ]
    rw, rgap, ry, rh = 22.1, 2.5, 9.0, 21.0
    bus_y = 34.6
    centres = []
    for k, (style, title, lines) in enumerate(routes):
        rx = 2 + k * (rw + rgap)
        box(ax, rx, ry, rw, rh, style, title, lines, title_pt=12.5, body_pt=10,
            lw=2.6 if k == 0 else 1.5, reserve=4.5)
        centres.append(rx + rw / 2)
        arrow(ax, (rx + rw / 2, bus_y), (rx + rw / 2, ry + rh + 0.2), lw=1.8)
    # Route 1 carries the rest of this talk
    pill_w = 18.0
    ax.add_patch(FancyBboxPatch((centres[0] - pill_w / 2, ry + 1.6), pill_w, 3.2,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                fc=PURPLE[1], ec="none", zorder=4))
    ax.text(centres[0], ry + 3.2, "Stehekin → next slides", ha="center", va="center",
            fontsize=10.5, weight="bold", color="white", zorder=5)
    # bus from the stack to every route
    stack_left_at_bus = sx0 - shear * (bus_y - (14.0 + 2 * sdy)) / sh
    ax.plot([centres[0], stack_left_at_bus - 0.3], [bus_y, bus_y], color=ARROW, lw=2.2,
            solid_capstyle="round", zorder=1)
    ax.text(50, bus_y, "  Downstream modeling routes  ", ha="center", va="center",
            fontsize=13, weight="bold", color=MUTED, zorder=3,
            bbox=dict(fc="white", ec="none", pad=0.5))

    # --- footer -----------------------------------------------------------------
    ax.plot([2, 158], [6.6, 6.6], color="#E3E7ED", lw=1.0)
    ax.text(2, 3.4, "A new fire is a new config file — not new code.", ha="left",
            va="center", fontsize=13, weight="bold", color=INK)
    ax.text(158, 3.4, REPO_URL, ha="right", va="center", fontsize=11, color=BLUE[1],
            weight="bold")
    ax.text(107, 3.4, COMMAND, ha="right", va="center", fontsize=9.5, family="Consolas",
            color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="#F1F3F6", ec="#D5DAE1"))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"pipeline_schematic.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"{n_fields} fields in {stack_dir}\nwrote {out_dir}/pipeline_schematic.{{png,pdf,svg}}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-dir", type=Path,
                    default=D / "ml_debris/pioneer/output/pipeline_v0.3.0",
                    help="landlab_data_prep output folder with the *.tif fields")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/clash_ppt/s13_pipeline")
    args = ap.parse_args(argv)
    make_figure(args.stack_dir, args.output_dir)


if __name__ == "__main__":
    main()
