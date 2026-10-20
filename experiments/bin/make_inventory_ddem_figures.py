"""Figures for the Eagle Creek inventory-vs-dDEM comparison (2018-2021).

  map_overview.png   whole fire: dDEM on hillshade, every polygon as a dot coloured by its median dz
  map_zoom.png       the densest 2 x 2 km: dDEM with polygon outlines, and the same polygons by agreement
  depth_scatter.png  mapped depth vs dDEM percentile, erosion and deposition
  by_class.png       the eight classes of the summary table: cells eroded, median dz, sign agreement
  by_class_dg.png    the same eight classes for the change in G
  by_class_dc.png    the same eight classes for the change in plan curvature

Reads Downloads/eagle_creek/inventory_vs_ddem_2018_2021/ (from inventory_dod_stats.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

D = Path("/mnt/c/Users/amehedi/Downloads/eagle_creek")
R = D / "inventory_vs_ddem_2018_2021"
FIG = R / "figures"
RED, BLUE, YELLOW, GRAY = "#e34948", "#2a78d6", "#eda100", "#f0efec"
INK, INK2 = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("loss_gain", ["#8c1d1c", RED, GRAY, BLUE, "#174d8f"])
LIM = 1.5
CLASSES = [("03", "Shallow landslide (evacuated)"), ("04", "Shallow landslide (deposition)"),
           ("09", "DF initiation, in channel"), ("14", "DF initiation, shallow landslide"),
           ("16", "DF transport, erosion"), ("17", "DF transport, deposition"),
           ("21", "DF deposition"), ("22", "DF levee")]
PROCESS = ["landslide", "df_pre_initiation", "df_initiation", "df_transport", "df_deposition"]
LABEL = {"landslide": "Landslide", "df_pre_initiation": "DF pre-initiation", "df_initiation": "DF initiation",
         "df_transport": "DF transport", "df_deposition": "DF deposition"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def read_window(src, bounds, stride):
    w = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
    shape = (max(1, int(w.height // stride)), max(1, int(w.width // stride)))
    a = src.read(1, window=w, out_shape=shape, resampling=Resampling.average, masked=True).astype("float64")
    t = src.window_transform(w)
    ext = (t.c, t.c + t.a * w.width, t.f + t.e * w.height, t.f)
    return np.ma.masked_invalid(a), ext


def hillshade(dem, cell):
    return LightSource(azdeg=315, altdeg=45).hillshade(np.ma.filled(dem, np.nan), vert_exag=1, dx=cell, dy=cell)


def draw_base(ax, dem, dz, ext, alpha=0.85):
    ax.imshow(hillshade(dem, (ext[1] - ext[0]) / dem.shape[1]), extent=ext, cmap="gray", vmin=0, vmax=1.4, zorder=1)
    im = ax.imshow(dz, extent=ext, cmap=DIVERGING, norm=Normalize(-LIM, LIM),
                   alpha=alpha, interpolation="nearest", zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return im


def scale_bar(ax, ext, length, label):
    x0 = ext[0] + 0.04 * (ext[1] - ext[0]); y0 = ext[2] + 0.05 * (ext[3] - ext[2])
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=3, solid_capstyle="butt", zorder=10)
    ax.text(x0 + length / 2, y0 + 0.012 * (ext[3] - ext[2]), label, ha="center", va="bottom", fontsize=9, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5), zorder=10)


CHANGE_LAYERS = {
    # layer: (title, x label, label format, x scale, negative / positive meaning, file)
    "dg": ("change in G", "ΔG (m), symmetric log axis", "{:+.2f}", "symlog",
           ("G decreased", "G increased"), "by_class_dg.png"),
    "dc": ("change in plan curvature", "ΔCurv (negative = more convergent)", "{:+.3f}", "linear",
           ("more convergent", "less convergent"), "by_class_dc.png"),
}


def by_class_change(g, summ, layer):
    """Same layout as by_class.png, for one change layer: share of cells < 0, median, sign of polygon medians."""
    title, xlabel, fmt, scale, (neg, pos), fname = CHANGE_LAYERS[layer]
    codes = [c for c, _ in CLASSES]
    y = np.arange(len(CLASSES))
    names = {c: next(k for k in summ.index if k.startswith(c + "_")) for c in codes}
    ylabels = [f"{c} {lab} ({int(summ.loc[names[c], 'n_polygons'])})" for c, lab in CLASSES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True)
    panels = [(axes[0], f"{layer}_frac_negative", f"Cells with {title} < 0", "share of cells", "linear"),
              (axes[1], f"{layer}_median", f"Median {title}", xlabel, scale)]
    for ax, col, ptitle, plabel, pscale in panels:
        data = [g.loc[g.code == c, col].dropna() for c in codes]
        ax.boxplot(data, vert=False, positions=y, widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#d9e6f5", edgecolor=INK2, lw=0.8), medianprops=dict(color=BLUE, lw=1.5),
                   whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
        rng = np.random.default_rng(0)
        for yi, d in zip(y, data):
            ax.scatter(d, yi + rng.uniform(-0.18, 0.18, len(d)), s=6, color=INK2, alpha=0.35, zorder=3)
        if pscale == "symlog":
            ax.set_xscale("symlog", linthresh=10)
            ticks = [-1e4, -1e2, 0, 1e2, 1e4]
            ax.set_xticks(ticks); ax.set_xticklabels(["−10⁴", "−10²", "0", "10²", "10⁴"])
            ax.minorticks_off()
        ax.set_yticks(y); ax.set_yticklabels(ylabels)
        ax.set_xlabel(plabel); ax.set_title(ptitle, loc="left", fontsize=11); ax.grid(axis="x", color="#e6e5e1", lw=0.6)
        ax.invert_yaxis()
    axes[0].axvline(0.5, color=INK2, lw=0.8, ls=":")
    axes[1].axvline(0, color=INK2, lw=0.8)
    sign = pd.crosstab(g.code, np.where(g[f"{layer}_median"] < 0, neg, pos)).reindex(codes).fillna(0)
    left = np.zeros(len(codes))
    for k, col in [(neg, RED), (pos, BLUE)]:
        v = sign[k].to_numpy() if k in sign else np.zeros(len(codes))
        axes[2].barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.6, label=f"polygon median: {k}")
        left += v
    for yi, c in enumerate(codes):
        n_neg = sign.loc[c].get(neg, 0)
        axes[2].text(left[yi] + 3, yi, f"{n_neg / left[yi]:.0%} {neg}", va="center", fontsize=8.5, color=INK)
    axes[2].set_yticks(y); axes[2].set_yticklabels(ylabels); axes[2].invert_yaxis()
    axes[2].set_xlabel("polygons"); axes[2].set_title(f"Sign of each polygon's median {title}", loc="left", fontsize=11)
    axes[2].set_xlim(0, left.max() * 1.45)
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h2, l2, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.0, -0.11, "Each grey dot is one polygon; the box shows their spread and its line the middle polygon.",
             fontsize=9, color=INK2)
    fig.savefig(FIG / fname, dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    g = gpd.read_file(R / "polygons_ddem_stats_2018_2021.gpkg")
    g["dod_sign"] = np.where(g.net_volume_m3 < 0, "erosion", "deposition")
    g["agree"] = np.select([(g.mapped_sign == "erosion") & (g.dod_sign == "erosion"),
                            (g.mapped_sign == "deposition") & (g.dod_sign == "deposition")],
                           ["erosion, agree", "deposition, agree"], "disagree")
    dz_src, dem_src = rasterio.open(D / "prep/dz_2m.tif"), rasterio.open(D / "prep/dem_pre_2m.tif")

    # ---- overview map
    b = g.total_bounds; m = 800
    bounds = (b[0] - m, b[1] - m, b[2] + m, b[3] + m)
    dz, ext = read_window(dz_src, bounds, 5); dem, _ = read_window(dem_src, bounds, 5)
    fig, ax = plt.subplots(figsize=(13, 6.6), constrained_layout=True)
    im = draw_base(ax, dem, dz, ext, alpha=0.75)
    pts = g.geometry.representative_point()
    order = np.argsort(np.abs(g.dz_median.to_numpy()))        # strongest on top
    ax.scatter(pts.x.to_numpy()[order], pts.y.to_numpy()[order], c=g.dz_median.to_numpy()[order], cmap=DIVERGING,
               norm=Normalize(-LIM, LIM), s=14, edgecolors="white", linewidths=0.4, zorder=5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title("Eagle Creek 2018–2021: dDEM (background) and the 967 mapped polygons (dots, median dz inside each)",
                 loc="left", fontsize=12, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01, extend="both")
    cb.set_label("Elevation change 2021 − 2018 (m); red = loss, blue = gain", color=INK2)
    ax.text(0.01, 0.01, "The red band along the lake is the Columbia River (water level), not ground", transform=ax.transAxes, fontsize=9, color=INK2)
    scale_bar(ax, ext, 5000, "5 km")
    fig.savefig(FIG / "map_overview.png", dpi=170); plt.close(fig)

    # ---- zoom: densest 2 km cell
    gx, gy = np.floor((pts.x - b[0]) / 2000), np.floor((pts.y - b[1]) / 2000)
    cell = pd.Series(list(zip(gx, gy))).value_counts().index[0]
    zb = (b[0] + cell[0] * 2000, b[1] + cell[1] * 2000, b[0] + cell[0] * 2000 + 2000, b[1] + cell[1] * 2000 + 2000)
    zdz, zext = read_window(dz_src, zb, 1); zdem, _ = read_window(dem_src, zb, 1)
    inside = g[g.intersects(gpd.GeoSeries.from_xy([zb[0]], [zb[1]], crs=g.crs).buffer(0).iloc[0].envelope.union(
        gpd.points_from_xy([zb[2]], [zb[3]], crs=g.crs)[0]).envelope)] if False else g.cx[zb[0]:zb[2], zb[1]:zb[3]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), constrained_layout=True)
    im = draw_base(axes[0], zdem, zdz, zext)
    inside.boundary.plot(ax=axes[0], color=INK, linewidth=0.7, zorder=5)
    axes[0].set_title(f"dDEM with the {len(inside)} mapped polygons outlined", loc="left", fontsize=11)
    axes[1].imshow(hillshade(zdem, 2.0), extent=zext, cmap="gray", vmin=0, vmax=1.4, zorder=1)
    colours = {"erosion, agree": RED, "deposition, agree": BLUE, "disagree": YELLOW}
    for k, col in colours.items():
        sel = inside[inside.agree == k]
        if len(sel):
            sel.plot(ax=axes[1], color=col, edgecolor="white", linewidth=0.4, alpha=0.9, zorder=5)
    axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("Same polygons: does the dDEM's net volume agree with the mapped class?", loc="left", fontsize=11)
    counts = inside.agree.value_counts()
    axes[1].legend(handles=[Line2D([], [], marker="s", ls="", ms=9, color=c, label=f"{k} ({counts.get(k, 0)})")
                            for k, c in colours.items()], loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    for ax in axes:
        ax.set_xlim(zext[0], zext[1]); ax.set_ylim(zext[2], zext[3])
    scale_bar(axes[0], zext, 500, "500 m")
    cb = fig.colorbar(im, ax=axes[0], location="bottom", shrink=0.7, pad=0.02, extend="both")
    cb.set_label("Elevation change (m)", color=INK2)
    fig.suptitle(f"Densest 2 × 2 km of the inventory (E {zb[0]:.0f}–{zb[2]:.0f}, N {zb[1]:.0f}–{zb[3]:.0f}, EPSG:6806)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(FIG / "map_zoom.png", dpi=170); plt.close(fig)

    # ---- depth scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, (mapped, dd, col, title) in zip(axes, [
            ("mapped_erosion_depth_m", "dz_p05", RED, "Erosion: mapped minimum depth vs dDEM 5th percentile"),
            ("mapped_deposition_depth_m", "dz_p95", BLUE, "Deposition: mapped maximum depth vs dDEM 95th percentile")]):
        d = g[[mapped, dd]].dropna(); d = d[(d[mapped] < 0) if col == RED else (d[mapped] > 0)]
        if col == BLUE:
            lim = (0, max(d[mapped].max(), d[dd].max()) * 1.05)
        else:
            lim = (np.floor(min(d[mapped].quantile(0.01), d[dd].quantile(0.01))), 0.3)
            beyond = int(((d[mapped] < lim[0]) | (d[dd] < lim[0])).sum())
            if beyond:
                ax.text(0.03, 0.80, f"{beyond} polygons deeper than {lim[0]:.0f} m not shown", transform=ax.transAxes,
                        fontsize=9, color=INK2)
        ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=1); ax.text(lim[1], lim[1], " 1:1", color=INK2, fontsize=9, va="bottom")
        ax.scatter(d[mapped], d[dd], s=14, color=col, alpha=0.55, edgecolors="white", linewidths=0.3, zorder=3)
        r = np.corrcoef(d[mapped], d[dd])[0, 1]
        ax.text(0.03, 0.95, f"n = {len(d)}   r = {r:.2f}\nmedian mapped {d[mapped].median():.2f} m, dDEM {d[dd].median():.2f} m",
                transform=ax.transAxes, va="top", fontsize=10, color=INK)
        ax.set_xlabel("Mapped depth (m)"); ax.set_ylabel("dDEM inside the polygon (m)"); ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.grid(color="#e6e5e1", lw=0.6)
    fig.savefig(FIG / "depth_scatter.png", dpi=170); plt.close(fig)

    # ---- by class: the same eight classes, labels and numbers as the class-summary table
    summ = pd.read_csv(R / "class_summary_2018_2021.csv")
    summ = summ[summ.group_by == "Primary"].set_index("class")
    g["code"] = g["Primary"].str[:2]
    codes = [c for c, _ in CLASSES]
    y = np.arange(len(CLASSES))
    names = {c: next(k for k in summ.index if k.startswith(c + "_")) for c in codes}
    ylabels = [f"{c} {lab} ({int(summ.loc[names[c], 'n_polygons'])})" for c, lab in CLASSES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True)
    for ax, col, table_col, title, xlabel in [
            (axes[0], "frac_cells_eroded", "frac_cells_eroded", "Cells eroded (dz < 0)", "share of cells"),
            (axes[1], "dz_median", "dz_median", "Median elevation change", "m, negative = erosion")]:
        data = [g.loc[g.code == c, col].dropna() for c in codes]
        ax.boxplot(data, vert=False, positions=y, widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#d9e6f5", edgecolor=INK2, lw=0.8), medianprops=dict(color=BLUE, lw=1.5),
                   whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
        rng = np.random.default_rng(0)
        for yi, d in zip(y, data):
            ax.scatter(d, yi + rng.uniform(-0.18, 0.18, len(d)), s=6, color=INK2, alpha=0.35, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(ylabels)
        ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontsize=11); ax.grid(axis="x", color="#e6e5e1", lw=0.6)
        ax.invert_yaxis()
    axes[1].axvline(0, color=INK2, lw=0.8)
    agree = pd.crosstab(g.code, g.agree).reindex(codes).fillna(0)
    left = np.zeros(len(codes))
    for k, col in [("erosion, agree", RED), ("deposition, agree", BLUE), ("disagree", YELLOW)]:
        v = agree.get(k, pd.Series(0, index=codes)).to_numpy()
        axes[2].barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.6, label=k)
        left += v
    for yi, (c, tot) in enumerate(zip(codes, left)):
        axes[2].text(tot + 3, yi, f"{summ.loc[names[c], 'sign_agreement_pct']:.0f}%", va="center", fontsize=9, color=INK)
    axes[2].set_yticks(y); axes[2].set_yticklabels(ylabels); axes[2].invert_yaxis()
    axes[2].set_xlabel("polygons"); axes[2].set_title("Agreement: sign of net volume vs mapped class", loc="left", fontsize=11)
    axes[2].set_xlim(0, left.max() * 1.2)
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h2, l2, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.0, -0.11, "Each grey dot is one polygon; the box shows their spread and its line the middle polygon.",
             fontsize=9, color=INK2)
    fig.savefig(FIG / "by_class.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    for layer in CHANGE_LAYERS:
        by_class_change(g, summ, layer)
    print("wrote", FIG, sorted(p.name for p in FIG.glob("*.png")), "| zoom window", zb)


if __name__ == "__main__":
    main()
