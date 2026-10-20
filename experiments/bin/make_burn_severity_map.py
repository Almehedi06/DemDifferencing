"""Simple map of the BAER soil burn severity for the Eagle Creek Fire.

Burn severity classes in the standard BAER colours on a hillshade of the USGS
10 m DEM, with the BAER burn-area outline, a legend giving each class's share
of the burned area, a scale bar and a north arrow.

    ~/miniconda3/envs/ml_debris/bin/python experiments/bin/make_burn_severity_map.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject

D = Path("/mnt/c/Users/amehedi/Downloads")
BAER = D / "eagle_creek/burn_severity/baer_2017"
PREFIX = "baer_or4563012190420170902_10004905_20160913_20170924"
SBS = BAER / f"{PREFIX}_sbs.tif"
OUTLINE = BAER / f"{PREFIX}_burn_area.shp"
DEM = D / "ml_debris/output/cache/USGS10m_45.48655824400533_-122.27029795775795_45.756698971571005_-121.64191726003173.tif"
OUT = D / "eagle_creek/burn_severity/burn_severity_map.png"

# Standard BAER soil burn severity colours
CLASSES = [(1, "Unburned / very low", "#006400"), (2, "Low", "#7fffd4"),
           (3, "Moderate", "#ffff00"), (4, "High", "#ff0000")]
INK, INK2 = "#0b0b0b", "#52514e"


def main():
    with rasterio.open(SBS) as s:
        sbs = s.read(1)
        grid = dict(dst_transform=s.transform, dst_crs=s.crs)
        shape, t, crs = s.shape, s.transform, s.crs
    ext = (t.c, t.c + t.a * shape[1], t.f + t.e * shape[0], t.f)

    # Hillshade at 10 m over the same extent, then shown under the 30 m classes
    fine_t = rasterio.Affine(10.0, 0, t.c, 0, -10.0, t.f)
    dem = np.full((shape[0] * 3, shape[1] * 3), np.nan, "float32")
    with rasterio.open(DEM) as d:
        reproject(rasterio.band(d, 1), dem, dst_transform=fine_t, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    hs = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)), vert_exag=2, dx=10, dy=10)

    burned = np.isin(sbs, [1, 2, 3, 4])
    cls = np.where(burned, sbs, np.nan).astype("float32")
    share = {c: (sbs == c).sum() / burned.sum() for c, _, _ in CLASSES}
    outline = gpd.read_file(OUTLINE).to_crs(crs)
    km2 = burned.sum() * abs(t.a * t.e) / 1e6

    fig, ax = plt.subplots(figsize=(13, 6.6), constrained_layout=True)
    ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1, zorder=1)
    ax.imshow(cls, extent=ext, cmap=ListedColormap([c for _, _, c in CLASSES]), vmin=0.5, vmax=4.5,
              interpolation="nearest", alpha=0.75, zorder=2)
    outline.boundary.plot(ax=ax, color=INK, linewidth=0.9, zorder=3)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    handles = [Patch(facecolor=col, edgecolor=INK2, linewidth=0.5, label=f"{lab}  ({share[c]:.0%})")
               for c, lab, col in CLASSES]
    handles.append(Patch(facecolor="none", edgecolor=INK, linewidth=0.9, label="BAER burn area"))
    ax.legend(handles=handles, title="Soil burn severity (share of mapped area)", loc="lower right",
              frameon=True, framealpha=0.92, fontsize=10, title_fontsize=10)

    # scale bar (5 km) and north arrow
    x0, y0 = ext[0] + 0.03 * (ext[1] - ext[0]), ext[2] + 0.06 * (ext[3] - ext[2])
    ax.plot([x0, x0 + 5000], [y0, y0], color=INK, lw=3, solid_capstyle="butt", zorder=5)
    ax.text(x0 + 2500, y0 + 0.02 * (ext[3] - ext[2]), "5 km", ha="center", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=5)
    ax.annotate("N", xy=(0.97, 0.93), xytext=(0.97, 0.80), xycoords="axes fraction", ha="center", fontsize=13,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5), zorder=5)

    ax.set_title("Eagle Creek Fire (2 Sep - 30 Nov 2017): BAER soil burn severity", loc="left", fontsize=13, color=INK)
    fig.text(0.0, -0.02, f"BAER, 30 m, imagery 13 Sep 2016 (pre) and 24 Sep 2017 (post, mid-fire: later burning may be "
             f"missing). Mapped area {km2:.0f} km². Hillshade: USGS 10 m DEM.", fontsize=9, color=INK2)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    print({lab: f"{share[c]:.1%}" for c, lab, _ in CLASSES}, f"mapped {km2:.0f} km2")


if __name__ == "__main__":
    main()
