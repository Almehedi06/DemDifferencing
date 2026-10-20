"""Landlab MWR modelled elevation change beside the dDEM, UAV window.

Two panels on one diverging scale, -1.5 to +1.5 m (red = loss, blue = deposition):
  a  Landlab MassWastingRunout elevation change, 10 m, on satellite imagery;
     cells with zero change are left transparent
  b  corrected dDEM, 1 m (ddem_correction/m6_gain_mask/dod_dndvi_test.tif: detilted, gains
     zeroed on hillslopes and where NDVI did not change), every surveyed cell drawn,
     on a plain OpenStreetMap basemap outside the surveyed area

MWR.asc declares no CRS. It is refused unless it sits on the landlab_data_prep
10 m lidar-footprint grid, whose CRS it then takes. Its 9999 flag cells (the
flag plus a little deposition, e.g. 9999.137) are masked and counted, never drawn.

Panel b's positive side is only partly corrected: the post-event DEM is SfM, so
standing vegetation can still read as gain. The figure says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from matplotlib.colors import Normalize
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_final_figures as M  # noqa: E402  shared style, tiles and furniture

D = Path("/mnt/c/Users/amehedi/Downloads")
P = D / "pioneer"
LIM = 1.5          # colour limits, metres
FLAG = 1000.0      # any value above this is a flag, not elevation change
STREETS = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def to_merc(arr, transform, crs, ext, px):
    """Nearest-neighbour onto a Web Mercator grid covering ext, so values stay exact."""
    W, H = int(np.ceil((ext[1] - ext[0]) / px)), int(np.ceil((ext[3] - ext[2]) / px))
    dst = np.full((H, W), np.nan, "float32")
    rasterio.warp.reproject(arr.astype("float32"), dst, src_transform=transform, src_crs=crs,
                            src_nodata=np.nan, dst_transform=from_origin(ext[0], ext[3], px, px),
                            dst_crs="EPSG:3857", dst_nodata=np.nan, resampling=Resampling.nearest)
    return dst, (ext[0], ext[0] + W * px, ext[3] - H * px, ext[3])


def read(path):
    with rasterio.open(path) as s:
        return s.read(1, masked=True).astype("float64").filled(np.nan), s.transform, s.crs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mwr", type=Path, default=P / "probability_fields/MWR.asc")
    ap.add_argument("--ddem", type=Path, default=P / "ddem_correction/m6_gain_mask/dod_dndvi_test.tif")
    ap.add_argument("--reference", type=Path,
                    default=P / "output/lidar_footprint_10m/topographic__elevation.tif")
    ap.add_argument("--output-dir", type=Path, default=P / "figures_final")
    args = ap.parse_args(argv)

    with rasterio.open(args.reference) as r:
        grid_crs, grid_t, grid_shape = r.crs, r.transform, r.shape
    with rasterio.open(args.mwr) as s:
        if s.shape != grid_shape or not s.transform.almost_equals(grid_t):
            raise ValueError(f"{args.mwr}: not on the 10 m lidar-footprint grid")
        dz = s.read(1).astype("float64")
    flagged = dz > FLAG
    print(f"  MWR: {int(flagged.sum())} flag cells (> {FLAG:g}) masked; "
          f"range without them {dz[~flagged].min():.2f} to {dz[~flagged].max():.2f} m")
    tif = args.mwr.with_suffix(".tif")
    with rasterio.open(tif, "w", driver="GTiff", height=grid_shape[0], width=grid_shape[1],
                       count=1, dtype="float32", crs=grid_crs, transform=grid_t, nodata=-9999.0,
                       compress="deflate") as dst:
        dst.write(np.where(flagged, -9999.0, dz).astype("float32"), 1)
        dst.update_tags(units="m", source=str(args.mwr),
                        meaning="Landlab MWR modelled elevation change: negative = erosion, "
                                "positive = deposition, 0 = no change modelled",
                        nodata_cells=f"{int(flagged.sum())} cells flagged 9999 in the .asc",
                        crs_note="the .asc declares none; taken from the 10 m lidar-footprint grid")
    print(f"  wrote {tif}")
    dz[flagged | (dz == 0)] = np.nan

    two, t_t, t_crs = read(args.ddem)           # every surveyed cell; outside the survey is NaN
    print(f"  dDEM: {args.ddem.name}, all changes drawn")

    _, ext = M.warp_to_merc(P / "probability_fields/P_Landslide.tif")      # UAV window
    ll = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326", ext[0], ext[2], ext[1], ext[3])
    lat = (ll[1] + ll[3]) / 2
    imagery, imagery_ext = M.fetch_tiles(ll, 15, M.IMAGERY)
    streets, streets_ext = M.fetch_tiles(ll, 15, STREETS)

    panels = [(*to_merc(dz, grid_t, grid_crs, ext, 15.0), "Landlab MWR, modelled",
               imagery, imagery_ext, M.ATTRIB, 0.92),
              (*to_merc(two, t_t, t_crs, ext, 1.5), "dDEM, erosion and deposition",
               streets, streets_ext, "© OpenStreetMap contributors", 1.0)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.2), constrained_layout=True)
    norm = Normalize(-LIM, LIM)
    im = None
    for ax, (arr, e, title, base, base_ext, credit, alpha), tag in zip(axes, panels, "ab"):
        ax.imshow(base, extent=base_ext, zorder=1, interpolation="bilinear")
        im = ax.imshow(arr, extent=e, cmap="RdBu", norm=norm, zorder=2, interpolation="nearest",
                       alpha=alpha)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, pad=10)
        M.panel_tag(ax, tag)
        ax.text(0.99, 0.012, credit, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="white", zorder=13, path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])
    M.scale_bar(axes[0], 1000, "1 km", lat)
    M.north_arrow(axes[1])
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.5, aspect=36, pad=0.015,
                      extend="both", ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    cb.set_label("Elevation change (m):  red = loss, blue = deposition", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.text(0.5, -0.02, "(b) Corrected dDEM: tilt removed, vegetation gains masked with NDVI. "
             "The post-event DEM is from drone photogrammetry,\nso some deposition may still be vegetation.",
             ha="center", va="top", fontsize=12.5, color="0.3")
    M.save(fig, args.output_dir, "mwr_vs_ddem")
    png = args.output_dir / "mwr_vs_ddem.png"
    Image.open(png).convert("RGB").save(png.with_suffix(".jpg"), quality=90)


if __name__ == "__main__":
    main()
