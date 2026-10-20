"""UAV-window layer figure (fig2) from the 10 m Landlab probability grids.

The grids arrive as P_*.asc with no CRS. They are converted to GeoTIFF beside
the .asc, taking CRS and grid from the landlab_data_prep 10 m lidar-footprint
DEM they were run on; an .asc on any other grid is refused rather than guessed.

The figure is written twice with the same style as make_final_figures:
  fig2_uav_layers_2d          every cell > 0.001
  fig2_uav_layers_2d_gt0p01   cells > 0.01 only (display choice, stated on the colour bar)

With --ortho the basemap is the UAV orthomosaic instead of Esri imagery (Esri
fills only what the drone did not cover), written as *_ortho, plus a JPG copy.
The 10 cm mosaic is uncompressed with no overviews, so it is read by nearest
sampling every 5th row/column (GDAL fetches only those strips) and then
alpha-weighted averaged over 4x4 samples: a 2 m basemap, finer than the ~2.6 m
per output pixel at 300 dpi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_final_figures as M  # noqa: E402

D = Path("/mnt/c/Users/amehedi/Downloads")


def asc_to_tif(asc, reference):
    with rasterio.open(reference) as ref:
        crs, transform, shape = ref.crs, ref.transform, ref.shape
    with rasterio.open(asc) as src:
        arr = src.read(1).astype("float32")
        if src.shape != shape or not src.transform.almost_equals(transform):
            raise ValueError(f"{asc.name}: grid {src.shape} {src.transform} does not match {reference}")
    if not (np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1):
        raise ValueError(f"{asc.name}: values outside [0, 1] or non-finite")
    tif = asc.with_suffix(".tif")
    with rasterio.open(tif, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(arr, 1)
    return tif


def read_ortho(path, step=5, block=4):
    """RGB in [0, 1] (NaN outside the mosaic) at step*block times the native pixel size."""
    with rasterio.open(path) as src:
        h = src.height // (step * block) * block
        w = src.width // (step * block) * block
        raw = src.read(window=Window(0, 0, w * step, h * step), out_shape=(4, h, w),
                       resampling=Resampling.nearest)
        transform, crs = src.transform * Affine.scale(step * block), src.crs

    def blocksum(x):
        return x.reshape(h // block, block, w // block, block).sum(axis=(1, 3))

    alpha = raw[3].astype("float32") / 255.0
    weight = blocksum(alpha)
    valid = weight / block ** 2 > 0.5
    rgb = np.stack([blocksum(raw[b].astype("float32") * alpha) for b in range(3)])
    rgb = rgb / np.where(weight > 0, weight, 1) / 255.0
    rgb[:, ~valid] = np.nan
    return rgb, transform, crs


def ortho_basemap(ortho_path, ext, img, img_ext, px=3.0):
    """Mosaic over Esri on a Web Mercator grid covering ext; px in Mercator metres (~2 m ground)."""
    rgb, o_transform, o_crs = read_ortho(ortho_path)
    left, top = ext[0] - 100, ext[3] + 100
    W, H = int(np.ceil((ext[1] + 100 - left) / px)), int(np.ceil((top - ext[2] + 100) / px))
    dst_t = from_origin(left, top, px, px)
    esri_t = from_bounds(img_ext[0], img_ext[2], img_ext[1], img_ext[3], img.shape[1], img.shape[0])
    out = np.zeros((3, H, W), "float32")
    for b in range(3):
        bg = np.zeros((H, W), "float32")
        rasterio.warp.reproject(img[..., b].astype("float32") / 255.0, bg, src_transform=esri_t,
                                src_crs="EPSG:3857", dst_transform=dst_t, dst_crs="EPSG:3857",
                                resampling=Resampling.bilinear)
        fg = np.full((H, W), np.nan, "float32")
        rasterio.warp.reproject(rgb[b], fg, src_transform=o_transform, src_crs=o_crs,
                                src_nodata=np.nan, dst_transform=dst_t, dst_crs="EPSG:3857",
                                dst_nodata=np.nan, resampling=Resampling.bilinear)
        out[b] = np.where(np.isfinite(fg), fg, 0.55 * bg + 0.45)   # Esri surround, faded
    covered = np.isfinite(fg).mean()
    return np.clip(out.transpose(1, 2, 0), 0, 1), (left, left + W * px, top - H * px, top), covered


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=Path, default=D / "pioneer/probability_fields")
    ap.add_argument("--reference", type=Path,
                    default=D / "pioneer/output/lidar_footprint_10m/topographic__elevation.tif")
    ap.add_argument("--points", type=Path,
                    default=D / "pioneer/Initiation_Points_final/Initiation_Points_final.shp")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures_final")
    ap.add_argument("--ortho", type=Path, default=None,
                    help="UAV orthomosaic to use as the basemap (writes *_ortho figures)")
    args = ap.parse_args(argv)

    for key, _ in M.LAYERS:
        print("  converted", asc_to_tif(args.layers / f"{key}.asc", args.reference).name)

    pts_m = gpd.read_file(args.points).to_crs(3857)
    _, ext = M.warp_to_merc(args.layers / "P_Landslide.tif")
    ll = rasterio.warp.transform_bounds("EPSG:3857", "EPSG:4326", ext[0], ext[2], ext[1], ext[3])
    lat = (ll[1] + ll[3]) / 2
    img, img_ext = M.fetch_tiles(ll, 15, M.IMAGERY)
    tag = ""
    if args.ortho is not None:
        img, img_ext, covered = ortho_basemap(args.ortho, ext, img, img_ext)
        print(f"  orthomosaic covers {covered:.0%} of the basemap grid")
        M.ATTRIB = "Imagery: UAV orthomosaic; surroundings Esri, Maxar, Earthstar Geographics"
        tag = "_ortho"

    for floor, stem in ((0.001, "fig2_uav_layers_2d"), (0.01, "fig2_uav_layers_2d_gt0p01")):
        stem = stem.replace("_2d", "_2d" + tag)
        M.panels_2d(args.layers, "", img, img_ext, ext, lat, pts_m, args.output_dir, stem,
                    1000, "1 km", True, floor=floor)
        if tag:
            png = args.output_dir / f"{stem}.png"
            Image.open(png).convert("RGB").save(png.with_suffix(".jpg"), quality=90)


if __name__ == "__main__":
    main()
