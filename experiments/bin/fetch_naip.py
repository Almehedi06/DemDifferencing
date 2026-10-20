"""NAIP aerial imagery (RGB) for one area and chosen years, for looking at the ground.

Lists NAIP tiles from Microsoft Planetary Computer, builds a mosaic of the
requested years, clips it to the AOI and writes one compressed, tiled GeoTIFF
per year with overviews, so QGIS opens it fast. Native CRS of the tiles is kept
(no reprojection); only the 0.3 m years are resampled, to --max-res.

    ~/miniconda3/envs/ml_debris/bin/python experiments/bin/fetch_naip.py \\
        --aoi /mnt/c/Users/amehedi/Downloads/eagle_creek/eaglecreek.shp \\
        --state or --years 2016 2020 2022 \\
        --out /mnt/c/Users/amehedi/Downloads/eagle_creek/exp_naip

Needs the ml_debris env (pystac_client, planetary_computer) for the listing and the
system gdalbuildvrt / gdalwarp / gdaladdo for the mosaic.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import planetary_computer
from pystac_client import Client

PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
GDAL_ENV = {"GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_HTTP_TIMEOUT": "120",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}


def run(cmd):
    env = {k: v for k, v in os.environ.items() if k not in ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA")} | GDAL_ENV
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", type=Path, required=True)
    ap.add_argument("--state", required=True, help="two-letter state code as NAIP records it, e.g. or")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--max-res", type=float, default=0.6, help="finest output pixel, m")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attempts", type=int, default=3, help="tries per year; each re-signs the tile links")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    aoi = gpd.read_file(args.aoi)
    bbox = list(aoi.to_crs(4326).total_bounds)
    client = Client.open(PLANETARY, modifier=planetary_computer.sign_inplace)
    manifest = args.out / "naip_manifest.json"
    record = json.loads(manifest.read_text()) if manifest.exists() else {
        "aoi": str(args.aoi), "state": args.state,
        "source": "NAIP via Microsoft Planetary Computer (collection 'naip')", "years": {}}

    def tiles(year):
        """List (and freshly sign) the year's tiles; signed links expire, so this runs per attempt."""
        return [it for it in client.search(collections=["naip"], bbox=bbox).items()
                if it.properties.get("naip:year") == str(year) and it.properties.get("naip:state") == args.state]

    for year in args.years:
        items = tiles(year)
        if not items:
            raise SystemExit(f"no NAIP tiles for {args.state} {year} over the AOI")
        gsd = float(items[0].properties["gsd"])
        res = max(gsd, args.max_res)
        dates = sorted({it.datetime.date().isoformat() for it in items})
        out = args.out / f"naip_{args.state}_{year}_rgb_{str(res).replace('.', 'p')}m.tif"
        print(f"{year}: {len(items)} tiles, {dates[0]}..{dates[-1]}, gsd {gsd} m -> {res} m", flush=True)
        if out.exists():                        # only complete files carry the final name
            print(f"  exists, kept: {out.name}", flush=True)
        else:
            part = out.with_suffix(".part.tif")
            for attempt in range(1, args.attempts + 1):
                listing = args.out / f"_tiles_{year}.txt"
                vrt = args.out / f"_naip_{year}.vrt"
                if attempt > 1:
                    items = tiles(year)
                listing.write_text("\n".join("/vsicurl/" + it.assets["image"].href for it in items))
                try:
                    run(["gdalbuildvrt", "-q", "-overwrite", "-b", "1", "-b", "2", "-b", "3", "-srcnodata", "0",
                         "-input_file_list", listing, vrt])
                    run(["gdalwarp", "-q", "-overwrite", "-cutline", args.aoi, "-crop_to_cutline", "-tr", res, res,
                         "-r", "average", "-dstnodata", "0", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                         "-co", "TILED=YES", "-co", "COMPRESS=JPEG", "-co", "JPEG_QUALITY=88",
                         "-co", "PHOTOMETRIC=YCBCR", "-co", "BIGTIFF=IF_SAFER", vrt, part])
                    run(["gdaladdo", "-q", "-r", "average", "--config", "COMPRESS_OVERVIEW", "JPEG",
                         "--config", "PHOTOMETRIC_OVERVIEW", "YCBCR", part, "2", "4", "8", "16", "32", "64"])
                    part.rename(out)
                    break
                except subprocess.CalledProcessError:
                    print(f"  attempt {attempt} failed (network); {'retrying' if attempt < args.attempts else 'giving up'}",
                          flush=True)
                    if attempt == args.attempts:
                        raise
                finally:
                    listing.unlink(missing_ok=True); vrt.unlink(missing_ok=True)   # hold expiring signed URLs
            print(f"  wrote {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
        record["years"][str(year)] = {"file": out.name, "tiles": len(items), "dates": dates, "native_gsd_m": gsd,
                                      "output_res_m": res, "crs": items[0].properties.get("proj:epsg"),
                                      "tile_ids": sorted(it.id for it in items)}
        record["updated"] = datetime.now().isoformat(timespec="seconds")
        manifest.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
