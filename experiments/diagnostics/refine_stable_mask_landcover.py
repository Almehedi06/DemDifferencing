"""Intersect an existing candidate-stable mask with selected land-cover classes.

Land cover is sampled with nearest-neighbour resampling onto the mask grid.
The output changes fitting eligibility only. It is not proof of stable ground,
and it neither edits the input mask nor filters a DEM/DoD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))   # pipeline helpers
from compare_dem_corrections import read, write  # noqa: E402


def refine(base, cover, classes, stable_value=1):
    if base.shape != cover.shape:
        raise ValueError('Mask and land cover must have the same grid shape')
    if not classes:
        raise ValueError('At least one land-cover class is required')
    valid_cover = np.isfinite(cover)
    if not np.equal(cover[valid_cover], np.floor(cover[valid_cover])).all():
        raise ValueError('Land-cover values must be integer class codes')
    valid = np.isfinite(base) & valid_cover
    selected = valid & (base == stable_value) & np.isin(cover, classes)
    result = np.full(base.shape, 255, dtype='uint8')
    result[valid] = 0
    result[selected] = 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable-mask', type=Path, required=True)
    parser.add_argument('--land-cover', type=Path, required=True)
    parser.add_argument('--classes', nargs='+', type=int, required=True)
    parser.add_argument('--stable-value', type=int, default=1)
    parser.add_argument('--min-stable-pixels', type=int, default=1000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError('Use a new, nonexistent output directory')
    if args.min_stable_pixels < 1:
        raise ValueError('Minimum stable pixels must be positive')
    if any(c < 0 or c >= 2**24 for c in args.classes):
        raise ValueError('Expected nonnegative class codes below 2**24')
    base, profile = read(args.stable_mask)
    with rasterio.open(args.land_cover) as src:
        if src.crs is None or src.count != 1:
            raise ValueError('Expected a single-band georeferenced land-cover raster')
        source_meta = dict(crs=str(src.crs), resolution=list(src.res), nodata=src.nodata,
                           size_bytes=args.land_cover.stat().st_size,
                           mtime_ns=args.land_cover.stat().st_mtime_ns)
        with WarpedVRT(src, crs=profile['crs'], transform=profile['transform'],
                       width=profile['width'], height=profile['height'],
                       resampling=Resampling.nearest, dtype='float32', nodata=np.nan) as vrt:
            cover = vrt.read(1, masked=True).astype('float32').filled(np.nan)
    result = refine(base, cover, args.classes, args.stable_value)
    count = int((result == 1).sum())
    if count < args.min_stable_pixels:
        raise ValueError(f'Only {count} candidate stable pixels remain')
    out = args.output_dir
    out.mkdir(parents=True)
    clean = profile.copy()
    for key in ['blockxsize', 'blockysize', 'tiled', 'interleave']:
        clean.pop(key, None)
    clean.update(driver='GTiff', dtype='uint8', nodata=255, count=1, compress='deflate')
    output_path = out / 'stable_open_ground.tif'
    with rasterio.open(output_path, 'w', **clean) as dst:
        dst.write(result, 1)
        dst.update_tags(base_mask=str(args.stable_mask.resolve()), classes=json.dumps(sorted(set(args.classes))),
                        stable_value=1, unstable_value=0, status='Candidate stable ground; not independently verified',
                        land_cover=str(args.land_cover.resolve()), resampling='nearest')
    write(out / 'landcover_on_grid.tif', cover, profile, source=str(args.land_cover.resolve()), resampling='nearest',
          status='Source classes sampled to reference grid; no increase in source information resolution')
    with rasterio.open(output_path) as src:
        if src.crs != profile['crs'] or src.transform != profile['transform'] or not np.array_equal(src.read(1), result):
            raise ValueError('Saved mask verification failed')
    valid = np.isfinite(base)
    report = dict(base_mask=str(args.stable_mask.resolve()),
                  base_mask_sha256=hashlib.sha256(args.stable_mask.read_bytes()).hexdigest(),
                  land_cover=str(args.land_cover.resolve()), land_cover_source=source_meta,
                  classes=sorted(set(args.classes)), input_stable_value=args.stable_value,
                  original_stable_pixels=int(((base == args.stable_value) & valid).sum()),
                  retained_stable_pixels=count, base_valid_pixels=int(valid.sum()),
                  retained_fraction_of_original_stable=float(count / ((base == args.stable_value) & valid).sum()),
                  stable_pixels_by_class={str(c): int(((result == 1) & (cover == c)).sum()) for c in sorted(set(args.classes))},
                  limitations=['Land cover does not establish that ground was unchanged.',
                               'Selection does not use elevation or DEM-difference thresholds.',
                               'Output mask refines fitting eligibility; DEM values and extent are untouched.'])
    (out / 'mask_report.json').write_text(json.dumps(report, indent=2))
    shutil.copy2(__file__, out / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, selected, title in [(axes[0], (base == args.stable_value), 'Original candidate stable mask'),
                                (axes[1], (result == 1), 'Retained mapped open-ground candidates')]:
        ax.imshow(np.where(valid, selected, np.nan)[::4, ::4], cmap='Greens', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(out / 'mask_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
