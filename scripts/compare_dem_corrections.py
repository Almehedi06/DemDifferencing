"""Compare vertical-shift and optional NuthKaab candidates with existing products.

Fits the shift on four fifths of spatial blocks of the supplied stable mask.
The remaining blocks check the new correction candidates. Existing correction
products may have used those blocks during fitting: their scores are diagnostics,
not an independent validation of those models. Inputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

NODATA = -9999.0


def read(path, reference=None):
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        if src.crs is None:
            raise ValueError(f"Missing CRS: {path}")
        if reference is not None and (src.crs != reference['crs'] or src.transform != reference['transform']
                                     or src.width != reference['width'] or src.height != reference['height']):
            raise ValueError(f"Grid mismatch: {path}")
        a = src.read(1, masked=True).astype('float32').filled(np.nan)
    return a, profile


def spatial_split(stable, resolution, block_size):
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError('Block size must be finite and positive')
    pixels = max(1, int(round(block_size / resolution)))
    rows = np.arange(stable.shape[0])[:, None] // pixels
    cols = np.arange(stable.shape[1])[None, :] // pixels
    held = (rows + 2 * cols) % 5 == 0
    return stable & ~held, stable & held


def stats(values):
    values = values[np.isfinite(values)].astype('float64')
    if values.size == 0:
        raise ValueError('No valid pixels for statistics')
    median = np.median(values)
    return dict(n=int(values.size), median_m=float(median),
                nmad_m=float(1.4826 * np.median(np.abs(values - median))),
                rmse_m=float(np.sqrt(np.mean(values ** 2))),
                p05_m=float(np.percentile(values, 5)), p95_m=float(np.percentile(values, 95)))


def write(path, array, profile, **tags):
    p = profile.copy()
    for key in ('blockxsize', 'blockysize', 'tiled', 'interleave'):
        p.pop(key, None)
    p.update(driver='GTiff', dtype='float32', count=1, nodata=NODATA, compress='deflate')
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(np.where(np.isfinite(array), array, NODATA).astype('float32'), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre', type=Path, required=True)
    parser.add_argument('--post', type=Path, required=True)
    parser.add_argument('--stable-mask', type=Path, required=True)
    parser.add_argument('--stable-value', type=int, default=1)
    parser.add_argument('--coreg-post', type=Path)
    parser.add_argument('--corrected-post', type=Path)
    parser.add_argument('--block-size-m', type=float, default=250)
    parser.add_argument('--fit-nuth-kaab', action='store_true',
                        help='Also fit horizontal/vertical translations on fit blocks, without ICP or bias surfaces')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}')
    pre, profile = read(args.pre)
    post, _ = read(args.post, profile)
    if not profile['crs'].is_projected or profile['crs'].linear_units_factor[1] != 1:
        raise ValueError('Spatial validation blocks require a projected metre CRS')
    transform = profile['transform']
    if transform.b or transform.d or transform.a <= 0 or not np.isclose(transform.a, -transform.e):
        raise ValueError('Expected an unrotated square grid')
    mask, _ = read(args.stable_mask, profile)
    raw = post - pre
    arrays = {'raw': raw}
    for name, path in [('existing_coreg', args.coreg_post), ('existing_full', args.corrected_post)]:
        if path:
            other, _ = read(path, profile)
            arrays[name] = other - pre
    common = np.isfinite(raw)
    for a in arrays.values():
        common &= np.isfinite(a)
    stable = (mask == args.stable_value) & common
    fit, validation = spatial_split(stable, transform.a, args.block_size_m)
    if fit.sum() < 100 or validation.sum() < 100:
        raise ValueError('Need at least 100 stable pixels in both fit and validation blocks')
    offset = -float(np.median(raw[fit]))
    arrays['vertical_only'] = raw + offset
    translated_post = None
    nuth_parameters = None
    if args.fit_nuth_kaab:
        import xdem
        ref_dem = xdem.DEM(args.pre)
        target_dem = xdem.DEM(args.post) + offset
        nuth = xdem.coreg.NuthKaab(subsample=100000)
        print('Fitting NuthKaab on stable fit blocks after initial vertical shift...', flush=True)
        nuth.fit(ref_dem, target_dem, inlier_mask=fit, random_state=42)
        translated = nuth.apply(target_dem)
        if translated.transform != ref_dem.transform or translated.shape != ref_dem.shape:
            translated = translated.reproject(ref_dem)
        translated_post = translated.data.astype('float32').filled(np.nan)
        arrays['nuth_kaab'] = translated_post - pre
        nuth_parameters = dict(initial_vertical_shift_m=offset, subsample=100000, random_state=42,
                               matrix_after_initial_vertical_shift=np.asarray(nuth.to_matrix()).tolist())
        # Score all candidates on exactly the same valid fit/check pixels.
        common &= np.isfinite(translated_post)
        fit &= common
        validation &= common
        stable &= common
    coreg_offset = None
    if 'existing_coreg' in arrays:
        coreg_offset = -float(np.median(arrays['existing_coreg'][fit]))
        arrays['existing_coreg_recentered'] = arrays['existing_coreg'] + coreg_offset
    rows = []
    for name, a in arrays.items():
        for subset, selected in [('fit', fit), ('validation_blocks', validation), ('all_stable', stable)]:
            row = {'candidate': name, 'subset': subset, **stats(a[selected])}
            rows.append(row)
            if subset == 'validation_blocks':
                print(f"{name}: median={row['median_m']:.3f} m, NMAD={row['nmad_m']:.3f} m, RMSE={row['rmse_m']:.3f} m", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    tags = dict(method='Constant vertical shift from stable fit blocks', vertical_shift_m=offset,
                vertical_datum_conversion='None; empirical correction only',
                status='Diagnostic candidate; G mask is not independently verified stable terrain')
    write(out/'post_vertical_shift.tif', post + offset, profile, **tags)
    write(out/'dod_vertical_shift.tif', arrays['vertical_only'], profile, operation='corrected post minus pre', **tags)
    if translated_post is not None:
        write(out/'post_nuth_kaab.tif', translated_post, profile, method='Initial median vertical shift + NuthKaab',
              parameters=json.dumps(nuth_parameters), status='Diagnostic candidate')
        write(out/'dod_nuth_kaab.tif', arrays['nuth_kaab'], profile, operation='corrected post minus pre',
              method='Initial median vertical shift + NuthKaab', status='Diagnostic candidate')
    split = np.where(common, 0., np.nan).astype('float32')
    split[fit], split[validation] = 1, 2
    write(out/'stable_fit_validation.tif', split, profile, fit_value=1, validation_value=2, other_valid_value=0)
    with (out/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = dict(inputs={key: str(getattr(args, key).resolve()) if getattr(args, key) else None
                            for key in ['pre', 'post', 'stable_mask', 'coreg_post', 'corrected_post']},
                    vertical_shift_m=offset, coreg_recenter_shift_m=coreg_offset,
                    nuth_kaab=nuth_parameters,
                    validation_block_size_m=args.block_size_m,
                    validation_note='New vertical shift and optional NuthKaab fits exclude validation blocks; existing corrections were fitted before this split and are not independently validated here.',
                    datum_note='This script does not verify survey dates or vertical datums; the fitted shift is empirical.',
                    common_valid_pixels=int(common.sum()), stable_pixels=int(stable.sum()), results=rows)
    (out/'comparison.json').write_text(json.dumps(manifest, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ['vertical_only', 'nuth_kaab', 'existing_coreg', 'existing_full']
    order = [name for name in order if name in arrays]
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 6), squeeze=False)
    for ax, name in zip(axes[0], order):
        im = ax.imshow(arrays[name][::5, ::5], cmap='RdBu', vmin=-3, vmax=3)
        ax.set_title(name.replace('_', ' ')); ax.set_axis_off()
    fig.colorbar(im, ax=list(axes[0]), label='Post minus pre (m)', shrink=.7)
    fig.savefig(out/'comparison.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    # Verify the saved candidate against the source grid and difference equation.
    saved_post, _ = read(out/'post_vertical_shift.tif', profile)
    saved_dod, _ = read(out/'dod_vertical_shift.tif', profile)
    valid = np.isfinite(pre) & np.isfinite(post)
    if not np.allclose(saved_post[valid] - pre[valid], saved_dod[valid], atol=0.0003):
        raise ValueError('Saved difference does not match saved corrected post minus pre')
    if translated_post is not None:
        saved_post, _ = read(out/'post_nuth_kaab.tif', profile)
        saved_dod, _ = read(out/'dod_nuth_kaab.tif', profile)
        valid = np.isfinite(pre) & np.isfinite(saved_post)
        if not np.array_equal(np.isfinite(saved_dod), valid) or not np.allclose(
                saved_post[valid] - pre[valid], saved_dod[valid], atol=0.0003):
            raise ValueError('Saved NuthKaab difference failed verification')
    print(f'Applied vertical offset: {offset:+.6f} m; saved diagnostic products to {out}', flush=True)


if __name__ == '__main__':
    main()
