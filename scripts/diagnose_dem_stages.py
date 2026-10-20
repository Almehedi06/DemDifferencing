"""Inspect extra corrections and provisional uncertainty after DEM alignment.

Use an existing fit/check mask (1=fit, 2=check). Additional stages are fitted
sequentially using fit pixels only. This does not establish that the selected
terrain is unchanged, modify the event pipeline, or promote a final product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_geom

from compare_dem_corrections import read, stats, write


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pd.DataFrame):
        return plain(value.to_dict(orient='records'))
    if callable(value):
        return getattr(value, '__name__', type(value).__name__)
    return str(value)


def array(dem):
    return np.ma.asarray(dem.data).astype('float32').filled(np.nan)


def masks(split, valid):
    codes = split[np.isfinite(split)]
    if not np.isin(codes, [0, 1, 2]).all():
        raise ValueError('Expected fit/check mask codes 0=other, 1=fit, 2=check')
    fit, check = (split == 1) & valid, (split == 2) & valid
    if fit.sum() < 100 or check.sum() < 100:
        raise ValueError('Need at least 100 valid fit pixels and 100 check pixels')
    return fit, check


def check_blocks(dod, check, profile, block_m):
    pixels = max(1, round(block_m / profile['transform'].a))
    rows, features = [], []
    transform = profile['transform']
    for r in range(0, dod.shape[0], pixels):
        for c in range(0, dod.shape[1], pixels):
            r1, c1 = min(r + pixels, dod.shape[0]), min(c + pixels, dod.shape[1])
            tile = dod[r:r1, c:c1]
            valid = check[r:r1, c:c1] & np.isfinite(tile)
            if valid.sum() < 100:
                continue
            record = dict(block_row=r // pixels, block_col=c // pixels, **stats(tile[valid]))
            rows.append(record)
            corners = [list(transform * xy) for xy in [(c, r), (c1, r), (c1, r1), (c, r1), (c, r)]]
            geom = transform_geom(profile['crs'], 'EPSG:4326', dict(type='Polygon', coordinates=[corners]))
            features.append(dict(type='Feature', geometry=geom, properties=record))
    return rows, dict(type='FeatureCollection', features=features)


def coverage(dod, sigma, selected, multiplier=1.959963984540054):
    valid = selected & np.isfinite(dod) & np.isfinite(sigma) & (sigma > 0)
    if valid.sum() < 100:
        raise ValueError('Insufficient pixels for uncertainty check')
    z = dod[valid] / sigma[valid]
    return dict(n=int(valid.sum()), standardized_median=float(np.median(z)),
                standardized_nmad=float(1.4826 * np.median(np.abs(z - np.median(z)))),
                within_nominal_95_fraction=float(np.mean(np.abs(z) <= multiplier)),
                sigma_median_m=float(np.median(sigma[valid])),
                threshold_median_m=float(np.median(multiplier * sigma[valid])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre', type=Path, required=True)
    parser.add_argument('--aligned-post', type=Path, required=True)
    parser.add_argument('--fit-check-mask', type=Path, required=True)
    parser.add_argument('--block-size-m', type=float, default=250)
    parser.add_argument('--assess-uncertainty', action='store_true')
    parser.add_argument('--skip-extra-stages', action='store_true',
                        help='Only inspect the supplied alignment and optionally its uncertainty')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.block_size_m) or args.block_size_m <= 0:
        raise ValueError('Block size must be positive and finite')
    if args.output_dir.exists():
        raise ValueError('Use a new, nonexistent output directory')
    pre, profile = read(args.pre)
    post, _ = read(args.aligned_post, profile)
    split, _ = read(args.fit_check_mask, profile)
    transform = profile['transform']
    if (not profile['crs'].is_projected or profile['crs'].linear_units_factor[1] != 1
            or transform.b or transform.d or transform.a <= 0 or not np.isclose(transform.a, -transform.e)):
        raise ValueError('Expected square, unrotated pixels in a projected metre CRS')
    fit, check = masks(split, np.isfinite(pre) & np.isfinite(post))

    import xdem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = args.output_dir
    out.mkdir(parents=True)
    (out / 'stages').mkdir()
    (out / 'source').mkdir()
    for name in ['diagnose_dem_stages.py', 'compare_dem_corrections.py']:
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    manifest = dict(status='running', xdem_version=xdem.__version__, random_state=42,
                    inputs={key: dict(path=str(getattr(args, key).resolve()), sha256=fingerprint(getattr(args, key)))
                            for key in ['pre', 'aligned_post', 'fit_check_mask']},
                    note='Sequential diagnostic refits after supplied alignment. Check blocks are model-comparison data, not a new independent final test.',
                    limitations=['Candidate stable mask has not been independently verified.',
                                 'Survey dates and vertical references remain unverified.',
                                 'No automatic method selection or LoD-filtered final product.'], stages=[])

    def save_manifest():
        (out / 'report.json').write_text(json.dumps(plain(manifest), indent=2, allow_nan=False))

    def emit(message):
        print(message, flush=True)
        with (out / 'run.log').open('a') as log:
            log.write(message + '\n')

    save_manifest()
    ref = xdem.DEM(args.pre)
    current = xdem.DEM(args.aligned_post)
    emit('Computing pre DEM slope, aspect and maximum curvature...')
    slope = array(ref.slope())
    aspect = array(ref.aspect())
    curvature = array(ref.max_curvature())
    stages = {'00_nuth_kaab': post - pre}
    deltas = {'00_nuth_kaab': np.zeros_like(pre)}
    recipes = [
        ('01_plus_icp', xdem.coreg.ICP()),
        ('02_plus_deramp', xdem.coreg.Deramp(poly_order=1)),
        ('03_plus_directional', xdem.coreg.DirectionalBias(angle=90, fit_or_bin='bin', bin_sizes=1000,
                                                         bin_apply_method='per_bin', bin_statistic=np.nanmedian)),
        ('04_plus_aspect', xdem.coreg.TerrainBias(terrain_attribute='aspect', fit_or_bin='bin',
                                               bin_sizes={'aspect': np.arange(0, 365, 5)},
                                               bin_apply_method='per_bin', bin_statistic=np.nanmedian)),
    ]
    if args.skip_extra_stages:
        recipes = []
    try:
        for name, model in recipes:
            emit(f'Fitting {name} on fit blocks only...')
            before = array(current)
            selected = fit & np.isfinite(before) & np.isfinite(slope)
            bias_vars = {'aspect': aspect} if name == '04_plus_aspect' else None
            model.fit(ref, current, inlier_mask=selected, random_state=42, bias_vars=bias_vars)
            current = model.apply(current, bias_vars=bias_vars)
            if current.transform != ref.transform or current.shape != ref.shape or current.crs != ref.crs:
                current = current.reproject(ref)
            after = array(current)
            stages[name] = after - pre
            deltas[name] = after - before
            write(out / 'stages' / f'{name}_post.tif', after, profile, status='Diagnostic candidate')
            write(out / 'stages' / f'{name}_dod.tif', stages[name], profile, operation='post minus pre', status='Diagnostic candidate')
            write(out / 'stages' / f'{name}_added_correction.tif', deltas[name], profile,
                  operation='this stage post minus previous stage post', status='Diagnostic candidate')
            saved, _ = read(out / 'stages' / f'{name}_dod.tif', profile)
            if not np.array_equal(np.isfinite(saved), np.isfinite(after - pre)) or not np.allclose(
                    saved[np.isfinite(saved)], (after - pre)[np.isfinite(saved)], atol=.0003):
                raise ValueError(f'Saved raster verification failed: {name}')
            manifest['stages'].append(dict(name=name, model=plain(model.meta)))
            v = check & np.isfinite(stages[name])
            summary = stats(stages[name][v])
            emit(f"{name}: check NMAD {summary['nmad_m']:.3f} m, median {summary['median_m']:.3f} m")
            save_manifest()

        common = np.logical_and.reduce([np.isfinite(a) for a in stages.values()])
        rows = []
        for name, dod in stages.items():
            for subset, selected in [('fit', fit), ('check', check)]:
                rows.append(dict(stage=name, subset=subset, **stats(dod[selected & common])))
        pd.DataFrame(rows).to_csv(out / 'stage_metrics.csv', index=False)
        manifest['stage_metrics'] = rows
        manifest['common_valid_pixels'] = int(common.sum())

        base = stages['00_nuth_kaab']
        block_rows, features = check_blocks(base, check, profile, args.block_size_m)
        pd.DataFrame(block_rows).to_csv(out / 'check_blocks.csv', index=False)
        (out / 'check_blocks.geojson').write_text(json.dumps(features, allow_nan=False))
        bins = []
        for attribute, var, edges in [('slope_deg', slope, [0, 5, 10, 20, 30, 40, 50, 60, 90]),
                                      ('aspect_deg', aspect, list(range(0, 361, 45)))]:
            for low, high in zip(edges[:-1], edges[1:]):
                selected = check & np.isfinite(base) & np.isfinite(var) & (var >= low) & (var < high)
                if selected.sum() >= 100:
                    bins.append(dict(attribute=attribute, lower=low, upper=high, **stats(base[selected])))
        pd.DataFrame(bins).to_csv(out / 'terrain_residuals.csv', index=False)
        fig, axes = plt.subplots(2, len(stages), figsize=(max(6, 4.4 * len(stages)), 9), squeeze=False)
        for col, (name, dod) in enumerate(stages.items()):
            im = axes[0, col].imshow(dod[::5, ::5], cmap='RdBu', vmin=-3, vmax=3)
            axes[0, col].set_title(name.replace('_', ' '))
            axes[1, col].imshow(deltas[name][::5, ::5], cmap='RdBu', vmin=-3, vmax=3)
            axes[1, col].set_title('Correction added at this step' if col else 'Baseline')
            for ax in axes[:, col]:
                ax.set_axis_off()
        fig.colorbar(im, ax=list(axes.ravel()), label='Metres', shrink=.7)
        fig.savefig(out / 'stage_comparison.png', dpi=130, bbox_inches='tight')
        plt.close(fig)

        if args.assess_uncertainty:
            emit('Assessing provisional DoD uncertainty for NuthKaab baseline...')
            terrain_valid = np.isfinite(slope) & np.isfinite(curvature) & np.isfinite(base)
            sigma, table, _ = xdem.spatialstats.infer_heteroscedasticity_from_stable(
                ref.copy(new_array=np.ma.masked_invalid(base)), [slope, curvature], stable_mask=fit & terrain_valid,
                list_var_names=['slope', 'max_curvature'], min_count=100, fac_spread_outliers=7)
            sigma = array(sigma)
            sigma[~terrain_valid | ~np.isfinite(sigma) | (sigma <= 0)] = np.nan
            report = {name: coverage(base, sigma, selected) for name, selected in [('fit', fit), ('check', check)]}
            report['method'] = 'xDEM robust NMAD model of DoD residuals versus pre slope and maximum curvature; fit blocks only'
            report['threshold_note'] = '1.959964 times model sigma is nominal 95% only under a suitable centered Gaussian error model; check observed coverage.'
            report['scope'] = 'Pointwise DoD error proxy, not individual-DEM accuracy or volume uncertainty.'
            report['check_coverage_within_2_percentage_points_of_95'] = abs(report['check']['within_nominal_95_fraction'] - .95) <= .02
            report['no_final_lod'] = 'Stable-terrain assumption remains unverified; no final LoD-filtered product is generated.'
            write(out / 'sigma_dod_candidate.tif', sigma, profile, status='Provisional error proxy', **report)
            write(out / 'nominal95_threshold_candidate.tif', 1.959963984540054 * sigma, profile,
                  status='Nominal threshold; see empirical coverage in report.json', **report)
            table.to_csv(out / 'uncertainty_bins.csv', index=False)
            manifest['uncertainty'] = report
            emit(f"Nominal 95% interval contains {100 * report['check']['within_nominal_95_fraction']:.1f}% of check pixels")
        manifest['status'] = 'complete'
        save_manifest()
        emit(f'Diagnostics saved: {out}')
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        save_manifest()
        raise


if __name__ == '__main__':
    main()
