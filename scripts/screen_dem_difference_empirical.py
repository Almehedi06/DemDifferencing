"""Create a provisional open-ground DoD screen from an empirical residual limit.

The cutoff is a quantile of absolute residuals on fit pixels (mask code 1).
Coverage is reported on check pixels (code 2). No Gaussian model is assumed.
Candidate stability, model generalization and acquisition dates remain external
assumptions: this is not a validated ground-change or confidence-limit product.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from compare_dem_corrections import read, stats, write
from diagnose_dem_stages import fingerprint, masks


def empirical_limit(dod, fit, check, quantile=.95, min_pixels=100):
    if not 0 < quantile < 1:
        raise ValueError('Quantile must be between zero and one')
    a = dod[fit & np.isfinite(dod)].astype('float64')
    b = dod[check & np.isfinite(dod)].astype('float64')
    if a.size < min_pixels or b.size < min_pixels:
        raise ValueError('Insufficient fit or check pixels')
    threshold = float(np.quantile(np.abs(a), quantile, method='linear'))
    if threshold <= 0:
        raise ValueError('A positive residual threshold is required')
    return threshold, dict(quantile=quantile, threshold_m=threshold,
                           fit={**stats(a), 'within_threshold_fraction': float(np.mean(np.abs(a) <= threshold))},
                           check={**stats(b), 'within_threshold_fraction': float(np.mean(np.abs(b) <= threshold))})


def screen(dod, eligible, threshold):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be positive and finite')
    valid = eligible & np.isfinite(dod)
    output = np.full(dod.shape, np.nan, dtype='float32')
    classes = np.full(dod.shape, np.nan, dtype='float32')
    output[valid] = np.where(np.abs(dod[valid]) > threshold, dod[valid], 0)
    classes[valid] = np.sign(output[valid])
    return output, classes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dod', type=Path, required=True)
    parser.add_argument('--fit-check-mask', type=Path, required=True)
    parser.add_argument('--land-cover-on-grid', type=Path, required=True)
    parser.add_argument('--classes', type=int, nargs='+', required=True)
    parser.add_argument('--quantile', type=float, default=.95)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError('Use a new, nonexistent output directory')
    dod, profile = read(args.dod)
    split, _ = read(args.fit_check_mask, profile)
    cover, _ = read(args.land_cover_on_grid, profile)
    eligible = np.isfinite(dod) & np.isfinite(cover) & np.isin(cover, args.classes)
    fit, check = masks(split, eligible)
    threshold, report = empirical_limit(dod, fit, check, args.quantile)
    output, change_class = screen(dod, eligible, threshold)
    out = args.output_dir
    out.mkdir(parents=True)
    common_tags = dict(status='Exploratory screen; unverified stability and vertical reference',
                       threshold_m=threshold, retained_land_cover_classes=json.dumps(sorted(set(args.classes))),
                       interpretation='Retained values are observed differences, not verified erosion/deposition')
    write(out / 'dod_open_ground_screened.tif', output, profile, **common_tags,
          zero_meaning='At or below empirical threshold; not proof of zero change',
          nodata_meaning='Outside selected land cover or missing input data')
    write(out / 'open_ground_change_class.tif', change_class, profile, **common_tags,
          negative_value=-1, below_threshold_value=0, positive_value=1)
    write(out / 'open_ground_analysis_mask.tif', np.where(np.isfinite(dod), eligible.astype(float), np.nan),
          profile, eligible_value=1, excluded_value=0, **common_tags)
    saved, _ = read(out / 'dod_open_ground_screened.tif', profile)
    if not np.array_equal(np.isfinite(saved), eligible):
        raise ValueError('Saved analysis extent differs from expected eligibility')
    retained = eligible & (np.abs(dod) > threshold)
    if not np.array_equal(saved[retained], dod[retained]) or not np.all(saved[eligible & ~retained] == 0):
        raise ValueError('Saved screen failed difference-preservation verification')
    report.update(inputs={key: dict(path=str(getattr(args, key).resolve()), sha256=fingerprint(getattr(args, key)))
                          for key in ['dod', 'fit_check_mask', 'land_cover_on_grid']},
                  classes=sorted(set(args.classes)), status='exploratory',
                  eligible_pixels=int(eligible.sum()), positive_pixels=int((change_class == 1).sum()),
                  negative_pixels=int((change_class == -1).sum()), subthreshold_pixels=int((change_class == 0).sum()),
                  limitations=['Fit/check mask is candidate stability, not independently verified unchanged ground.',
                               'Threshold is empirical from model-fit pixels; check blocks have been reused for model comparisons.',
                               'Observed check coverage does not establish a general 95% confidence guarantee.',
                               'Land-cover selection changes the analysis extent; it does not reconstruct bare earth beneath vegetation.',
                               'No survey-date or vertical-reference verification; no volume uncertainty calculation.',
                               'Real changes smaller than the cutoff are removed from this screened output.'])
    (out / 'screening_report.json').write_text(json.dumps(report, indent=2))
    shutil.copy2(__file__, out / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [('Full corrected DoD', dod), ('Mapped open ground', np.where(eligible, dod, np.nan)),
              (f'Exploratory screen: |DoD| > {threshold:.2f} m', output)]
    for ax, (title, a) in zip(axes, panels):
        im = ax.imshow(a[::4, ::4], cmap='RdBu', vmin=-5, vmax=5)
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(im, ax=list(axes), label='Post minus pre (m)', shrink=.7)
    fig.savefig(out / 'screening_comparison.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Empirical cutoff: {threshold:.6f} m; check coverage: {100 * report["check"]["within_threshold_fraction"]:.2f}%', flush=True)
    print(f'Exploratory screening outputs: {out}', flush=True)


if __name__ == '__main__':
    main()
