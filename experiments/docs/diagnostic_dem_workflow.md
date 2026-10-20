# Inspect correction stages before filtering a DoD

The experimental scripts operate separately from `run_event_pipeline.py` and
its Montecito correction settings. Use new output directories. The existing
restore point is recorded in `RESTORE_POINTS.md`.

## Stage diagnostic

First produce an alignment and spatial fit/check mask with
`scripts/compare_dem_corrections.py --fit-nuth-kaab`. Then run:

```bash
python -u scripts/diagnose_dem_stages.py \
  --pre /path/to/pre.tif \
  --aligned-post /path/to/translation_review/post_nuth_kaab.tif \
  --fit-check-mask /path/to/translation_review/stable_fit_validation.tif \
  --assess-uncertainty \
  --output-dir /path/to/new_stage_diagnostics
```

The supplied alignment is the baseline. Additional stages are cumulative:
ICP, first-order deramp, 90-degree directional correction with 1,000 bins,
then aspect correction with 5-degree bins. The last two apply per-bin medians.
Their parameters follow the current Montecito production settings; fits use
only mask code 1 and random seed 42. The checks use code 2. These are diagnostic
refits after the supplied alignment, not exact replays of an earlier run.

To assess only the existing alignment, add `--skip-extra-stages`.

Outputs include each corrected post DEM, each DoD, the correction added at each
step, fit metadata, common-pixel metrics, and `stage_comparison.png`. Output DoDs
retain the pre grid and use post minus pre. Source scripts and input checksums
are saved inside the output folder. Each saved stage is checked against its
in-memory difference and grid; original inputs are never written.

`check_blocks.geojson` contains 250 m blocks with at least 100 check pixels;
load it in QGIS and style by `median_m` or `rmse_m`. `terrain_residuals.csv`
summarizes baseline check residuals by slope and aspect. Neither dataset
establishes that selected terrain is unchanged.

## Provisional uncertainty

With `--assess-uncertainty`, the script models robust spread of baseline DoD
residuals from slope and maximum curvature on the fit pixels, using xDEM's
`infer_heteroscedasticity_from_stable`. Thus sigma describes a DoD error proxy;
no assumption of equal precision between the two input DEMs is needed. xDEM's
model defaults include an NMAD spread statistic, a minimum bin count of 100,
and a seven-spread outlier screen during standardization. This screen applies
to the uncertainty fit, not to the output DoD or the untrimmed check pixels.

`sigma_dod_candidate.tif` and `nominal95_threshold_candidate.tif` are diagnostic
rasters. The latter equals 1.959964 times sigma. It represents a nominal 95%
threshold only under a suitable centered Gaussian error model. The report
gives actual coverage of untrimmed fit/check differences. The comparison with
95% +/- 2 percentage points is a diagnostic flag, not a formal significance
test; spatial dependence and whether terrain is static remain unresolved.
Model interpolation/extrapolation also does not establish local data support.

The script does not automatically select a method, filter the DoD, establish
a vertical datum, verify acquisition dates, or calculate volume uncertainty.
A repeated comparison against these same check blocks is model assessment;
it should not be presented as a new independent final validation.

## Pioneer findings on 2026-09-15

All stages below were scored on the same 346,347 check pixels:

| Cumulative correction | Median (m) | NMAD (m) | RMSE (m) |
| --- | ---: | ---: | ---: |
| Initial vertical shift + NuthKaab | -0.017 | 0.399 | 6.315 |
| Add ICP | -0.587 | 0.959 | 5.996 |
| Add deramp | -1.293 | 1.471 | 5.829 |
| Add directional bias | 0.008 | 1.580 | 6.040 |
| Add aspect bias | 0.004 | 1.580 | 6.038 |

The saved incremental maps show horizontal bands introduced by directional
correction. The baseline has the lowest robust scatter, although its RMSE is
higher than the other stages. It remains a provisional alignment candidate.

The uncertainty model's nominal 95% threshold covered only 77.44% of the
baseline check pixels. Its median threshold there was 0.737 m. These results
do not support presenting the threshold as a calibrated 95% LoD.

The existing 2021 NLCD raster was separately sampled with nearest-neighbour
resampling. Check-pixel RMSE was 9.44 m in evergreen forest, 2.38 m in shrub,
1.32 m in grassland, and 0.98 m in barren land. This is an association using
30 m land-cover classes, not proof of vegetation contamination or stable ground.
It supports testing a separate mask of candidate open ground before using a
final detection limit. Such a test must retain the original G mask and results.

Data outputs are under `Downloads/pioneer/stage_diagnostics_20260915` and
`Downloads/pioneer/uncertainty_diagnostics_20260915`. Stage fitting completed in
the first directory, but its optional uncertainty step failed because xDEM
0.2.2 required a georeferenced Raster input. This was fixed and only the
uncertainty/baseline assessment was rerun successfully in the second directory;
the first report retains its original failure record.

Method references:

- https://xdem.readthedocs.io/en/stable/gen_modules/xdem.spatialstats.infer_heteroscedasticity_from_stable.html
- https://xdem.readthedocs.io/en/stable/static_surfaces.html
- https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description
