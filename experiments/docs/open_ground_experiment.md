# Pioneer open-ground mask experiment

This experiment preserves the original pipeline and `G_pre < 5` mask. It
creates a separate fitting mask from `G_pre < 5` intersected with mapped barren
land and grassland (NLCD classes 31 and 71). No elevation, slope, or raw-DoD
threshold is added. Land cover is sampled with nearest-neighbour resampling;
the original 30 m classification does not gain information at the 1 m grid.
Neither class guarantees unchanged ground: barren land includes active surfaces,
and grassland still contains vegetation and may change.

## Run the experiment

Activate `xdem_env` and run from the DemDifferencing repository. Choose a new
experiment directory for a repeat run:

```bash
conda activate xdem_env
cd /home/abdullah/DemDifferencing
PIONEER_DATA=/mnt/c/Users/amehedi/Downloads/pioneer
PIONEER_EXPERIMENT="$PIONEER_DATA/open_ground_test_run2"

python -u experiments/diagnostics/refine_stable_mask_landcover.py \
  --stable-mask "$PIONEER_DATA/terrain_stable_run/stable_cells/masks/pioneer_stable_cells.tif" \
  --land-cover "$PIONEER_DATA/unzipped_nlcd_2021/Annual_NLCD_LndCov_2021_CU_C1V1.tif" \
  --classes 31 71 \
  --output-dir "$PIONEER_EXPERIMENT/mask"

python -u scripts/compare_dem_corrections.py \
  --pre "$PIONEER_DATA/pre.tif" \
  --post "$PIONEER_DATA/post.tif" \
  --stable-mask "$PIONEER_EXPERIMENT/mask/stable_open_ground.tif" \
  --coreg-post "$PIONEER_DATA/translation_review/post_nuth_kaab.tif" \
  --fit-nuth-kaab \
  --output-dir "$PIONEER_EXPERIMENT/alignment"

python -u scripts/diagnose_dem_stages.py \
  --pre "$PIONEER_DATA/pre.tif" \
  --aligned-post "$PIONEER_EXPERIMENT/alignment/post_nuth_kaab.tif" \
  --fit-check-mask "$PIONEER_EXPERIMENT/alignment/stable_fit_validation.tif" \
  --skip-extra-stages --assess-uncertainty \
  --output-dir "$PIONEER_EXPERIMENT/uncertainty"

python -u scripts/screen_dem_difference_empirical.py \
  --dod "$PIONEER_EXPERIMENT/alignment/dod_nuth_kaab.tif" \
  --fit-check-mask "$PIONEER_EXPERIMENT/alignment/stable_fit_validation.tif" \
  --land-cover-on-grid "$PIONEER_EXPERIMENT/mask/landcover_on_grid.tif" \
  --classes 31 71 \
  --output-dir "$PIONEER_EXPERIMENT/empirical_screen"
```

The existing `--coreg-post` option is used to compare the previous NuthKaab
alignment. In comparison.csv it is labelled `existing_coreg`; this input is
not the earlier ICP result. The new alignment starts from the original post.
The 250 m spatial fit/check partition and seed 42 are retained. Previous
comparison scripts, diagnostic scripts and event settings were not modified.

## Results saved under Downloads/pioneer/open_ground_test

The refined mask contains 378,826 pixels (21.66% of the original candidate
mask): 12,866 barren and 365,960 grassland pixels. It is spatially uneven.

Fair comparison on the same 75,763 open-ground check pixels:

| Alignment | Median (m) | NMAD (m) | RMSE (m) |
| --- | ---: | ---: | ---: |
| Previous G-only NuthKaab | 0.0813 | 0.3033 | 1.3145 |
| New open-ground NuthKaab | -0.0172 | 0.3057 | 1.2894 |

The improvement is small and primarily a vertical recentering. On the broader
original G check set, median shifts from -0.0175 to -0.1177 m and NMAD changes
from 0.3991 to 0.4032 m. This is not evidence of uniformly improved alignment.
Do not compare the old all-G RMSE of 6.32 m directly with the new open-ground
RMSE of 1.29 m: those evaluate different pixels.

The new robust uncertainty model's nominal 95% threshold covered 80.97% of
open-ground check pixels. It is not established as a calibrated 95% LoD.

## Exploratory empirical screen

`screen_dem_difference_empirical.py` uses the 95th percentile of **absolute
residuals on fitting pixels**, including their tails. This avoids assuming a
Gaussian residual distribution. The resulting symmetric threshold is
3.287622 m. It covers 97.21% of reserved open-ground check pixels. Threshold
estimation uses the same fitting pixels as alignment; check blocks have also
been reused across model comparisons. Coverage does not prove a general
95% confidence guarantee or independently verify the stability assumption.

`empirical_screen/dod_open_ground_screened.tif` has these semantics:

- Within mapped classes 31 and 71, values with |DoD| > 3.287622 m are retained
  exactly; all smaller or equal differences are zero.
- Pixels outside these classes, or with missing data, are NoData. This is an
  analysis-area restriction, not a conversion of the surface to bare earth.
- G is a fitting criterion only; the analysis can include mapped open ground
  with G >= 5, including channels and other potentially changing terrain.
- Zero means below the screening threshold, not proof of zero physical change.
- Genuine changes smaller than 3.29 m are removed from this screen. The
  threshold may be inflated by real changes or vegetation in fitting pixels.

The analysis covers 2.668245 km2. It retains 155,398 positive and 132 negative
pixels, with 2,512,715 subthreshold pixels. The strongly positive remainder
still needs physical interpretation; retained differences are not verified
deposition/erosion. Source dates and vertical references remain unverified.

The full corrected raster remains in `alignment/dod_nuth_kaab.tif`; it retains
the full valid DEM overlap. Previous full-raster results remain unchanged in
`translation_review/`. All outputs use the original pre grid and metre units.

Supporting files include `paired_comparison.csv`, `mask/mask_comparison.png`,
fit/check masks, input fingerprints, correction parameters, uncertainty reports,
and `empirical_screen/screening_comparison.png`.

NLCD definitions: https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description
