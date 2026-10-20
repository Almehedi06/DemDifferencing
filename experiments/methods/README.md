# Methods M1–M4

Four ways of turning the Pioneer/Stehekin pre/post DEM pair into an erosion product.
They share one correction engine in `scripts/`; only the *detection* step differs.
Every method is validated below against the same independent ground truth: the 89
mapped debris-flow initiation points in the USGS data release (doi:10.5066/P1GNNSZU).

**M4 is the current recommendation.** M1 failed on this event, M2 is refuted, M3 is
superseded.

| | detection rule | initiation points detected | coverage |
| --- | --- | ---: | ---: |
| M1 | none (no LoD) | — | 100% |
| M2 | \|DoD\| > 3.28 m, one threshold | **2 / 89** | 100% |
| M3 | per-land-cover asymmetric limits, forest excluded | **10 / 89** | 53% |
| **M4** | **DoD < −0.50 m** | **69 / 89** | **100%** |

M4 by mechanism: landslide **47/47 (100%)**, runoff 13/22 (59%), firehose 9/20 (45%).

## Why the input constrains all of this

`post.tif` derives from UAV **structure-from-motion** (Agisoft Metashape, per the
data release). SfM reconstructs only what the cameras see, so under surviving canopy
it returns treetops and no ground at all. The DoD therefore carries a large, strictly
**positive** tail wherever vegetation survived — 16.8% of NLCD evergreen exceeds
+3.28 m, with a 95th percentile of +19 m.

The negative side is uncontaminated: its 1st percentile sits between −0.94 and
−1.03 m in *every* land-cover class, forest included. That asymmetry is the single
fact that decides the method.

## M1 — original (`M1_original/`)

The committed original at `294e90f`, frozen. Fits on `|raw DoD| < 5 m`.

**It failed on this event.** With an 18.7 m systematic offset, that rule selected the
379,042 pixels where roughly 18 m of canopy or error coincidentally cancelled the
offset. Stable NMAD ran 3.54 → 5.16 → 4.81 m across stages, so its corrections made
scatter *worse*, and its output DoD remains offset by −15.5 m. Kept for provenance.

Results: `Downloads/pioneer/Montecito_params/`

## M2 — translation + empirical LoD (`M2_translation/`)

Median vertical shift + Nuth–Kääb, fitted on `G_pre < 5` intersected with NLCD
barren/grassland. Its alignment is the best available — check-block median −0.015 m,
NMAD 0.30 m — and M3 and M4 both build on it.

Its **detection step is refuted**. The 3.284814 m threshold is the 95th percentile of
*absolute* stable residuals, so it measured the vegetation tail and then applied that
figure symmetrically to the clean side. It retained 1,487,195 positive and 11,046
negative pixels (median retained value +10.5 m — canopy) and detected 2 of 89
initiation points.

```bash
python -u scripts/run_event_pipeline.py --config configs/pioneer.yaml
```

Results: `Downloads/pioneer/main_pipeline_run/`. Its `dod_corrected.tif` is the
continuous corrected DoD that M3 and M4 consume. **Do not use its
`dod_lod_filtered.tif` for anything.**

## M3 — domain-restricted asymmetric limits (`M3_domain_lod/`)

Separate negative and positive limits per land-cover class, calibrated on spatial
blocks that fitted no alignment, with evergreen excluded as NoData. Limits came out
at −0.826/+3.474 m (open) and −0.705/+5.666 m (shrub).

**Superseded on two counts.** It gated the domain on NLCD **2021**, which predates the
2024 fire, so it excluded terrain by pre-fire vegetation rather than by the actual
post-fire surface. And 41 of the 89 initiation points fall in that excluded evergreen
class, so the exclusion discarded 46% of the known events to protect against a
one-sided contamination that simply dropping the positive side removes for free.

Retained for the method comparison and the block-split machinery.

## M4 — single negative threshold (`M4_erosion_threshold/`)

Keep `DoD < −0.50 m` on the continuous corrected DoD. No land-cover mask, no domain
restriction, no per-class calibration, full coverage.

Three reasons for −0.50 m:

1. It maximises skill (detection rate minus false-positive rate) at 68.9, against
   51.6 at M3's −0.826 m.
2. It is the **largest** threshold that still detects every landslide-initiated
   debris flow, 47/47. At −0.60 m that becomes 46/47.
3. Its 8.7% false-positive rate is an upper bound — the reference terrain is
   `G_pre < 5` candidate-stable, never verified unchanged, and some of it eroded.

```bash
python -u experiments/methods/M4_erosion_threshold/m4_erosion_threshold.py \
  --dod        ".../main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif" \
  --threshold-m 0.50 \
  --initiation-points ".../Initiation_Points_final/Initiation_Points_final.shp" \
  --output-dir ".../m4_erosion"
```

Results: `Downloads/pioneer/m4_erosion/` — 889,477 px, 0.889 km², median retained
depth −0.64 m, deepest −11.01 m.

## M5 and M6 — tilt removal and gain mask (the dDEM correction test)

Full methods, tests, results and interpretation are in one document:
`Downloads/pioneer/ddem_correction/dDEM_correction_ndvi_green.md`. In short:

- **M5** (`M5_detilt/`) subtracts a plane fitted on 250 m block histogram peaks from the
  M2 dDEM. The histogram peak moves from −0.22 to −0.02 m.
- **M6** (`M6_gain_mask/`) keeps gains only in channels (G > 200) or on slopes < 15°, and
  not green in the UAV orthophoto; losses are untouched. It also writes Erkan's NDVI
  rule (`dod_ndvi_test.tif`) for comparison. Area raised > 0.5 m: 30% → 1.5% (NDVI
  rule: 17.5%). It also writes a dNDVI variant (`dod_dndvi_test.tif`) for the record.

```bash
W=.../Downloads/pioneer/ddem_correction
python -u experiments/methods/M5_detilt/m5_detilt.py \
  --dod ".../main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif" \
  --initiation-points ".../Initiation_Points_final/Initiation_Points_final.shp" \
  --output-dir "$W/m5_detilt"
python -u experiments/methods/M6_gain_mask/m6_gain_mask.py \
  --dod "$W/m5_detilt/dod_detilted.tif" \
  --driving-index ".../main_pipeline_run/stable_cells/terrain/pioneer/G_pre_dinf.tif" \
  --slope-percent ".../main_pipeline_run/stable_cells/terrain/pioneer/pre_slope_percent.tif" \
  --ortho ".../Pioneer-Ortho-10cm-UTM_NAVD88.tif" \
  --ndvi "$W/ndvi/s2_ndvi_summer2026.tif" \
  --ndvi-before "$W/ndvi/s2_ndvi_summer2025.tif" \
  --output-dir "$W/m6_gain_mask"
```

M6 needs the NDVI from `experiments/bin/fetch_ndvi.py`; figures come from
`experiments/bin/make_ddem_correction_figures.py`.

## Limits that apply to every method

- **Erosion under surviving canopy is undetectable**, because SfM provides no ground
  surface there. M4 reports it as zero, which means "not resolved", not "no change".
- **Detection is mechanism-dependent.** Runoff and firehose initiation do not deplete
  half a metre at a point. Quote detection by mechanism, never as one rate.
- **M4 maps erosion only.** Deposition is not a product of this method.
- The vertical datum is declared NAVD88 but no geoid model is documented, and the
  −18.4 m offset against 3DEP resembles a geoid separation. A constant offset cancels
  in differencing; the measured residual tilt is 0.567 m across the scene.
- The 3DEP acquisition date is unknown, so the change interval is unverified. The
  debris flows occurred 8–12 December 2025; the UAV survey is 15–17 June 2026.
- Candidate-stable terrain is `G_pre < 5`, an assumption, not verified stable ground.

Full analysis: `Downloads/pioneer/M1_M2_M3_comparison_20260916.md`.
