# bin/ — hazard-layer validation and figures

Standalone analysis for the Pioneer/Stehekin debris-flow work. **Separate from
the M1–M4 dDEM pipeline** in `scripts/` and `methods/`: nothing here is
imported by that pipeline, and nothing here writes into its outputs. Everything
reads only and writes to a fresh directory.

Two live scripts. Run from the repo root with `xdem_env` active.

---

## `validate_hazard_layers.py` — performance

Scores the modelled probability layers and the dDEM erosion layer against the
89 USGS-mapped debris-flow initiation points.

```bash
conda activate xdem_env
python experiments/bin/validate_hazard_layers.py                       # defaults: single-cell
python experiments/bin/validate_hazard_layers.py --window-m 30 \
    --output-dir .../hazard_layer_validation_window30m     # 30 m window
```

**Positives** are the mapped initiation points, scored per mechanism (LS n=47,
Runoff n=22, Firehose n=20) rather than pooled — the three have very different
n and pooling hides that.
**Negatives** are `negative_points/random90_unbiased.shp`: 90 points >50 m from
every mapped initiation point, shared across all three mechanisms, since
"nothing was mapped here" is a negative for all of them equally.

Two validations, deliberately assuming different things:

| | needs absence points? | what it reports |
| --- | --- | --- |
| Confusion matrix | yes | TP/FP/FN/TN, precision, recall, specificity, F1, CSI, MCC at a fixed cutoff |
| Success-rate curve | **no** | fraction of points captured vs. fraction of area flagged (Chung & Fabbri) |

The success-rate table sidesteps the pseudo-absence-quality question entirely,
which is why it is the headline. ROC-AUC/PR-AUC are computed but kept as
secondary columns: they average over thresholds nobody would operationally use
and inflate under spatially clustered test points, which these are.

Useful options:

- `--window-m 30` — score each point as the max within a 30 m footprint
  instead of the single nearest cell. Applied to positive **and** absence
  points; an asymmetric window inflates recall for free.
- `--domain-raster G_pre_dinf.tif --domain-max 500` — drop points outside a
  terrain envelope (the "uplands" test).
- `--mechanisms LS` — restrict to one mechanism.

### Things in here that are easy to get wrong

- **Mid-rank tie handling.** `P_Landslide` is 92% exact zero. A naive
  descending scan puts the 10%-area threshold inside that tied block, so every
  point scores as "captured in the top 10%" — an artifact of the tie, not
  skill. `area_fraction_needed()` mid-ranks ties, the same convention AUC uses
  internally. Do not replace it with a threshold scan.
- **Zero is data, not nodata.** The `.asc` grids carry no NODATA value. Edges
  are uniformly zero (a domain mask) but so is ~88% of the interior, so zero
  cannot be dropped.
- **Class ratio varies by mechanism.** The absence set is fixed at 90 while
  positives differ, so precision/F1/CSI are not comparable *across*
  mechanisms. Recall is; specificity is identical by construction.

---

## `make_final_figures.py` — figures

Six publication figures, 300 dpi PNG + vector-text PDF, composed from tile
services rather than screenshotted so they are reproducible and correctly
projected.

```bash
python experiments/bin/make_final_figures.py     # -> Downloads/pioneer/figures_final/
```

| figure | content |
| --- | --- |
| fig1 | regional, 2D, all three modelled layers on satellite |
| fig2 | UAV window, 2D, all three layers + initiation points |
| fig3 | regional, 3D oblique, all three layers |
| fig4 | UAV window, 3D oblique, all three layers |
| fig5 | UAV window, 2D, dDEM erosion + points |
| fig6 | UAV window, 3D oblique, dDEM erosion |

Imagery is Esri World Imagery; terrain for the 3D panels is AWS terrarium
tiles, decoded `(R*256 + G + B/256) - 32768` metres. fig6 uses the local 1 m
pre-event DEM instead.

### Things in here that are easy to get wrong

- **Scale bars are Mercator-corrected.** Tiles are EPSG:3857, which stretches
  distance by 1/cos(latitude) — about 1.5× at 48.3°N. A bar drawn in raw map
  units reads ~50% short.
- **3D panels carry no scale bar,** by design: a perspective view has no
  single scale, so a bar would be right at one depth and wrong everywhere
  else. They state scene width and vertical exaggeration instead.
- **NaN must stay NaN in the 3D z-array.** `plot_surface` skips NaN quads.
  Filling them with the minimum elevation draws a cliff "skirt" around every
  non-rectangular extent.

---

## Slide and one-off figure scripts (added 2026-09-23/24)

| script | makes |
|---|---|
| `make_pipeline_schematic.py` | slide 13 landlab_data_prep schematic (real Pioneer layer stack) |
| `make_cascade_slide.py` | slide 14: burn severity, Washington locators (flat + topographic), PRISM storm forcing, timeline |
| `make_uav_layer_figures.py` | fig2 from the 10 m `.asc` layers, all values and > 0.01 |
| `make_usgs_hazard_map.py` | official USGS basin hazard map (`pioneer/usgs_hazard_pio2024`), `--rain 16..40` |
| `make_pooled_auc_figure.py` | AUC and ROC for all 89 initiations pooled (checked against `cellwise_metrics.csv`); also `roc_two_negative_sets`: that ROC next to the Strauch et al. (2018) absence cells from an `auc_runs/` run |

Both scoring scripts now score a pooled class `All` (every initiation, type
ignored) next to LS, Runoff and Firehose.

## `pioneer_auc.yaml` — scoring the layers against the points (updated 2026-10-20)

The scoring is the general package `landlab-grid-validation` (`python -m
landlab_grid_validation config.yaml`); this folder only keeps the Pioneer config.
Every rule (slope domain, surveyed area, event footprint, distance from points, G
limits, how many absence cells, stratification) is a key in the YAML, and unknown
keys are errors. Each run writes `Downloads/pioneer/auc_runs/<run_name>/`: metrics
with a 95% interval from resampling the points, `roc.png` and `roc_curves.csv`, the
absence cells as a GeoPackage, the settings and input checksums.

```bash
cd ~/landlab-grid-validation
PYTHONPATH=src python -m landlab_grid_validation ~/DemDifferencing/experiments/bin/pioneer_auc.yaml
```

Runs so far (made with the retired `superseded/auc_experiment.py`, which the package
reproduces to four decimals): `no_G_rule` (the Strauch et al. 2018 protocol, main
result), `G5-150_both_classes` and `G5-150_negatives_only` (sensitivity), and the
same three under `auc_runs_peri/` with the domain clipped to the validation perimeter.

## dDEM correction test: NDVI and figures (added 2026-10-01)

Details, results and interpretation: `Downloads/pioneer/ddem_correction/dDEM_correction_ndvi_green.md`.

```bash
conda activate ml_debris
python experiments/bin/fetch_ndvi.py --output-dir .../Downloads/pioneer/ddem_correction/ndvi
python experiments/bin/fetch_ndvi.py --output-dir .../Downloads/pioneer/ddem_correction/ndvi --delta 2023 2025
python experiments/bin/make_ddem_correction_figures.py      # -> ddem_correction/figures/
```

- `fetch_ndvi.py` (data only): Sentinel-2 summer NDVI 2023–2026 on the Landlab 10 m grid
  extended 500 m, NAIP 2023 NDVI on the 1 m dDEM grid extended 500 m, the scene time
  series and a manifest. The 2024 window is 15 Aug–30 Sep, after the burn.
  Every scene in each window is saved under `scenes/<year>/`; scenes with < 50% clear land,
  smoke (median AOT > 0.25) or a robust-outlier median NDVI are dropped from the composite.
  It also writes `vegetation__ndvi_summer<year>.asc`, Landlab-ready grids identical in
  header and domain to `lidar_footprint_10m/topographic__elevation.asc` (`--landlab-only`
  writes just these from NDVI already downloaded).
  `--delta 2023 2025` writes the NDVI change, post minus pre (negative = vegetation lost),
  from summers already downloaded: `s2_delta_ndvi_2023_to_2025.tif` on the extended grid,
  `vegetation__delta_ndvi_2023_to_2025.tif` and `.asc` on the Landlab grid, and an entry
  under `delta` in the manifest.
- `make_ddem_correction_figures.py`: `ndvi_maps.png` (six NDVI maps), `ndvi_scenes_<year>.png`
  (every scene, dropped ones in red), `slide_ndvi.png` and `slide_ddem.png` (16:9 slides).

## `superseded/`

Three earlier figure scripts, kept for provenance, not maintained:
`make_publication_figures.py` (hillshade basemap), `make_satellite_figures.py`
(first satellite version), `make_3d_figures.py` (first 3D version). All three
are replaced by `make_final_figures.py`, which covers every extent and layer
they did and fixes the skirt and colourbar-collision bugs.

---

## Standing caveats for anything produced here

1. **The dDEM shares data lineage with the ground truth.** The initiation
   points were mapped remotely from the same UAV survey's topographic
   products that the post-DEM comes from. The Landlab layers were built
   independently. The comparison is therefore not like-for-like, and the
   dDEM's advantage is partly structural. State this in any caption.
2. **Susceptibility vs. change detection are different questions.** The
   probability layers predict potential; the dDEM measures what moved.
   Scoring both against "where things actually failed" favours the dDEM close
   to tautologically.
3. Absence points are buffered pseudo-absences, never field-verified.
4. Mapped completeness is claimed as "fully complete" in the USGS metadata,
   but mapping was remote and ~46% of points sit in evergreen, where SfM sees
   no ground at all.
5. Esri imagery predates the 2024 fire in places — a location basemap, not
   evidence of the scars.

## Offline baseline maps with a 0.01 display cutoff

`make_baseline_censored_figures.py` creates the matching regional and UAV-window
baseline panels in `Downloads/pioneer/regional_comparison/figures/`:

```bash
python experiments/bin/make_baseline_censored_figures.py
```

Outputs are `baseline_regional_layers_2d_gt0p01.{png,pdf}` and
`baseline_uav_window_2d_gt0p01.{png,pdf}`. PNGs are 300 dpi. The script uses
the original baseline TIFFs in `Downloads/regional_baseline/regional_baseline/`
and shows only reprojected probabilities >0.01, using the existing panel style.
This is a display filter; it does not change model values or evaluation scores.

The clean RGB basemaps are extracted from separate image objects in the existing
summer comparison PDFs, with their PDF placement mapped back to the reference
extents. No new imagery is fetched. This additional utility requires `pypdf`
(already installed in the current base Python environment). A JSON manifest
records source/model/output hashes and extents. Existing target filenames are
protected from overwriting; pass a new `--output-dir` to make another copy.

## `inventory_dod_stats.py` — Eagle Creek inventory polygons vs the dDEM

For one epoch of `Downloads/eaglecreek_inventory.gpkg`, every mapped polygon gets the
dDEM statistics inside it (cells, mean, median, percentiles, erosion/deposition volume,
share of cells eroded) beside its inventory attributes, depths converted from feet.
dDEM: `eagle_creek/21_18.tif` as prepared (`prep/dz_2m.tif`). Both inputs are taken as
final; no threshold is applied.

```bash
python experiments/bin/inventory_dod_stats.py --epoch 2018_2021
# -> Downloads/eagle_creek/inventory_vs_ddem_2018_2021/ (csv, gpkg, inputs.json, README)
```

`make_inventory_ddem_figures.py` draws the four figures for that comparison into
`inventory_vs_ddem_2018_2021/figures/` (overview map, densest-2 km zoom, depth scatter, the eight table classes).
