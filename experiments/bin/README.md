# bin/ — hazard-layer validation and figures

Standalone analysis for the Pioneer/Stehekin debris-flow work. **Separate from
the M1–M4 dDEM pipeline** in `scripts/` and `methods/`: nothing here is
imported by that pipeline, and nothing here writes into its outputs. Everything
reads only and writes to a fresh directory.

Figures, data fetches and the build of Erkan's evaluation bundle. The scoring itself is the
`landlab-grid-validation` package; the record of the validation is
`Downloads/pioneer/VALIDATION_MASTER.md`. Run from the repo root.

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

The September sensitivity runs (`no_G_rule`, `G5-150_both_classes`,
`G5-150_negatives_only`, and the same with the domain clipped to the validation
perimeter) were deleted on 21 Oct 2026; their numbers are in
`Downloads/pioneer/VALIDATION_MASTER.md`, and this config remakes any of them under a
new `run_name`.

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
