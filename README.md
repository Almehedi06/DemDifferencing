# DemDifferencing

## Overview

This repository contains scripts, outputs, and summary products used to evaluate and improve DEM coregistration and bias correction for the Montecito study area.

The workflow uses xDEM to:

* Coregister DEMs using Nuth & Kääb + ICP
* Apply polynomial deramping
* Apply directional bias correction
* Apply terrain-based aspect bias correction
* Evaluate uncertainty using the Hugonnet et al. (2022) framework
* Compare correction strategies through sensitivity analyses
* Export dDEM values for histogram-based benchmarking

---

## Layout

```text
scripts/        the event pipeline: run_event_pipeline.py and what it imports
configs/        one YAML per event (pioneer.yaml, eagle_creek_..., event.example.yaml)
tests/          pipeline tests:  python -m unittest discover -s tests
experiments/    everything exploratory, in one place, to be promoted or dropped later
  methods/        M1-M6 detection and correction methods (README inside)
  bin/            Pioneer validation, figures, slides, NDVI download (README inside)
  diagnostics/    land-cover mask refinement, Thomas aspect sensitivity runs (+ tests)
  docs/           diagnostic workflow notes
  archive/        earlier Montecito outputs: histogram plots, metric tables, figures
```

Only `scripts/`, `configs/` and `tests/` are the pipeline. Nothing in `experiments/`
is imported by it.

### Files kept in Git

Source code, event configs, tests, environment definitions, documentation, and
the benchmark figures and tables in `experiments/archive/` are versioned.
Keep local input data in `data/` and new run products in `outputs/` (as in
`configs/event.example.yaml`), or outside the repository. These two local
directories, the default `Production_o1_a90_ASPECT/` and `evaluation/` output
directories, Python caches, and local environments are ignored. PNG and CSV
files are not ignored globally, so selected reference results can still be
shared in the archive.

---

## Repository Contents

### Scripts

**Sensitivity Analysis (Aspect Bias Included)**

* Tests multiple deramp orders and directional bias angles
* Includes aspect bias correction
* Generates performance metrics used to identify the preferred correction workflow

**Production Processing Script (Montecito preset)**

* Applies the preferred correction workflow identified through sensitivity testing:

  * Nuth & Kääb + ICP coregistration
  * First-order deramp
  * Directional bias correction (90°)
  * Aspect bias correction

**Histogram Analysis Script**

* Reads exported dDEM values
* Generates histogram plots
* Identifies histogram peak locations
* Produces summary statistics for comparison with collaborator workflows

---

## Included Outputs

### Histogram Products

`experiments/archive/histogram_peak_plots_all_and_stable/`

Contains histogram plots for:

* All valid pixels within the AOI
* Stable-terrain pixels only

### Summary Tables

`experiments/archive/histogram_peak_summary_all_and_stable.csv`

* Histogram peak location
* Mean
* Median
* Standard deviation
* Selected percentiles
* Reported separately for all pixels and stable terrain

`experiments/archive/xdem_run_metrics_raw.csv`

* Per-run performance metrics for all sensitivity-analysis runs

`experiments/archive/xdem_run_metrics_ranked.csv`

* Ranked performance table used to identify preferred correction parameters

`experiments/archive/xdem_uncertainty_summary.csv`

* Uncertainty estimates and associated summary statistics

---

## Notes

* File paths and local machine information have been removed from outputs prior to sharing.
* Stable terrain was defined using a threshold on the initial raw dDEM.
* Histogram outputs are provided both for benchmarking against collaborator workflows and for evaluating correction performance.

---

## Main event pipeline: one command

Use `scripts/run_event_pipeline.py` for stable-cell generation, correction,
uncertainty summaries and optional LoD filtering. It calls the existing
`scripts/xDEM_pipeline_Thomas_production.py`; diagnostic scripts do not need
separate runs. The settings are in one event YAML file.

### Pioneer

```bash
conda activate xdem_env
cd /home/abdullah/DemDifferencing
python scripts/run_event_pipeline.py --config configs/pioneer.yaml --dry-run
python -u scripts/run_event_pipeline.py --config configs/pioneer.yaml
```

`configs/pioneer.yaml` now uses:

1. Pre as the reference, with the existing matched 1 m pre/post rasters.
2. Candidate stable cells: `G_pre < 5`, with **no elevation or raw-DoD filter**.
3. Fitting candidates restricted to NLCD barren/grassland (classes 31/71).
   This restriction affects fitting only, not the output footprint.
4. Reserved check blocks: one fifth of 250 m blocks. Corrections and threshold
   estimation use the other four fifths.
5. `corrections.preset: translation`: median vertical-offset correction followed
   by Nuth–Kaab (100,000 samples, seed 42). ICP, deramp, directional and aspect
   corrections are disabled for this preset because the Pioneer diagnostics
   showed artifacts from the additional stages.
6. `lod.method: empirical`: calculate the 95th percentile of absolute corrected
   residuals on fitting pixels, report coverage on check pixels, and apply the
   threshold across the full valid corrected DEM overlap.

Values above the threshold retain their signed elevation difference; values
at or below it become zero. Input/overlap NoData remains NoData. **Zero means
below the detection threshold, not proof of zero physical change.**

The threshold is recalculated for each run; it is not hard-coded to 3.29 m.
It is an empirical LoD, not an established 95% confidence bound over the whole
AOI. The fitting terrain may contain real change; an open-ground calibration
is not validated for forest. Earlier model comparisons reused these check
blocks. Survey chronology and vertical references still require verification.
H2022 uncertainty is assessed on the final corrected DEM for the translation
preset (at all original stages for Montecito). Its nominal
Gaussian limits do not drive this empirical filter.

### Results

Pioneer writes to a new directory:
`/mnt/c/Users/amehedi/Downloads/pioneer/main_pipeline_run`.
Under `corrections/pioneer/final_dem_products/`:

| File | Meaning |
| --- | --- |
| `post_coreg_bias.tif` | Post DEM after vertical-offset correction and alignment |
| `dod_corrected.tif` | Corrected post minus pre, unfiltered |
| `dod_lod_filtered.tif` | Corrected DoD after empirical threshold filtering, full valid overlap |
| `lod_threshold.tif` | Applied threshold in metres on the same grid |
| `lod_report.json` | Threshold, fit/check statistics, counts and limitations |

Raw/coregistered products, plots and CSV summaries are also retained.
`stable_fit_check.tif` records 1=fit, 2=check, 0=other valid pixels.
`pipeline_metadata.json`, `resolved_config.yaml`, and `event_summary.json`
record processing settings, transformations and completion/failure status.
A nonempty output directory stops the run; choose a new `batch.output_dir`
when rerunning. Existing Pioneer inputs and results are preserved.

### Original Montecito behavior

The existing correction sequence remains available through:

```yaml
corrections:
  preset: montecito
fitting:
  land_cover: null
lod:
  method: none
```

This uses Nuth–Kaab + ICP, first-order deramp, 90-degree directional correction
(1000 bins), and aspect correction in 5-degree bins. These are also the defaults
when the new config sections are omitted. The event runner supplies the
configured G-based mask; direct production-script runs without `--stable-mask`
retain the original `abs(raw DoD) < 5 m` fitting selection. The original output
filenames are preserved for the Montecito preset. LoD filtering can also be
enabled after that sequence by setting `lod.method: empirical`.

### Another event

Copy `configs/event.example.yaml` and change the pair paths, event ID, output
directory and applicable fitting settings. The example keeps the original
Montecito corrections and disables LoD until explicitly selected. Set
`fitting.land_cover: null` for G-only fitting. Optional thresholds use `null`
to disable a rule. Land-cover class codes must match that event's classification.

Relative paths resolve against the config directory. Paired-directory discovery
is also supported. IDs must be unique; missing partners or invalid settings
stop processing. G is D-infinity specific contributing area multiplied by
slope (rise/run) to power 1.167. Computing G and making spatial check blocks
require square, unrotated pixels in a projected metre CRS. Elevations must be
in metres. Existing G rasters can be supplied with `terrain.g_source: raster`,
`g_dir` and `g_template: "{id}_G.tif"`.

For a new machine, create the environment with
`conda env create -f environment.yml`. Whitebox needs its platform executable.
Run the checks with `python -m unittest discover -s tests -v`.

The pre-consolidation repository is backed up under
`/home/abdullah/DemDifferencing_backups/20260915T212712Z_consolidation`.
See [RESTORE_POINTS.md](RESTORE_POINTS.md) for preservation details.

## Earlier diagnostics (retained for reproducibility)

These scripts explain the earlier experiments. They are optional diagnostics,
not additional steps needed to run the main pipeline above.

## Diagnose banding or a large vertical offset

`scripts/compare_dem_corrections.py` provides a separate diagnostic comparison.
It estimates one median vertical offset from four fifths of 250 m spatial blocks
of an existing stable mask. With `--fit-nuth-kaab`, it then fits Nuth–Kääb
translations to the same fit pixels, using 100,000 sampled points and seed 42.
It does not apply ICP, deramping, directional bias or aspect bias. The remaining
blocks evaluate the new candidates. These blocks remain G-selected candidate
stable terrain; they are not independent evidence that no surface change occurred.

Example (use a fresh output directory if rerunning):

```bash
python -u scripts/compare_dem_corrections.py \
  --pre /mnt/c/Users/amehedi/Downloads/pioneer/pre.tif \
  --post /mnt/c/Users/amehedi/Downloads/pioneer/post.tif \
  --stable-mask /mnt/c/Users/amehedi/Downloads/pioneer/terrain_stable_run/stable_cells/masks/pioneer_stable_cells.tif \
  --fit-nuth-kaab \
  --output-dir /mnt/c/Users/amehedi/Downloads/pioneer/translation_check
```

Optional `--coreg-post` and `--corrected-post` compare previous corrected DEMs
on the same evaluation pixels. Those previous models may already have fitted
the evaluation pixels, so their scores are diagnostic, not independent validation.

Outputs: `post_vertical_shift.tif`, `dod_vertical_shift.tif`, optional
`post_nuth_kaab.tif`/`dod_nuth_kaab.tif`, a spatial fit/validation mask, comparison
CSV/JSON, and a plot using the same colour scale. Transform parameters are saved
in JSON. All DoDs use corrected post minus pre; output grids match pre exactly.
The vertical shift is empirical and does not establish or convert a vertical
datum. This comparison does not change the event runner's Montecito defaults,
replace original inputs, or perform LoD filtering.
