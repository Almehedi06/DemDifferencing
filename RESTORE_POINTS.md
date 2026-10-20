# Latest checkpoint: before main-pipeline consolidation

`/home/abdullah/DemDifferencing_backups/20260915T212712Z_consolidation`
contains the complete repository before adding configurable corrections and
integrated empirical LoD. All 134 copied files were SHA-256 verified, including
Git history and uncommitted/untracked work. See its `RESTORE.md` and
`manifest.json`. Restore into a new folder; do not reset the active repository.

The active `configs/pioneer.yaml` now selects the translation preset and
empirical LoD, writing to `Downloads/pioneer/main_pipeline_run`.
The original correction methods remain available with `preset: montecito`,
`fitting.land_cover: null`, and `lod.method: none`. Previous result directories
and DEM inputs have not been modified. Earlier complete data checkpoints
remain in the backup directory.

---

# Pioneer DEM restore point

Created 20260915T195548Z at the user's request to preserve earlier work.
Verified backup location: `/home/abdullah/DemDifferencing_backups/20260915T195548Z` (see its manifest.json and RESTORE.md).

## Workflows retained

- Main pipeline: scripts/run_event_pipeline.py with configs/pioneer.yaml.
  Corrections remain NuthKaab + ICP, first-order deramp, 90-degree directional
  bias, and 5-degree aspect bins. Stable selection is G_pre < 5, with no
  elevation threshold. Pre remains the reference.
- Previous output: /mnt/c/Users/amehedi/Downloads/pioneer/terrain_stable_run.
- Earlier Montecito-parameter output: /mnt/c/Users/amehedi/Downloads/pioneer/Montecito_params.
- Separate experiments: correction_review/ and translation_review/ under
  /mnt/c/Users/amehedi/Downloads/pioneer. The latter uses an initial vertical offset plus NuthKaab only.
- Latest experiment added scripts/compare_dem_corrections.py and its tests
  and README documentation; it did not replace the main correction method.

## Preservation

The checkpoint copies the complete current repository, including uncommitted
and untracked work and Git history. It also stores the committed HEAD source
separately. HEAD 294e90fe6134cbf896e6e3ab98ed6789cea77beb predates the recent generalized pipeline: it is not
the same as the code used for the completed Pioneer run.

DEM inputs, source rasters, and the named DEM result directories are copied
separately and verified with SHA-256 hashes. This is a checkpoint of current
files, not a claim that an exact historical snapshot exists for every earlier
edit. Existing source files and result directories are retained in place.

Consult RESTORE.md in the backup to inspect or restore into a NEW folder.
Preserve this checkpoint before any future experiments or method changes.
