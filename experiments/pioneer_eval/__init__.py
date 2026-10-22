"""Pioneer evaluation: score Landlab outputs against what happened in Dec 2025.

    from pioneer_eval import evaluate_hazard, evaluate_mwr
    evaluate_hazard(".", run_name="baseline")     # after P_*.asc are written
    evaluate_mwr("MWR.asc", run_name="baseline")  # after MWR.asc is written

Results go to evaluation/<run_name>/ next to the notebook, plus one row per run in
evaluation/runs_log.csv (hazard) or evaluation/mwr_runs_log.csv (runout). Each run
also writes maps (PNG, the points coloured by failure type) and GeoTIFFs for QGIS: the
observations as yes/no rasters and the agreement maps (0 correct no, 1 false alarm,
2 miss, 3 hit). Observations are in data/: the 89 mapped initiation points and the
corrected dDEM.

This file holds only the Pioneer settings (paths, rules, thresholds). The scoring is
landlab_grid_validation, the general package bundled beside it.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))  # the bundled landlab_grid_validation

CRS = "EPSG:6339"  # NAD83(2011) UTM 10N; the .asc files do not record it
LAYERS = ["P_Landslide", "P_Channel_ini", "P_Combined_H"]
POINTS = DATA / "Initiation_Points_final.shp"
DDEM = DATA / "dod_dndvi_test.tif"
ELEVATION = DATA / "topographic__elevation.asc"

# Strauch et al. (2018), adapted to points and the dDEM. Do not change between runs,
# or scores stop being comparable. Rules are explained in landlab_grid_validation.rules.
SETTINGS = {
    "observed": {"type_column": "FailureTyp", "tolerance_cells": 1},         # 3 x 3 cells (30 m) per point
    "domain": [                                                               # where both classes come from
        {"slope_of": str(ELEVATION), "min": 17},                              # slopes that can fail
        {"raster": str(DDEM), "finite": True, "resampling": "average"},       # inside the drone survey
    ],
    "negatives": {
        "exclude": [{"raster": str(DDEM), "abs_min": 0.5, "buffer_cells": 1,  # cells the event changed,
                     "resampling": "average"}],                               # plus one cell around them
        "min_distance_m": 50,                                                 # from any point
        "n": "all", "seed": 42,                                               # every qualifying cell
    },
    "bootstrap": 1000, "seed": 42,                                            # 95% interval by resampling the points
    "thresholds": [0.5],                                                      # cutoffs for precision, recall, f1, csi, area flagged
    "optimal_threshold": True,                                                # also at the Youden cutoff (max TPR - FPR)
    "fpr_levels": [0.10, 0.05],                                               # TPR at these false-positive rates (first one is printed)
}


def _run_name(run_name, prefix):
    return run_name or datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def _log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def evaluate_hazard(model_dir=".", run_name=None, params=None, out_dir="evaluation",
                    maps=True, map_threshold=0.5):
    """Score P_Landslide, P_Channel_ini and P_Combined_H (.asc in model_dir) against the
    mapped initiation points. Prints, by failure type: AUC [95% interval], TPR at a 10%
    false-positive rate, the area flagged and the precision at P >= 0.5 (and as a
    multiple of the base rate), the Brier score and the share of positive cells the
    model scores 0; shows the ROC curves and the sampling map; logs the run. The full
    table (confusion counts, f1, csi, the Youden cutoff, ...) is metrics.csv.
    params: optional dict of model settings to keep with the scores, e.g.
    {"Csoil": Csoil, "tau_c_low": 1.1}.

    maps=True also draws each layer against the points on the Landlab grid and writes
    GeoTIFFs: the observations as yes/no rasters and one agreement map per layer, with
    a cell called "yes" where the probability is at least map_threshold. A point is
    "hit" when any cell within 30 m of it is flagged; the hit rate of a random location
    under the same rule is reported beside it. The AUC does not use a threshold, so
    map_threshold changes the maps and the hit rates only, never the scores."""
    from landlab_grid_validation import (Absences, evaluate_features, grid_from_raster, mask_from_rules,
                                         raster_to_node_field)

    run_name = _run_name(run_name, "run")
    model_dir, out_dir = Path(model_dir).resolve(), Path(out_dir).resolve()
    run_dir = out_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already holds a run; choose another run_name")

    grid, crs = grid_from_raster(model_dir / f"{LAYERS[0]}.asc", crs=CRS)
    layers = {name: raster_to_node_field(grid, crs, model_dir / f"{name}.asc", raster_crs=CRS) for name in LAYERS}
    obs, neg = SETTINGS["observed"], SETTINGS["negatives"]
    ev = evaluate_features(
        grid, crs, layers, POINTS, tolerance_cells=obs["tolerance_cells"], group_by=[obs["type_column"]],
        domain=mask_from_rules(grid, crs, SETTINGS["domain"]),
        absences=Absences(exclude=mask_from_rules(grid, crs, neg["exclude"], how="any"),
                          min_distance_m=neg["min_distance_m"], n=neg["n"], seed=neg["seed"]),
        bootstrap=SETTINGS["bootstrap"], seed=SETTINGS["seed"], thresholds=tuple(SETTINGS["thresholds"]),
        optimal_threshold=SETTINGS["optimal_threshold"], fpr_levels=tuple(SETTINGS["fpr_levels"]))
    import geopandas as gpd

    ev.write(run_dir, settings={"run_name": run_name, "model": {n: str(model_dir / f"{n}.asc") for n in LAYERS},
                                "points": str(POINTS), **SETTINGS},
             inputs=[model_dir / f"{n}.asc" for n in LAYERS] + [POINTS, ELEVATION, DDEM],
             points=gpd.read_file(POINTS).to_crs(crs),                       # sampling_map.png: the cells used
             background=raster_to_node_field(grid, crs, ELEVATION, raster_crs=CRS))
    _display(run_dir / "sampling_map.png")

    m = pd.DataFrame(ev.table)
    print(ev.summary())
    _display(run_dir / "roc.png")

    hits = _hazard_maps(model_dir, run_dir, map_threshold, grid, crs, layers) if maps else {}

    # one row per run, the three numbers to compare runs by (all points): AUC, the optimal
    # probability (highest TPR - FPR) and the precision there; every other score is in metrics.csv
    row = {"run": run_name, **(params or {})}
    for k in LAYERS:
        r = m.query("group == 'All' and layer == @k").iloc[0]
        row.update({f"AUC_{k}": r.auc, f"optimal_P_{k}": r.optimal_threshold,
                    f"precision_at_optimal_{k}": r.precision_at_optimal,
                    f"precision_x_base_{k}": r.precision_lift_at_optimal})
    _log(out_dir / "runs_log.csv", row)
    return m


def _hazard_maps(model_dir, run_dir, threshold, grid, crs, layers):
    """Each layer against the points on the Landlab grid, as PNG and GeoTIFF, and the
    point-level score: a point is hit when any cell within its block is flagged,
    compared with the hit rate of a random location. Returns {layer: (hit, chance)}."""
    import matplotlib.pyplot as plt

    from landlab_grid_validation import (FieldSpec, chance_hit_rate, compare_fields_on_grid, feature_hits,
                                         plot_comparison, raster_to_node_field, vector_to_feature_cells,
                                         write_field_geotiff)

    surveyed = np.isfinite(raster_to_node_field(grid, crs, DDEM, resampling="average"))
    points, observed = _observations(grid, crs)
    # 1 = within 30 m of a point, 0 = not, empty where the drone did not fly
    observed = {code: np.where(surveyed, values, np.nan) for code, values in observed.items()}
    type_column, k = SETTINGS["observed"]["type_column"], SETTINGS["observed"]["tolerance_cells"]
    cells = vector_to_feature_cells(grid, crs, points, tolerance_cells=k)   # each point's block
    kinds = points[type_column].astype(str).to_numpy()

    for code, values in observed.items():
        write_field_geotiff(grid, crs, run_dir / f"obs_points_{code}.tif", values)

    rows, summary = [], {}
    for name, layer in layers.items():
        result = compare_fields_on_grid(
            grid, layer, observed["All"],
            FieldSpec(name=name, kind="continuous", units="probability", threshold=threshold),
            FieldSpec(name="mapped initiations (all types)", kind="binary"),
        )
        write_field_geotiff(grid, crs, run_dir / f"agree_{name}.tif", result.agreement())
        hits = feature_hits(result.predicted_yes, cells)
        chance = chance_hit_rate(grid, result.predicted_yes, k)
        flagged = float(np.nanmean(result.predicted_yes))                  # share of surveyed cells
        for code in ["All", *sorted(set(kinds))]:
            sel = np.ones(len(points), bool) if code == "All" else kinds == code
            rows.append({"layer": name, "threshold": threshold, "flagged_share_of_surveyed_cells": round(flagged, 4),
                         "chance_hit_rate": round(chance, 4), "type": code, "points": int(sel.sum()),
                         "points_hit": int(hits[sel].sum()), "share_hit": round(float(hits[sel].mean()), 4)})
        summary[name] = (float(hits.mean()), chance)
        by_type = ", ".join(f"{code} {hits[kinds == code].mean():.0%}" for code in sorted(set(kinds)))
        print(f"{name} at {threshold:g}: hits {hits.mean():.0%} of the {len(points)} points ({by_type}); "
              f"a random location would be hit {chance:.0%} of the time")
        fig = plot_comparison(grid, result, points=points, colour_by=type_column, feature_hits=hits,
                              chance_hit_rate=chance)
        # the AUC in the table uses only slopes over 17 degrees; this map covers the whole survey
        fig.suptitle(f"{name} vs the {len(points)} mapped initiations, whole drone survey, "
                     f"'yes' where P >= {threshold:g}", fontsize=12)
        fig.savefig(run_dir / f"map_{name}.png", dpi=150)
        plt.close(fig)
        _display(run_dir / f"map_{name}.png")
    pd.DataFrame(rows).to_csv(run_dir / "point_hits.csv", index=False)
    print(f"maps, point_hits.csv and GeoTIFFs written to {run_dir}")
    return summary


def _observations(grid, crs):
    """The mapped points, and yes/no node fields: all types together, then each type."""
    import geopandas as gpd

    from landlab_grid_validation import vector_to_node_field

    points = gpd.read_file(POINTS).to_crs(crs)
    k, column = SETTINGS["observed"]["tolerance_cells"], SETTINGS["observed"]["type_column"]
    observed = {"All": vector_to_node_field(grid, crs, points, tolerance_cells=k)}
    for code in sorted(points[column].unique()):
        observed[code] = vector_to_node_field(grid, crs, points[points[column] == code],
                                              tolerance_cells=k)
    return points, observed


def evaluate_mwr(mwr_asc="MWR.asc", run_name=None, params=None, out_dir="evaluation", change_m=0.5):
    """Compare MWR elevation change (MWR.asc, m) with the dDEM, for erosion and for
    deposition. A cell is eroded where the dDEM drops more than change_m and deposited
    where it rises more than change_m; the MWR likewise. Prints AUC, recall and
    precision for each (exact, and allowing a one-cell offset), maps hits and misses,
    writes GeoTIFFs (dDEM erosion and deposition as yes/no, and the agreement maps) and
    logs the run. Erosion columns keep their old names; deposition ones end in
    _deposition."""
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from rasterio.warp import Resampling, reproject
    from scipy.ndimage import binary_dilation

    from landlab_grid_validation import grid_from_raster, roc_auc, write_field_geotiff
    from landlab_grid_validation.plot import OUTCOMES

    run_name = _run_name(run_name, "mwr")
    with rasterio.open(ELEVATION) as src:
        transform, shape, dem = src.transform, src.shape, src.read(1)
    with rasterio.open(mwr_asc) as src:
        if src.shape != shape or not src.transform.almost_equals(transform):
            raise ValueError(f"{mwr_asc} is not on the Pioneer 10 m grid")
        dz = src.read(1).astype("float64")
    ddem = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(DDEM) as src:
        reproject(rasterio.band(src, 1), ddem, dst_transform=transform, dst_crs=src.crs,
                  dst_nodata=np.nan, resampling=Resampling.average)

    valid = (dem != -9999) & np.isfinite(ddem) & (np.abs(dz) < 50)   # surveyed; drops 9999 flags
    near = np.ones((3, 3), bool)
    out = Path(out_dir).resolve() / run_name
    out.mkdir(parents=True, exist_ok=True)
    grid, crs = grid_from_raster(ELEVATION, crs=CRS)
    res = {"run": run_name, **(params or {})}
    sides = {"erosion": (-dz, ddem < -change_m, ""), "deposition": (dz, ddem > change_m, "_deposition")}
    for side, (score, observed_change, suffix) in sides.items():
        o = observed_change & valid                       # observed
        p = (score > change_m) & valid                    # modelled
        res.update({
            f"AUC{suffix}": round(float(roc_auc(score[valid], o[valid].astype(float))), 4),
            f"recall{suffix}": round(float((o & p).sum() / max(o.sum(), 1)), 4),
            f"precision{suffix}": round(float((o & p).sum() / max(p.sum(), 1)), 4),
            f"recall_1cell{suffix}": round(float((o & binary_dilation(p, near)).sum() / max(o.sum(), 1)), 4),
            f"precision_1cell{suffix}": round(float((p & binary_dilation(o, near)).sum() / max(p.sum(), 1)), 4)})
        # 0 correct no, 1 false alarm, 2 miss, 3 hit: the same codes as the hazard maps
        agree = np.where(valid, p + 2 * o, np.nan)
        name = "mwr_vs_ddem" + suffix
        write_field_geotiff(grid, crs, out / f"obs_ddem_{side}.tif", np.where(valid, o, np.nan)[::-1].ravel())
        write_field_geotiff(grid, crs, out / f"agree_MWR_vs_ddem{suffix}.tif", agree[::-1].ravel())   # rows north-up
        plt.figure(figsize=(8, 7))
        plt.imshow(agree, cmap=ListedColormap([c for _, c in OUTCOMES]),
                   norm=BoundaryNorm(np.arange(-0.5, 4), 4), interpolation="nearest")
        cb = plt.colorbar(ticks=[0, 1, 2, 3], shrink=0.7)
        cb.set_ticklabels(["neither (correct no)", "MWR only (false alarm)", "dDEM only (miss)", "both (hit)"])
        plt.title(f"MWR {side} vs dDEM {side} > {change_m} m   AUC {res[f'AUC{suffix}']:.2f}")
        plt.axis("off")
        plt.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.show()
    print(res)
    print(f"maps and GeoTIFFs written to {out}")
    _log(Path(out_dir).resolve() / "mwr_runs_log.csv", res)
    return res


def _show(table, png):
    try:
        from IPython.display import display
        display(table)
    except ImportError:
        print(table)
    _display(png)


def _display(png):
    try:
        from IPython.display import Image, display
        display(Image(str(png)))
    except ImportError:
        pass
