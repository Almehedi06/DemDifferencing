"""Pioneer evaluation: score Landlab outputs against what happened in Dec 2025.

    from pioneer_eval import evaluate_hazard, evaluate_mwr
    evaluate_hazard(".", run_name="baseline")     # after P_*.asc are written
    evaluate_mwr("MWR.asc", run_name="baseline")  # after MWR.asc is written

Results go to evaluation/<run_name>/ next to the notebook, plus one row per run in
evaluation/runs_log.csv (hazard) or evaluation/mwr_runs_log.csv (runout). Each run
also writes maps (PNG) and GeoTIFFs for QGIS: the observations as yes/no rasters and
the agreement maps (0 correct no, 1 false alarm, 2 miss, 3 hit).
Observations are in data/: the 89 mapped initiation points and the corrected dDEM.
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))  # the bundled landlab_grid_validation and auc_experiment

CRS = "EPSG:6339"  # NAD83(2011) UTM 10N; the .asc files do not record it
LAYERS = ["P_Landslide", "P_Channel_ini", "P_Combined_H"]
POINTS = DATA / "Initiation_Points_final.shp"
DDEM = DATA / "dod_dndvi_test.tif"
ELEVATION = DATA / "topographic__elevation.asc"

# Strauch et al. (2018), adapted to points and the dDEM. Do not change between runs,
# or scores stop being comparable.
SETTINGS = {
    "observed": {"type_column": "FailureTyp", "block_cells": 1},  # 3 x 3 cells (30 m) per point
    "domain": {"slope_min_deg": 17, "slope_max_deg": None, "G_min": None, "G_max": None,
               "perimeter": None},
    "negatives": {"G_min": None, "G_max": None,
                  "min_distance_m": 50,          # absences at least 50 m from any point
                  "footprint_threshold_m": 0.5,  # and outside cells the event changed by > 0.5 m
                  "footprint_buffer_cells": 1, "n": "all", "seed": 42},
    "uncertainty": {"bootstrap": 1000, "seed": 42},
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
    mapped initiation points. Prints AUC [95% interval] by failure type, shows the ROC
    and success-rate curves, and logs the run. params: optional dict of model settings
    to keep with the scores, e.g. {"Csoil": Csoil, "tau_c_low": 1.1}.

    maps=True also draws each layer against the points on the Landlab grid and writes
    GeoTIFFs: the observations as yes/no rasters and one agreement map per layer, with
    a cell called "yes" where the probability is at least map_threshold. The AUC does
    not use a threshold, so map_threshold changes the maps only, never the scores."""
    import yaml
    import auc_experiment

    run_name = _run_name(run_name, "run")
    model_dir, out_dir = Path(model_dir).resolve(), Path(out_dir).resolve()
    cfg = {"run_name": run_name, "output_dir": str(out_dir), "crs": CRS,
           "model": {name: str(model_dir / f"{name}.asc") for name in LAYERS},
           "observed": {"points": str(POINTS), **SETTINGS["observed"]},
           "terrain": {"elevation": str(ELEVATION), "G": None,
                       "G_resampling": "nearest", "ddem": str(DDEM)},
           **{k: SETTINGS[k] for k in ("domain", "negatives", "uncertainty")}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    auc_experiment.main([f.name])

    run_dir = out_dir / run_name
    m = pd.read_csv(run_dir / "metrics.csv")
    m["AUC [95%]"] = (m.auc.map("{:.2f}".format) + " [" + m.auc_low.map("{:.2f}".format)
                      + "-" + m.auc_high.map("{:.2f}".format) + "]")
    table = m.pivot(index="type", columns="layer", values="AUC [95%]")[LAYERS]
    _show(table, run_dir / "roc_success.png")

    if maps:
        _hazard_maps(model_dir, run_dir, map_threshold)

    row = {"run": run_name, **(params or {})}
    row.update({f"AUC_{k}": m.query("type == 'All' and layer == @k").auc.item() for k in LAYERS})
    _log(out_dir / "runs_log.csv", row)
    return m


def _hazard_maps(model_dir, run_dir, threshold):
    """Each layer against the points on the Landlab grid, as PNG and GeoTIFF."""
    import matplotlib.pyplot as plt

    from landlab_grid_validation import (FieldSpec, compare_fields_on_grid, grid_from_raster,
                                         plot_comparison, raster_to_node_field,
                                         write_field_geotiff)

    grid, crs = grid_from_raster(model_dir / f"{LAYERS[0]}.asc", crs=CRS)
    surveyed = np.isfinite(raster_to_node_field(grid, crs, DDEM, resampling="average"))
    points, observed = _observations(grid, crs)
    # 1 = within 30 m of a point, 0 = not, empty where the drone did not fly
    observed = {code: np.where(surveyed, values, np.nan) for code, values in observed.items()}

    for code, values in observed.items():
        write_field_geotiff(grid, crs, run_dir / f"obs_points_{code}.tif", values)

    for name in LAYERS:
        layer = raster_to_node_field(grid, crs, model_dir / f"{name}.asc", raster_crs=CRS)
        result = compare_fields_on_grid(
            grid, layer, observed["All"],
            FieldSpec(name=name, kind="continuous", units="probability", threshold=threshold),
            FieldSpec(name="mapped initiations (all types)", kind="binary"),
        )
        write_field_geotiff(grid, crs, run_dir / f"agree_{name}.tif", result.agreement())
        fig = plot_comparison(grid, result, points=points)
        # the scores are in the table: this map covers the whole survey, the AUC only
        # slopes over 17 degrees, so a number here would disagree with the table
        fig.suptitle(f"{name} vs the 89 mapped initiations, whole drone survey, "
                     f"'yes' where P >= {threshold:g}", fontsize=12)
        fig.savefig(run_dir / f"map_{name}.png", dpi=150)
        plt.close(fig)
        _display(run_dir / f"map_{name}.png")
    print(f"maps and GeoTIFFs written to {run_dir}")


def _observations(grid, crs):
    """The mapped points, and yes/no node fields: all types together, then each type."""
    import geopandas as gpd

    from landlab_grid_validation import vector_to_node_field

    points = gpd.read_file(POINTS).to_crs(crs)
    k, column = SETTINGS["observed"]["block_cells"], SETTINGS["observed"]["type_column"]
    observed = {"All": vector_to_node_field(grid, crs, points, tolerance_cells=k)}
    for code in sorted(points[column].unique()):
        observed[code] = vector_to_node_field(grid, crs, points[points[column] == code],
                                              tolerance_cells=k)
    return points, observed


def evaluate_mwr(mwr_asc="MWR.asc", run_name=None, params=None, out_dir="evaluation", erosion_m=0.5):
    """Compare MWR erosion (MWR.asc, elevation change in m) with dDEM erosion: a cell is
    eroded where the dDEM drops more than erosion_m. Prints AUC, recall and precision
    (exact, and allowing a one-cell offset), maps hits and misses, and logs the run.
    Also writes GeoTIFFs: dDEM erosion as yes/no, and the agreement map."""
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

    score = -dz                                   # erosion depth, m
    valid = (dem != -9999) & np.isfinite(ddem) & (np.abs(dz) < 50)   # surveyed; drops 9999 flags
    o = (ddem < -erosion_m) & valid               # observed erosion
    p = (score > erosion_m) & valid               # modelled erosion
    near = np.ones((3, 3), bool)
    res = {"run": run_name, **(params or {}),
           "AUC": round(float(roc_auc(score[valid], o[valid].astype(float))), 4),
           "recall": round(float((o & p).sum() / o.sum()), 4),
           "precision": round(float((o & p).sum() / max(p.sum(), 1)), 4),
           "recall_1cell": round(float((o & binary_dilation(p, near)).sum() / o.sum()), 4),
           "precision_1cell": round(float((p & binary_dilation(o, near)).sum() / max(p.sum(), 1)), 4)}
    print(res)

    # 0 correct no, 1 false alarm, 2 miss, 3 hit: the same codes as the hazard maps
    agree = np.where(valid, p + 2 * o, np.nan)
    eroded = np.where(valid, o, np.nan)

    out = Path(out_dir).resolve() / run_name
    out.mkdir(parents=True, exist_ok=True)
    grid, crs = grid_from_raster(ELEVATION, crs=CRS)
    for name, rows in (("obs_ddem_erosion", eroded), ("agree_MWR_vs_ddem", agree)):
        write_field_geotiff(grid, crs, out / f"{name}.tif", rows[::-1].ravel())   # rows north-up

    plt.figure(figsize=(8, 7))
    plt.imshow(agree, cmap=ListedColormap([c for _, c in OUTCOMES]),
               norm=BoundaryNorm(np.arange(-0.5, 4), 4), interpolation="nearest")
    cb = plt.colorbar(ticks=[0, 1, 2, 3], shrink=0.7)
    cb.set_ticklabels(["neither (correct no)", "MWR only (false alarm)",
                       "dDEM only (miss)", "both (hit)"])
    plt.title(f"MWR erosion vs dDEM erosion > {erosion_m} m   AUC {res['AUC']:.2f}")
    plt.axis("off")
    plt.savefig(out / "mwr_vs_ddem.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"map and GeoTIFFs written to {out}")
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
