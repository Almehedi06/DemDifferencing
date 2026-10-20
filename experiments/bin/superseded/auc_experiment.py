"""AUC experiment: score Landlab layers against mapped initiation points.

One run = one config file (see pioneer_auc.yaml); change settings there, not here.
The protocol follows Strauch et al. (2018): initiation (yes) and absence (no) cells
come from the same domain, where slopes can fail, and absence cells are drawn at
random away from the initiation points and outside what the event changed.

Each run writes its own folder, output_dir/run_name:

  metrics.csv       AUC with a 95% interval and success-rate scores, per layer and type
  roc.png           one ROC curve panel per observation type
  roc_curves.csv    the ROC curves as numbers, for plotting elsewhere
  negatives.gpkg    the absence cells used, with their G, slope and dDEM
  config.yaml       this run's settings, copied
  run_info.json     cell counts, versions and input checksums

    ~/landlab-grid-validation/.venv/bin/python experiments/bin/auc_experiment.py \
        experiments/bin/pioneer_auc.yaml [--overwrite]

Run from ~/DemDifferencing with the landlab-grid-validation environment, which
has the package and PyYAML.

The 95% interval resamples the initiation points (each with its block) and keeps
the absence cells fixed: the points, not the cells, are the independent samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA = {
    "run_name": None, "output_dir": None, "crs": None, "model": None,
    "observed": {"points", "type_column", "block_cells"},
    "terrain": {"elevation", "G", "G_resampling", "ddem"},
    "domain": {"slope_min_deg", "slope_max_deg", "G_min", "G_max", "perimeter"},
    "negatives": {"G_min", "G_max", "min_distance_m", "footprint_threshold_m", "footprint_buffer_cells", "n", "seed"},
    "uncertainty": {"bootstrap", "seed"},
}
PATHS = [("observed", "points"), ("terrain", "elevation"), ("terrain", "G"), ("terrain", "ddem"),
         ("domain", "perimeter")]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # one per model layer, in order


def load_config(path: Path) -> dict:
    """Read the YAML config; every key must be known and present. Relative paths
    resolve against the config's folder."""
    import yaml

    cfg = yaml.safe_load(path.read_text())
    problems = [f"unknown key: {k}" for k in cfg if k not in SCHEMA]
    problems += [f"missing key: {k}" for k in SCHEMA if k not in cfg]
    for section, keys in SCHEMA.items():
        if keys and isinstance(cfg.get(section), dict):
            problems += [f"unknown key: {section}.{k}" for k in cfg[section] if k not in keys]
            problems += [f"missing key: {section}.{k}" for k in keys if k not in cfg[section]]
    if problems:
        raise SystemExit(f"{path}:\n  " + "\n  ".join(problems))

    def resolve(p):
        return None if p is None else str((path.parent / p).resolve()) if not Path(p).is_absolute() else p

    cfg["model"] = {name: resolve(p) for name, p in cfg["model"].items()}
    for section, key in PATHS:
        cfg[section][key] = resolve(cfg[section][key])
    cfg["output_dir"] = resolve(cfg["output_dir"])
    return cfg


def within(values, low, high):
    """values > low and values < high; a None limit is ignored. NaN fails both."""
    keep = np.isfinite(values)
    if low is not None:
        keep &= values > low
    if high is not None:
        keep &= values < high
    return keep


def perimeter_cells(grid, crs, path):
    """Cells whose centre falls inside the polygon.

    Hand-drawn perimeters often self-intersect, which makes clipping unreliable,
    so the shape is repaired before use instead of trusted as drawn.
    """
    import geopandas as gpd
    from shapely.validation import make_valid

    from landlab_grid_validation import vector_to_node_field

    shape = gpd.read_file(path).to_crs(crs)
    repaired = gpd.GeoDataFrame(geometry=[make_valid(shape.union_all())], crs=shape.crs)
    return vector_to_node_field(grid, crs, repaired, all_touched=False) == 1


def slope_degrees(grid, elevation):
    gy, gx = np.gradient(elevation.reshape(grid.shape), grid.dy, grid.dx)
    return np.degrees(np.arctan(np.hypot(gx, gy))).reshape(-1)


def curve(pos, ref):
    """For every threshold t, x = share of ref with score >= t and y = share of pos
    with score >= t. ref = absence cells gives the ROC curve; ref = the whole domain
    gives the success-rate curve. Ties join with a straight line, worth half."""
    t = np.unique(np.r_[pos, ref])[::-1]
    y = 1 - np.searchsorted(np.sort(pos), t, "left") / pos.size
    x = 1 - np.searchsorted(np.sort(ref), t, "left") / ref.size
    return np.r_[0.0, x], np.r_[0.0, y]


def area(x, y):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def bootstrap_auc(scores, point_cells, neg_scores, n_boot, seed):
    """95% interval of the AUC from resampling the initiation points with replacement.
    Each positive cell's placement (share of absence cells below it, ties half) is
    computed once, so each resample is a weighted mean. Same seed for every layer,
    so layers are compared on the same resamples."""
    neg = np.sort(neg_scores)
    sums, counts = [], []
    for cells in point_cells:
        s = scores[cells]
        placement = (np.searchsorted(neg, s, "left") + np.searchsorted(neg, s, "right")) / 2 / neg.size
        sums.append(placement.sum())
        counts.append(cells.size)
    sums, counts = np.array(sums), np.array(counts)
    pick = np.random.default_rng(seed).integers(0, sums.size, size=(n_boot, sums.size))
    aucs = sums[pick].sum(1) / counts[pick].sum(1)
    return np.percentile(aucs, [2.5, 97.5])


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def figure(curves, types, layers, n_points, path):
    """One ROC panel per observation type (true-positive rate against false-positive
    rate), up to three panels per row. The success-rate curve is not drawn: with
    absence cells making up nearly the whole domain it repeats the ROC; its area is
    still in metrics.csv.

    Built with Figure directly, not pyplot: this runs inside notebooks, and switching
    the backend or leaving a pyplot figure open would stop the notebook's later plots."""
    from matplotlib.figure import Figure

    ncols = min(3, len(types))
    nrows = -(-len(types) // ncols)
    fig = Figure(figsize=(4.8 * ncols, 4.4 * nrows), constrained_layout=True)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.ravel()[len(types):]:
        ax.set_visible(False)
    for ax, code in zip(axes.ravel(), types):
        ax.plot([0, 1], [0, 1], "--", color="#8a8a85", lw=1)
        for colour, layer in zip(COLOURS, layers):
            x, y, value = curves[(code, layer, "roc")]
            ax.plot(x, y, color=colour, lw=2, label=f"{layer}  (AUC {value:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("false-positive rate (share of absence cells flagged)")
        ax.set_ylabel("true-positive rate (share of initiation cells flagged)")
        ax.set_title(f"{code}, {n_points[code]} points", loc="left", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(color="#e6e6e3", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    out = Path(cfg["output_dir"]) / cfg["run_name"]
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} already holds a run; choose a new run_name or pass --overwrite")

    import geopandas as gpd
    import landlab
    from scipy.ndimage import binary_dilation, distance_transform_edt

    import landlab_grid_validation as lgv
    from landlab_grid_validation import grid_from_raster, raster_to_node_field, roc_auc

    # ---- everything onto the model grid
    layer_names = list(cfg["model"])
    grid, crs = grid_from_raster(cfg["model"][layer_names[0]], crs=cfg["crs"])

    def read(path, **kw):
        return raster_to_node_field(grid, crs, path, raster_crs=cfg["crs"], **kw)

    layers = {name: read(path) for name, path in cfg["model"].items()}
    terrain = cfg["terrain"]
    slope = slope_degrees(grid, read(terrain["elevation"]))
    G = read(terrain["G"], resampling=terrain["G_resampling"]) if terrain["G"] else None
    ddem = read(terrain["ddem"], resampling="average") if terrain["ddem"] else None

    def need(values, what):
        if values is None:
            raise SystemExit(f"{what} is set in the config but its raster is null")
        return values

    # ---- the domain: where both classes may come from
    valid = np.isfinite(slope) & np.all([np.isfinite(v) for v in layers.values()], axis=0)
    if ddem is not None:
        valid &= np.isfinite(ddem)                     # the surveyed area: only there were points mapped
    dom = cfg["domain"]
    domain = valid & within(slope, dom["slope_min_deg"], dom["slope_max_deg"])
    if dom["G_min"] is not None or dom["G_max"] is not None:
        domain &= within(need(G, "domain G limit"), dom["G_min"], dom["G_max"])
    if dom["perimeter"]:
        domain &= perimeter_cells(grid, crs, dom["perimeter"])

    # ---- initiation points as blocks of cells
    obs = cfg["observed"]
    points = gpd.read_file(obs["points"]).to_crs(crs)
    node = grid.find_nearest_node((points.geometry.x.to_numpy(), points.geometry.y.to_numpy()))
    rows, cols = np.divmod(node, grid.shape[1])
    k = int(obs["block_cells"])
    blocks_full = []
    for r, c in zip(rows, cols):
        rr, cc = np.meshgrid(np.arange(r - k, r + k + 1), np.arange(c - k, c + k + 1), indexing="ij")
        inside = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
        blocks_full.append((rr[inside] * grid.shape[1] + cc[inside]).ravel())
    blocks = [b[domain[b]] for b in blocks_full]       # only the block cells inside the domain
    any_block = np.zeros(grid.number_of_nodes, bool)
    any_block[np.concatenate(blocks_full)] = True

    kinds = points[obs["type_column"]].astype(str).to_numpy()
    types = ["All"] + sorted(set(kinds))
    members = {code: [i for i in range(len(points)) if code == "All" or kinds[i] == code] for code in types}

    # ---- absence cells: domain, plus the rules for negatives only
    neg_cfg = cfg["negatives"]
    pool = domain & ~any_block
    if neg_cfg["G_min"] is not None or neg_cfg["G_max"] is not None:
        pool &= within(need(G, "negatives G limit"), neg_cfg["G_min"], neg_cfg["G_max"])
    if neg_cfg["min_distance_m"]:
        at_point = np.zeros(grid.shape, bool)
        at_point[rows, cols] = True
        distance = distance_transform_edt(~at_point, sampling=(grid.dy, grid.dx)).reshape(-1)
        pool &= distance >= neg_cfg["min_distance_m"]
    footprint = np.zeros(grid.number_of_nodes, bool)
    if neg_cfg["footprint_threshold_m"] is not None:
        moved = np.nan_to_num(np.abs(need(ddem, "footprint_threshold_m"))) > neg_cfg["footprint_threshold_m"]
        if neg_cfg["footprint_buffer_cells"]:
            moved = binary_dilation(moved.reshape(grid.shape), structure=np.ones((3, 3), bool),
                                    iterations=int(neg_cfg["footprint_buffer_cells"])).reshape(-1)
        footprint = moved
        pool &= ~footprint
    candidates = np.flatnonzero(pool)
    if neg_cfg["n"] == "all":
        negatives = candidates
    else:
        n = int(neg_cfg["n"])
        if n > candidates.size:
            raise SystemExit(f"negatives.n = {n} but only {candidates.size} cells qualify")
        negatives = np.sort(np.random.default_rng(neg_cfg["seed"]).choice(candidates, n, replace=False))
    if negatives.size == 0:
        raise SystemExit("no absence cells qualify; loosen the negatives rules")

    # ---- score
    unc = cfg["uncertainty"]
    results, curves, n_points = [], {}, {}
    for code in types:
        point_cells = [blocks[i] for i in members[code] if blocks[i].size]
        n_points[code] = len(point_cells)
        yes = np.zeros(grid.number_of_nodes, bool)
        yes[np.concatenate(point_cells)] = True
        for name, values in layers.items():
            pos, neg = values[yes], values[negatives]
            auc = roc_auc(np.r_[pos, neg], np.r_[np.ones(pos.size), np.zeros(neg.size)])
            x, y = curve(pos, neg)
            if not np.isclose(area(x, y), auc):
                raise RuntimeError(f"ROC area {area(x, y)} disagrees with the rank AUC {auc}")
            low, high = bootstrap_auc(values, point_cells, neg, int(unc["bootstrap"]), unc["seed"])
            sx, sy = curve(pos, values[domain])
            curves[(code, name, "roc")] = (x, y, auc)
            curves[(code, name, "sr")] = (sx, sy, area(sx, sy))
            results.append({
                "layer": name, "type": code, "points": len(point_cells), "yes_cells": int(yes.sum()),
                "no_cells": int(negatives.size), "auc": round(auc, 4), "auc_low": round(float(low), 4),
                "auc_high": round(float(high), 4), "success_rate_area": round(area(sx, sy), 4),
                "captured_in_top_10pct_area": round(float(np.interp(0.10, sx, sy)), 4),
                "captured_in_top_20pct_area": round(float(np.interp(0.20, sx, sy)), 4),
            })

    # ---- write the run
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    figure(curves, types, layer_names, n_points, out / "roc.png")
    with open(out / "roc_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "layer", "auc", "fpr", "tpr"])
        for (code, name, kind), (x, y, value) in curves.items():
            if kind == "roc":
                writer.writerows([code, name, round(value, 6), round(float(a), 6), round(float(b), 6)]
                                 for a, b in zip(x, y))
    table = {"slope_deg": slope[negatives]}
    if G is not None:
        table["G"] = G[negatives]
    if ddem is not None:
        table["ddem_m"] = ddem[negatives]
    gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(grid.x_of_node[negatives], grid.y_of_node[negatives]),
                     crs=crs).to_file(out / "negatives.gpkg")
    shutil.copyfile(args.config, out / "config.yaml")

    inputs = list(cfg["model"].values()) + [cfg[s][k] for s, k in PATHS if cfg[s][k]]
    outside_neg_G = None
    if G is not None and (neg_cfg["G_min"] is not None or neg_cfg["G_max"] is not None):
        yes_all = np.zeros(grid.number_of_nodes, bool)
        yes_all[np.concatenate(blocks)] = True
        outside_neg_G = round(float(np.mean(~within(G[yes_all], neg_cfg["G_min"], neg_cfg["G_max"]))), 4)
    info = {
        "run": cfg["run_name"], "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "cells": {"grid": int(grid.number_of_nodes), "valid": int(valid.sum()), "domain": int(domain.sum()),
                  "footprint": int((footprint & domain).sum()), "negatives_pool": int(candidates.size),
                  "negatives_used": int(negatives.size)},
        "points": {code: {"listed": len(members[code]), "with_cells_in_domain": n_points[code]} for code in types},
        "initiation_cells_outside_negatives_G_range": outside_neg_G,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "landlab": landlab.__version__, "landlab_grid_validation": lgv.__version__},
        "inputs": {p: {"bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in inputs},
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))

    print(f"run {cfg['run_name']}: {info['cells']['domain']} domain cells, {negatives.size} absence cells")
    if outside_neg_G is not None:
        print(f"note: {outside_neg_G:.0%} of initiation cells lie outside the negatives' G range")
    print(f"{'type':10s}" + "".join(f"{n:>26s}" for n in layer_names))
    for code in types:
        cells = [r for r in results if r["type"] == code]
        print(f"{code:10s}" + "".join(f"{r['auc']:>11.3f} ({r['auc_low']:.2f}-{r['auc_high']:.2f})  " for r in cells))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
