"""Build stable masks, then run the main corrections and optional LoD per pair.

One YAML config supports an explicit list of pairs or paired directories.
Relative paths are resolved against the config file's directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import rasterio
import yaml

from build_stable_cells_batch import (
    discover_pairs, grids_match, load_config, process_pair, write_summaries,
)


def preflight(cfg, pairs):
    """Check all pairs before creating products or launching expensive tools."""
    fitting = cfg.get("fitting", {})
    if fitting.get("land_cover"):
        with rasterio.open(fitting["land_cover"]) as cover:
            if cover.crs is None or cover.count != 1:
                raise ValueError("Fitting land cover needs a single band and CRS")
    for pair in pairs:
        with rasterio.open(pair.pre_path) as pre, rasterio.open(pair.post_path) as post:
            if pre.crs is None or post.crs is None:
                raise ValueError(f"{pair.pair_id}: both DEMs need a CRS")
            if not grids_match(pre.profile, post.profile) and not cfg.get("processing", {}).get("reproject_post_to_pre", False):
                raise ValueError(f"{pair.pair_id}: pre/post grids differ")
            if (cfg.get("terrain", {}).get("g_source", "compute_dinf") == "compute_dinf"
                    or cfg.get("lod", {}).get("method", "none") != "none"):
                if not pre.crs.is_projected or pre.crs.linear_units_factor[1] != 1:
                    raise ValueError(f"{pair.pair_id}: computing G requires a projected metre CRS")
                if pre.transform.b or pre.transform.d or pre.res[0] != pre.res[1]:
                    raise ValueError(f"{pair.pair_id}: computing G requires an unrotated square grid")
        terrain = cfg.get("terrain", {})
        if terrain.get("g_source") == "raster":
            if not terrain.get("g_dir"):
                raise ValueError("terrain.g_dir is required for g_source=raster")
            g = Path(terrain["g_dir"]) / terrain.get("g_template", "{id}_G.tif").format(id=pair.pair_id)
            if not g.is_file():
                raise FileNotFoundError(g)
    required = ["xdem", "geoutils", "seaborn"]
    if cfg.get("terrain", {}).get("g_source", "compute_dinf") == "compute_dinf":
        required.append("whitebox")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing packages: {missing}; activate xdem_env on this machine")


def correction_command(pair, mask, destination, cfg):
    quality = cfg.get("quality", {})
    lod = cfg.get("lod", {})
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("xDEM_pipeline_Thomas_production.py")),
        "--pre", str(pair.pre_path), "--post", str(pair.post_path),
        "--stable-mask", str(mask), "--stable-value", str(cfg.get("stable_cells", {}).get("stable_value", 1)),
        "--min-stable-pixels", str(quality.get("min_stable_pixels", 1000)),
        "--min-stable-fraction", str(quality.get("min_stable_fraction", 0.01)),
        "--output-dir", str(destination),
        "--correction-preset", cfg.get("corrections", {}).get("preset", "montecito"),
        "--lod-method", lod.get("method", "none"),
        "--lod-quantile", str(lod.get("quantile", .95)),
        "--validation-block-size-m", str(lod.get("block_size_m", 250.0)),
    ]
    fitting = cfg.get("fitting", {})
    if fitting.get("land_cover"):
        command += ["--fit-land-cover", fitting["land_cover"], "--fit-land-cover-classes"]
        command += [str(c) for c in fitting.get("land_cover_classes", [31, 71])]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs without writing or processing")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    pairs = discover_pairs(cfg)
    preflight(cfg, pairs)
    out = Path(cfg["batch"]["output_dir"])
    print(f"Event: {cfg['batch'].get('name', args.config.stem)}; pairs: {len(pairs)}", flush=True)
    print(f"Stable rules: {cfg.get('stable_cells', {})}", flush=True)
    print(f"Correction preset: {cfg.get('corrections', {}).get('preset', 'montecito')}", flush=True)
    print(f"Fitting land cover: {cfg.get('fitting', {}).get('land_cover') or 'unrestricted'}", flush=True)
    print(f"LoD: {cfg.get('lod', {}).get('method', 'none')}; output uses full valid DEM overlap", flush=True)
    for pair in pairs:
        print(f"  {pair.pair_id}: {pair.pre_path} -> {pair.post_path}", flush=True)
    print(f"Output: {out}", flush=True)
    if args.dry_run:
        print("Preflight passed. No processing or writes performed.")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output is not empty: {out}. Choose a new batch.output_dir for a new run.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rows = []
    continue_on_error = cfg.get("processing", {}).get("continue_on_error", False)
    for pair in pairs:
        row = {"pair_id": pair.pair_id, "status": "failed"}
        try:
            print(f"\n[{pair.pair_id}] Building stable cells", flush=True)
            row.update(process_pair(pair, cfg, out / "stable_cells"))
            row["status"] = "mask_created"
            destination = out / "corrections" / pair.pair_id
            print(f"[{pair.pair_id}] Running corrections", flush=True)
            subprocess.run(correction_command(pair, row["stable_cells_path"], destination, cfg), check=True)
            row["status"] = "ok"
            row["correction_output_dir"] = str(destination)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(row["error"], file=sys.stderr, flush=True)
        rows.append(row)
        write_summaries(rows, out / "stable_cells")
        (out / "event_summary.json").write_text(json.dumps(rows, indent=2))
        if row["status"] == "failed" and not continue_on_error:
            break
    failed = [r["pair_id"] for r in rows if r["status"] == "failed"]
    if failed:
        raise SystemExit(f"Failed pairs: {failed}; see {out / 'event_summary.json'}")
    print(f"Completed {len(rows)} pairs. Results: {out}")
    if cfg.get("lod", {}).get("method", "none") == "none":
        print("LoD filtering is disabled in this configuration.")
    else:
        print("Empirical LoD-filtered DoD and threshold report: corrections/<pair>/final_dem_products/")


if __name__ == "__main__":
    main()
