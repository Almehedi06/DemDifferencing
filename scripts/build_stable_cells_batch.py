"""Build stable-cell masks for a batch of pre/post DEM pairs.

The mask is intended for xDEM coregistration/uncertainty calibration. It does
not change downstream C2/C3 evidence logic; it only defines candidate stable
terrain used upstream.

Default Eagle Creek rule:
    stable = valid(pre, post, G) AND G_pre < 5 AND pre_elevation >= 600

Run:
    python scripts/build_stable_cells_batch.py \
        --config configs/eagle_creek_2018_2021_stable_cells.yaml

Dry run:
    python scripts/build_stable_cells_batch.py \
        --config configs/eagle_creek_2018_2021_stable_cells.yaml --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject


@dataclass(frozen=True)
class DemPair:
    pair_id: str
    pre_path: Path
    post_path: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = path.resolve().parent
    def resolve(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else base / p)
    batch = cfg["batch"]
    for key in ("pre_dir", "post_dir", "output_dir"):
        if batch.get(key):
            batch[key] = resolve(batch[key])
    for pair in batch.get("pairs", []):
        for key in ("pre", "post"):
            pair[key] = resolve(pair[key])
    terrain = cfg.get("terrain", {})
    if terrain.get("g_dir"):
        terrain["g_dir"] = resolve(terrain["g_dir"])
    fitting = cfg.get("fitting", {})
    if fitting.get("land_cover"):
        fitting["land_cover"] = resolve(fitting["land_cover"])
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if not cfg.get("batch", {}).get("output_dir"):
        raise ValueError("batch.output_dir is required")
    processing = cfg.get("processing", {})
    if processing.get("reference", "pre") != "pre":
        raise ValueError("processing.reference must be pre: G and masks are defined on the pre grid")
    terrain = cfg.get("terrain", {})
    if terrain.get("g_source", "compute_dinf") not in ("compute_dinf", "raster"):
        raise ValueError("terrain.g_source must be compute_dinf or raster")
    alpha = terrain.get("alpha", 1.167)
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("terrain.alpha must be finite and positive")
    stable = cfg.get("stable_cells", {})
    allowed = {"g_min", "g_max", "elev_min", "elev_max", "slope_min", "slope_max",
               "dh_abs_max", "stable_value", "unstable_value", "nodata_value"}
    if set(stable) - allowed:
        raise ValueError(f"Unknown stable_cells keys: {sorted(set(stable) - allowed)}")
    for key in allowed - {"stable_value", "unstable_value", "nodata_value"}:
        value = stable.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
            raise ValueError(f"stable_cells.{key} must be a finite number or null")
    for prefix in ("g", "elev", "slope"):
        low, high = stable.get(prefix + "_min"), stable.get(prefix + "_max")
        if low is not None and high is not None and low >= high:
            raise ValueError(f"stable_cells.{prefix}_min must be less than {prefix}_max")
    if stable.get("dh_abs_max") is not None and stable["dh_abs_max"] <= 0:
        raise ValueError("stable_cells.dh_abs_max must be positive or null")
    codes = [stable.get(k, d) for k, d in (("stable_value", 1), ("unstable_value", 0), ("nodata_value", 255))]
    if any(type(v) is not int or not 0 <= v <= 255 for v in codes) or len(set(codes)) != 3:
        raise ValueError("Mask codes must be three distinct integers from 0 to 255")
    quality = cfg.get("quality", {})
    pixels, fraction = quality.get("min_stable_pixels", 1000), quality.get("min_stable_fraction", 0.01)
    if type(pixels) is not int or pixels < 1:
        raise ValueError("quality.min_stable_pixels must be a positive integer")
    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError("quality.min_stable_fraction must be between 0 and 1")
    dirname = cfg["batch"].get("mask_dirname", "masks")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dirname):
        raise ValueError("batch.mask_dirname must be a simple folder name")
    sections = {"corrections": {"preset"}, "fitting": {"land_cover", "land_cover_classes"},
                "lod": {"method", "quantile", "block_size_m"}}
    for name, keys in sections.items():
        if not isinstance(cfg.get(name, {}), dict) or set(cfg.get(name, {})) - keys:
            raise ValueError(f"Unknown or invalid {name} settings")
    if cfg.get("corrections", {}).get("preset", "montecito") not in ("montecito", "translation"):
        raise ValueError("corrections.preset must be montecito or translation")
    lod = cfg.get("lod", {})
    if lod.get("method", "none") not in ("none", "empirical"):
        raise ValueError("lod.method must be none or empirical")
    for key, default in (("quantile", .95), ("block_size_m", 250.0)):
        value = lod.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"lod.{key} must be positive and finite")
    if lod.get("quantile", .95) >= 1:
        raise ValueError("lod.quantile must be less than one")
    classes = cfg.get("fitting", {}).get("land_cover_classes", [31, 71])
    if not isinstance(classes, list) or not classes or any(type(c) is not int or c < 0 or c >= 2**24 for c in classes):
        raise ValueError("fitting.land_cover_classes must contain nonnegative integer class codes below 2**24")


def discover_pairs(cfg: dict[str, Any]) -> list[DemPair]:
    batch = cfg["batch"]
    if "pairs" in batch:
        if any(k in batch for k in ("pre_dir", "post_dir")):
            raise ValueError("Use batch.pairs or directory discovery, not both")
        pairs = [DemPair(str(p["id"]), Path(p["pre"]), Path(p["post"])) for p in batch["pairs"]]
        ids = [p.pair_id for p in pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate pair IDs")
    else:
        pairs = _discover_directory_pairs(batch)
    for pair in pairs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", pair.pair_id):
            raise ValueError(f"Invalid pair ID: {pair.pair_id!r}")
        for p in (pair.pre_path, pair.post_path):
            if not p.is_file():
                raise FileNotFoundError(p)
        if pair.pre_path.resolve() == pair.post_path.resolve():
            raise ValueError(f"Pair {pair.pair_id}: pre and post must differ")
    requested = batch.get("pair_ids")
    if requested:
        requested = set(map(str, requested))
        missing = requested - {p.pair_id for p in pairs}
        if missing:
            raise ValueError(f"Requested pairs not found: {sorted(missing)}")
        pairs = [p for p in pairs if p.pair_id in requested]
    if not pairs:
        raise ValueError("No DEM pairs found; check filenames and pair configuration")
    return sorted(pairs, key=lambda p: _natural_key(p.pair_id))


def _discover_directory_pairs(batch: dict) -> list[DemPair]:
    pre_dir = Path(batch["pre_dir"]).expanduser()
    post_dir = Path(batch["post_dir"]).expanduser()
    pre_regex = re.compile(str(batch.get("pre_id_regex", r"^(?P<id>.+)_Pre\.tif$")))
    post_regex = re.compile(str(batch.get("post_id_regex", r"^(?P<id>.+)_Post\.tif$")))

    pre = _index_files(pre_dir, str(batch.get("pre_glob", "*_Pre.tif")), pre_regex)
    post = _index_files(post_dir, str(batch.get("post_glob", "*_Post.tif")), post_regex)

    ids = set(pre) & set(post)
    if set(pre) != set(post):
        raise ValueError(f"Unmatched DEM IDs: pre-only={sorted(set(pre)-set(post))}, post-only={sorted(set(post)-set(pre))}")

    return [
        DemPair(pair_id=i, pre_path=pre[i], post_path=post[i])
        for i in sorted(ids, key=_natural_key)
    ]


def _index_files(folder: Path, pattern: str, regex: re.Pattern[str]) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    out: dict[str, Path] = {}
    for path in folder.glob(pattern):
        match = regex.match(path.name)
        if match is None:
            continue
        pair_id = str(match.group("id"))
        if pair_id in out:
            raise ValueError(f"Duplicate DEM ID {pair_id} in {folder}")
        out[pair_id] = path
    return out


def _natural_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = src.profile.copy()
    return arr, profile


def write_uint8_mask(path: Path, mask: np.ndarray, valid: np.ndarray, profile: dict[str, Any], cfg: dict[str, Any]) -> None:
    stable_cfg = cfg["stable_cells"]
    stable_value = int(stable_cfg.get("stable_value", 1))
    unstable_value = int(stable_cfg.get("unstable_value", 0))
    nodata_value = int(stable_cfg.get("nodata_value", 255))

    out = np.full(mask.shape, nodata_value, dtype="uint8")
    out[valid] = unstable_value
    out[mask] = stable_value

    prof = _clean_profile(profile)
    prof.update(driver="GTiff", dtype="uint8", count=1, nodata=nodata_value, compress="lzw")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.update_tags(stable_value=str(stable_value), unstable_value=str(unstable_value),
                        selection=json.dumps(stable_cfg, sort_keys=True),
                        terrain=json.dumps(cfg.get("terrain", {}), sort_keys=True))


def write_float_raster(path: Path, arr: np.ndarray, profile: dict[str, Any]) -> None:
    prof = _clean_profile(profile)
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def stable_mask_dir(out_dir: Path, cfg: dict[str, Any]) -> Path:
    return out_dir / str(cfg.get("batch", {}).get("mask_dirname", "masks"))


def _clean_profile(profile: dict[str, Any]) -> dict[str, Any]:
    prof = profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        prof.pop(key, None)
    return prof


def grids_match(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-6) -> bool:
    if a.get("crs") != b.get("crs"):
        return False
    if int(a.get("height")) != int(b.get("height")) or int(a.get("width")) != int(b.get("width")):
        return False
    at = a["transform"]
    bt = b["transform"]
    return all(abs(x - y) <= tol for x, y in zip(at[:6], bt[:6]))


def reproject_to_profile(src_path: Path, dst_profile: dict[str, Any]) -> np.ndarray:
    nodata = -9999.0
    dst = np.full((dst_profile["height"], dst_profile["width"]), nodata, dtype="float32")
    with rasterio.open(src_path) as src:
        src_arr = src.read(1, masked=True).astype("float32").filled(nodata)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
    dst[dst == nodata] = np.nan
    return dst.astype("float64")


def compute_g_for_pair(pair: DemPair, pre_arr: np.ndarray, pre_profile: dict[str, Any], out_dir: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    terrain = cfg.get("terrain", {})
    source = str(terrain.get("g_source", "compute_dinf")).lower()

    if source == "raster":
        g_dir = terrain.get("g_dir")
        if not g_dir:
            raise ValueError("terrain.g_source='raster' requires terrain.g_dir")
        g_path = Path(g_dir).expanduser() / str(terrain.get("g_template", "{id}_G.tif")).format(id=pair.pair_id)
        if not g_path.exists():
            raise FileNotFoundError(f"G raster not found for pair {pair.pair_id}: {g_path}")
        g_arr, g_profile = read_raster(g_path)
        if not grids_match(pre_profile, g_profile):
            g_arr = reproject_to_profile(g_path, pre_profile)
        return g_arr, None

    if source != "compute_dinf":
        raise ValueError(f"Unsupported terrain.g_source: {source!r}")

    return compute_g_dinf(pair.pre_path, out_dir / "terrain" / pair.pair_id, float(terrain.get("alpha", 1.167)), bool(terrain.get("keep_intermediates", True)))


def compute_g_dinf(dem_path: Path, work_dir: Path, alpha: float, keep_intermediates: bool) -> tuple[np.ndarray, np.ndarray]:
    try:
        import whitebox
    except ImportError as exc:
        raise ImportError(
            "whitebox is required for terrain.g_source='compute_dinf'. "
            "Install it in the active environment or provide precomputed G rasters."
        ) from exc

    with rasterio.open(dem_path) as src:
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units_factor[1] != 1.0:
            raise ValueError("Computing G requires a projected CRS with metre units and elevations in metres")
        if src.transform.b != 0 or src.transform.d != 0 or not np.isclose(src.res[0], src.res[1]):
            raise ValueError("Computing G requires an unrotated grid with square pixels")
    work_dir = work_dir.resolve()
    dem_path = dem_path.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    filled = work_dir / "pre_filled.tif"
    slope_pct_path = work_dir / "pre_slope_percent.tif"
    sca_path = work_dir / "pre_dinf_sca.tif"
    g_path = work_dir / "G_pre_dinf.tif"

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    def run(tool, output, *args, **kwargs):
        output.unlink(missing_ok=True)
        status = tool(*args, **kwargs)
        if status != 0 or not output.is_file():
            raise RuntimeError(f"Whitebox {tool.__name__} failed (status {status}); expected {output}")
    run(wbt.fill_depressions_wang_and_liu, filled, str(dem_path), str(filled), fix_flats=True)
    run(wbt.slope, slope_pct_path, str(filled), str(slope_pct_path), units="percent")
    run(wbt.d_inf_flow_accumulation, sca_path, str(filled), str(sca_path), out_type="sca")

    slope_pct, profile = read_raster(slope_pct_path)
    sca, _ = read_raster(sca_path)
    slope = slope_pct / 100.0
    g_arr = sca * np.power(slope, alpha)
    g_arr[~np.isfinite(g_arr)] = np.nan
    g_arr[g_arr < 0] = np.nan

    write_float_raster(g_path, g_arr, profile)

    if not keep_intermediates:
        for path in (filled, slope_pct_path, sca_path):
            path.unlink(missing_ok=True)

    return g_arr, slope


def slope_from_dem(dem: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    dx = abs(float(profile["transform"].a))
    dy = abs(float(profile["transform"].e))
    dz_dy, dz_dx = np.gradient(dem, dy, dx)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    slope[~np.isfinite(slope)] = np.nan
    return slope


def build_stable_mask(
    pre: np.ndarray,
    post: np.ndarray,
    g_arr: np.ndarray,
    slope: np.ndarray | None,
    pre_profile: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stable_cfg = cfg["stable_cells"]
    valid = np.isfinite(pre) & np.isfinite(post) & np.isfinite(g_arr)
    stable = valid.copy()

    rules: dict[str, Any] = {}
    stable = _apply_min(stable, g_arr, stable_cfg.get("g_min"), "g_min", rules)
    stable = _apply_max(stable, g_arr, stable_cfg.get("g_max"), "g_max", rules)
    stable = _apply_min(stable, pre, stable_cfg.get("elev_min"), "elev_min", rules)
    stable = _apply_max(stable, pre, stable_cfg.get("elev_max"), "elev_max", rules)

    if stable_cfg.get("slope_min") is not None or stable_cfg.get("slope_max") is not None:
        if slope is None:
            slope = slope_from_dem(pre, pre_profile)
        valid &= np.isfinite(slope)
        stable &= np.isfinite(slope)
        stable = _apply_min(stable, slope, stable_cfg.get("slope_min"), "slope_min", rules)
        stable = _apply_max(stable, slope, stable_cfg.get("slope_max"), "slope_max", rules)

    dh_abs_max = stable_cfg.get("dh_abs_max")
    if dh_abs_max is not None:
        dh = post - pre
        stable &= np.isfinite(dh) & (np.abs(dh) < float(dh_abs_max))
        rules["dh_abs_max"] = float(dh_abs_max)

    valid_count = int(np.count_nonzero(valid))
    stable_count = int(np.count_nonzero(stable))
    stable_fraction = stable_count / valid_count if valid_count else np.nan

    stats = {
        "valid_pixels": valid_count,
        "stable_pixels": stable_count,
        "stable_fraction": stable_fraction,
        "rules": rules,
    }
    return stable, valid, stats


def _apply_min(mask: np.ndarray, arr: np.ndarray, value: Any, label: str, rules: dict[str, Any]) -> np.ndarray:
    if value is None:
        return mask
    rules[label] = float(value)
    return mask & (arr >= float(value))


def _apply_max(mask: np.ndarray, arr: np.ndarray, value: Any, label: str, rules: dict[str, Any]) -> np.ndarray:
    if value is None:
        return mask
    rules[label] = float(value)
    return mask & (arr < float(value))


def process_pair(pair: DemPair, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    pre, pre_profile = read_raster(pair.pre_path)
    post, post_profile = read_raster(pair.post_path)

    processing = cfg.get("processing", {})
    if not grids_match(pre_profile, post_profile):
        if bool(processing.get("reproject_post_to_pre", False)):
            post = reproject_to_profile(pair.post_path, pre_profile)
        else:
            raise ValueError("pre/post grids do not match; set processing.reproject_post_to_pre=true to resample post to pre")

    g_arr, slope = compute_g_for_pair(pair, pre, pre_profile, out_dir, cfg)
    stable, valid, stats = build_stable_mask(pre, post, g_arr, slope, pre_profile, cfg)

    quality = cfg.get("quality", {})
    min_pixels = int(quality.get("min_stable_pixels", 1000))
    min_fraction = float(quality.get("min_stable_fraction", 0.01))
    warning = ""
    if stats["stable_pixels"] < min_pixels:
        warning = f"stable_pixels_below_{min_pixels}"
    elif np.isfinite(stats["stable_fraction"]) and stats["stable_fraction"] < min_fraction:
        warning = f"stable_fraction_below_{min_fraction}"
    if warning or stats["valid_pixels"] == 0:
        raise ValueError(f"Pair {pair.pair_id}: insufficient stable terrain ({stats}); {warning}")

    stable_path = stable_mask_dir(out_dir, cfg) / f"{pair.pair_id}_stable_cells.tif"
    write_uint8_mask(stable_path, stable, valid, pre_profile, cfg)

    return {
        "pair_id": pair.pair_id,
        "status": "ok",
        "warning": warning,
        "pre_path": str(pair.pre_path),
        "post_path": str(pair.post_path),
        "stable_cells_path": str(stable_path),
        "valid_pixels": stats["valid_pixels"],
        "stable_pixels": stats["stable_pixels"],
        "stable_fraction": stats["stable_fraction"],
        "rules": stats["rules"],
    }


def write_summaries(rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = summary_dir / "stable_cells_summary.csv"
    fieldnames = [
        "pair_id",
        "status",
        "warning",
        "valid_pixels",
        "stable_pixels",
        "stable_fraction",
        "pre_path",
        "post_path",
        "stable_cells_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    json_path = summary_dir / "stable_cells_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    pair_index = summary_dir / "xdem_pair_index.csv"
    with pair_index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "pre_path", "post_path", "stable_cells_path"])
        writer.writeheader()
        for row in rows:
            if row.get("status") == "ok":
                writer.writerow(
                    {
                        "pair_id": row["pair_id"],
                        "pre_path": row["pre_path"],
                        "post_path": row["post_path"],
                        "stable_cells_path": row["stable_cells_path"],
                    }
                )


def parse_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stable-cell masks for pre/post DEM pairs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ids", default=None, help="Comma-separated pair IDs to run, overriding config batch.pair_ids.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N discovered pairs.")
    parser.add_argument("--dry-run", action="store_true", help="Print matched pairs without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing stable-cell masks.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ids:
        cfg.setdefault("batch", {})["pair_ids"] = parse_ids(args.ids)
    if args.overwrite:
        cfg.setdefault("processing", {})["overwrite"] = True

    pairs = discover_pairs(cfg)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    out_dir = Path(cfg["batch"]["output_dir"]).expanduser()
    print(f"Batch      : {cfg['batch'].get('name', args.config.stem)}")
    print(f"Pairs found: {len(pairs)}")
    print(f"Output     : {out_dir}")

    if args.dry_run:
        for pair in pairs:
            print(f"  {pair.pair_id}: {pair.pre_path.name} + {pair.post_path.name}")
        return

    rows: list[dict[str, Any]] = []
    continue_on_error = bool(cfg.get("processing", {}).get("continue_on_error", True))
    overwrite = bool(cfg.get("processing", {}).get("overwrite", False))

    # Only named products are overwritten; never remove the output directory.

    for i, pair in enumerate(pairs, start=1):
        stable_path = stable_mask_dir(out_dir, cfg) / f"{pair.pair_id}_stable_cells.tif"
        if stable_path.exists() and not overwrite:
            print(f"[{i}/{len(pairs)}] {pair.pair_id}: exists, skip")
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "status": "skipped_exists",
                    "warning": "",
                    "pre_path": str(pair.pre_path),
                    "post_path": str(pair.post_path),
                    "stable_cells_path": str(stable_path),
                }
            )
            continue

        print(f"[{i}/{len(pairs)}] {pair.pair_id}: building stable cells")
        try:
            row = process_pair(pair, cfg, out_dir)
            rows.append(row)
            frac = row["stable_fraction"]
            print(f"  stable={row['stable_pixels']:,}/{row['valid_pixels']:,} ({100 * frac:.2f}%) {row['warning']}")
        except Exception as exc:
            row = {
                "pair_id": pair.pair_id,
                "status": "failed",
                "warning": "",
                "pre_path": str(pair.pre_path),
                "post_path": str(pair.post_path),
                "stable_cells_path": str(stable_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            print(f"  FAILED: {row['error']}")
            if not continue_on_error:
                write_summaries(rows, out_dir)
                raise

    write_summaries(rows, out_dir)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    skipped = sum(1 for row in rows if row.get("status") == "skipped_exists")
    print(f"Done. ok={ok}, failed={failed}, skipped={skipped}")
    print(f"Summary: {out_dir / 'summaries' / 'stable_cells_summary.csv'}")
    print(f"xDEM pair index: {out_dir / 'summaries' / 'xdem_pair_index.csv'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
