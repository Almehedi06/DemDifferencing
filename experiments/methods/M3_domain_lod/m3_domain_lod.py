"""M3: domain-restricted, asymmetric, per-land-cover detection limits for a corrected DoD.

Reads an existing corrected DoD and candidate-stable mask. Writes a new directory.
It does not re-run or modify any correction pipeline, and never writes to its inputs.

Rationale (Pioneer, 2026-09-16):
  The corrected DoD carries a one-sided positive tail from vegetation left in the
  post-DEM ground classification. A two-sided threshold estimated from absolute
  residuals is therefore inflated by contamination and destroys erosion sensitivity.
  M3 estimates the negative and positive limits separately, per land-cover class,
  on spatial blocks never used to fit the alignment, and validates coverage on a
  further disjoint block set. Classes whose validated limit is not usable are
  excluded from the detection domain as NoData rather than reported as zero change.

Three disjoint spatial block sets (250 m blocks, same geometry as the alignment run):
  alignment  (r_b + 2 c_b) % 5 != 0   already consumed by coregistration; unused here
  calibration(r_b + 2 c_b) % 5 == 0 and (r_b + c_b) % 2 == 0   estimates the limits
  validation (r_b + 2 c_b) % 5 == 0 and (r_b + c_b) % 2 == 1   measures coverage only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

NODATA = -9999.0
MIN_PIXELS = 5000          # per class, per split; below this a class is not calibrated


def read(path, reference=None):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {path}")
        profile = src.profile.copy()
        if reference is not None and (src.crs != reference["crs"]
                                     or src.transform != reference["transform"]
                                     or src.shape != (reference["height"], reference["width"])):
            raise ValueError(f"Grid mismatch against reference: {path}")
        return src.read(1, masked=True).astype("float64").filled(np.nan), profile


def write(path, array, profile, **tags):
    p = {k: v for k, v in profile.items()
         if k not in ("blockxsize", "blockysize", "tiled", "interleave")}
    p.update(driver="GTiff", dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def three_way_split(shape, resolution, block_size_m):
    if not np.isfinite(block_size_m) or block_size_m <= 0:
        raise ValueError("Block size must be finite and positive")
    px = max(1, int(round(block_size_m / resolution)))
    rb = np.arange(shape[0])[:, None] // px
    cb = np.arange(shape[1])[None, :] // px
    held = ((rb + 2 * cb) % 5 == 0)
    even = ((rb + cb) % 2 == 0)
    return ~held, held & even, held & ~even      # alignment, calibration, validation


def robust(values):
    v = values[np.isfinite(values)]
    if v.size == 0:
        raise ValueError("No finite values for statistics")
    med = float(np.median(v))
    return dict(n=int(v.size), median_m=med,
                nmad_m=float(1.4826 * np.median(np.abs(v - med))),
                rmse_m=float(np.sqrt(np.mean(v ** 2))),
                p05_m=float(np.percentile(v, 5)), p95_m=float(np.percentile(v, 95)),
                p01_m=float(np.percentile(v, 1)), p99_m=float(np.percentile(v, 99)))


def candidate_limits(cal, target):
    """Asymmetric limit candidates. The negative side is vegetation-free by construction;
    the positive side is not, so the two are never forced to share a value.

    The per-tail estimators split the excluded mass evenly between the tails, so they
    reach the nominal coverage while letting each side find its own magnitude."""
    med = float(np.median(cal))
    nmad = float(1.4826 * np.median(np.abs(cal - med)))
    out = {
        "quantile_2sided_95": (abs(float(np.quantile(np.abs(cal), .95))),) * 2,
        "nmad_1p96": (1.959963984540054 * nmad,) * 2,
    }
    for nominal in (target, 0.99):
        half = 100 * (1 - nominal) / 2
        out[f"quantile_per_tail_{nominal:.3f}".rstrip('0').rstrip('.')] = (
            abs(float(np.percentile(cal, half))), float(np.percentile(cal, 100 - half)))
    return out


def coverage(values, lo, hi):
    """Total coverage plus each tail separately. The negative exceedance rate is the
    false-erosion rate on terrain taken to be stable, which is the number that limits
    erosion detection; the positive rate is dominated by vegetation, not by noise."""
    return dict(total=float(np.mean((values >= -lo) & (values <= hi))),
                negative_exceedance=float(np.mean(values < -lo)),
                positive_exceedance=float(np.mean(values > hi)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dod", type=Path, required=True, help="Corrected DoD (post minus pre)")
    ap.add_argument("--stable-mask", type=Path, required=True, help="Candidate-stable mask raster")
    ap.add_argument("--stable-value", type=int, default=1)
    ap.add_argument("--land-cover", type=Path, required=True)
    ap.add_argument("--domain", type=str, default="open=31,71;shrub=52",
                    help="Semicolon-separated name=class,class groups forming the detection domain")
    ap.add_argument("--excluded", type=str, default="evergreen=42",
                    help="Groups reported for comparison but excluded from the domain")
    ap.add_argument("--block-size-m", type=float, default=250.0)
    ap.add_argument("--target-coverage", type=float, default=0.95)
    ap.add_argument("--max-coverage-shortfall", type=float, default=0.02,
                    help="A class is excluded if no estimator reaches target minus this on validation")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")
    if not 0 < args.target_coverage < 1:
        ap.error("Target coverage must lie strictly between zero and one")

    def parse_groups(spec):
        groups = {}
        for part in filter(None, (s.strip() for s in spec.split(";"))):
            name, _, codes = part.partition("=")
            if not name or not codes:
                ap.error(f"Malformed group specification: {part!r}")
            groups[name.strip()] = [int(c) for c in codes.split(",")]
        return groups

    domain_groups = parse_groups(args.domain)
    excluded_groups = parse_groups(args.excluded)
    overlap = set().union(*domain_groups.values()) & set().union(*excluded_groups.values())
    if overlap:
        ap.error(f"Classes appear in both domain and excluded: {sorted(overlap)}")

    dod, profile = read(args.dod)
    stable_raw, _ = read(args.stable_mask, profile)
    with rasterio.open(args.land_cover) as src:
        if src.crs is None or src.count != 1:
            ap.error("Land cover needs a single band and a CRS")
        with WarpedVRT(src, crs=profile["crs"], transform=profile["transform"],
                       height=profile["height"], width=profile["width"],
                       resampling=Resampling.nearest, dtype="float32", nodata=np.nan) as vrt:
            lc = vrt.read(1, masked=True).filled(np.nan).astype("float64")
    finite_lc = np.isfinite(lc)
    if not np.equal(lc[finite_lc], np.floor(lc[finite_lc])).all():
        ap.error("Land-cover values must be integer class codes")

    valid = np.isfinite(dod)
    stable = valid & np.isfinite(stable_raw) & (stable_raw == args.stable_value)
    if not stable.any():
        ap.error("Candidate-stable mask selected no pixels; check --stable-value")
    resolution = abs(float(profile["transform"].a))
    _, cal_blocks, val_blocks = three_way_split(dod.shape, resolution, args.block_size_m)

    report = {"inputs": {k: str(getattr(args, k).resolve())
                         for k in ("dod", "stable_mask", "land_cover")},
              "block_size_m": args.block_size_m, "target_coverage": args.target_coverage,
              "valid_pixels": int(valid.sum()), "candidate_stable_pixels": int(stable.sum()),
              "classes": {}}

    selected, all_groups = {}, {**domain_groups, **excluded_groups}
    for name, codes in all_groups.items():
        in_class = valid & finite_lc & np.isin(lc, codes)
        cal = dod[stable & cal_blocks & in_class]
        val = dod[stable & val_blocks & in_class]
        entry = {"codes": codes, "class_pixels": int(in_class.sum()),
                 "calibration_pixels": int(cal.size), "validation_pixels": int(val.size),
                 "eligible_for_domain": name in domain_groups}
        if cal.size < MIN_PIXELS or val.size < MIN_PIXELS:
            entry["status"] = "not_calibrated_insufficient_pixels"
            report["classes"][name] = entry
            continue
        entry["calibration"] = robust(cal)
        entry["validation"] = robust(val)
        entry["estimators"] = {}
        for est, (lo, hi) in candidate_limits(cal, args.target_coverage).items():
            entry["estimators"][est] = {
                "negative_limit_m": lo, "positive_limit_m": hi,
                "calibration": coverage(cal, lo, hi),
                "validation": coverage(val, lo, hi),
            }
        # Prefer the tightest limits that still reach target coverage on validation.
        floor = args.target_coverage - args.max_coverage_shortfall
        passing = {k: v for k, v in entry["estimators"].items()
                   if v["validation"]["total"] >= floor}
        if not passing:
            entry["status"] = "excluded_no_estimator_reached_target"
            best = max(entry["estimators"].items(), key=lambda kv: kv[1]["validation"]["total"])
            entry["best_validation_coverage"] = best[1]["validation"]["total"]
            report["classes"][name] = entry
            continue
        chosen = min(passing.items(),
                     key=lambda kv: (kv[1]["negative_limit_m"], kv[1]["positive_limit_m"]))
        entry["chosen_estimator"] = chosen[0]
        entry["negative_limit_m"] = chosen[1]["negative_limit_m"]
        entry["positive_limit_m"] = chosen[1]["positive_limit_m"]
        entry["validation_coverage"] = chosen[1]["validation"]["total"]
        entry["false_erosion_rate"] = chosen[1]["validation"]["negative_exceedance"]
        entry["status"] = "calibrated" if name in domain_groups else "calibrated_but_excluded"
        if name in domain_groups:
            selected[name] = (in_class, chosen[1]["negative_limit_m"], chosen[1]["positive_limit_m"])
        report["classes"][name] = entry

    if not selected:
        raise SystemExit("No land-cover group produced a validated detection limit; "
                         "no M3 product written. See report.json.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain = np.zeros(dod.shape, dtype=bool)
    lo_map = np.full(dod.shape, np.nan)
    hi_map = np.full(dod.shape, np.nan)
    for name, (in_class, lo, hi) in selected.items():
        domain |= in_class
        lo_map[in_class], hi_map[in_class] = lo, hi
    domain &= valid & np.isfinite(lo_map)

    filtered = np.full(dod.shape, np.nan)
    detected = domain & ((dod < -lo_map) | (dod > hi_map))
    filtered[domain] = 0.0
    filtered[detected] = dod[detected]

    tags = dict(method="M3 domain-restricted asymmetric per-land-cover empirical detection limits",
                operation="corrected post minus pre",
                zero_meaning="Inside detection domain, within limits; not proof of zero change",
                nodata_meaning="Outside the validated detection domain, or missing DoD",
                asymmetry_reason="Positive limits absorb vegetation left in the post ground classification",
                status="Provisional; ground classification and vertical reference unverified")
    write(args.output_dir / "dod_m3_filtered.tif", filtered, profile, **tags)
    write(args.output_dir / "m3_limit_negative.tif", np.where(domain, lo_map, np.nan), profile, **tags)
    write(args.output_dir / "m3_limit_positive.tif", np.where(domain, hi_map, np.nan), profile, **tags)
    write(args.output_dir / "m3_detection_domain.tif",
          np.where(valid, domain.astype(float), np.nan), profile,
          domain_value=1, excluded_value=0, **tags)
    split_map = np.where(valid, 0.0, np.nan)
    split_map[stable & cal_blocks] = 1
    split_map[stable & val_blocks] = 2
    write(args.output_dir / "m3_calibration_validation.tif", split_map, profile,
          calibration_value=1, validation_value=2,
          note="Alignment-fitting blocks are code 0; they are excluded from both sets", **tags)

    # Verify what was written rather than trusting the arrays in memory.
    back, _ = read(args.output_dir / "dod_m3_filtered.tif", profile)
    if not np.array_equal(np.isfinite(back), domain):
        raise ValueError("Saved product extent does not match the detection domain")
    if not np.allclose(back[detected], dod[detected], atol=3e-4):
        raise ValueError("Saved product altered detected magnitudes")
    if not np.all(back[domain & ~detected] == 0):
        raise ValueError("Saved product did not zero sub-limit pixels inside the domain")

    report.update(
        domain_pixels=int(domain.sum()),
        domain_fraction_of_valid=float(domain.sum() / valid.sum()),
        detected_pixels=int(detected.sum()),
        detected_negative_pixels=int((detected & (dod < 0)).sum()),
        detected_positive_pixels=int((detected & (dod > 0)).sum()),
        detected_area_km2=float(detected.sum() * resolution ** 2 / 1e6),
        limitations=[
            "Candidate-stable terrain is not independently verified unchanged ground.",
            "Limits are asymmetric because vegetation contamination is one-sided; the positive "
            "limit is inflated by that contamination and is not a noise estimate.",
            "Calibration and validation blocks are disjoint and neither fitted the alignment, "
            "but both derive from the same candidate-stable mask.",
            "Excluded classes are NoData: undetectable here, not unchanged.",
            "Post-DEM ground classification is unresolved; residual vegetation remains the "
            "dominant error source.",
            "A residual spatial tilt of about 0.57 m across the scene is absorbed into the "
            "limits by block-reserved calibration, not corrected.",
            "Survey chronology and vertical datum remain unverified.",
        ])
    (args.output_dir / "m3_report.json").write_text(json.dumps(report, indent=2))

    print(f"Detection domain: {report['domain_pixels']:,} px "
          f"({100 * report['domain_fraction_of_valid']:.1f}% of valid)")
    for name, e in report["classes"].items():
        if e.get("status") == "calibrated":
            print(f"  {name:<10} -{e['negative_limit_m']:.3f} / +{e['positive_limit_m']:.3f} m "
                  f"[{e['chosen_estimator']}] validation coverage "
                  f"{100 * e['validation_coverage']:.2f}%")
        else:
            print(f"  {name:<10} {e['status']}")
    print(f"Detected: {report['detected_pixels']:,} px "
          f"({report['detected_negative_pixels']:,} negative, "
          f"{report['detected_positive_pixels']:,} positive), "
          f"{report['detected_area_km2']:.3f} km2")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
