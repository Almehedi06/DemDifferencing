"""Score every correction/filtering variant on one common set of evaluation pixels.

Existing per-run CSVs each report on that run's own fitting mask, so their numbers are
not comparable. This script evaluates all variants on identical pixels: the M3
validation blocks, which fitted no alignment and calibrated no threshold.

Reads only; writes a comparison table, a report and figures to a new directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path, ref=None):
    with rasterio.open(path) as s:
        if ref is not None and (s.crs != ref["crs"] or s.transform != ref["transform"]
                                or s.shape != (ref["height"], ref["width"])):
            raise ValueError(f"Grid mismatch: {path}")
        return s.read(1, masked=True).astype("float64").filled(np.nan), s.profile.copy()


def score(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite evaluation pixels")
    med = float(np.median(v))
    return dict(n=int(v.size), median_m=med,
                nmad_m=float(1.4826 * np.median(np.abs(v - med))),
                rmse_m=float(np.sqrt(np.mean(v ** 2))),
                p05_m=float(np.percentile(v, 5)), p95_m=float(np.percentile(v, 95)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pioneer-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {a.output_dir}")
    R = a.pioneer_root

    dods = {
        "M1_montecito":   R/"Montecito_params/final_dem_products/post-pre_COREG_BIAS_ASPECT_o1_a90_dDEM.tif",
        "M1.5_g_mask":    R/"terrain_stable_run/corrections/pioneer/final_dem_products/post-pre_COREG_BIAS_ASPECT_o1_a90_dDEM.tif",
        "M2_translation": R/"main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif",
    }
    products = {
        "M2_lod_filtered": R/"main_pipeline_run/corrections/pioneer/final_dem_products/dod_lod_filtered.tif",
        "M3_domain_lod":   R/"m3_final_candidate/dod_m3_filtered.tif",
    }
    missing = [str(p) for p in list(dods.values()) + list(products.values()) if not p.is_file()]
    if missing:
        ap.error("Missing inputs:\n  " + "\n  ".join(missing))

    base, prof = read(dods["M2_translation"])
    arrays = {k: (base if k == "M2_translation" else read(v, prof)[0]) for k, v in dods.items()}
    filt = {k: read(v, prof)[0] for k, v in products.items()}
    split, _ = read(R/"m3_final_candidate/m3_calibration_validation.tif", prof)
    domain, _ = read(R/"m3_final_candidate/m3_detection_domain.tif", prof)
    with rasterio.open(R/"unzipped_nlcd_2021/Annual_NLCD_LndCov_2021_CU_C1V1.tif") as src, \
         WarpedVRT(src, crs=prof["crs"], transform=prof["transform"], height=prof["height"],
                   width=prof["width"], resampling=Resampling.nearest,
                   dtype="float32", nodata=np.nan) as vrt:
        lc = vrt.read(1, masked=True).filled(np.nan).astype("float64")

    # Common evaluation set: validation blocks where EVERY variant has a value.
    ev = (split == 2)
    for arr in arrays.values():
        ev &= np.isfinite(arr)
    if ev.sum() < 1000:
        raise ValueError(f"Only {int(ev.sum())} common evaluation pixels")

    rows, report = [], {"common_evaluation_pixels": int(ev.sum()),
                        "evaluation_set": "M3 validation blocks; fitted no alignment, "
                                          "calibrated no threshold",
                        "methods": {}}
    for name, arr in arrays.items():
        s = score(arr[ev])
        s.update(method=name, kind="corrected dDEM", lod_product="not produced",
                 detected_negative="n/a", detected_positive="n/a", domain_pct=100.0)
        rows.append(s); report["methods"][name] = s
    for name, arr in filt.items():
        f = np.isfinite(arr)
        det = f & (arr != 0)
        s = score(arrays["M2_translation"][ev])          # same underlying dDEM
        s.update(method=name, kind="filtered product",
                 lod_product="produced",
                 detected_negative=int((det & (arr < 0)).sum()),
                 detected_positive=int((det & (arr > 0)).sum()),
                 domain_pct=float(100 * f.sum() / np.isfinite(base).sum()))
        rows.append(s); report["methods"][name] = s

    a.output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["method", "kind", "n", "median_m", "nmad_m", "rmse_m", "p05_m", "p95_m",
            "domain_pct", "lod_product", "detected_negative", "detected_positive"]
    with (a.output_dir/"method_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- Figure 1: histograms on a common clipped axis, log counts
    fig, axes = plt.subplots(1, len(arrays), figsize=(4.6*len(arrays), 4), sharey=True)
    bins = np.linspace(-5, 5, 161)
    for ax, (name, arr) in zip(np.atleast_1d(axes), arrays.items()):
        v = arr[ev]
        ax.hist(v[(v >= -5) & (v <= 5)], bins=bins, color="#4a7ba7", edgecolor="none")
        ax.set_yscale("log"); ax.axvline(0, color="k", lw=.6)
        ax.set_title(name, fontsize=10); ax.set_xlabel("dh (m)")
    np.atleast_1d(axes)[0].set_ylabel("count (log)")
    fig.suptitle("Corrected dDEM on common evaluation pixels (range ±5 m; tails omitted, not clipped)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(a.output_dir/"fig1_histograms_common.png", dpi=150); plt.close(fig)

    # ---- Figure 2: per-class distributions with the M2 and M3 limits drawn
    m3 = json.loads((R/"m3_final_candidate/m3_report.json").read_text())
    m2thr = json.loads((R/"main_pipeline_run/corrections/pioneer/final_dem_products/lod_report.json"
                        ).read_text())["threshold_m"]
    groups = [("open (31,71)", [31, 71]), ("shrub (52)", [52]), ("evergreen (42)", [42])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (label, codes) in zip(axes, groups):
        v = base[np.isfinite(base) & np.isin(lc, codes)]
        ax.hist(v[(v >= -6) & (v <= 6)], bins=np.linspace(-6, 6, 193),
                color="#7a9e7e", edgecolor="none")
        ax.set_yscale("log")
        ax.axvline(-m2thr, color="#c0392b", ls="--", lw=1.2)
        ax.axvline(m2thr, color="#c0392b", ls="--", lw=1.2, label=f"M2 ±{m2thr:.2f} m")
        key = {"open (31,71)": "open", "shrub (52)": "shrub", "evergreen (42)": "evergreen"}[label]
        e = m3["classes"].get(key, {})
        if e.get("status") == "calibrated":
            ax.axvline(-e["negative_limit_m"], color="#1f6f4a", lw=1.5)
            ax.axvline(e["positive_limit_m"], color="#1f6f4a", lw=1.5,
                       label=f"M3 −{e['negative_limit_m']:.2f}/+{e['positive_limit_m']:.2f} m")
        else:
            ax.text(.5, .88, "excluded from M3 domain", transform=ax.transAxes,
                    ha="center", color="#c0392b", fontsize=9)
        ax.set_title(label, fontsize=10); ax.set_xlabel("dh (m)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("count (log)")
    fig.suptitle("Corrected dDEM by land cover, with M2 and M3 detection limits "
                 "(range ±6 m; tails omitted, not clipped)", fontsize=11)
    fig.tight_layout(); fig.savefig(a.output_dir/"fig2_limits_by_landcover.png", dpi=150); plt.close(fig)

    # ---- Figure 3: maps on one colour scale
    panels = [("M1 montecito", arrays["M1_montecito"]), ("M1.5 G mask", arrays["M1.5_g_mask"]),
              ("M2 corrected", arrays["M2_translation"]), ("M2 LoD filtered", filt["M2_lod_filtered"]),
              ("M3 domain LoD", filt["M3_domain_lod"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1*len(panels), 4.6))
    for ax, (t, arr) in zip(axes, panels):
        im = ax.imshow(arr[::6, ::6], cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(t, fontsize=10); ax.set_axis_off()
    fig.colorbar(im, ax=list(axes), label="post − pre (m)", shrink=.7)
    fig.savefig(a.output_dir/"fig3_maps.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 4: detected erosion vs deposition
    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = list(filt); xs = np.arange(len(names))
    neg = [report["methods"][n]["detected_negative"] for n in names]
    pos = [report["methods"][n]["detected_positive"] for n in names]
    ax.bar(xs-.18, neg, .36, label="detected negative (erosion)", color="#c0392b")
    ax.bar(xs+.18, pos, .36, label="detected positive", color="#2c6fad")
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("pixels (log)"); ax.legend(fontsize=9)
    ax.set_title("Detections by sign", fontsize=11)
    fig.tight_layout(); fig.savefig(a.output_dir/"fig4_detections.png", dpi=150); plt.close(fig)

    report["m2_threshold_m"] = m2thr
    report["m3_limits"] = {k: {"negative_m": v.get("negative_limit_m"),
                               "positive_m": v.get("positive_limit_m"),
                               "status": v.get("status")}
                           for k, v in m3["classes"].items()}
    report["notes"] = [
        "M1 fitted on |raw DoD| < 5 m, which cannot cope with the 18.7 m offset on this "
        "event; its products remain offset and it is a demonstrated failure here, not a "
        "competing candidate.",
        "M1 and M1.5 produce no LoD product; those cells read 'not produced'.",
        "Corrected-dDEM rows share the M2 alignment where the product is a filter of it, "
        "so their scatter statistics are identical by construction.",
        "Detection counts cover the whole raster, not just evaluation pixels.",
    ]
    (a.output_dir/"comparison_report.json").write_text(json.dumps(report, indent=2))

    print(f"Common evaluation pixels: {int(ev.sum()):,}\n")
    print(f"{'method':<18}{'median':>9}{'NMAD':>8}{'RMSE':>9}{'domain%':>9}{'det -':>9}{'det +':>10}")
    for r in rows:
        dn = r['detected_negative']; dp = r['detected_positive']
        print(f"{r['method']:<18}{r['median_m']:>9.3f}{r['nmad_m']:>8.3f}{r['rmse_m']:>9.3f}"
              f"{r['domain_pct']:>9.1f}"
              f"{(f'{dn:,}' if dn!='n/a' else '-'):>9}{(f'{dp:,}' if dp!='n/a' else '-'):>10}")
    print(f"\nOutputs: {a.output_dir}")


if __name__ == "__main__":
    main()
