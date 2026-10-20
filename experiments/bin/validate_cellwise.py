"""Cell-wise scoring of modelled layers against rasterised initiation evidence.

Companion to validate_hazard_layers.py, which scores *points*. This scores
*cells*: the evidence rasters from make_evidence_rasters.py expand each mapped
point to a 3x3 block (30 m at 10 m resolution), and every cell is an
observation. That is the usual way to build a confusion matrix between two
rasters, and it gives enough samples for well-behaved ROC and PR curves.

The two tests answer different questions and will not agree exactly:
  point-wise  "did the model flag near this failure?"
  cell-wise   "how well do the two maps agree spatially?"

Negative-class variants, both reported
--------------------------------------
balanced  negatives are the pseudo-absence blocks (~808 cells). Comparable to
          the point-wise tables; roughly 1:2 positive:negative.
domain    negatives are every other valid cell in the model domain (~200k).
          Realistic base rate, but precision and CSI collapse for every layer
          because FP counts explode -- that is prevalence, not a change in
          model quality. Recall and AUC stay interpretable. Never compare a
          `domain` precision against a `balanced` precision.

In both variants, cells carrying evidence of *any* mechanism are excluded from
the negative pool, so a Runoff cell is never counted as a true negative when
scoring LS.

Pooled class "All": the union of the three mechanisms' evidence cells, scored
as one positive class -- "did something initiate here", type ignored. LS has
the most points, so the pooled result leans towards the LS result; read it
beside the per-mechanism rows, never instead of them.

Shared statistics (confusion_metrics, area_fraction_needed, capture_curve) are
imported from validate_hazard_layers rather than reimplemented, so the
mid-rank tie handling cannot drift between the two scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_hazard_layers import area_fraction_needed, capture_curve, confusion_metrics

MECHANISMS = ["LS", "Runoff", "Firehose"]
POOLED = "All"
CLASSES = MECHANISMS + [POOLED]
LAYERS = [("P_Landslide", "Landslide"),
          ("P_Channel_ini", "Channel initiation"),
          ("P_Combined_H", "Combined hillslope")]
COLOURS = {"P_Landslide": "#8c8c8c", "P_Channel_ini": "#3d8ce8",
           "P_Combined_H": "#e8a33d", "dod_erosion": "#c1272d"}
AREA_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read_on_grid(path, ref_transform, ref_shape, ref_crs):
    """Read a raster, reprojecting onto the reference grid if it differs."""
    with rasterio.open(path) as src:
        same = (src.crs == ref_crs and src.transform.almost_equals(ref_transform)
                and src.shape == ref_shape)
        if same:
            return src.read(1, masked=True).filled(np.nan).astype("float64")
        dst = np.full(ref_shape, np.nan, "float64")
        rasterio.warp.reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=ref_transform, dst_crs=ref_crs, dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear)
        return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    ap.add_argument("--evidence-dir", type=Path, default=D / "pioneer/evidence_rasters")
    ap.add_argument("--layers-dir", type=Path, default=D / "pioneer/probability_fields")
    ap.add_argument("--layer-suffix", default="",
                    help="Appended to each layer stem, e.g. _regional_baseline_Cs_1k")
    ap.add_argument("--skip-dod", action="store_true",
                    help="Score only the modelled layers. The dDEM derives from the same "
                         "survey as the evidence, so it is a second evidence stream rather "
                         "than a competing model.")
    ap.add_argument("--dod", type=Path,
                    default=D / "pioneer/main_pipeline_run/corrections/pioneer/"
                                "final_dem_products/dod_corrected.tif")
    ap.add_argument("--probability-cutoff", type=float, default=0.5)
    ap.add_argument("--dod-cutoff-m", type=float, default=0.50)
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/cellwise_validation")
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.evidence_dir / "evidence_LS.tif") as src:
        transform, shape, crs = src.transform, src.shape, src.crs

    def ev(name):
        with rasterio.open(args.evidence_dir / f"evidence_{name}.tif") as s:
            return s.read(1) == 1

    evidence = {m: ev(m) for m in MECHANISMS}
    absence = ev("absence")
    any_evidence = np.logical_or.reduce(list(evidence.values()))
    absence = absence & ~any_evidence          # never score evidence as a negative
    evidence[POOLED] = any_evidence

    # Unified scoring layers: score raster (higher = more positive-like) + cutoff.
    scored = []
    for stem, label in LAYERS:
        scored.append((stem, label,
                       read_on_grid(args.layers_dir / f"{stem}{args.layer_suffix}.tif",
                                    transform, shape, crs),
                       args.probability_cutoff))
    if not args.skip_dod:
        scored.append(("dod_erosion", "dDEM erosion",
                       -read_on_grid(args.dod, transform, shape, crs), args.dod_cutoff_m))

    rows, capture_rows, curves = [], [], {}
    for stem, label, score, cutoff in scored:
        finite = np.isfinite(score)
        domain_neg = finite & ~any_evidence & ~absence
        for variant, neg in (("balanced", absence & finite), ("domain", domain_neg)):
            curves.setdefault(variant, {})
            for mech in CLASSES:
                pos = evidence[mech] & finite
                y = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
                s = np.r_[score[pos], score[neg]]
                pred = s >= cutoff
                row = dict(layer=stem, mechanism=mech, negatives=variant,
                           n_positive_cells=int(pos.sum()), n_negative_cells=int(neg.sum()),
                           cutoff=cutoff, **confusion_metrics(y, pred),
                           roc_auc=float(roc_auc_score(y, s)),
                           pr_auc=float(average_precision_score(y, s)))
                rows.append(row)
                if mech == "LS":
                    fpr, tpr, _ = roc_curve(y, s)
                    prec, rec, _ = precision_recall_curve(y, s)
                    curves[variant][stem] = dict(fpr=fpr, tpr=tpr, prec=prec, rec=rec,
                                                 auc=row["roc_auc"], ap=row["pr_auc"],
                                                 label=label)
                # success-rate uses the whole domain as the area denominator
                needed = area_fraction_needed(score[finite], score[pos])
                cap = capture_curve(needed, np.array(AREA_FRACTIONS))
                if variant == "balanced":
                    capture_rows.append(dict(
                        layer=stem, mechanism=mech, n_positive_cells=int(pos.sum()),
                        **{f"capture_at_{int(f*100)}pct_area": round(float(v), 4)
                           for f, v in zip(AREA_FRACTIONS, cap)}))

    with (args.output_dir / "cellwise_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (args.output_dir / "cellwise_success_rate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(capture_rows[0]))
        w.writeheader(); w.writerows(capture_rows)

    # ---------------- figures ----------------
    for variant in ("balanced", "domain"):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
        axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=1.0, label="No skill")
        for stem, c in curves[variant].items():
            axes[0].plot(c["fpr"], c["tpr"], lw=2.0, color=COLOURS[stem],
                         label=f"{c['label']} (AUC {c['auc']:.2f})")
            axes[1].plot(c["rec"], c["prec"], lw=2.0, color=COLOURS[stem],
                         label=f"{c['label']} (AP {c['ap']:.2f})")
        base = rows[0]
        axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
        axes[0].set_title(f"ROC — landslide cells, {variant} negatives")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title(f"Precision–recall — landslide cells, {variant} negatives")
        for ax in axes:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.grid(lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.legend(loc="lower right")
        fig.savefig(args.output_dir / f"roc_pr_{variant}.png")
        fig.savefig(args.output_dir / f"roc_pr_{variant}.pdf")
        plt.close(fig)

    # bar plot: metrics per layer x mechanism, balanced variant
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(20.5, 4.8), constrained_layout=True,
                             sharey=True)
    metrics = ["recall", "F1", "CSI"]
    for ax, mech in zip(axes, CLASSES):
        sub = [r for r in rows if r["mechanism"] == mech and r["negatives"] == "balanced"]
        x = np.arange(len(metrics)); width = 0.2
        for i, r in enumerate(sub):
            ax.bar(x + (i - 1.5) * width, [r[m] for m in metrics], width,
                   color=COLOURS[r["layer"]], edgecolor="black", linewidth=0.5,
                   label=r["layer"] if mech == "LS" else None)
        ax.set_xticks(x); ax.set_xticklabels(["Recall", "F1", "CSI"])
        ax.set_ylim(0, 1); ax.grid(axis="y", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        name = "All types" if mech == POOLED else mech
        ax.set_title(f"{name} ({sub[0]['n_positive_cells']} cells)")
    axes[0].set_ylabel("Score"); axes[0].legend(loc="upper left")
    fig.suptitle("Cell-wise performance, balanced negatives", fontsize=14)
    fig.savefig(args.output_dir / "cellwise_bars.png")
    fig.savefig(args.output_dir / "cellwise_bars.pdf")
    plt.close(fig)

    (args.output_dir / "cellwise_report.json").write_text(json.dumps(dict(
        method="cell-wise, 3x3 (30 m) evidence blocks on the 10 m model grid",
        variants={"balanced": "negatives = pseudo-absence blocks",
                  "domain": "negatives = all other valid domain cells"},
        caveats=[
            "Precision/CSI are prevalence-sensitive: the 'domain' variant's low values "
            "reflect a 1:many base rate, not worse models. Never compare precision "
            "across variants.",
            "Cells carrying evidence of any mechanism are excluded from every negative pool.",
            "Point-wise and cell-wise tests answer different questions and will not agree "
            "exactly; neither supersedes the other.",
            "The dDEM shares data lineage with the mapped evidence (both derive from the "
            "same UAV survey); the Landlab layers do not. Not a like-for-like comparison.",
        ]), indent=2))

    hdr = (f"{'layer':<14}{'mech':<10}{'neg':<10}{'nPos':>6}{'nNeg':>8}"
           f"{'recall':>8}{'F1':>7}{'CSI':>7}{'AUC':>7}{'AP':>7}")
    print(hdr)
    for r in rows:
        print(f"{r['layer']:<14}{r['mechanism']:<10}{r['negatives']:<10}"
              f"{r['n_positive_cells']:>6}{r['n_negative_cells']:>8}"
              f"{r['recall']:>8.2f}{r['F1']:>7.2f}{r['CSI']:>7.2f}"
              f"{r['roc_auc']:>7.2f}{r['pr_auc']:>7.2f}")
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
