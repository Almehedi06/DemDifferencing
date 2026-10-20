"""AUC of the three modelled layers against all mapped debris-flow initiations, pooled.

The pooled class is the union of every mechanism's evidence blocks, scored
against the pseudo-absence blocks: the "All" / balanced rows of
validate_cellwise.py. The AUCs are recomputed here from the rasters and must
match that CSV, so the figure cannot drift from the scored numbers.

Wording on the figure is deliberately type-free ("debris-flow initiations").

A second figure, roc_two_negative_sets, puts this ROC next to the one with the
absence cells of Strauch et al. (2018): every cell steeper than 17 degrees outside
the event footprint, read from an auc_experiment.py run (--strauch-run).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_final_figures as M  # noqa: E402  shared style (rcParams) and save()

D = Path("/mnt/c/Users/amehedi/Downloads")
LAYERS = [("P_Landslide", "Landslide", "#8c8c8c"),
          ("P_Channel_ini", "Channel initiation", "#3d8ce8"),
          ("P_Combined_H", "Hazard", "#e8a33d")]
MECHANISMS = ["LS", "Runoff", "Firehose"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pioneer", type=Path, default=D / "pioneer")
    ap.add_argument("--output-dir", type=Path, default=D / "pioneer/figures_final")
    ap.add_argument("--strauch-run", type=Path, default=D / "pioneer/auc_runs/no_G_rule",
                    help="auc_experiment.py run folder with roc_curves.csv")
    args = ap.parse_args(argv)
    ev_dir, layers_dir = args.pioneer / "evidence_rasters", args.pioneer / "probability_fields"

    with rasterio.open(ev_dir / "evidence_LS.tif") as src:
        grid = (src.transform, src.shape)

    def evidence(name):
        with rasterio.open(ev_dir / f"evidence_{name}.tif") as s:
            return s.read(1) == 1

    pos = np.logical_or.reduce([evidence(m) for m in MECHANISMS])
    neg = evidence("absence") & ~pos
    report = json.loads((ev_dir / "evidence_report.json").read_text())
    n_sites = sum(v["n_points"] for v in report["mechanisms"].values())
    n_absence = report["absence"]["n_points"]

    scored = {(r["layer"], r["mechanism"]): float(r["roc_auc"])
              for r in csv.DictReader(open(args.pioneer / "cellwise_validation/cellwise_metrics.csv"))
              if r["negatives"] == "balanced"}

    results = []
    for stem, label, colour in LAYERS:
        with rasterio.open(layers_dir / f"{stem}.tif") as src:
            if (src.transform, src.shape) != grid:
                raise ValueError(f"{stem}: not on the evidence grid")
            score = src.read(1, masked=True).filled(np.nan).astype("float64")
        f = np.isfinite(score)
        y = np.r_[np.ones(int((pos & f).sum())), np.zeros(int((neg & f).sum()))]
        v = np.r_[score[pos & f], score[neg & f]]
        auc = float(roc_auc_score(y, v))
        if abs(auc - scored[(stem, "All")]) > 1e-9:
            raise ValueError(f"{stem}: AUC {auc} differs from scored {scored[(stem, 'All')]}")
        fpr, tpr, _ = roc_curve(y, v)
        results.append((label, colour, auc, fpr, tpr))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(results))
    for i, (label, colour, auc, _, _) in enumerate(results):
        ax.bar(i, auc, 0.6, color=colour, edgecolor="black", linewidth=0.7)
        ax.text(i, auc + 0.006, f"{auc:.2f}", ha="center", va="bottom", fontsize=14,
                fontweight="bold")
    ax.axhline(0.5, ls="--", color="0.35", lw=1.3)
    ax.set_xlim(-0.55, len(results) - 0.1)
    ax.text(len(results) - 0.14, 0.503, "chance", ha="right", va="bottom", fontsize=11, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in results])
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("AUC", labelpad=8)
    ax.set_title("Where debris flows start: model skill", pad=10)
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.2, label="No skill")
    for label, colour, auc, fpr, tpr in results:
        ax.plot(fpr, tpr, linewidth=2.4, color=colour, label=f"{label}  ({auc:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate", labelpad=8)
    ax.set_title("ROC curve  (AUC in brackets)", pad=10)
    ax.legend(loc="lower right", framealpha=0.96, borderpad=0.6)
    ax.grid(linewidth=0.5, alpha=0.5)

    for ax, tag in zip(axes, "ab"):
        ax.text(0.022, 0.975, tag, transform=ax.transAxes, ha="left", va="top", fontsize=18,
                fontweight="bold", bbox=dict(boxstyle="square,pad=0.32", facecolor="white",
                                             edgecolor="0.25", linewidth=1.0))
    fig.text(0.5, -0.045,
             f"All {n_sites} mapped debris-flow initiations (30 m around each) against "
             f"{n_absence} random sites away from them.  AUC 0.5 = chance, 1 = perfect.",
             ha="center", va="top", fontsize=12, color="0.25")
    M.save(fig, args.output_dir, "auc_all_debris_flows")
    for label, _, auc, _, _ in results:
        print(f"    {label:<20} AUC {auc:.3f}")
    two_negative_sets(results, n_sites, n_absence, args.strauch_run, args.output_dir)


def two_negative_sets(results, n_sites, n_absence, run, out_dir):
    """ROC with our absence points next to ROC with the Strauch et al. (2018) absence cells."""
    rows = [r for r in csv.DictReader(open(run / "roc_curves.csv")) if r["type"] == "All"]
    n_no = int(next(csv.DictReader(open(run / "metrics.csv")))["no_cells"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    titles = [f"Absence: {n_absence} random points (30 m around each)",
              f"Absence: all {n_no:,} cells over 17° outside the event\n(Strauch et al. 2018)"]
    for ax, title, tag in zip(axes, titles, "ab"):
        ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.2, label="No skill")
        for stem, label, colour in LAYERS:
            if tag == "a":
                fpr, tpr, auc = next((r[3], r[4], r[2]) for r in results if r[0] == label)
            else:
                mine = [r for r in rows if r["layer"] == stem]
                fpr = [float(r["fpr"]) for r in mine]
                tpr = [float(r["tpr"]) for r in mine]
                auc = float(mine[0]["auc"])
            ax.plot(fpr, tpr, linewidth=2.4, color=colour, label=f"{label}  ({auc:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate", labelpad=8)
        ax.set_title(title, pad=10, fontsize=13)
        ax.legend(loc="lower right", framealpha=0.96, borderpad=0.6)
        ax.grid(linewidth=0.5, alpha=0.5)
        ax.text(0.022, 0.975, tag, transform=ax.transAxes, ha="left", va="top", fontsize=18,
                fontweight="bold", bbox=dict(boxstyle="square,pad=0.32", facecolor="white",
                                             edgecolor="0.25", linewidth=1.0))
    fig.text(0.5, -0.045, f"All {n_sites} mapped debris-flow initiations (30 m around each) as the yes cells.  "
             "AUC in brackets: 0.5 = chance, 1 = perfect.", ha="center", va="top", fontsize=12, color="0.25")
    M.save(fig, out_dir, "roc_two_negative_sets")


if __name__ == "__main__":
    main()
