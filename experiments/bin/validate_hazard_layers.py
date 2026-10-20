"""Score modeled hazard layers and the dDEM erosion layer against observed points.

Standalone validation utility. Not part of the M1-M4 pipeline in scripts/ or
methods/, reads only, and writes to a new output directory.

Inputs
------
Modeled layers  P_Landslide.tif, P_Combined_H.tif, P_Channel_ini.tif -- Landlab
                probability-of-failure grids, 10 m, 474x423, converted from
                Erkan's .asc exports (CRS EPSG:32610 assigned since the .asc
                itself declares none -- confirmed by extent match against the
                equivalent RGBA renders used in an earlier pass of this
                script). Real single-band float grids: no reconstruction
                needed. See ASC-INPUT NOTE below.
                Names match two of the three observed mechanisms directly
                (Landslide/LS, Channel_ini/Firehose); Combined_H is presumed
                to mean "hillslope, combined [mechanisms]" but that is not
                confirmed from the filename alone -- the cross-tab output
                lets the reader check it against the data instead of trusting
                the name.
Erosion layer   the continuous corrected dDEM (dod_corrected.tif), scored at
                a -0.50 m cutoff (see experiments/methods/M4_erosion_threshold/), folded
                into the same "layers" list as the probability grids so every
                layer is scored identically.
Observed points Initiation_Points_final.shp -- 47 LS / 22 Runoff / 20
                Firehose, remotely mapped by USGS from the same UAV survey
                (not field-verified; see the USGS metadata for the mapping
                method). Each mechanism is scored as its own positive class,
                separately, because the three have very different n and
                mixing them would hide that (Erkan, 2026-09-22).
Absence points  random90_unbiased.shp -- 90 points >50 m from every mapped
                initiation point of any mechanism. Already built for this
                purpose; reused as-is here, not regenerated. Pseudo-absences,
                not field-verified negatives -- used only for the confusion
                matrix, not for the success-rate table (see below).

ASC-INPUT NOTE -- zero is treated as data, not nodata
------------------------------------------------------
The .asc grids carry no NODATA_value line. Edges are uniformly zero
(consistent with a domain mask) but so is ~88% of the interior on
P_Landslide, so zero cannot be dropped as "outside the model" -- it is kept
as a genuine low-probability value.

Two validations, deliberately different in what they assume
-------------------------------------------------------------
1. Confusion matrix (TP/FP/FN/TN, precision, recall, specificity, accuracy,
   F1, CSI, MCC) at each layer's fixed cutoff, scored separately per
   mechanism against the shared absence points. This depends on the
   pseudo-absence points being reasonable negatives.
2. Success-rate / prediction-rate curve (Chung & Fabbri 2003, standard in the
   landslide-susceptibility literature): rank every cell in a common area by
   descending score, and read off what fraction of a mechanism's observed
   points fall inside the top X% of that ranked area. This needs no absence
   points at all -- the "area" denominator is the whole raster -- so it
   sidesteps the pseudo-absence-quality question entirely. Per Erkan's stated
   preference, this is the headline comparison; ROC-AUC/PR-AUC are still
   computed and kept in the CSV, but as secondary columns, not the summary
   number, because a single AUC value averages over thresholds nobody would
   operationally use and is known to inflate under spatially clustered test
   points -- which these are (chained along channels and scarps).

The dDEM's native grid (1 m) is far finer than the probability grids' (10 m,
2.006 km^2 footprint); success-rate area fractions are computed only within
that shared footprint so "top 10% of area" means the same physical area for
every layer.

Deliberately NOT done here: no window/neighbourhood scoring around each
point (single nearest-cell sampling only). Erkan raised this
(2026-09-22) and it is a legitimate next step, but the user asked to finish
this pass with single-cell scoring first and apply a window as a separate,
explicit step -- see the confusion_metrics() call sites if/when that lands,
and apply any window to the absence points too, symmetrically, or FP/
specificity stop meaning anything.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from sklearn.metrics import average_precision_score, roc_auc_score

CMAP_STEPS = 256
MECHANISMS = ["LS", "Runoff", "Firehose"]
POOLED = "All"          # every initiation point, type ignored
AREA_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)


def read_probability_layer(path, cmap_name="YlOrRd", steps=CMAP_STEPS, assumed_crs="EPSG:32610"):
    """Read a probability layer as a [0, 1] array plus its transform and CRS.

    Single-band input (a real model grid, e.g. an .asc with no declared CRS)
    is read directly; `assumed_crs` is applied only when the file itself
    declares none. 4-band RGBA input (a rendered image, no surviving raw
    value) is inverted against a reference colormap instead -- kept as a
    fallback for when no raw grid is available; prefer a real grid whenever
    one exists, as here. Returns max_color_error as NaN for a real grid.
    """
    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs or rasterio.crs.CRS.from_string(assumed_crs)
    if data.shape[0] == 1:
        return data[0].astype(float), transform, crs, np.nan
    if data.shape[0] < 4:
        raise ValueError(f"{path}: expected 1 band (real grid) or 4 (RGBA render), got {data.shape[0]}")
    ref = (matplotlib.colormaps[cmap_name](np.linspace(0, 1, steps))[:, :3] * 255).round()
    r, g, b, alpha = data
    valid = alpha > 0
    value = np.full(r.shape, np.nan)
    rgb = np.stack([r[valid], g[valid], b[valid]], axis=1).astype(int)
    dist = np.abs(rgb[:, None, :] - ref[None, :, :]).sum(axis=2)
    step = dist.argmin(axis=1)
    value[valid] = step / (steps - 1)
    return value, transform, crs, int(dist[np.arange(len(step)), step].max())


def sample_at_points(array, transform, points):
    """Nearest-pixel value at each point; NaN for points outside the grid."""
    rows, cols = rasterio.transform.rowcol(transform, points.geometry.x, points.geometry.y)
    out = np.full(len(points), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if 0 <= r < array.shape[0] and 0 <= c < array.shape[1]:
            out[i] = array[r, c]
    return out


def window_max(array, transform, points, window_m):
    """Max value within a window_m x window_m physical box around each
    point's nearest cell. The window is expressed in metres and converted to
    an odd cell count per layer's own resolution, so "30 m" is the same
    physical footprint on a 1 m grid (31x31 cells) and a 10 m grid (3x3
    cells) -- not the same pixel count. Returns (values, cells_per_side,
    actual_window_m). Apply to positive AND absence points identically: an
    asymmetric window inflates recall without costing specificity.
    """
    cellsize = abs(transform.a)
    cells = max(1, round(window_m / cellsize))
    if cells % 2 == 0:
        cells += 1
    radius = cells // 2
    rows, cols = rasterio.transform.rowcol(transform, points.geometry.x, points.geometry.y)
    out = np.full(len(points), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        r0, r1 = max(0, r - radius), min(array.shape[0], r + radius + 1)
        c0, c1 = max(0, c - radius), min(array.shape[1], c + radius + 1)
        window = array[r0:r1, c0:c1]
        if np.isfinite(window).any():
            out[i] = np.nanmax(window)
    return out, cells, cells * cellsize


def crop_to_bounds(array, transform, bounds):
    """Slice array/transform to a bounding box, for a fair cross-resolution area comparison."""
    window = from_bounds(*bounds, transform=transform).round_lengths().round_offsets()
    row0, row1 = max(0, window.row_off), min(array.shape[0], window.row_off + window.height)
    col0, col1 = max(0, window.col_off), min(array.shape[1], window.col_off + window.width)
    return array[int(row0):int(row1), int(col0):int(col1)]


def confusion_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan          # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan   # a.k.a. threat score / Jaccard
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=precision, recall=recall,
                specificity=specificity, accuracy=accuracy, F1=f1, CSI=csi, MCC=mcc)


def threshold_free_scores(y_true, score):
    """ROC-AUC and PR-AUC; score must increase with predicted positive-ness.

    Secondary metrics here, not the headline -- see module docstring.
    """
    valid = np.isfinite(score)
    y, s = np.asarray(y_true, bool)[valid], score[valid]
    if y.sum() == 0 or (~y).sum() == 0:
        return dict(roc_auc=np.nan, pr_auc=np.nan)
    return dict(roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)))


def area_fraction_needed(domain_scores, point_scores):
    """For each point, the share of the domain (ranked by descending score)
    that must be flagged before that point's score is included.

    Ties are mid-ranked: a point tied with k other domain cells at the same
    value needs half that tied block flagged, not the far edge of it. This
    matters here specifically -- P_Landslide is 92% exact zero, so a naive
    descending-scan threshold at the 10% area mark lands exactly on that tied
    block, and without mid-ranking every zero-or-higher point (i.e. every
    point, since the grid's minimum is 0) would score as "captured within the
    top 10%", which is an artifact of the tie, not a claim about the model.
    """
    domain = np.sort(domain_scores[np.isfinite(domain_scores)])
    n = len(domain)
    le = np.searchsorted(domain, point_scores, side="right")   # count(domain <= score)
    lt = np.searchsorted(domain, point_scores, side="left")    # count(domain < score)
    needed = (n - le + 0.5 * (le - lt)) / n
    return np.where(np.isfinite(point_scores), needed, np.nan)


def capture_curve(needed, fractions):
    """Fraction of points captured at each area fraction, from area_fraction_needed()."""
    needed = needed[np.isfinite(needed)]
    if len(needed) == 0:
        return np.full(len(fractions), np.nan)
    return np.array([float(np.mean(needed <= f)) for f in fractions])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/mnt/c/Users/amehedi/Downloads")
    P = D / "pioneer"
    PF = D / "pioneer/probability_fields"
    ap.add_argument("--probability-layers", nargs="+",
                    default=[str(PF / "P_Landslide.tif"), str(PF / "P_Combined_H.tif"),
                             str(PF / "P_Channel_ini.tif")],
                    help="Probability rasters, single-band real grid or 4-band RGBA render")
    ap.add_argument("--dod-continuous", type=Path,
                    default=P / "main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif")
    ap.add_argument("--dod-erosion-cutoff-m", type=float, default=0.50,
                    help="Positive depth in metres; a cell counts as erosion if DoD < -this value")
    ap.add_argument("--initiation-points", type=Path,
                    default=P / "Initiation_Points_final/Initiation_Points_final.shp")
    ap.add_argument("--absence-points", type=Path, default=P / "negative_points/random90_unbiased.shp")
    ap.add_argument("--domain-raster", type=Path, default=None,
                    help="If set with --domain-max, drop positive AND absence points where this "
                         "raster exceeds the max before scoring (e.g. G, to exclude lowland/valley "
                         "-floor terrain implausible for initiation). Not applied to the success-rate "
                         "area-ranking step in this version -- see module docstring.")
    ap.add_argument("--domain-max", type=float, default=None)
    ap.add_argument("--mechanisms", nargs="+", default=MECHANISMS + [POOLED],
                    help=f"FailureTyp values to score; {POOLED!r} pools every initiation point")
    ap.add_argument("--probability-cutoff", type=float, default=0.5)
    ap.add_argument("--window-m", type=float, default=None,
                    help="If set, score each point as the max within this physical footprint "
                         "(applied to positive AND absence points alike) instead of the single "
                         "nearest cell.")
    ap.add_argument("--output-dir", type=Path, default=P / "hazard_layer_validation")
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"Output directory is not empty: {args.output_dir}")

    ip = gpd.read_file(args.initiation_points)
    absence = gpd.read_file(args.absence_points)
    domain_note = None
    if args.domain_raster is not None:
        if args.domain_max is None:
            ap.error("--domain-raster requires --domain-max")
        with rasterio.open(args.domain_raster) as src:
            dvalue, dtransform, dcrs = src.read(1, masked=True).filled(np.nan), src.transform, src.crs
        def in_domain(points):
            v = sample_at_points(dvalue, dtransform, points.to_crs(dcrs))
            return np.isfinite(v) & (v <= args.domain_max)
        n_ip_before, n_absence_before = len(ip), len(absence)
        ip = ip[in_domain(ip)].reset_index(drop=True)
        absence = absence[in_domain(absence)].reset_index(drop=True)
        domain_note = (f"{args.domain_raster.name} <= {args.domain_max}: kept {len(ip)}/{n_ip_before} "
                       f"initiation points, {len(absence)}/{n_absence_before} absence points")
        print(domain_note)

    def of_type(points, m):
        return points if m == POOLED else points[points.FailureTyp == m]

    n_by_mechanism = {m: len(of_type(ip, m)) for m in args.mechanisms}
    for m, n in n_by_mechanism.items():
        if n == 0:
            ap.error(f"No points with FailureTyp == {m!r}")

    # Build one uniform "layers" list: every entry is a score raster (higher
    # = more positive-like) plus the cutoff that turns it into a prediction.
    # The dDEM joins this list as -depth so it is scored exactly like the
    # probability grids, rather than as a special case.
    layers = []
    colormap_errors = {}
    for layer_path in args.probability_layers:
        layer_path = Path(layer_path)
        value, transform, crs, max_err = read_probability_layer(layer_path)
        colormap_errors[layer_path.stem] = max_err
        kind = "real single-band grid" if np.isnan(max_err) else "reconstructed probability (RGBA render)"
        layers.append(dict(name=layer_path.stem, score=value, transform=transform, crs=crs,
                           cutoff=args.probability_cutoff, kind=kind))

    with rasterio.open(args.dod_continuous) as src:
        dod = src.read(1, masked=True).filled(np.nan)
        dod_transform, dod_crs = src.transform, src.crs
    layers.append(dict(name="dod_erosion", score=-dod, transform=dod_transform, crs=dod_crs,
                       cutoff=args.dod_erosion_cutoff_m,
                       kind=f"continuous DoD, erosion side at -{args.dod_erosion_cutoff_m:.2f} m"))

    # Common footprint for area-fraction comparisons: the probability grids'
    # extent (2.006 km^2), which sits entirely inside the dDEM's (13.27 km^2).
    common_bounds = rasterio.transform.array_bounds(*layers[0]["score"].shape, layers[0]["transform"])

    def sample(arr, transform, pts):
        if args.window_m is None:
            return sample_at_points(arr, transform, pts)
        return window_max(arr, transform, pts, args.window_m)[0]

    confusion_rows, capture_rows, curves = [], [], {}
    window_sizes = {}

    for layer in layers:
        pts_absence = absence.to_crs(layer["crs"])
        absence_score = sample(layer["score"], layer["transform"], pts_absence)
        absence_pred = absence_score >= layer["cutoff"]
        if args.window_m is not None:
            _, cells, actual_m = window_max(layer["score"], layer["transform"], pts_absence, args.window_m)
            window_sizes[layer["name"]] = f"{cells}x{cells} cells = {actual_m:.1f} m"

        domain = crop_to_bounds(layer["score"], layer["transform"], common_bounds)
        curves[layer["name"]] = {}

        for mech in args.mechanisms:
            positive = of_type(ip, mech).to_crs(layer["crs"])
            score = sample(layer["score"], layer["transform"], positive)
            pred = score >= layer["cutoff"]

            y_true = np.r_[np.ones(len(positive)), np.zeros(len(absence))]
            y_pred = np.r_[pred, absence_pred]
            y_score = np.r_[score, absence_score]
            row = dict(layer=layer["name"], mechanism=mech, kind=layer["kind"], n_positive=len(positive),
                      n_absence=len(absence), cutoff=layer["cutoff"],
                      **confusion_metrics(y_true, y_pred), **threshold_free_scores(y_true, y_score))
            confusion_rows.append(row)

            needed = area_fraction_needed(domain, score)
            fine_fractions = np.linspace(0.01, 1.0, 200)
            curves[layer["name"]][mech] = (fine_fractions, capture_curve(needed, fine_fractions))
            at_f = dict(zip(AREA_FRACTIONS, capture_curve(needed, np.array(AREA_FRACTIONS))))
            capture_rows.append(dict(layer=layer["name"], mechanism=mech, n_positive=len(positive),
                                     **{f"capture_at_{int(f*100)}pct_area": v for f, v in at_f.items()}))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "validation_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(confusion_rows[0])); w.writeheader(); w.writerows(confusion_rows)
    with (args.output_dir / "success_rate_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(capture_rows[0])); w.writeheader(); w.writerows(capture_rows)

    # Cross-tab: mean score per probability layer x observed mechanism (all points).
    # Does not assume what Combined_H represents -- lets that be read off the numbers.
    crosstab = {}
    for layer in layers:
        s = sample_at_points(layer["score"], layer["transform"], ip.to_crs(layer["crs"]))
        crosstab[layer["name"]] = {m: float(np.nanmean(s[ip.FailureTyp.values == m])) for m in MECHANISMS}
    (args.output_dir / "layer_vs_mechanism_crosstab.json").write_text(json.dumps(crosstab, indent=2))

    reconstructed = sorted({r["layer"] for r in confusion_rows if "reconstructed" in r["kind"]})
    caveats = [
        "Sample size differs sharply by mechanism (LS=47, Runoff=22, Firehose=20); "
        "n_positive is reported on every row -- weight Runoff/Firehose results accordingly.",
        "Single nearest-cell sampling only, no neighbourhood/window around each point. "
        "A window pass (applied symmetrically to positive AND absence points) is a planned "
        "next step, not done here.",
        "Absence points are spatially buffered pseudo-absences, not field-verified; the "
        "confusion-matrix table depends on their quality. The success-rate table does not "
        "use them at all, by design.",
        "Initiation points are remotely mapped from the same UAV survey used to build the "
        "hazard layers and the dDEM, not independently field-observed.",
        "The probability grids declare no CRS and no nodata value; EPSG:32610 was assumed "
        "and zero was treated as a genuine low-probability value, not nodata -- see the "
        "ASC-INPUT NOTE in the module docstring.",
        "Layer name -> mechanism correspondence (e.g. Combined_H) is not independently "
        "confirmed; see layer_vs_mechanism_crosstab.json.",
        "ROC-AUC/PR-AUC are reported but are secondary: they average over thresholds nobody "
        "would operationally use and are known to inflate under spatially clustered test "
        "points, which these are. The success-rate table is the headline comparison.",
    ]
    if reconstructed:
        caveats.insert(0, f"{reconstructed} were reconstructed from RGBA colormap renders, not "
                          "raw probability grids. Confirm normalisation with Erkan.")
    if domain_note:
        caveats.insert(0, domain_note)
        caveats.insert(1, "The domain filter was applied to POINTS only (both positive and "
                          "absence, symmetrically), not to the success-rate area-ranking step -- "
                          "that step still ranks the whole raster, not just the restricted "
                          "domain. Treat the success-rate table as not yet consistent with this "
                          "run's confusion-matrix table.")
        caveats.insert(2, "G is highly right-skewed and mechanism-dependent: Firehose/Runoff "
                          "initiate in channels (physically higher G) while LS initiates on "
                          "hillslopes (lower G), so a single G cutoff does not exclude 'lowlands' "
                          "evenly across mechanisms -- it will tend to thin Firehose/Runoff more "
                          "than LS. See the G-threshold sweep run alongside this analysis.")
    report = dict(mechanisms=n_by_mechanism,
                 absence_class=f"pseudo-absence, >50 m from any initiation point (n={len(absence)})",
                 sampling="single nearest cell" if args.window_m is None
                          else f"max within {args.window_m} m window, applied to positive and absence points alike",
                 window_sizes_by_layer=window_sizes,
                 common_area_footprint_km2=round(
                     (common_bounds[2] - common_bounds[0]) * (common_bounds[3] - common_bounds[1]) / 1e6, 3),
                 caveats=caveats)
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2))

    # Success-rate curves: one panel per mechanism, every layer overlaid, plus the
    # no-skill diagonal. This is the headline figure.
    fig, axes = plt.subplots(1, len(args.mechanisms), figsize=(5.2 * len(args.mechanisms), 4.6), sharey=True)
    for ax, mech in zip(np.atleast_1d(axes), args.mechanisms):
        ax.plot([0, 1], [0, 1], "k--", lw=.8, label="no skill")
        for layer in layers:
            af, cap = curves[layer["name"]][mech]
            ax.plot(af, cap, label=layer["name"])
        ax.set_title(f"{mech} (n={n_by_mechanism[mech]})")
        ax.set_xlabel("fraction of area flagged"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("fraction of points captured")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Prediction-rate (success-rate) curves, by mechanism")
    fig.tight_layout()
    fig.savefig(args.output_dir / "success_rate_curves.png", dpi=150)
    plt.close(fig)

    # Confusion-matrix summary bars, one row per mechanism.
    fig, axes = plt.subplots(len(args.mechanisms), 1, figsize=(8, 3.2 * len(args.mechanisms)))
    for ax, mech in zip(np.atleast_1d(axes), args.mechanisms):
        sub = [r for r in confusion_rows if r["mechanism"] == mech]
        x = np.arange(len(sub)); width = .25
        for off, metric in zip([-width, 0, width], ["CSI", "F1", "recall"]):
            ax.bar(x + off, [r[metric] for r in sub], width, label=metric)
        ax.set_xticks(x); ax.set_xticklabels([r["layer"] for r in sub], rotation=20, fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(f"{mech} (n={n_by_mechanism[mech]})"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "layer_comparison.png", dpi=150)
    plt.close(fig)

    print(f"absence: pseudo-absence (n={len(absence)}, shared across mechanisms)\n")
    hdr = (f"{'layer':<14}{'mech':<10}{'n':>4}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}"
          f"{'recall':>8}{'F1':>7}{'CSI':>7}{'ROC-AUC':>9}")
    print(hdr)
    for r in confusion_rows:
        print(f"{r['layer']:<14}{r['mechanism']:<10}{r['n_positive']:>4}{r['TP']:>4}{r['FP']:>4}"
              f"{r['FN']:>4}{r['TN']:>4}{r['recall']:>8.2f}{r['F1']:>7.2f}{r['CSI']:>7.2f}{r['roc_auc']:>9.2f}")
    print("\nsuccess-rate: fraction of points captured within the top 10% / 20% of area, by layer x mechanism")
    for r in capture_rows:
        print(f"  {r['layer']:<14}{r['mechanism']:<10}n={r['n_positive']:<4}"
              f"10%: {r['capture_at_10pct_area']:.2f}   20%: {r['capture_at_20pct_area']:.2f}")
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
