"""Build the pioneer_eval folder and zip for Erkan from their sources.

Never edit the copy in Downloads by hand: change the sources below and rebuild.

    python experiments/bin/build_pioneer_eval.py

Sources:
    experiments/pioneer_eval/            __init__.py and README.txt
    experiments/bin/auc_experiment.py    the scoring
    ~/landlab-grid-validation/src/landlab_grid_validation   the package, copied in
    the Pioneer data folder              points, corrected dDEM, model DEM
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIONEER = Path("/mnt/c/Users/amehedi/Downloads/pioneer")
SOURCES = {
    "__init__.py": HERE.parent / "pioneer_eval" / "__init__.py",
    "README.txt": HERE.parent / "pioneer_eval" / "README.txt",
    "auc_experiment.py": HERE / "auc_experiment.py",
}
PACKAGE = Path.home() / "landlab-grid-validation" / "src" / "landlab_grid_validation"
DATA = [PIONEER / "obs_data" / f"Initiation_Points_final.{ext}"
        for ext in ("shp", "shx", "dbf", "prj", "cpg")]
DATA += [PIONEER / "ddem_correction" / "m6_gain_mask" / "dod_dndvi_test.tif",
         PIONEER / "lidar_footprint_10m" / "topographic__elevation.asc"]


def build(out_dir: Path) -> Path:
    missing = [p for p in [*SOURCES.values(), PACKAGE, *DATA] if not p.exists()]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(map(str, missing)))

    folder = out_dir / "pioneer_eval"
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "data").mkdir(parents=True)
    (folder / "landlab_grid_validation").mkdir()
    for name, src in SOURCES.items():
        shutil.copy2(src, folder / name)
    for src in sorted(PACKAGE.glob("*.py")):
        shutil.copy2(src, folder / "landlab_grid_validation" / src.name)
    for src in DATA:
        shutil.copy2(src, folder / "data" / src.name)

    zip_path = out_dir / "pioneer_eval.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(out_dir))
    return zip_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=PIONEER / "hazard_model")
    args = ap.parse_args(argv)
    zip_path = build(args.out_dir)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    main()
