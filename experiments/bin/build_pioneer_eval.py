"""Build Erkan's pioneer_eval bundle from its sources: the zip to send, and the copy
that the notebooks in his model folder import.

Never edit either copy by hand: change the sources below and rebuild.

    python experiments/bin/build_pioneer_eval.py

Sources:
    experiments/pioneer_eval/            __init__.py (the Pioneer settings) and README.txt
    ~/landlab-grid-validation/src/landlab_grid_validation   the scoring package, copied in
    the Pioneer data folder              points, corrected dDEM, model DEM

Outputs:
    Downloads/pioneer/hazard_model/pioneer_eval.zip                the bundle to send
    Downloads/pioneer/hazard_model/hazard_model/pioneer_eval/      the copy the notebooks import
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIONEER = Path("/mnt/c/Users/amehedi/Downloads/pioneer")
SOURCES = {
    "__init__.py": HERE.parent / "pioneer_eval" / "__init__.py",
    "README.txt": HERE.parent / "pioneer_eval" / "README.txt",
}
PACKAGE = Path.home() / "landlab-grid-validation" / "src" / "landlab_grid_validation"
DATA = [PIONEER / "obs_data" / f"Initiation_Points_final.{ext}"
        for ext in ("shp", "shx", "dbf", "prj", "cpg")]
DATA += [PIONEER / "ddem_correction" / "m6_gain_mask" / "dod_dndvi_test.tif",
         PIONEER / "lidar_footprint_10m" / "topographic__elevation.asc"]


def assemble(folder: Path) -> None:
    """Lay the bundle out in folder: the two sources, the package, the data."""
    (folder / "data").mkdir(parents=True)
    (folder / "landlab_grid_validation").mkdir()
    for name, src in SOURCES.items():
        shutil.copy2(src, folder / name)
    for src in sorted(PACKAGE.glob("*.py")):
        shutil.copy2(src, folder / "landlab_grid_validation" / src.name)
    for src in DATA:
        shutil.copy2(src, folder / "data" / src.name)


def build(zip_path: Path, notebook_dir: Path | None) -> Path:
    """Write the zip and, unless notebook_dir is None, replace notebook_dir/pioneer_eval."""
    missing = [p for p in [*SOURCES.values(), PACKAGE, *DATA] if not p.exists()]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(map(str, missing)))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "pioneer_eval"
        assemble(folder)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(folder.parent))
        if notebook_dir is not None:
            target = notebook_dir / "pioneer_eval"
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(folder, target)
    return zip_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=PIONEER / "hazard_model" / "pioneer_eval.zip",
                    help="where to write the bundle to send")
    ap.add_argument("--notebook-dir", type=Path, default=PIONEER / "hazard_model" / "hazard_model",
                    help="folder of Erkan's notebooks; its pioneer_eval/ is replaced by the new build")
    ap.add_argument("--no-install", action="store_true", help="write the zip only")
    args = ap.parse_args(argv)
    zip_path = build(args.zip, None if args.no_install else args.notebook_dir)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    if not args.no_install:
        print(f"replaced {args.notebook_dir / 'pioneer_eval'}")


if __name__ == "__main__":
    main()
