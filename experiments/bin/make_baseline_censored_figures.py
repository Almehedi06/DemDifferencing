"""Create baseline P>0.01 comparison maps using locally saved PDF basemaps.

Offline: no tile downloads. Reuses the clean, separate background-image XObject
in each existing summer PDF, not a screenshot or a model-overlay composite.
Model values come from the original baseline TIFFs. The display operation is
the same bilinear Web Mercator reprojection followed by censoring used by the
existing panels_2d function. No source raster or metric is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
from pypdf import PdfReader

import make_final_figures as figures


def file_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def extract_basemap(pdf_path, map_extent):
    """Map the embedded RGB image's PDF rectangle into the known map extent.

    Matplotlib rounds image placement to output pixels. Account for the exact
    image rectangle versus axes clipping rectangle rather than stretching the
    extracted image to the axes and introducing an avoidable small shift.
    """
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    obj = page['/Resources']['/XObject']['/I1'].get_object()
    if obj.get('/ColorSpace') != '/DeviceRGB' or obj.get('/SMask') is not None:
        raise ValueError(f"{pdf_path}: /I1 is not the expected opaque RGB basemap")
    operations = page.get_contents().operations
    image_index = next(i for i, (args, op) in enumerate(operations)
                       if op == b'Do' and str(args[0]) == '/I1')
    preceding = operations[:image_index]
    image_matrix = next(args for args, op in reversed(preceding) if op == b'cm')
    axes_rect = next(args for args, op in reversed(preceding) if op == b're')
    width, shear_y, shear_x, height, left, bottom = map(float, image_matrix)
    ax_left, ax_bottom, ax_width, ax_height = map(float, axes_rect)
    if shear_x != 0 or shear_y != 0 or min(width, height, ax_width, ax_height) <= 0:
        raise ValueError('Unexpected rotated/invalid PDF image placement')
    x0, x1, y0, y1 = map_extent
    if not np.isclose(ax_width/ax_height, (x1-x0)/(y1-y0), rtol=1e-5):
        raise ValueError('PDF map aspect ratio does not match the reference geographic extent')
    def x(value):
        return x0 + (value-ax_left)/ax_width*(x1-x0)
    def y(value):
        return y0 + (value-ax_bottom)/ax_height*(y1-y0)
    extent = (x(left), x(left+width), y(bottom), y(bottom+height))
    image = np.asarray(page.images['/I1'].image.convert('RGB')).copy()
    record = {**file_record(pdf_path), "pdf_image": "/I1", "size_pixels": list(image.shape[:2]),
              "axes_rect_pdf_points": list(map(float, axes_rect)),
              "image_matrix_pdf_points": list(map(float, image_matrix)),
              "extracted_basemap_extent_epsg3857": extent}
    return image, extent, record


def main():
    downloads = Path('/mnt/c/Users/amehedi/Downloads')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=downloads/'pioneer/regional_baseline/regional_baseline')
    parser.add_argument('--template-model', type=Path, default=downloads/'regional_summer/P_Landslide_regional_summer_Cs_1k.tif')
    parser.add_argument('--local-reference', type=Path, default=downloads/'probability_fields/P_Landslide.tif')
    parser.add_argument('--points', type=Path, default=downloads/'pioneer/Initiation_Points_final/Initiation_Points_final.shp')
    parser.add_argument('--output-dir', type=Path, default=downloads/'pioneer/regional_comparison/figures')
    parser.add_argument('--template-dir', type=Path, default=downloads/'pioneer/regional_comparison/figures')
    args = parser.parse_args()
    stems = ['baseline_regional_layers_2d_gt0p01', 'baseline_uav_window_2d_gt0p01']
    outputs = [args.output_dir/f'{stem}.{ext}' for stem in stems for ext in ['png', 'pdf']]
    manifest_path = args.output_dir/'baseline_gt0p01_manifest.json'
    if any(path.exists() for path in outputs+[manifest_path]):
        parser.error('A target baseline output already exists; use a new output directory')
    paths = [args.baseline_dir/f'{name}_regional_baseline.tif' for name, _ in figures.LAYERS]
    with rasterio.open(args.template_model) as template:
        for path in paths:
            with rasterio.open(path) as src:
                if (src.crs != template.crs or src.shape != template.shape
                        or not src.transform.almost_equals(template.transform)):
                    raise ValueError('Baseline and summer regional grids differ; cannot infer the template map extent')
                data = src.read(1, masked=True).compressed()
                if src.count != 1 or not len(data) or not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
                    raise ValueError(f'Invalid numeric probability raster: {path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = gpd.read_file(args.points).to_crs(3857)
    _, regional_extent = figures.warp_to_merc(paths[0])
    _, local_extent = figures.warp_to_merc(args.local_reference)
    records = []
    specs = [(stems[0], 'summer_regional_layers_2d_gt0p01.pdf', regional_extent, False, 5000, '5 km'),
             (stems[1], 'summer_uav_window_2d_gt0p01.pdf', local_extent, True, 1000, '1 km')]
    for stem, template_name, extent, show_points, bar_m, bar_label in specs:
        image, image_extent, provenance = extract_basemap(args.template_dir/template_name, extent)
        lonlat = rasterio.warp.transform_bounds(3857, 4326, extent[0], extent[2], extent[1], extent[3])
        latitude = (lonlat[1]+lonlat[3])/2
        figures.panels_2d(args.baseline_dir, '_regional_baseline', image, image_extent,
                          extent, latitude, points, args.output_dir, stem,
                          bar_m, bar_label, show_points, floor=.01)
        records.append({"figure": stem, "map_extent_epsg3857": extent, "basemap": provenance,
                        "show_observation_points": show_points})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "display_rule": "Only reprojected baseline probabilities > 0.01 are shown; <= 0.01 is transparent",
                "model_values_source": "Original baseline GeoTIFFs; probability values are not taken from PDFs",
                "network_access": False, "basemap_note": "Existing PDF background-image layer at its saved display resolution",
                "scripts": [file_record(__file__), file_record(figures.__file__)],
                "models": [file_record(path) for path in paths],
                "extent_references": [file_record(args.template_model), file_record(args.local_reference)],
                "observation_files": [file_record(args.points.with_suffix(ext)) for ext in ['.shp', '.shx', '.dbf', '.prj']],
                "figures": records, "outputs": [file_record(path) for path in outputs]}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
