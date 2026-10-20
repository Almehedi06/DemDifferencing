import sys
import unittest
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_dem_stages import check_blocks, coverage, masks


class DiagnosticTests(unittest.TestCase):
    def test_fit_check_masks_exclude_invalid_without_cross_contamination(self):
        split = np.ones((30, 30), dtype=float)
        split[15:] = 2
        split[0, 0] = np.nan
        valid = np.ones(split.shape, dtype=bool)
        valid[20, 20] = False
        fit, check = masks(split, valid)
        self.assertFalse((fit & check).any())
        self.assertFalse(fit[0, 0])
        self.assertFalse(check[20, 20])
        self.assertEqual(int(fit.sum()), 449)
        self.assertEqual(int(check.sum()), 449)
        split[0, 1] = 255
        with self.assertRaisesRegex(ValueError, 'mask codes'):
            masks(split, valid)

    def test_uncertainty_check_detects_large_unexplained_residuals(self):
        dod = np.zeros(1000, dtype=float)
        dod[:200] = 4
        sigma = np.ones_like(dod)
        sigma[999] = np.nan
        result = coverage(dod, sigma, np.ones(dod.shape, dtype=bool))
        self.assertEqual(result['n'], 999)
        self.assertAlmostEqual(result['within_nominal_95_fraction'], 799 / 999)
        self.assertLess(result['within_nominal_95_fraction'], .93)

    def test_block_report_uses_check_pixels_and_exports_lon_lat(self):
        dod = np.full((20, 20), 50., dtype=float)
        check = np.zeros(dod.shape, dtype=bool)
        check[:10, :10] = True
        dod[:10, :10] = .25
        profile = dict(transform=from_origin(500000, 4100000, 1, 1), crs=CRS.from_epsg(32610))
        rows, features = check_blocks(dod, check, profile, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['n'], 100)
        self.assertEqual(rows[0]['median_m'], .25)
        xy = features['features'][0]['geometry']['coordinates'][0][0]
        self.assertAlmostEqual(xy[0], -123., places=5)
        self.assertTrue(36 < xy[1] < 38)


if __name__ == '__main__':
    unittest.main()
