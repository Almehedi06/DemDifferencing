import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))   # pipeline helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))               # the diagnostic scripts
import refine_stable_mask_landcover as landcover


class LandCoverMaskTests(unittest.TestCase):
    def test_refinement_cannot_add_pixels_or_accept_missing_landcover(self):
        base = np.array([[1, 0, 1], [np.nan, 1, 1]], dtype=float)
        cover = np.array([[71, 71, 42], [31, np.nan, 31]], dtype=float)
        actual = landcover.refine(base, cover, [31, 71])
        expected = np.array([[1, 0, 0], [255, 255, 1]], dtype='uint8')
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(base[actual == 1] == 1))

    def test_rejects_continuous_values_mistaken_for_classes(self):
        with self.assertRaisesRegex(ValueError, 'integer class codes'):
            landcover.refine(np.ones((2, 2)), np.full((2, 2), 31.4), [31])

    def test_nearest_resampling_and_original_files_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = dict(driver='GTiff', count=1, width=4, height=4, dtype='uint8',
                           crs='EPSG:32610', nodata=255, transform=from_origin(500000, 4100000, 1, 1))
            base = np.ones((4, 4), dtype='uint8')
            base[0, 0] = 0
            base[2, 0] = 255
            with rasterio.open(root / 'mask.tif', 'w', **profile) as dst:
                dst.write(base, 1)
            coarse = dict(profile, width=2, height=2, nodata=250,
                          transform=from_origin(500000, 4100000, 2, 2))
            with rasterio.open(root / 'cover.tif', 'w', **coarse) as dst:
                dst.write(np.array([[71, 42], [31, 250]], dtype='uint8'), 1)
            before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ['mask.tif', 'cover.tif']}
            args = ['--stable-mask', str(root / 'mask.tif'), '--land-cover', str(root / 'cover.tif'),
                    '--classes', '31', '71', '--min-stable-pixels', '1', '--output-dir', str(root / 'out')]
            landcover.main(args)
            with rasterio.open(root / 'out/stable_open_ground.tif') as src:
                self.assertEqual(src.transform, profile['transform'])
                self.assertEqual(src.crs, rasterio.crs.CRS.from_epsg(32610))
                actual = src.read(1)
            expected = np.array([[0, 1, 0, 0], [1, 1, 0, 0], [255, 1, 255, 255], [1, 1, 255, 255]], dtype='uint8')
            np.testing.assert_array_equal(actual, expected)
            with self.assertRaisesRegex(ValueError, 'nonexistent'):
                landcover.main(args)
            for name, digest in before.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
