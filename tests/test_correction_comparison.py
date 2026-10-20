import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_dem_corrections as compare


class ComparisonTests(unittest.TestCase):
    def test_spatial_fit_recovers_shift_without_removing_local_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = dict(driver='GTiff', count=1, width=40, height=40, dtype='float32',
                     nodata=-9999, crs='EPSG:32610', transform=from_origin(500000, 4100000, 1, 1))
            pre = np.full((40, 40), 700., dtype='float32')
            post = pre - 18.
            post[15:20, 15:20] += 10.
            mask = np.ones((40, 40), dtype='float32')
            mask[15:20, 15:20] = 0
            post[0, 0] = -9999
            for name, a in [('pre', pre), ('post', post), ('mask', mask)]:
                with rasterio.open(root/f'{name}.tif', 'w', **p) as dst:
                    dst.write(a, 1)
            argv = ['--pre', str(root/'pre.tif'), '--post', str(root/'post.tif'),
                    '--stable-mask', str(root/'mask.tif'), '--block-size-m', '5', '--output-dir', str(root/'out')]
            compare.main(argv)
            a, profile = compare.read(root/'out/dod_vertical_shift.tif')
            self.assertTrue(np.isnan(a[0, 0]))
            self.assertEqual(a[1, 1], 0)
            self.assertEqual(a[16, 16], 10)
            self.assertEqual(profile['transform'], p['transform'])
            with self.assertRaisesRegex(ValueError, 'not empty'):
                compare.main(argv)

    def test_spatial_blocks_are_disjoint_and_preserve_stable_selection(self):
        stable = np.ones((50, 50), dtype=bool)
        stable[:4, :4] = False
        fit, validation = compare.spatial_split(stable, 1, 10)
        self.assertFalse((fit & validation).any())
        np.testing.assert_array_equal(fit | validation, stable)
        self.assertGreater(validation.sum(), 0)


if __name__ == '__main__':
    unittest.main()
