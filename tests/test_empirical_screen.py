import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from screen_dem_difference_empirical import empirical_limit, screen


class EmpiricalScreenTests(unittest.TestCase):
    def test_threshold_uses_fit_pixels_only(self):
        data = np.r_[np.linspace(-2, 2, 200), np.zeros(200)]
        fit = np.arange(400) < 200
        check = ~fit
        threshold, initial = empirical_limit(data, fit, check)
        data[check] = 100
        repeated, changed = empirical_limit(data, fit, check)
        self.assertEqual(threshold, repeated)
        self.assertEqual(initial['check']['within_threshold_fraction'], 1)
        self.assertEqual(changed['check']['within_threshold_fraction'], 0)

    def test_screen_preserves_detected_values_and_excluded_nodata(self):
        dod = np.array([-4, -3, 0, 3, 4, 7, np.nan], dtype='float32')
        eligible = np.array([True, True, True, True, True, False, True])
        output, classes = screen(dod, eligible, 3)
        np.testing.assert_array_equal(output, np.array([-4, 0, 0, 0, 4, np.nan, np.nan], dtype='float32'))
        np.testing.assert_array_equal(classes, np.array([-1, 0, 0, 0, 1, np.nan, np.nan], dtype='float32'))


if __name__ == '__main__':
    unittest.main()
