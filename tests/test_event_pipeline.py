"""Offline tests; run with python -m unittest discover -s tests -v."""
import sys
import tempfile
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_stable_cells_batch as builder
import run_event_pipeline as event


def write(path, values, transform=None, nodata=-9999):
    with rasterio.open(path, 'w', driver='GTiff', count=1, height=values.shape[0], width=values.shape[1],
                       dtype=values.dtype, crs='EPSG:32610', transform=transform or from_origin(500000, 4100000, 1, 1), nodata=nodata) as dst:
        dst.write(values, 1)
    return path


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pre = write(self.root/'pre.tif', np.full((8, 8), 700, dtype='float32'))
        # A 10 m difference must be eligible when the raw DoD screen is disabled.
        self.post = write(self.root/'post.tif', np.full((8, 8), 710, dtype='float32'))
        self.g = write(self.root/'pair_G.tif', np.ones((8, 8), dtype='float32'))
        self.cfg = {'batch': {'pairs': [{'id': 'pair', 'pre': 'pre.tif', 'post': 'post.tif'}], 'output_dir': 'out'},
                    'terrain': {'g_source': 'raster', 'g_dir': '.'},
                    'stable_cells': {'g_max': 5, 'elev_min': 600, 'dh_abs_max': None},
                    'quality': {'min_stable_pixels': 2, 'min_stable_fraction': .01}}
        self.config = self.root/'event.yaml'
        self.config.write_text(yaml.safe_dump(self.cfg))

    def test_explicit_paths_resolve_relative_to_config_and_mask_rules(self):
        cfg = builder.load_config(self.config)
        pair, = builder.discover_pairs(cfg)
        self.assertEqual(pair.pre_path, self.pre)
        row = builder.process_pair(pair, cfg, self.root/'out')
        self.assertEqual(row['stable_pixels'], 64)
        cfg['stable_cells']['dh_abs_max'] = 5
        with self.assertRaisesRegex(ValueError, 'insufficient stable'):
            builder.process_pair(pair, cfg, self.root/'screened')

    def test_threshold_boundaries_and_nodata(self):
        pre = np.array([[599., 600., 601., 700., 700.]])
        post = pre.copy()
        g = np.array([[1., 1., 5., 4.9, np.nan]])
        stable, valid, _ = builder.build_stable_mask(pre, post, g, None, {}, self.cfg)
        np.testing.assert_array_equal(stable, [[False, True, False, True, False]])
        self.assertFalse(valid[0, -1])

    def test_bad_ids_and_typo_fail(self):
        cfg = builder.load_config(self.config)
        cfg['batch']['pairs'][0]['id'] = '../escape'
        with self.assertRaisesRegex(ValueError, 'Invalid pair ID'):
            builder.discover_pairs(cfg)
        self.cfg['stable_cells']['g_mx'] = 5
        with self.assertRaisesRegex(ValueError, 'Unknown stable_cells'):
            builder.validate_config(self.cfg)

    def test_directory_discovery_rejects_missing_partners(self):
        cfg = {'batch': {'pre_dir': str(self.root), 'post_dir': str(self.root),
                         'pre_glob': '*_Pre.tif', 'post_glob': '*_Post.tif'}}
        write(self.root/'2_Pre.tif', np.ones((8, 8), dtype='float32'))
        with self.assertRaisesRegex(ValueError, 'Unmatched DEM IDs'):
            builder.discover_pairs(cfg)
        write(self.root/'2_Post.tif', np.ones((8, 8), dtype='float32'))
        self.assertEqual(builder.discover_pairs(cfg)[0].pair_id, '2')

    def test_runner_records_correction_failure(self):
        with patch.object(event, 'preflight'), patch.object(event.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'correction')):
            with self.assertRaisesRegex(SystemExit, 'Failed pairs'):
                event.main(['--config', str(self.config)])
        rows = json.loads((self.root/'out/event_summary.json').read_text())
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertIn('CalledProcessError', rows[0]['error'])

    def test_mismatch_fails_even_if_old_require_flag_disabled(self):
        cfg = builder.load_config(self.config)
        cfg['processing'] = {'require_same_grid': False}
        write(self.post, np.ones((8, 8), dtype='float32'), from_origin(500001, 4100000, 1, 1))
        pair, = builder.discover_pairs(cfg)
        with self.assertRaisesRegex(ValueError, 'grids do not match'):
            builder.process_pair(pair, cfg, self.root/'out')

    def test_integer_raster_nodata(self):
        p = write(self.root/'int.tif', np.array([[1, 255]], dtype='uint8'), nodata=255)
        a, _ = builder.read_raster(p)
        self.assertEqual(a[0, 0], 1)
        self.assertTrue(np.isnan(a[0, 1]))

    def test_runner_passes_generated_mask_and_saves_summary(self):
        with patch.object(event, 'preflight'), patch.object(event.subprocess, 'run') as launch:
            event.main(['--config', str(self.config)])
        cmd = launch.call_args.args[0]
        mask = Path(cmd[cmd.index('--stable-mask') + 1])
        self.assertTrue(mask.exists())
        with rasterio.open(mask) as src:
            self.assertEqual(int((src.read(1) == 1).sum()), 64)
        self.assertTrue((self.root/'out/event_summary.json').exists())
        with patch.object(event, 'preflight'):
            with self.assertRaisesRegex(ValueError, 'not empty'):
                event.main(['--config', str(self.config)])

    def test_production_external_mask_preserves_eligible_large_differences(self):
        import xdem
        import matplotlib.pyplot as plt
        import xDEM_pipeline_Thomas_production as production
        values = np.zeros((8, 8), dtype='uint8')
        values[2:5, 2:5] = 1
        values[0, 0] = 255
        mask_path = write(self.root/'mask.tif', values, nodata=255)
        pipeline = production.CoregBiasPipeline('pre', stable_mask_path=mask_path, min_stable_pixels=2)
        pipeline.ref = xdem.DEM(self.pre)
        with patch.object(production.plt, 'show', side_effect=lambda: plt.close('all')):
            mask = pipeline.build_stable_mask('pair', xdem.DEM(self.post))
        self.assertEqual(int(mask.sum()), 9)
        self.assertFalse(mask[0, 0])
        np.testing.assert_array_equal(mask, values == 1)
        # A shifted mask must fail, even when its shape matches.
        write(mask_path, values, from_origin(500001, 4100000, 1, 1), nodata=255)
        with self.assertRaisesRegex(ValueError, 'reference grid'):
            pipeline.build_stable_mask('pair', xdem.DEM(self.post))

    def test_consolidated_settings_and_legacy_defaults(self):
        cfg = builder.load_config(self.config)
        pair, = builder.discover_pairs(cfg)
        cmd = event.correction_command(pair, 'mask.tif', 'output', cfg)
        self.assertEqual(cmd[cmd.index('--correction-preset') + 1], 'montecito')
        self.assertEqual(cmd[cmd.index('--lod-method') + 1], 'none')
        self.cfg.update(corrections={'preset': 'translation'}, lod={'method': 'empirical', 'quantile': .9},
                        fitting={'land_cover': 'cover.tif', 'land_cover_classes': [31, 71]})
        self.config.write_text(yaml.safe_dump(self.cfg))
        cfg = builder.load_config(self.config)
        cmd = event.correction_command(pair, 'mask.tif', 'output', cfg)
        self.assertEqual(cmd[cmd.index('--fit-land-cover') + 1], str(self.root/'cover.tif'))
        self.assertEqual(cmd[cmd.index('--lod-quantile') + 1], '0.9')
        cfg['lod']['methd'] = 'empirical'
        with self.assertRaisesRegex(ValueError, 'lod'):
            builder.validate_config(cfg)

    def test_main_pipeline_corrects_offset_filters_lod_and_keeps_full_extent(self):
        import xdem
        import matplotlib.pyplot as plt
        import xDEM_pipeline_Thomas_production as production
        y, x = np.mgrid[:96, :96]
        pre = (700 + 20*np.sin(x/10) + 12*np.cos(y/13)).astype('float32')
        post = pre - 18 + np.random.default_rng(7).normal(0, .08, pre.shape).astype('float32')
        post[40:60, 75:85] += 5  # Real synthetic change outside the fitting land cover.
        post[10:15, 75:80] = -9999
        write(self.pre, pre)
        write(self.post, post)
        cover = np.full(pre.shape, 71, dtype='uint8')
        cover[:, 64:] = 42
        cover_path = write(self.root/'cover.tif', cover, nodata=250)
        stable_path = write(self.root/'stable.tif', np.ones(pre.shape, dtype='uint8'), nodata=255)
        original = {p: p.read_bytes() for p in [self.pre, self.post, cover_path, stable_path]}
        out = self.root/'main'
        out.mkdir()
        pipeline = production.CoregBiasPipeline('pre', correction_preset='translation',
                    stable_mask_path=stable_path, fit_land_cover=cover_path,
                    min_stable_pixels=100, lod_method='empirical', validation_block_size_m=20)
        with patch.object(production, 'output_dir', out), \
                patch.object(production.plt, 'show', side_effect=lambda: plt.close('all')), \
                patch.object(pipeline, 'analyze'), patch.object(pipeline, 'analyze_uncertainty'), \
                patch.object(production.xdem.coreg, 'ICP', side_effect=AssertionError('ICP must be disabled')), \
                patch.object(production.xdem.coreg, 'Deramp', side_effect=AssertionError('Deramp must be disabled')):
            pipeline.load_dems({'pre': self.pre, 'post': self.post})
            pipeline.run_pair('post-pre', pipeline.dems['post'])
        with rasterio.open(out/'final_dem_products/dod_corrected.tif') as src:
            corrected = src.read(1, masked=True).filled(np.nan)
            transform, crs = src.transform, src.crs
        with rasterio.open(out/'final_dem_products/post_coreg_bias.tif') as src:
            final_post = src.read(1, masked=True).filled(np.nan)
        np.testing.assert_allclose(corrected, final_post - pre, equal_nan=True, atol=.0001)
        with rasterio.open(out/'final_dem_products/dod_lod_filtered.tif') as src:
            filtered = src.read(1, masked=True).filled(np.nan)
            self.assertEqual(src.transform, transform)
            self.assertEqual(src.crs, crs)
        np.testing.assert_array_equal(np.isfinite(filtered), np.isfinite(corrected))
        self.assertTrue(np.isfinite(filtered[50, 80]))  # Forest is not removed from output.
        self.assertGreater(filtered[50, 80], 4.5)
        self.assertTrue(np.isnan(filtered[12, 77]))
        report = json.loads((out/'final_dem_products/lod_report.json').read_text())
        threshold = report['threshold_m']
        self.assertLess(threshold, .3)
        selected = np.isfinite(corrected) & (np.abs(corrected) > threshold)
        np.testing.assert_array_equal(filtered[selected], corrected[selected])
        self.assertTrue(np.all(filtered[np.isfinite(corrected) & ~selected] == 0))
        with rasterio.open(out/'stable_fit_check.tif') as src:
            split = src.read(1)
        self.assertTrue(np.any(split == 1))
        self.assertTrue(np.any(split == 2))
        self.assertFalse(np.any((split[:, 64:] == 1) | (split[:, 64:] == 2)))
        self.assertEqual(pipeline.metadata['production_bias']['status'], 'disabled')
        for p, content in original.items():
            self.assertEqual(p.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
