# Frozen

`xDEM_pipeline_Thomas_production_ORIGINAL.py` is the committed original at
`294e90f`, byte-identical. It is kept for provenance and is never edited.

The live engine, carrying both the `montecito` and `translation` presets, is
`scripts/xDEM_pipeline_Thomas_production.py`. Change that one.

Verify this copy at any time:

```bash
git show 294e90f:scripts/xDEM_pipeline_Thomas_production.py \
  | diff - methods/M1_original/xDEM_pipeline_Thomas_production_ORIGINAL.py && echo identical
```
