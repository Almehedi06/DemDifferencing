PIONEER EVALUATION
Scores the Landlab outputs against what happened in the December 2025 debris flows.
Nothing to install: it uses the packages your model notebooks already use.


HOW TO USE

1. Put this pioneer_eval folder next to your notebooks.

2. Last cell of Hazard_Probability_cont, after the P_*.asc files are written:

       from pioneer_eval import evaluate_hazard
       evaluate_hazard(".", run_name="baseline", params={"Csoil": Csoil, "Cmult": Cmult})

3. Last cell of MWR_alone, after MWR.asc is written:

       from pioneer_eval import evaluate_mwr
       evaluate_mwr("MWR.asc", run_name="baseline")

For each parameter experiment, change the parameters and give the run a new run_name.
"params" is optional; whatever you put there is saved with the scores.


HOW TO READ A RUN

Three numbers per layer, printed and logged for every run:
  AUC              do failure cells get higher P than the other cells? It uses no
                   cutoff, so flagging more cells cannot change it by itself.
  optimal P        the cutoff with the highest true-positive and lowest false-positive
                   rate. Used instead of a fixed 0.5.
  precision there  the share of cells flagged at that cutoff that are failure cells;
                   "x base" divides it by the base rate (1 = random). Compare "x base":
                   precision itself depends on how many absence cells are scored.
A change is an improvement when AUC rises and the precision at the optimal P does not fall.

The baseline of 21 Oct 2026 scores AUC 0.57 (landslide), 0.67 (channel initiation) and
0.68 (combined), with precision at the optimal P of 2.1, 2.7 and 2.4 times the base rate.
MWR is at chance: AUC 0.53 for erosion, 0.49 for deposition.


WHAT YOU GET (folder "evaluation" next to the notebook)

  evaluation/runs_log.csv       one row per hazard run: your params and the three numbers
  evaluation/mwr_runs_log.csv   one row per MWR run: AUC, recall, precision, erosion and deposition
  evaluation/<run_name>/        metrics.csv (every score, per layer and failure type),
                                roc.png, sampling_map.png (which cells were used),
                                map_P_<layer>.png and the MWR maps, point_hits.csv,
                                GeoTIFFs for QGIS (observations as yes/no; agreement maps
                                coded 0 correct no, 1 false alarm, 2 miss, 3 hit)

The maps call a cell "yes" where P >= 0.5 (change with map_threshold=...). A point counts
as hit when a flagged cell lies within 30 m; compare that with the "random location" rate
shown next to it.


HOW IT SCORES (Strauch et al. 2018, adapted to points and the dDEM)

  - failure cells: the 89 mapped points (47 landslide, 22 runoff, 20 firehose), each a
    3 x 3 block of 10 m cells
  - both classes only on slopes steeper than 17 degrees inside the drone survey
  - absence cells: every other cell more than 50 m from a point and outside the area the
    event changed (dDEM change > 0.5 m, plus one cell)
  - 95% interval by resampling the points
  - MWR: eroded where the dDEM drops more than 0.5 m, deposited where it rises more than 0.5 m

The rules are at the top of __init__.py (SETTINGS). The scoring is landlab_grid_validation,
a general package bundled here.


DATA (data/, EPSG:6339)

  Initiation_Points_final.shp   the 89 mapped initiation points
  dod_dndvi_test.tif            corrected dDEM, 1 m
  topographic__elevation.asc    the 10 m model DEM (for slope)
