PIONEER EVALUATION
Scores the Landlab outputs against what happened in the December 2025 debris flows.
Nothing to install: it uses the packages your model notebooks already use
(numpy, pandas, scipy, matplotlib, rasterio, geopandas, landlab).


HOW TO USE

1. Put this pioneer_eval folder next to your notebooks.

2. At the end of Hazard_Probability_cont, after P_Landslide.asc, P_Channel_ini.asc
   and P_Combined_H.asc are written, add one cell:

       from pioneer_eval import evaluate_hazard
       evaluate_hazard(".", run_name="baseline", params={"Csoil": Csoil, "Cmult": Cmult})

3. At the end of MWR_alone, after MWR.asc is written, add one cell:

       from pioneer_eval import evaluate_mwr
       evaluate_mwr("MWR.asc", run_name="baseline")

To compare parameter sets: change parameters, give the run a new run_name, rerun.
"params" is optional; whatever you put there is saved next to the scores.


WHAT YOU GET (in an "evaluation" folder next to the notebook)

  evaluation/<run_name>/      metrics.csv (AUC with its 95% interval, per layer and
                              failure type), roc.png (one ROC panel per type),
                              point_hits.csv, negatives.gpkg (the absence cells),
                              settings.json, run_info.json
  evaluation/runs_log.csv     one row per hazard run: your params, the AUC of each
                              layer, the share of points hit at the map threshold and
                              the share a random location would get
  evaluation/mwr_runs_log.csv one row per MWR run: params, then AUC, recall, precision
                              for erosion (old names) and for deposition (_deposition)

Maps, in each evaluation/<run_name>/ (also shown in the notebook):

  map_P_<layer>.png           three panels on the Landlab grid: the layer, the mapped
                              points (one colour per failure type, counts in the
                              legend), and the flagged cells with each point marked
                              hit or miss; that panel's title gives the share of
                              points hit and the share a random location would get
  mwr_vs_ddem.png             MWR erosion against dDEM erosion
  mwr_vs_ddem_deposition.png  MWR deposition against dDEM deposition

GeoTIFFs for QGIS, in the same folder (EPSG:6339, 10 m):

  obs_points_All.tif          the points as yes/no: 1 within 30 m of a mapped point,
  obs_points_LS.tif             0 not, empty outside the drone survey. One file for
  obs_points_Runoff.tif         all types and one per type.
  obs_points_Firehose.tif
  obs_ddem_erosion.tif        1 where the dDEM dropped more than 0.5 m, 0 not
  obs_ddem_deposition.tif     1 where the dDEM rose more than 0.5 m, 0 not
  agree_P_<layer>.tif         0 correct no, 1 false alarm, 2 miss, 3 hit
  agree_MWR_vs_ddem.tif         (same codes for every agreement map; the MWR
  agree_MWR_vs_ddem_deposition.tif   file without a suffix is erosion)

The hazard maps call a cell "yes" where the probability is at least 0.5. Change
it with evaluate_hazard(..., map_threshold=0.3). The maps cover the whole drone
survey; the AUC uses its slopes steeper than 17 degrees. The threshold changes the
maps and the point hits only; the AUC needs no threshold. maps=False skips them.

A point is "hit" when any cell within 30 m of it is flagged. Compare that with the
"random location" rate printed beside it (the share of cells whose 30 m block holds a
flagged cell), not with the share of cells flagged: a block catches a flagged cell far
more often than one cell does. The agreement GeoTIFF compares cell by cell, so a thin
flagged channel through a 30 m block shows mostly "miss" even where the point was found.


HOW IT SCORES

Hazard layers (AUC, 0.5 = chance, 1 = perfect), following Strauch et al. (2018):
  - initiation: the 89 mapped points (47 landslide, 22 runoff, 20 firehose),
    each a 3 x 3 block of cells (30 m)
  - both classes come only from slopes steeper than 17 degrees inside the survey
  - absences: every other cell more than 50 m from a point and outside the area the
    event changed (dDEM change > 0.5 m, plus one cell)
  - 95% interval by resampling the points
  Your 23 Sep layers score 0.57 (landslide), 0.66 (channel initiation),
  0.68 (combined), all types together.

MWR: a cell is eroded where the dDEM drops more than 0.5 m, deposited where it rises
more than 0.5 m; compared with MWR erosion and deposition depth. The 24 Sep MWR
scores AUC 0.53 for erosion (chance level).

The rules live at the top of pioneer_eval/__init__.py (SETTINGS). The scoring itself
is landlab_grid_validation, a general package bundled in this folder; it works for
any site, any Landlab grid and any point or polygon inventory.


DATA (in data/)

  Initiation_Points_final.shp   89 mapped initiation points (EPSG:6339)
  dod_dndvi_test.tif            corrected dDEM, 1 m: tilt removed, vegetation gains masked
  topographic__elevation.asc    the 10 m model DEM (for slope)

All files are on the 10 m Pioneer lidar-footprint grid, EPSG:6339 (UTM 10N).
