# mobility-emissions
Collection of methods that compute emissions starting from mobility trajectories.

Note: mainly relies on the Python libraries [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility) and [OSMnx](https://github.com/gboeing/osmnx).

### Description
The methods collected in `mobair` allow:
* map-matching: match points of a `TrajDataFrame` to the edges (roads) or nodes (crossroads) of a road network took from OpenStreetMap;
* speed/acceleration computation: compute the values of speed and acceleration in each point of the `TrajDataFrame`;
* trajectory time filtering: filter the points of the `TrajDataFrame` s.t. the time interval between the points of the resulting sub-trajectories is not greater than a threshold;
* computation of emissions: compute the instantaneous emissions of 4 air pollutants (CO2, NOx, PM, VOC) in each point of the `TrajDataFrame`.
* visualize the total quantity of each pollutant in a road network.

### Example
See the notebook `example_in_Rome.ipynb` in the `notebooks` folder to get an idea of a step-by-step procedure to compute and plot emissions.

### Article
Pre-print version of the paper at https://arxiv.org/abs/2107.03282.

The code used for the figures is in `notebooks/paper_figures.ipynb`.
