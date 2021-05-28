# mobility-emissions
Collection of methods that compute emissions starting from mobility trajectories.

Note: it relies on the library [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility).

### Description
`mobility-emissions` allows to:
* map-matching: match points of a `TrajDataFrame` to the edges (roads) or nodes (crossroads) of a road network took from OpenStreetMap;
* speed/acceleration computation: compute the values of speed and acceleration in each point of the `TrajDataFrame`;
* trajectory time filtering: filter the points of the `TrajDataFrame` s.t. the time interval between the points of the resulting sub-trajectories is not greater than a threshold;
* computation of emissions: compute the instantaneous emissions of 4 air pollutants (CO$_2$, NO$_x$, PM, VOC) in each point of the `TrajDataFrame`.
* visualize the total quantity of each pollutant in a road network.

### Example
See the notebook `example_in_Rome.ipynb` in the `notebooks` folder to get an idea of a step-by-step procedure to compute and plot emissions.
