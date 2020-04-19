# mobility-airpollution
Collection of methods that compute emissions starting from mobility trajectories.

Note: it relies on the library [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility).

### Description
`mobility-airpollution` allows to:
* match points of a `TrajDataFrame` to the edges (roads) of a road network;
* compute the values of speed and acceleration for each point;
* filter the points of the trajectories s.t. the time interval between the points of the resulting sub-trajectories is not greater than a threshold;
* compute the instantaneous emissions of 4 pollutants for each point;
* represent in a plot the total quantity of each pollutant on the edges in the road network.

### Example
See the notebook example_in_Rome to get an idea of a step-by-step procedure to compute and plot emissions.
