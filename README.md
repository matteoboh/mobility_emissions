# mobility-airpollution
Collection of methods that compute emissions starting from mobility trajectories.

Note: it relies on the library [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility).

### Description
`mobility-airpollution` allows to:
* match points of a `TrajDataFrame` to the edges (roads) of a road network;
* compute the values of speed and acceleration for each point;
* filter the points of the trajectories s.t. the time interval between the points of the resulting sub-trajectories is not greater than a threshold;
* compute the instantaneous emissions of 4 pollutants for each point;
* visualize the total quantity of each pollutant in a road network;
* plot and fit the distribution of each pollutant per road/vehicle;
* compute some statistics about the emissions over the network.

### Example
See the notebook `example_in_Rome.ipynb` in the `example` folder to get an idea of a step-by-step procedure to compute and plot emissions.

### Pipeline
1. `./get_data/`
    1. prepare the `config.json` file in order to get data from the db;
    2. `get_trajectories.py` downloads the data from the db. The user must specify: the table from which to take the trajectories ('uk' or 'italy'), the starting and ending date of observation;
    3. `select_trajectories_within_tessellation.py` takes the `csv` file coming from (ii) and selects the trajectories that fall within the specified `REGION`.
2. `./process_data/`
    1. `filtering_mapmatching_and_pollution.py` takes the selected trajectories and does all the job.
3. once the emissions are computed, the user can:
    * plot them onto a road network (`./process_data/plotter.py`), even for a subregion (e.g. a district in a city)
    * compute some statistics, correlations, and/or plot and fit the distributions of pollution (`./process_data/compute_statistics.py`) 

#### Notes
* always look at the `PARAMETERS` to be set at the very beginning of each file `.py` before the execution: some paths where to take/save the input/output files have to be specified;
* `select_trajectories_within_tessellation.py` also saves the road network of the region as a `.graphml` file: change the related parameter at the beginning of the file if this is not desired.