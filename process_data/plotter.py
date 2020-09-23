import skmob
import pandas as pd
import numpy as np
import osmnx as ox
from .all_methods import *

###### PARAMETERS
PATH_TO_INPUT_FILE = './output_files/'
NAME_OF_INPUT_FILE = 'uk_5125users_2017-01-01_2017-02-02__greater_london_2800users__filtered_2570users_120sec__WITH_EMISSIONS.csv'
PATH_TO_ROAD_NETWORKS = './input_files/road_nets/'
region = 'Greater London'
subregion = None #'Quartiere XXXII Europa' #'Municipio Roma I' #'Rione XV Esquilino'
bbox_subregion = None #[41.904815, 41.886851, 12.495463, 12.471533]
pollutant_to_plot = 'CO_2' # one of: CO_2, NO_x, PM, VOC
######

######
normalization = None
######

### Loading tdf
print('Loading tdf...')
#df = pd.read_csv(PATH_TO_INPUT_FILE+NAME_OF_INPUT_FILE, header=0)
#tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lng', datetime='datetime', user_id='uid', trajectory_id='tid')
tdf = skmob.read(PATH_TO_INPUT_FILE + NAME_OF_INPUT_FILE)


### Loading road network
print('Loading road network...')
region_name = region.lower().replace(" ", "_")
try:
	### Load from (previously saved) graphml file
	if subregion == None:
		graphml_filename = '%s_network.graphml' % (region_name)
	else:
		subregion_name = subregion.lower().replace(" ", "_")
		graphml_filename = '%s_%s_network.graphml' % (region_name, subregion_name)
	road_network = ox.save_load.load_graphml(graphml_filename, folder=PATH_TO_ROAD_NETWORKS)
except:
	### ...or directly from osmnx:
	if subregion == None:
		road_network = ox.graph_from_place(region, network_type = 'drive_service')
	else:
		road_network = ox.graph_from_place(subregion + ', ' + region, network_type = 'drive_service')

### Saving road network (if not previously saved)
#ox.save_graphml(road_network, filename=graphml_filename, folder=PATH_TO_ROAD_NETWORKS)

### Normalizing emissions s.t. we can compare different regions
#print('Normalizing emissions...')
#tdf = normalize_emissions(tdf, percentage=True)


### Plotting
fig, ax = plot_road_network_with_emissions(tdf, road_network,
										   normalization_factor = normalization,
										   name_of_pollutant=pollutant_to_plot,
										   color_map='autumn_r',
										   bounding_box=bbox_subregion,
										   save_fig=True)