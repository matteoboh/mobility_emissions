import osmnx as ox
import skmob
import pandas as pd
import tarfile
from skmob.preprocessing import filtering
from .all_methods import *

###### PARAMETERS
PATH_TO_INPUT_FILE = './input_files/'
NAME_OF_INPUT_FILE = 'italy_27837users_2017-01-01_2017-02-02__rome_8299users.csv'
PATH_TO_OUTPUT_FILE = './output_files/'
PATH_TO_ROAD_NETWORKS = './input_files/road_nets/'  # folder where to save the road net as a .graphml file (if not wanted, write None)
max_interval = 120  # see description of all_methods.filter_on_time_interval
region = 'Rome'
PATH_TO_TABLE_WITH_INFO_ON_VEHICLES = 'util_files/modelli_auto.tar.xz'
PATH_TO_TABLE_WITH_EMISSION_FUNCTIONS = 'util_files/emission_functions.csv'
######


### Loading tdf
print('Loading tdf...')
df = pd.read_csv(PATH_TO_INPUT_FILE+NAME_OF_INPUT_FILE, header=0)
tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lng', datetime='datetime', user_id='uid', trajectory_id='tid')
#tdf = skmob.read(PATH_TO_INPUT_FILE + NAME_OF_INPUT_FILE)

set_of_uid_in_input = set(tdf['uid'])
num_of_uid_in_input = len(set_of_uid_in_input)

### Filtering..
# on time
print('Filtering on time...')
tdf_filtered_time = filter_on_time_interval(tdf, max_interval)

# on speed
print('Filtering on speed...')
tdf_filtered_speed = filtering.filter(tdf_filtered_time, max_speed_kmh = 300)

# on acceleration
print('Computing acceleration and filtering...')
tdf_with_speed_and_acc = compute_acceleration_from_tdf(tdf_filtered_speed)
ftdf = tdf_with_speed_and_acc[tdf_with_speed_and_acc['acceleration'] < 10]


### Loading road network
print('Loading road network...')
region_name = region.lower().replace(" ", "_")
road_network = ox.graph_from_place(region, network_type = 'drive_service')
graphml_filename = '%s_network.graphml' % (region_name)
if PATH_TO_ROAD_NETWORKS != None:
    ox.save_graphml(road_network, filename=graphml_filename, folder=PATH_TO_ROAD_NETWORKS)

### Map-matching
print('Map-matching...')
#ftdf_final = find_nearest_nodes_in_network(road_network, ftdf, return_tdf_with_new_col=True)
ftdf_final = find_nearest_edges_in_network(road_network, ftdf, return_tdf_with_new_col=True)

set_of_uid_final = set(ftdf_final['uid'])
num_of_uid_final = len(set_of_uid_final)
print('There have been', num_of_uid_in_input-num_of_uid_final, 'uid lost in the filtering process.')


### Compute emissions
print('Computing emissions...')
tar = tarfile.open(PATH_TO_TABLE_WITH_INFO_ON_VEHICLES, "r:xz")
for x in tar.getmembers():
    file = tar.extractfile(x)
    modelli_auto = pd.read_csv(file, names=['vid', 'manufacturer', 'type'], usecols = [0,1,2])

emissions = pd.read_csv(PATH_TO_TABLE_WITH_EMISSION_FUNCTIONS)

df_vehicles_info = modelli_auto.loc[modelli_auto['vid'].isin(set_of_uid_final)]
dict_vehicle_fuel_type = match_vehicle_to_fuel_type(ftdf_final, modelli_auto, ['PETROL', 'DIESEL', 'LPG'])
tdf_with_emissions = compute_emissions(ftdf_final, emissions, dict_vehicle_fuel_type)
print('End.')


###### Name of output file:
NAME_OF_OUTPUT_FILE = NAME_OF_INPUT_FILE[:-4] + '__filtered_%susers_%ssec__WITH_EMISSIONS.csv' % (num_of_uid_final, max_interval)
#NAME_OF_OUTPUT_FILE = str('final_traj_%s_%s_uid_%s_sec__WITH_EMISSIONS.json' %(region, num_of_uid_final, max_interval))
######

print('Saving results in', PATH_TO_OUTPUT_FILE)
#tdf_with_emissions.to_csv(PATH_TO_OUTPUT_FILE + NAME_OF_OUTPUT_FILE, index=False)
skmob.write(tdf_with_emissions, PATH_TO_OUTPUT_FILE + NAME_OF_OUTPUT_FILE)