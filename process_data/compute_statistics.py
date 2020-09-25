import skmob
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from .all_methods import *
from PROCESS_DATA.stats_utils import *

###### NOTE
## This module can be used to obtain various statistics on emissions / road network.
######

###### PARAMETERS
PATH_TO_INPUT_FILE = './output_files/'
NAME_OF_INPUT_FILE = 'italy_27837users_2017-01-01_2017-02-02__rome_8299users__filtered_6772users_120sec__WITH_EMISSIONS.csv'
PATH_TO_ROAD_NETWORKS = './input_files/road_nets/'
region = 'Rome'
subregion = None  # 'Quartiere XXXII Europa' #'Municipio Roma I' #'Rione XV Esquilino'
list_of_pollutants = ['CO_2', 'NO_x', 'PM', 'VOC']
region_area = 102410000  # land area in squared meters, needed to compute density stats for the network
# Florence: 102410000, Rome: 1285000000, London: 1572000000

# set to True/False based on what you want to obtain or not:
compute_correlations = False
print_emissions_statistics = True
print_network_statistics = False
######

### Loading tdf
tdf = skmob.read(PATH_TO_INPUT_FILE + NAME_OF_INPUT_FILE)

###########################################################################################################
################################### PLOT and FIT DISTRIBUTIONS ############################################
###########################################################################################################

### Extract overall distribution of pollution
dict_pollutant_to_entire_distribution = {}
for c_pollutant in list_of_pollutants:
	dict_pollutant_to_entire_distribution[c_pollutant] = np.array(tdf[c_pollutant])

### Loading road network (needed to compute the distribution of pollutant per road)
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
		road_network = ox.graph_from_place(region, network_type='drive_service')
	else:
		road_network = ox.graph_from_place(subregion + ', ' + region, network_type='drive_service')
###


###########################################################################################################
##################################### COMPUTE CORRELATIONS ################################################
###########################################################################################################

if compute_correlations:
	df_corrs = compute_corrs(tdf, list_of_pollutants)
	df_corrs.to_csv(PATH_TO_INPUT_FILE + '/correlations__%s.csv' % region_name, index=True)

###########################################################################################################
###################################### COMPUTE STATISTICS #################################################
###########################################################################################################

if print_emissions_statistics:

	num_vehicles = len(set(tdf['uid']))
	num_roads = len(set(tuple(i) for i in tdf['road_link']))  ## road links are lists of two elements each (non-hashable, so it is necessary to create a set of tuples instead)

	for c_pollutant in list_of_pollutants:

		print('------------------------------')
		print('Some stats for %s emissions:' % c_pollutant)
		print('------------------------------')

		## stats distribution PER VEHICLE ##
		dict_vehicle_to_emissions = map_vehicle_to_emissions(tdf, name_of_pollutant=c_pollutant)
		dict_vehicle_to_total_emissions = {}
		emissions = []
		max_sum = 0
		min_sum = float('inf')
		for c_uid, c_array in dict_vehicle_to_emissions.items():
			c_sum = np.sum(c_array)
			dict_vehicle_to_total_emissions[c_uid] = c_sum
			if c_sum > max_sum:
				max_sum = c_sum
				uid_max = c_uid
			if c_sum < min_sum:
				min_sum = c_sum
				uid_min = c_uid
			emissions.extend([c_sum])

		print('----- stats per vehicle ------')

		overall_sum = np.sum(emissions)
		print('Overall value of emission: ', overall_sum)
		overall_mean = np.mean(emissions)
		print('Mean value of emissions per vehicle: %s (the %s of total emissions)' % (
			overall_mean, overall_mean / overall_sum))
		overall_median = np.median(emissions)
		print('Median value of emissions per vehicle: ', overall_median)
		rate_max = max_sum / overall_sum
		rate_min = min_sum / overall_sum
		print('User %s emitted the most (%s grams, the %s of total emissions in the network).' % (
			uid_max, max_sum, rate_max))
		print('User %s emitted the least (%s grams, the %s of total emissions in the network).' % (
			uid_min, min_sum, rate_min))
		num_vehicles_10_percent = int(num_vehicles / 100 * 10)
		sum_10_percent = 0
		for c_uid, c_tot in sorted(dict_vehicle_to_total_emissions.items(), key=lambda item: item[1], reverse=True)[
							0:num_vehicles_10_percent]:
			sum_10_percent += c_tot
		print('10%% of the vehicles are responsible for the %s of total emissions.' % (sum_10_percent / overall_sum))

		## stats distribution PER ROAD ##
		dict_road_to_emissions = map_road_to_emissions(tdf, road_network, name_of_pollutant=c_pollutant)
		dict_road_to_total_emissions = {}
		emissions = []
		for c_rid, c_array in dict_road_to_emissions.items():
			c_sum = np.sum(c_array)
			dict_road_to_total_emissions[c_rid] = c_sum
			emissions.extend([c_sum])

		print('------- stats per road -------')

		overall_sum = np.sum(emissions)
		# print('Overall value of emission: ', overall_sum)
		overall_mean = np.mean(emissions)
		print('Mean value of emissions per road: %s (the %s of total emissions)' % (
			overall_mean, overall_mean / overall_sum))
		overall_median = np.median(emissions)
		print('Median value of emissions per road: ', overall_median)
		max_cumulate_emissions = max(dict_road_to_total_emissions.values())
		rid_max = max(dict_road_to_total_emissions, key=dict_road_to_total_emissions.get)
		min_cumulate_emissions = min(dict_road_to_total_emissions.values())
		rid_min = min(dict_road_to_total_emissions, key=dict_road_to_total_emissions.get)
		rate_max = max_cumulate_emissions / overall_sum
		rate_min = min_cumulate_emissions / overall_sum

		name_road_with_max_emissions = road_network.get_edge_data(rid_max[0], rid_max[1], key=0, default='-noname-').get('name', '-noname-')
		name_road_with_min_emissions = road_network.get_edge_data(rid_min[0], rid_min[1], key=0, default='-noname-').get('name', '-noname-')
		print('%s has the maximum quantity of emissions (%s grams, the %s of total emissions in the network).' % (
			name_road_with_max_emissions, max_sum, rate_max))
		print('%s has the minimum quantity of emissions (%s grams, the %s of total emissions in the network).' % (
			name_road_with_max_emissions, min_sum, rate_min))
		#
		print('-- edge attributes --')
		print('attributes found for %s:' %name_road_with_max_emissions)
		print(road_network.get_edge_data(rid_max[0], rid_max[1], key=0, default='-noname-'))
		print('attributes found for %s:' %name_road_with_min_emissions)
		print(road_network.get_edge_data(rid_min[0], rid_min[1], key=0, default='-noname-'))
		print('---------------------')
		#
		num_roads_10_percent = int(num_roads / 100 * 10)
		sum_10_percent = 0
		for c_rid, c_tot in sorted(dict_road_to_total_emissions.items(), key=lambda item: item[1], reverse=True)[
							0:num_roads_10_percent]:
			sum_10_percent += c_tot
		print('10%% of the roads have the %s of total emissions.' % (sum_10_percent / overall_sum))

###########################

if print_network_statistics:

	print('------------------------------')
	print('Some stats for the road network of %s:' % region)
	print('------------------------------')

	dict_stats_road_network = compute_stats_for_network(road_network, area=region_area, circuity_dist='gc')

	for stat, value in dict_stats_road_network.items():
		print('%s : %s' % (stat, value))
