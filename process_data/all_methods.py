import numpy as np
import osmnx as ox
import networkx as nx
import skmob
from skmob.utils import gislib, utils, constants
from skmob.core.trajectorydataframe import *
from skmob.measures.individual import *
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import matplotlib.dates as dt


###########################################################################################################
##################################### FILTERING POINTS BY TIME ############################################
###########################################################################################################

def filter_on_time_interval(tdf, max_time=30):
	"""Time filtering.

	For each trajectory in a TrajDataFrame, retains only the subset of consecutive points
	(i.e. the sub-trajectories) that are distant no more than max_time (in seconds) from each other.

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the individuals.

	max_time : int
		the maximum number of seconds between two consecutive points;
		the default is 30.

	Returns
	-------
	TrajDataFrame
		the TrajDataFrame with the filtered sub-trajectories.
	"""

	tdf.sort_by_uid_and_datetime()

	max_tid = -1 # this is needed afterwards for the creation of new 'tid' for the filtered sub-trajectories
	groupby = []

	if utils.is_multi_user(tdf):
		if len(set(tdf[constants.UID])) != 1:
			groupby.append(constants.UID)
	if utils.is_multi_trajectory(tdf):
		max_tid = max(set(tdf[constants.TID]))
		if len(set(tdf[constants.TID])) != 1:
			groupby.append(constants.TID)

	if len(groupby) > 0:
		tdf_filtered = tdf.groupby(groupby, group_keys=False, as_index=False).apply(filter_on_time_interval_for_one_id,
																					max_tid,
																					max_time)
	else:
		tdf_filtered = filter_on_time_interval_for_one_id(tdf, max_tid, max_time)

	return tdf_filtered.sort_by_uid_and_datetime()


def filter_on_time_interval_for_one_id(tdf, max_tid = -1, max_time=30):

	tdf_filtered = tdf.copy()
	tdf_filtered = tdf_filtered.sort_by_uid_and_datetime()
	tdf_filtered.reset_index(inplace=True, drop=True)
	set_of_uid = set(tdf_filtered['uid'])

	if max_tid == -1:
		c_tid = 0
		max_tid = 0
	else:
		set_of_tid = set(tdf_filtered['tid'])
		if len(set_of_tid) != 1:
			print('Called function on more than one "tid".')
			return
		else:
			c_tid = tdf_filtered['tid'].iloc[0]

	if len(set_of_uid) != 1:
		print('Called function on more than one "uid".')
		return

	else:
		dict_subtraj = {}
		set_points_of_subtraj = set()
		previous_point_was_ok = False

		for c_row in range(0, tdf_filtered.shape[0] - 1):
			t_0 = tdf_filtered['datetime'].loc[c_row]
			t_1 = tdf_filtered['datetime'].loc[c_row + 1]
			d_time = utils.diff_seconds(t_0, t_1)

			# Now, at each step, two points of the trajectory are compared, p_i and p_i+1.
			# (1) if they are close to each other no more than max_time:
			#    (1.1) if p_i was already added to the sub-trajectory, add p_i+1 to the sub-trajectory
			#    (1.2) if p_i was not added to the sub-trajectory (i.e. p_i is the starting point of the sub-trajectory), add both p_i and p_i+1 to the sub-trajectory
			# (2) if they are distant more than max_time:
			#    (2.1) if p_i was added to the sub-trajectory (i.e. p_i is the last point of the sub-trajectory), add the sub-trajectory to the dictionary
			
			if d_time < max_time: # (1)
				if previous_point_was_ok: # (1.1)
					set_points_of_subtraj.add(c_row + 1)
				else: # (1.2)
					set_points_of_subtraj = set()
					set_points_of_subtraj.add(c_row)
					set_points_of_subtraj.add(c_row + 1)
				previous_point_was_ok = True
			else: # (2)
				if previous_point_was_ok: # (2.1)
					dict_subtraj[c_row] = set_points_of_subtraj
				previous_point_was_ok = False

		# For each trajectory, retain all the sub-trajectories found.
		# (note that each sub-trajectory needs a new 'tid')
		dict_subtraj_with_new_tid = {}
		i=0
		for c_subtraj in dict_subtraj.values():
			new_tid = c_tid + max_tid + i # the new 'tid' given to the sub-trajectory
			dict_subtraj_with_new_tid[new_tid] = c_subtraj
			i+=1

		# Appending all the new sub-trajectories to a new tdf:
		df_filtered_with_new_tid = pd.DataFrame()
		for tid, subtraj in dict_subtraj_with_new_tid.items():
			slice_of_df_filtered_with_new_tid = tdf_filtered.loc[list(subtraj)]
			slice_of_df_filtered_with_new_tid['tid'] = tid
			df_filtered_with_new_tid = df_filtered_with_new_tid.append(slice_of_df_filtered_with_new_tid)

	tdf_filtered_with_new_tid = skmob.TrajDataFrame(df_filtered_with_new_tid, latitude='lat', longitude='lng', datetime='datetime', user_id='uid')

	return tdf_filtered_with_new_tid


###########################################################################################################
################################# COMPUTE SPEED AND ACCELERATION ##########################################
###########################################################################################################

def compute_speed_from_tdf(tdf):
	"""Compute speed.

	For each point in a TrajDataFrame, computes the speed (in m/s) as the Haversine distance from the
	previous point over the time interval between them.

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the individuals.

	Returns
	-------
	TrajDataFrame
		the TrajDataFrame with 1 more column collecting the value of speed for each point.
	"""

	tdf.sort_by_uid_and_datetime()

	groupby = []

	if utils.is_multi_user(tdf):
		if len(set(tdf[constants.UID])) != 1:
			groupby.append(constants.UID)
	if utils.is_multi_trajectory(tdf):
		if len(set(tdf[constants.TID])) != 1:
			groupby.append(constants.TID)

	if len(groupby) > 0:
		tdf_with_speed = tdf.groupby(groupby, group_keys=False, as_index=False).apply(compute_speed_from_tdf_for_one_id)
	else:
		tdf_with_speed = compute_speed_from_tdf_for_one_id(tdf)

	return tdf_with_speed


def compute_speed_from_tdf_for_one_id(tdf):

	tdf.sort_by_uid_and_datetime()
	set_of_ids = set(tdf['uid'])

	if len(set_of_ids) != 1:
		print('Called function on more than one ID: use compute_speed_from_tdf instead.')
		return

	else:
		list_of_speed = [0]
		for c_row in range(0, tdf.shape[0] - 1):
			loc_0 = (tdf['lat'].iloc[c_row], tdf['lng'].iloc[c_row])
			loc_1 = (tdf['lat'].iloc[c_row + 1], tdf['lng'].iloc[c_row + 1])
			c_dist = gislib.getDistanceByHaversine(loc_0, loc_1)  # returns distance in km
			c_dist = c_dist * 1000  # converting in meters

			t_0 = tdf['datetime'].iloc[c_row]
			t_1 = tdf['datetime'].iloc[c_row + 1]
			d_time = utils.diff_seconds(t_0, t_1)  # in sec

			if d_time != 0:
				c_speed = c_dist / d_time  # in m/s
			else:
				c_speed = list_of_speed[c_row - 1]
			list_of_speed.append(c_speed)

		tdf_with_speed = tdf.copy()
		tdf_with_speed.loc[:, 'speed'] = list_of_speed

	return tdf_with_speed

#####################################

def compute_acceleration_from_tdf(tdf):
	"""Compute acceleration.

	For each point in a TrajDataFrame, computes the acceleration (in m/s^2) based on the speed
	and the time interval between the point and the previous one.

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the individuals.

	Returns
	-------
	TrajDataFrame
		the TrajDataFrame with 1 more column collecting the value of acceleration for each point.

	Warnings
	--------
	if speed has not been previously computed, a function is firstly called to compute it.
	"""

	tdf.sort_by_uid_and_datetime()

	groupby = []

	if utils.is_multi_user(tdf):
		if len(set(tdf[constants.UID])) != 1:
			groupby.append(constants.UID)
	if utils.is_multi_trajectory(tdf):
		if len(set(tdf[constants.TID])) != 1:
			groupby.append(constants.TID)

	if len(groupby) > 0:
		tdf_with_acc = tdf.groupby(groupby, group_keys=False, as_index=False).apply(
			compute_acceleration_from_tdf_for_one_id)
	else:
		tdf_with_acc = compute_acceleration_from_tdf_for_one_id(tdf)

	return tdf_with_acc


def compute_acceleration_from_tdf_for_one_id(tdf):
	# This function computes acceleration (in m/s^2) for each point of a trajectory
	# based on speed and time interval.
	# If the tdf does not have the column 'speed', it first computes it calling compute_speed_from_tdf.
	# N.B.: this function correctly works only if there is exactly one 'uid' in the tdf.
	# To compute the acceleration from a tdf with more than one 'uid', use the general function compute_acceleration_from_tdf.

	tdf.sort_by_uid_and_datetime()
	set_of_ids = set(tdf['uid'])

	if 'speed' not in tdf.columns:
		tdf = compute_speed_from_tdf(tdf)

	if len(set_of_ids) != 1:
		print('Called function on more than one ID: use compute_acceleration_from_tdf instead.')
		return

	else:
		list_of_acc = [0]
		for c_row in range(0, tdf.shape[0] - 1):
			speed_0 = tdf['speed'].iloc[c_row]
			speed_1 = tdf['speed'].iloc[c_row + 1]
			d_speed = speed_1 - speed_0

			t_0 = tdf['datetime'].iloc[c_row]
			t_1 = tdf['datetime'].iloc[c_row + 1]
			d_time = utils.diff_seconds(t_0, t_1)  # in sec

			if d_time != 0:
				c_acc = d_speed / d_time  # in m/s^2
			else:
				c_acc = list_of_acc[c_row - 1]
			list_of_acc.append(c_acc)

		tdf_with_acc = tdf.copy()
		tdf_with_acc.loc[:, 'acceleration'] = list_of_acc

	return tdf_with_acc


###########################################################################################################
########################################## MAP-MATCHING ###################################################
###########################################################################################################

def find_nearest_nodes_in_network(road_network, tdf, return_tdf_with_new_col=False):
	"""Map-matching.

	For each point in a TrajDataFrame, it finds the nearest node in a road network.

	Parameters:
	----------
	road_network : networkx MultiDiGraph
		the road network on which to map the points.

	tdf : TrajDataFrame
		the trajectories of the individuals.

	return_tdf_with_new_col : boolean
		if False (default), returns the list of the nearest nodes (as OSM IDs);
		if True, returns a copy of the original TrajDataFrame with one more column called 'node_id'.

	Returns
	-------
	list (if return_tdf_with_new_col = True)
		list of the nearest nodes.

	TrajDataFrame (if return_tdf_with_new_col = False)
		the TrajDataFrame with 1 more column collecting the nearest node's ID for each point.
	"""

	tdf.sort_by_uid_and_datetime()

	# extracting arrays of lat and lng for the vehicle:
	vec_of_longitudes = np.array(tdf['lng'])
	vec_of_latitudes = np.array(tdf['lat'])

	# for each (lat,lon), find the nearest node in the road network:
	list_of_nearest_nodes = ox.get_nearest_nodes(road_network, X=vec_of_longitudes, Y=vec_of_latitudes,
												 method='balltree')
	###
	# method (str {None, 'kdtree', 'balltree'}) – Which method to use for finding nearest node to each point.
	# If None, we manually find each node one at a time using osmnx.utils.get_nearest_node and haversine.
	# If ‘kdtree’ we use scipy.spatial.cKDTree for very fast euclidean search.
	# If ‘balltree’, we use sklearn.neighbors.BallTree for fast haversine search.
	###

	if return_tdf_with_new_col:
		tdf_with_nearest_nodes = tdf.copy()#.drop(columns=['lat', 'lng'])
		tdf_with_nearest_nodes['node_id'] = list_of_nearest_nodes
		return tdf_with_nearest_nodes

	return list_of_nearest_nodes


def find_nearest_edges_in_network(road_network, tdf, return_tdf_with_new_col=False):
	"""Map-matching.

	For each point in a TrajDataFrame, it finds the nearest edge in a road network.
	Each edge is represented as the couple of OSM ids of its starting and ending nodes.

	Parameters:
	----------
	road_network : networkx MultiDiGraph
		the road network on which to map the points.

	tdf : TrajDataFrame
		the trajectories of the individuals.

	return_tdf_with_new_col : boolean
		if False (default), returns the list of the nearest edges;
		if True, returns a copy of the original TrajDataFrame with one more column called 'road_link'.

	Returns
	-------
	list (if return_tdf_with_new_col = True)
		list of the nearest edges.

	TrajDataFrame (if return_tdf_with_new_col = False)
		the TrajDataFrame with 1 more column collecting the couple of OSM ids of the two nodes.
	"""

	tdf.sort_by_uid_and_datetime()

	# extracting arrays of lat and lng for the vehicle:
	vec_of_longitudes = np.array(tdf['lng'])
	vec_of_latitudes = np.array(tdf['lat'])

	# for each (lat,lon), find the nearest node in the road network:
	array_of_nearest_edges = ox.get_nearest_edges(road_network, X=vec_of_longitudes, Y=vec_of_latitudes,
												 method='balltree')
	###
	# method (str {None, 'kdtree', 'balltree'}) – Which method to use for finding nearest node to each point.
	# If None, we manually find each node one at a time using osmnx.utils.get_nearest_node and haversine.
	# If ‘kdtree’ we use scipy.spatial.cKDTree for very fast euclidean search.
	# If ‘balltree’, we use sklearn.neighbors.BallTree for fast haversine search.
	###

	list_of_nearest_edges = []
	for index in range(0, array_of_nearest_edges.shape[0]):
		list_of_nearest_edges.append([array_of_nearest_edges[index][0], array_of_nearest_edges[index][1]])

	if return_tdf_with_new_col:
		tdf_with_nearest_edges = tdf.copy()#.drop(columns=['lat', 'lng'])
		tdf_with_nearest_edges['road_link'] = list_of_nearest_edges
		return tdf_with_nearest_edges

	return list_of_nearest_edges


def compute_route_of_one_vehicle(road_network, tdf, vehicle_id):
	"""Computes the (shortest) route of a specific vehicle in the tdf.

	This function computes a route (as a sequence of nodes in the given road network) starting from a
	trajectory dataframe for one specific vehicle ('uid').
	For each couple of points in the tdf, their nearest nodes in the network are located,
	and the shortest path between them is computed and added to the route.
	N.B.: the shortest path is defined as the sequence of roads in the road network that minimizes the
	sum of their lengths.

	Parameters
	----------
	road_network : networkx MultiDiGraph.

	tdf : TrajDataFrame
		the trajectories of the vehicles.

	vehicle_id : str
		ID of the vehicle.

	Returns
	-------
	List
		list of the nodes in the network describing the shortest route that connects all of them.
	"""

	tdf.sort_by_uid_and_datetime()
	set_of_ids = set(tdf['uid'])
	if str(vehicle_id) not in set_of_ids:
		print('ID ' + vehicle_id + ' not found.')
		return

	# building a tdf with only the points belonging to the specified vehicle:
	tdf_one_id = tdf.loc[tdf['uid'] == str(vehicle_id)]

	# compute a list of nearest nodes to the points in the tdf:
	list_of_nearest_nodes = find_nearest_nodes_in_network(road_network, tdf_one_id, return_tdf_with_new_col=False)

	# Building the route as composed by shortest paths between the nodes in list_of_nearest_nodes:
	ent_route = []
	for index in range(0, len(list_of_nearest_nodes) - 1):
		origin_node = list_of_nearest_nodes[index]
		destination_node = list_of_nearest_nodes[index + 1]
		if origin_node != destination_node:
			c_route = nx.shortest_path(road_network, origin_node, destination_node, weight='length')
			# with weight='length', shortest path is computed as the path that minimizes the sum of the lengths of the roads in it
			# with weight=None, it gives the path with minimum number of roads in it (i.e. weight=1)
			if index != 0:
				c_route = c_route[1:]  # cutting the first node as it is equal to the last of the previous c_route
			ent_route.extend(c_route)
	return ent_route


###########################################################################################################
####################################### COMPUTE EMISSIONS #################################################
###########################################################################################################

def match_vehicle_to_fuel_type(tdf, df_with_all_vehicles, list_of_fuel_types=['PETROL', 'DIESEL', 'LPG']):
	"""Matching each vehicle to its fuel type.

	For each 'uid' in a TrajDataFrame, recovers the fuel type of the vehicle.

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the individuals.

	df_with_all_vehicles : DataFrame
		the vehicle types in the entire data set.
		The DataFrame must cointains columns 'vid' and 'type',
		where 'vid' corresponds to 'uid' in the TrajDataFrame,
		and 'type' is a string containing information on the vehicle (including the fuel type).

	list_of_fuel_types: list
		names of fuel types.

	Returns
	-------
	Dictionary
		a dictionary mapping each 'uid' in the TrajDataFrame to its fuel type.

	Warnings
	--------
	if df_with_all_vehicles['type'] does not contain one of the fuel types in list_of_fuel_types,
	the fuel type is set to 'PETROL'.
	"""

	set_of_uid = set(tdf['uid'])
	df_subset_of_vehicles = df_with_all_vehicles.loc[df_with_all_vehicles['vid'].isin(set_of_uid)]

	map_uid__fuel_type = {}

	for fuel_type in list_of_fuel_types:
		c_df = df_subset_of_vehicles[df_subset_of_vehicles['type'].str.contains(fuel_type)]
		c_set_uid = set(c_df['vid'])
		for c_uid in c_set_uid:
			map_uid__fuel_type[c_uid] = fuel_type

	for c_uid in set_of_uid:
		if c_uid not in map_uid__fuel_type.keys():
			map_uid__fuel_type[c_uid] = 'PETROL'

	return map_uid__fuel_type


def compute_emissions(tdf, df_with_emission_functions, dict_of_fuel_types_in_tdf):
	"""Compute instantaneous emissions with equations used in [Nyhan et al. (2016)].

	For each point in a TrajDataFrame, computes instantaneous emissions of 4 pollutants (CO_2, NO_x, PM, VOC).

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the vehicles.

	df_with_emission_functions : DataFrame
		the emission functions took from Table 2 in [Int Panis et al. (2006)]

	dict_of_fuel_types_in_tdf : dictionary
		maps each 'uid' in the TrajDataFrame to its fuel_type, with fuel_type in {'PETROL', 'DIESEL', 'LPG'}

	Returns
	-------
	TrajDataFrame
		the TrajDataFrame with 4 more columns collecting the instantaneous emissions for each point.

	Warnings
	--------
	if speed and acceleration have not been previously computed, a function is firstly called to compute them, and the execution will be slower.

	References
	----------
	[Nyhan et al. (2016)] M. Nyhan, S. Sobolevsky, C. Kang, P. Robinson, A. Corti, M. Szell, D. Streets, Z. Lu, R. Britter, S.R.H. Barrett, C. Ratti, Predicting vehicular emissions in high spatial resolution using pervasively measured transportation data and microscopic emissions model, Atmospheric Environment, Volume 140, 2016, https://doi.org/10.1016/j.atmosenv.2016.06.018.
	[Int Panis et al. (2006)] L. Int Panis, S. Broekx, R. Liu, Modelling instantaneous traffic emission and the influence of traffic speed limits, Science of The Total Environment, Volume 371, Issues 1–3, 2006, https://doi.org/10.1016/j.scitotenv.2006.08.017.

	"""

	if 'acceleration' not in tdf.columns:
		tdf = compute_acceleration_from_tdf(tdf)

	set_of_pollutants = set(df_with_emission_functions['pollutant'])

	dict_pollutant_to_emission = {}

	for c_row in range(0, tdf.shape[0]):
		c_uid = tdf['uid'].iloc[c_row]
		c_fuel_type = dict_of_fuel_types_in_tdf[c_uid]
		c_acc = tdf['acceleration'].iloc[c_row]
		c_speed = tdf['speed'].iloc[c_row]

		for c_pollutant in set_of_pollutants:
			input_df = df_with_emission_functions[(df_with_emission_functions['pollutant'] == c_pollutant) &
												  (df_with_emission_functions['fuel_type'] == c_fuel_type)]
			if c_pollutant in {'CO_2', 'PM'}:
				inst_emission = compute_emission_for_CO2_and_PM(input_df, c_speed, c_acc)
			else:
				# if c_pollutant in {'NO_x, VOC'}:
				inst_emission = compute_emission_for_NOx_and_VOC(input_df, c_speed, c_acc)

			dict_pollutant_to_emission.setdefault(c_pollutant, []).append(inst_emission)

	# Adding columns with emissions for each pollutant to the tdf
	tdf_with_emissions = tdf.copy()
	for pollutant, emissions in dict_pollutant_to_emission.items():
		tdf_with_emissions[pollutant] = emissions

	return tdf_with_emissions


def compute_emission_for_CO2_and_PM(df_with_emission_functions, speed_value, acceleration_value):
	v_t = speed_value
	a_t = acceleration_value

	f = np.array(df_with_emission_functions[['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6']]).flatten()

	E_1 = f[0] + f[1] * v_t + f[2] * (v_t ** 2) + f[3] * a_t + f[4] * (a_t ** 2) + f[5] * v_t * a_t

	ER = max(0, E_1)

	return ER


def compute_emission_for_NOx_and_VOC(df_with_emission_functions, speed_value, acceleration_value):
	v_t = speed_value
	a_t = acceleration_value

	if a_t >= -0.5:
		sel_row_of_df = df_with_emission_functions[df_with_emission_functions['acceleration'] == '>= -0.5']
		f = np.array(sel_row_of_df[['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6']]).flatten()

	else:
		sel_row_of_df = df_with_emission_functions[df_with_emission_functions['acceleration'] == '< -0.5']
		f = np.array(sel_row_of_df[['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6']]).flatten()

	E_1 = f[0] + f[1] * v_t + f[2] * (v_t ** 2) + f[3] * a_t + f[4] * (a_t ** 2) + f[5] * v_t * a_t

	ER = max(0, E_1)

	return ER


def map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant='CO_2'):
	"""Map each road to its emissions.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	name_of_pollutant : string
		the name of the pollutant. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].

	Returns
	-------
	Dictionary
		a dictionary with road links as keys and the list of emissions on that road as value.
		E.g. {(node_0, node_1): [emission_0, ..., emission_n]}

	"""

	if name_of_pollutant not in tdf_with_emissions.columns:
		print('Emissions have not been previously computed: use compute_emissions first.')
		return
	if 'road_link' not in tdf_with_emissions.columns:
		print('Points of TrajDataFrame have not been previously map-matched: use find_nearest_edges_in_network first.')

	road_links = list(tdf_with_emissions['road_link'])
	emissions = list(tdf_with_emissions[name_of_pollutant])

	dict_road_to_emissions = {}

	for index in range(0, len(road_links)):

		c_road = (road_links[index][0], road_links[index][1])

		if c_road in road_network.edges():
			dict_road_to_emissions.setdefault(c_road, []).append(emissions[index])

	return dict_road_to_emissions


def map_vehicle_to_emissions(tdf_with_emissions, name_of_pollutant='CO_2'):
	"""Map each vehicle to its emissions.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	name_of_pollutant : string
		the name of the pollutant. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].

	Returns
	-------
	Dictionary
		a dictionary with 'uid' in the TrajDataFrame as keys and the list of emissions of that vehicle as value.
		E.g. {(uid): [emission_0, ..., emission_n]}

	"""

	set_of_vehicles = set(tdf_with_emissions['uid'])

	dict_vehicle_to_emissions = {}

	for c_vehicle in set_of_vehicles:

		c_emissions = np.array(tdf_with_emissions[tdf_with_emissions['uid'] == c_vehicle][name_of_pollutant])
		dict_vehicle_to_emissions[c_vehicle] = c_emissions

	return dict_vehicle_to_emissions



###########################################################################################################
############################################ PLOTTING #####################################################
###########################################################################################################

def plot_road_network_with_emissions(tdf_with_emissions, road_network, normalization_factor = None,
									 name_of_pollutant='CO_2', color_map='autumn_r', bounding_box=None, save_fig=False):
	"""Plot emissions

	Plotting emissions of one of four pollutants using the module osmnx.plot_graph_routes.
	Colors indicate intensity of cumulate emissions on each road.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	normalization_factor : str
		the type of normalization wanted. It can be None, 'tot_emissions' or 'road_length'.

	name_of_pollutant : string
		the name of the pollutant to plot. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].
		Default is 'CO_2'.

	color_map : str
		name of the colormap to use.
		Default is 'autumn_r'.

	bounding_box : list
		the bounding box as north, south, east, west, if one wants to plot the emissions only in a certain bbox of the network.
		Default is None.

	save_fig : bool
		whether or not to save the figure.
		Default is False.

	Returns
	-------
	fig, ax
	"""

	if name_of_pollutant not in tdf_with_emissions.columns:
		print('Emissions have not been previously computed: use compute_emissions first.')
		return
	if 'road_link' not in tdf_with_emissions.columns:
		print('Points of TrajDataFrame have not been previously map-matched: use find_nearest_edges_in_network first.')

	list_all_emissions = list(tdf_with_emissions[name_of_pollutant])

	dict_road_to_emissions = map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant)

	# extracting the list of roads and creating a list of colors to color them:
	list_roads = []
	list_road_to_emissions_cumulates = []  # this is used to create the list of colors
	list_all_normalized_emissions = []  # this is used to set the colormap (with cm.ScalarMappable)

	for road, emission in dict_road_to_emissions.items():
		list_roads.append(list(road))
		if normalization_factor == None:
			list_road_to_emissions_cumulates.extend([list(road) + [sum(emission)]])
			colorbar_label = r'$%s$ (g)' % name_of_pollutant
		else:
			if normalization_factor == 'road_length':
				road_length = road_network.get_edge_data(road[0], road[1], key=0, default=80)['length']  # default set to 80 meters (~mean road length) --> can be improved TODO
				normalized_emissions = sum(emission) / road_length #* 100  # quantity of emissions per 100 meters on that road
				#colorbar_label = r'$%s$ (grams per 100 meters of road)' % name_of_pollutant
				colorbar_label = r'$%s$ (grams per meter of road)' % name_of_pollutant
			if normalization_factor == 'tot_emissions':
				normalized_emissions = sum(emission) / sum(list_all_emissions) * 100  # emission percentage of the gross total
				colorbar_label = '% ' + r'$%s$' % name_of_pollutant
			list_all_normalized_emissions.extend([normalized_emissions])
			list_road_to_emissions_cumulates.extend([list(road) + [normalized_emissions]])

	edge_cols = get_edge_colors_from_list(list_road_to_emissions_cumulates, cmap=color_map, num_bins=3)

	if normalization_factor != None:
		list_all_emissions = list_all_normalized_emissions
	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min(list_all_emissions),
																 vmax=max(list_all_emissions)))

	fig, ax = ox.plot_graph_routes(road_network,
								   list_roads,
								   bbox=bounding_box,
								   fig_height=20,
								   route_color=edge_cols,
								   route_linewidth=3,
								   orig_dest_node_alpha=0,
								   show=False, close=False)

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max')
	cbar.set_label(colorbar_label, size=25,
				   labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=20)

	if save_fig:
		filename = str('plot_emissions_%s.png' % name_of_pollutant)
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close(fig)
	else:
		fig.show()

	return fig, ax


def plot_road_network_with_emissions__OLD(tdf_with_emissions, road_network, name_of_pollutant='CO_2',
									 color_map='autumn_r', bounding_box=None, save_fig=False):
	"""Plot emissions

	Plotting emissions of one of four pollutants using the module osmnx.plot_graph_routes.
	Colors indicate intensity of cumulate emissions on each road.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	name_of_pollutant : string
		the name of the pollutant to plot. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].
		Default is 'CO_2'.

	color_map : str
		name of the colormap to use.
		Default is 'autumn_r'.

	bounding_box : list
		the bounding box as north, south, east, west, if one wants to plot the emissions only in a certain bbox of the network.
		Default is None.

	save_fig : bool
		whether or not to save the figure.
		Default is False.

	Returns
	-------
	fig, ax
	"""

	if name_of_pollutant not in tdf_with_emissions.columns:
		print('Emissions have not been previously computed: use compute_emissions first.')
		return
	if 'road_link' not in tdf_with_emissions.columns:
		print('Points of TrajDataFrame have not been previously map-matched: use find_nearest_edges_in_network first.')

	emissions = list(tdf_with_emissions[name_of_pollutant])

	dict_road_to_emissions = map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant)

	# extracting the list of roads and creating a list of colors to color them:
	list_roads = []
	list_road_to_emissions_cumulates = []  # this is used to create the list of colors

	for road, emission in dict_road_to_emissions.items():
		list_roads.append(list(road))
		list_road_to_emissions_cumulates.extend([list(road) + [sum(emission)]])

	edge_cols = get_edge_colors_from_list(list_road_to_emissions_cumulates,
										  cmap=color_map, num_bins=3)

	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min(emissions), vmax=max(emissions)))

	fig, ax = ox.plot_graph_routes(road_network,
								   list_roads,
								   bbox=bounding_box,
								   fig_height=20,
								   route_color=edge_cols,
								   route_linewidth=3,
								   orig_dest_node_alpha=0,
								   show=False, close=False)

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max')
	cbar.set_label('% '+'%s' % name_of_pollutant, size=25,
				   labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=20)

	if save_fig:
		filename = str('plot_emissions_%s.png' % name_of_pollutant)
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close(fig)
	else:
		fig.show()

	return fig, ax


def plot_road_network_with_emissions__OLD_OLD(tdf_with_emissions, road_network, name_of_pollutant='CO_2',
										  color_map='autumn_r', bounding_box=None, save_fig=False):
	"""Plot emissions

	Plotting emissions of one of four pollutants using the module osmnx.plot_graph_routes.
	Colors indicate intensity of cumulate emissions on each road.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	name_of_pollutant : string
		the name of the pollutant to plot. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].

	Returns
	-------
	fig, ax
	"""

	if name_of_pollutant not in tdf_with_emissions.columns:
		print('Emissions have not been previously computed: use compute_emissions first.')
		return
	if 'road_link' not in tdf_with_emissions.columns:
		print('Points of TrajDataFrame have not been previously map-matched: use find_nearest_edges_in_network first.')

	dict_road_to_emissions = map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant)

	# extracting the list of roads and creating a list of colors to color them:
	list_roads = []
	list_road_to_emissions_cumulates = []  # this is used to create the list of colors

	for road, emission in dict_road_to_emissions.items():
		list_roads.append(list(road))
		list_road_to_emissions_cumulates.extend([list(road) + [sum(emission)]])

	edge_cols = get_edge_colors_from_list(list_road_to_emissions_cumulates,
										  cmap=color_map, num_bins=3)

	if save_fig:
		fig, ax = ox.plot_graph_routes(road_network,
									   list_roads,
									   bbox=bounding_box,
									   fig_height=20,
									   route_color=edge_cols,
									   route_linewidth=2,
									   orig_dest_node_alpha=0,
									   show=False, save=True, file_format='png',
									   filename=str('plot_emissions_%s' % name_of_pollutant))
	else:
		fig, ax = ox.plot_graph_routes(road_network,
									   list_roads,
									   bbox=bounding_box,
									   # fig_height=20,
									   route_color=edge_cols,
									   route_linewidth=2,
									   orig_dest_node_alpha=0)

	return fig, ax


def get_edge_colors_from_list(list_of_emissions_per_edge, num_bins=3, cmap='summer', start=0, stop=1, na_color='none'):
	"""
	Get a list of edge colors by binning some continuous-variable attribute into
	quantiles.

	Parameters
	----------
	list_of_emissions_per_edge : list
		list in the form [u, v, cont_var], where u, v are OSM ids specifying an edge,
		and cont_var is the value of the continuous variable on that edge
	num_bins : int
		how many quantiles
	cmap : string
		name of a colormap
	start : float
		where to start in the colorspace
	stop : float
		where to end in the colorspace
	na_color : string
		what color to assign nodes with null attribute values

	Returns
	-------
	list

	References
	----------
	took from get_edge_colors_by_attr in the osmnx.plot module, and modified.
	"""
	if num_bins is None:
		num_bins = len(list_of_emissions_per_edge)
	bin_labels = range(num_bins)
	attr_values = pd.Series([co2 for u, v, co2 in list_of_emissions_per_edge])
	cats = pd.qcut(x=attr_values, q=num_bins, labels=bin_labels)
	colors = ox.get_colors(num_bins, cmap, start, stop)
	edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
	return edge_colors


def plot_speed_over_time_for_one_id(tdf, uid):
	"""Plot speed over time.

	For a given individual ('uid') in a TrajDataFrame, plots its speed over time.

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the individuals.

	uid : int
		the uid of the individual for which we want the plot.

	Returns
	-------

	Warnings
	--------
	if the time interval of the trajectory is >24 hours, probably the x labels will be unreadable;
	if speed has not been previously computed, a function is firstly called to compute it.
	"""

	set_of_ids = set(tdf['uid'])
	if str(uid) not in set_of_ids:
		print('ID ' + uid + ' not found.')
		return

	if 'speed' not in tdf.columns:
		tdf_with_speed = compute_speed_from_tdf(tdf)
	else:
		tdf_with_speed = tdf.copy()

	tdf_one_id = tdf_with_speed.loc[tdf_with_speed['uid'] == str(uid)]

	seconds = dt.SecondLocator()
	minutes = dt.MinuteLocator()
	hours = dt.HourLocator()
	date_fmt = dt.DateFormatter('%H:%M')

	fig, ax = plt.subplots()
	ax.plot_date(tdf_one_id['datetime'], tdf_one_id['speed'], fmt='-')

	ax.xaxis.set_major_locator(hours)
	ax.xaxis.set_major_formatter(date_fmt)

	ax.set_ylabel('speed')

	fig.autofmt_xdate()  # rotate x labels to make them readable
	plt.title('Speed over time for one vehicle')
	plt.show()

	return


###########################################################################################################
####################################### COMPUTE STATISTICS ################################################
###########################################################################################################

def compute_corrs(tdf_with_emissions, list_of_pollutants=['CO_2', 'NO_x', 'PM', 'VOC']):
	"""Compute correlation coefficients between emissions and (1) radius of gyration and (2) uncorrelated entropy.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	list_of_pollutants : list
		the list of pollutants for which one wants to compute the correlations.

	Returns
	-------
	DataFrame
		a DataFrame containing the computed coefficients for each of the pollutants.
	"""

	print('Computing radius of gyration...')
	rg_df = radius_of_gyration(tdf_with_emissions)
	print('Computing uncorrelated entropy...')
	ue_df = uncorrelated_entropy(tdf_with_emissions, normalize=True)

	corr_coefs = pd.DataFrame(columns=list_of_pollutants, index=['r_gyr', 'un_entropy'])

	for c_pollutant in list_of_pollutants:

		dict_vehicle_to_emissions = map_vehicle_to_emissions(tdf_with_emissions, name_of_pollutant=c_pollutant)

		df_rows = []
		for c_uid, c_array in dict_vehicle_to_emissions.items():
			c_rg = float(rg_df[rg_df['uid'] == c_uid]['radius_of_gyration'])
			c_ue = float(ue_df[ue_df['uid'] == c_uid]['norm_uncorrelated_entropy'])
			c_row = [c_uid, np.sum(c_array), c_rg, c_ue]
			df_rows.append(c_row)

		df = pd.DataFrame(df_rows, columns=['uid', 'emissions', 'r_gyr', 'un_entropy'])

		corr_coefs.loc['r_gyr', c_pollutant] = [np.corrcoef(df['emissions'], df['r_gyr'])[1][0]]
		corr_coefs.loc['un_entropy', c_pollutant] = [np.corrcoef(df['emissions'], df['un_entropy'])[1][0]]

	print("Pearson's correlation coeffs:")
	print()
	print(corr_coefs)

	return corr_coefs


def normalize_emissions(tdf_with_emissions, percentage=True, list_of_pollutants=['CO_2', 'NO_x', 'PM', 'VOC']):
	"""Normalize values of emissions of the pollutants.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	percentage : bool
		whether one wants the result as a percentage or not.

	list_of_pollutants : list
		the list of pollutants for which one wants to normalize.

	Returns
	-------
	DataFrame
		a DataFrame containing the normalized values of emissions for each of the pollutants.
	"""

	tdf_with_normalized_emissions = tdf_with_emissions.copy()
	for c_pollutant in list_of_pollutants:
		tot_emissions = float(np.sum(tdf_with_emissions[c_pollutant]))
		if percentage:
			tdf_with_normalized_emissions[c_pollutant] = tdf_with_emissions[c_pollutant] / tot_emissions * 100
		else:
			tdf_with_normalized_emissions[c_pollutant] = tdf_with_emissions[c_pollutant] / tot_emissions
	return tdf_with_normalized_emissions


def compute_stats_for_network(road_network, area=None, circuity_dist='gc'):
	"""Calculate basic descriptive metric and topological stats for a graph.

	See basic_stats in osmnx.stats module for details.
	For an unprojected lat-lng graph, tolerance and graph units should be in degrees, and circuity_dist should be ‘gc’.
	For a projected graph, tolerance and graph units should be in meters (or similar) and circuity_dist should be ‘euclidean’.

	Parameters
	----------
	road_network : networkx MultiDiGraph

	area : numeric
		the area covered by the street network, in square meters (typically land area);
		if none, will skip all density-based metrics.

	circuity_dist : str
		 ‘gc’ or ‘euclidean’, how to calculate straight-line distances for circuity measurement;
		 use former for lat-lng networks and latter for projected networks.

	Returns
	-------
	Dictionary
		dictionary of network stats (see osmnx documentation for details).
	"""

	dict_stats = ox.stats.basic_stats(road_network, area=area,
									  clean_intersects=False, tolerance=15,
									  circuity_dist=circuity_dist)
	return dict_stats