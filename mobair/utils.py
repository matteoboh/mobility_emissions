import numpy as np
import osmnx as ox
import networkx as nx
from .emissions import *

###########################################################################################################
############################################# UTILS #######################################################
###########################################################################################################


def map_road_list_to_attribute(list_roads, road_network, attribute_name, default_value):
	"""Maps each road of a list to the given attribute.

		Parameters
		----------
		list_roads : list
			a list of lists of nodes IDs.

		road_network : networkx MultiDiGraph
			the road network from which to extract the road attributes.

		attribute_name : str
			the name of the attribute.

		default_value : string or int
			the value to return if the attribute is not found.

		Returns
		-------
		Dictionary
			dictionary that maps each road (u,v,key) to its value of the attribute.
		"""

	set_roads = set(tuple(i) for i in list_roads)
	dict_road_to_attribute = {}
	for c_road in set_roads:

		c_dict_attributes = road_network.get_edge_data(c_road[0], c_road[1], c_road[2], default=default_value)

		if type(c_dict_attributes) == dict:   # i.e. if the edge does exist in the network...
			c_attribute = c_dict_attributes.get(attribute_name, default_value)
		else:
			c_attribute = default_value
		dict_road_to_attribute[c_road] = c_attribute

	return dict_road_to_attribute


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


def map_road_to_cumulate_emissions(tdf_with_emissions, road_network, name_of_pollutant='CO_2', normalization_factor=None):
	"""Outputs a dict of type {(u,v,key) : cumulate_emissions}, with cumulate_emissions being normalised or not.

	Parameters:
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	name_of_pollutant : str
		the name of the pollutant for which one wants the list. Must be in {'CO_2', 'NO_x', 'PM', 'VOC'}.
		Default is 'CO_2'.

	normalization_factor : str
		whether one wants to normalise the emissions on each road or not.
		Must be in {'None', 'tot_emissions', 'road_length'}.
		If 'tot_emissions', the resulting cumulate_emissions can be interpreted as the share of the total emissions
		of the network in that road.
		If 'road_length', the resulting cumulate_emissions can be interpreted as the quantity of emissions per meter
		on that road.

	Returns:
	-------
	dict
		a dictionary of each edge with the (eventually normalised) cumulate emissions estimated on that edge.
	label
		a label to use for plotting.
	"""

	if normalization_factor not in list([None, 'tot_emissions', 'road_length']):
		print('normalization_factor must be one of [None, tot_emissions, road_length]')
		return

	map__road__emissions = map_road_to_emissions(tdf_with_emissions, name_of_pollutant)
	array_all_emissions = np.array(tdf_with_emissions[name_of_pollutant])
	map__road__cumulate_emissions = []
	label = ''

	if normalization_factor == None:
		label = r'$%s$ (g)' % name_of_pollutant
		map__road__cumulate_emissions = {road: np.sum(em) for road, em in
										 map__road__emissions.items()}
	else:
		if normalization_factor == 'road_length':
			label = r'$%s$ (grams per meter of road)' % name_of_pollutant
			map__road__length = nx.get_edge_attributes(road_network, 'length')
			map__road__cumulate_emissions = {road: np.sum(em) / map__road__length[road] for road, em in
											 map__road__emissions.items() if road in map__road__length.keys()}

		if normalization_factor == 'tot_emissions':
			label = '% ' + r'$%s$' % name_of_pollutant
			sum_all_emissions = np.sum(array_all_emissions)
			map__road__cumulate_emissions = {road: np.sum(em) / sum_all_emissions * 100 for road, em in
											 map__road__emissions.items()}

	return map__road__cumulate_emissions, label


def create_list_cumulate_emissions_per_vehicle(tdf_with_emissions, name_of_pollutant):
	"""Outputs the list of cumulate emissions for each vehicle.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	name_of_pollutant : str
		the name of the pollutant for which one wants the list. Must be in {'CO_2', 'NO_x', 'PM', 'VOC'}.
		Default is 'CO_2'.

	Returns
	-------
	list
		the list of cumulate emissions.
	label
		a label to use for plotting.
	"""
	dict_road_to_vehicle = map_vehicle_to_emissions(tdf_with_emissions, name_of_pollutant)
	list_cumulate_emissions = [np.sum(em) for em in dict_road_to_vehicle.values()]
	label = r'$%s$ (g)' % name_of_pollutant

	return list_cumulate_emissions, label


def add_edge_emissions(dict__road__emissions, road_network, name_of_pollutant='CO_2'):
	"""Add the value of emissions as a new attribute to the edges of the road network.

	Parameters
	----------
	dict__road__emissions : dict
		dict of type {(u,v,key) : emissions} (as returned by create_dict_road_to_cumulate_emissions).

	road_network : networkx MultiGraph

	name_of_pollutant : string
		the name of the pollutant to plot. Must be in {'CO_2', 'NO_x', 'PM', 'VOC'}.
		Default is 'CO_2'.

	Returns
	-------
	networkx MultiDiGraph
		road network with the new attribute on its edges.
		Note that for the edges with no value of emissions the attribute is set to None.
	"""
	for u, v, key, data in road_network.edges(keys=True, data=True):
		try:
			c_emissions = dict__road__emissions[(u, v, key)]
			data[name_of_pollutant] = c_emissions
		except KeyError:
			data[name_of_pollutant] = None

	return road_network


def add_edge_centrality_measures(road_network):
	"""Computes and add centrality measures as new attributes to the edges of the road network.
	The centrality measures are: degree, closeness centrality, betweenness centrality.
	Note that the first two are computed using the line version of the graph, while for the latter there is a networkx
	function that directly computes it for the edges of the original graph.

	Parameters
	----------
	road_network : networkx MultiDiGraph

	Returns
	-------
	networkx MultiDiGraph
		the road network with the new attributes on its edges.
	"""

	# line version of the network:
	road_network_line = nx.line_graph(road_network)

	### Closeness centrality:
	# 1. compute closeness centrality of the edges in the line version of the network
	edge_ccentrality = nx.closeness_centrality(road_network_line)
	# 2. add it as new attribute on the edges of the original network
	nx.set_edge_attributes(road_network, edge_ccentrality, 'closeness_centrality')

	### Degree centrality:
	# 1. compute degree of the edges in the line version of the network
	edge_degree = nx.degree_centrality(road_network_line)
	# 2. add it as new attribute on the edges of the original network
	nx.set_edge_attributes(road_network, edge_degree, 'degree_centrality')
	'''
	### Clustering coefficient:
	# 1. compute clustering coefficient of the edges in the line version of the network
	edge_clustering = nx.clustering(road_network_line)
	# 2. add it as new attribute on the edges of the original network
	nx.set_edge_attributes(road_network, edge_clustering, 'clustering_coeff')
	'''
	### Betweenness centrality:
	# 1. compute betweenness centrality (directly on the original network, as there is a function for doing that)
	edge_bcentrality = nx.edge_betweenness_centrality(road_network, normalized=True, weight='length')
	# 1.1 correct the resulting dictionary's keys
	edge_bcentrality__corrected = {key: (edge_bcentrality[key[:-1]] if key[:-1] in edge_bcentrality.keys() else None)
								   for key in road_network.edges(keys=True)}
	# 2. add it as new attribute on the edges of the original network
	nx.set_edge_attributes(road_network, edge_bcentrality__corrected, 'betweenness_centrality')

	return road_network


def add_edge_nearest_POIs(road_network, region, radius):
	### POIs ###
	food_amenities = ['pub', 'bar', 'restaurant', 'cafe', 'food_court']  # https://wiki.openstreetmap.org/wiki/Key:amenity
	education_amenities = ['college', 'kindergarten', 'library', 'school', 'university']
	service_amenities = ['bank', 'clinic', 'hospital', 'pharmacy', 'marketplace', 'post_office']
	shops = ['department_store', 'mall', 'supermarket']  # https://wiki.openstreetmap.org/wiki/Key:shop
	leisure = ['stadium', 'park']  # https://wiki.openstreetmap.org/wiki/Key:leisure
	railway = ['station']  # https://wiki.openstreetmap.org/wiki/Key:railway
	aeroway = ['aerodrome']  # https://wiki.openstreetmap.org/wiki/Key:aeroway
	highway = ['traffic_signals', 'stop', 'crossing']  # https://wiki.openstreetmap.org/wiki/Key:highway

	map__poi_type__poi = {'amenity': food_amenities + education_amenities + service_amenities,
						  'leisure': leisure,
						  'shop': shops,
						  'railway': railway,
						  'aeroway': aeroway,
						  'highway': highway}

	map__poi__poi_category = {'pub': 'food',
							  'bar': 'food',
							  'restaurant': 'food',
							  'cafe': 'food',
							  'food_court': 'food',
							  'college': 'education',
							  'kindergarten': 'education',
							  'library': 'education',
							  'school': 'education',
							  'university': 'education',
							  'bank': 'service',
							  'clinic': 'service',
							  'hospital': 'service',
							  'pharmacy': 'service',
							  'marketplace': 'service',
							  'post_office': 'service',
							  'stadium': 'leisure',
							  'park': 'leisure',
							  'department_store': 'retail',
							  'mall': 'retail',
							  'supermarket': 'retail',
							  'station': 'transport',
							  'aerodrome': 'transport',
							  'traffic_signals': 'signage',
							  'stop': 'signage',
							  'crossing': 'signage'}

	###
	#start_time = time.time()

	# querying all the POIs of certain types in the region
	gdf_pois = ox.geometries_from_place(region, map__poi_type__poi)
	# gdf_pois = ox.geometries_from_bbox(north, south, east, west, map__poi_type__poi)

	set_edges_missing_geometry = set()
	for u, v, key, edge_data in road_network.edges(keys=True, data=True):
		# taking the centroid of the road
		try:
			road_centroid = edge_data['geometry'].centroid
		except KeyError:
			set_edges_missing_geometry.add((u, v, key))
			# adding as attributes to the edge with None
			for poi_cat in set(map__poi__poi_category.values()):
				edge_data[poi_cat] = None
			continue

		# taking all the POIs in the gdf that are distant no more than radius from the centroid
		gdf_pois['dist'] = list(map(lambda k: ox.distance.great_circle_vec(gdf_pois.loc[k]['geometry'].centroid.y,
																		   gdf_pois.loc[k]['geometry'].centroid.x,
																		   road_centroid.y, road_centroid.x),
									gdf_pois.index))
		gdf_nearest_pois = gdf_pois[gdf_pois['dist'] <= radius]

		# creating dictionary with categories of POIs and counts
		map__poi_cat__num_of_poi = {
			'food': 0,
			'education': 0,
			'service': 0,
			'leisure': 0,
			'retail': 0,
			'transport': 0,
			'signage': 0
		}
		for poi_type, poi_list in map__poi_type__poi.items():
			if poi_type in gdf_nearest_pois.columns:
				c_df = getattr(gdf_nearest_pois, poi_type).value_counts()
				for poi in [poi for poi in c_df.index if poi in poi_list]:
					poi_category = map__poi__poi_category[poi]
					# if poi_category not in map__poi_cat__num_of_poi:
					#    map__poi_cat__num_of_poi[poi_category] = 0
					map__poi_cat__num_of_poi[poi_category] += c_df[poi]

		# adding as attributes to the edge
		for poi_cat, num_of_poi in map__poi_cat__num_of_poi.items():
			edge_data[poi_cat] = num_of_poi

	#runtime = time.time() - start_time
	#print('runtime: ', time.time() - start_time)
	#print('')
	print('> There were %s out of %s total edges with missing geometry attribute.' % (
	len(set_edges_missing_geometry), road_network.size()))

	return road_network


def split_trajectories_in_tdf(tdf, stop_tdf):
	"""Cluster the points of a TrajDataFrame into trajectories by using stop locations.

	Parameters
	----------
	tdf : TrajDataFrame
		original trajectories
	stop_tdf : TrajDataFrame
		the output of skmob.preprocessing.detection.stops, containing the stop locations of the users in the tdf

	Returns
	-------
	TrajDataFrame
		the TrajDataFrame with a new column 'tid' collecting the unique identifier of the trajectory to which the point
		belongs.
	"""
	tdf_with_tid = tdf.groupby('uid').apply(_split_trajectories, stop_tdf)
	return tdf_with_tid.reset_index(drop=True)


def _split_trajectories(tdf, stop_tdf):
	c_uid = tdf.uid[:1].item()
	stop_tdf_current_user = stop_tdf[stop_tdf.uid == c_uid]
	if stop_tdf_current_user.empty:
		return
	else:
		first_traj = [tdf[tdf.datetime <= stop_tdf_current_user.datetime[:1].item()]]
		last_traj = [tdf[tdf.datetime >= stop_tdf_current_user.leaving_datetime[-1:].item()]]
		all_other_traj = [tdf[(tdf.datetime >= start_traj_time) & (tdf.datetime <= end_traj_time)] for end_traj_time, start_traj_time in zip(stop_tdf_current_user['datetime'][1:], stop_tdf_current_user['leaving_datetime'][:-1])]
		all_traj = first_traj + all_other_traj + last_traj
		tdf_with_tid = pd.concat(all_traj)
		list_tids = [list(np.repeat(i, len(df))) for i, df in zip(range(1,len(all_traj)+1), all_traj)]
		list_tids_ravel = [item for sublist in list_tids for item in sublist]
		tdf_with_tid['tid'] = list_tids_ravel
		return tdf_with_tid
