import numpy as np
import osmnx as ox
from skmob.core.trajectorydataframe import *

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