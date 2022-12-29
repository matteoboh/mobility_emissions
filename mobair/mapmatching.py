import skmob
import numpy as np
import networkx as nx
import osmnx as ox
from skmob.core.trajectorydataframe import *

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
	list_of_nearest_nodes = ox.nearest_nodes(road_network, X=vec_of_longitudes, Y=vec_of_latitudes)
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
	Each edge is represented as a list of type [u,v,key],
	where u,v are respectively the OSM ids of its starting and ending node,
	and the key discriminates between parallel edges (if present).

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
		the TrajDataFrame with 1 more column collecting the lists [u,v,key] identifying the edges.
	"""
	tdf.sort_by_uid_and_datetime()

	# extracting arrays of lat and lng for the vehicle:
	vec_of_longitudes = np.array(tdf['lng'])
	vec_of_latitudes = np.array(tdf['lat'])

	# for each (lat,lon), find the nearest edge in the road network:
	array_of_nearest_edges = ox.nearest_edges(road_network, X=vec_of_longitudes, Y=vec_of_latitudes)
	###
	# method (str {None, 'kdtree', 'balltree'}) – Which method to use for finding nearest node to each point.
	# If None, we manually find each node one at a time using osmnx.utils.get_nearest_node and haversine.
	# If ‘kdtree’ we use scipy.spatial.cKDTree for very fast euclidean search.
	# If ‘balltree’, we use sklearn.neighbors.BallTree for fast haversine search.
	###

	list_of_nearest_edges = [(edge[0],edge[1],edge[2]) for edge in array_of_nearest_edges]

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

	vehicle_id : int
		ID of the vehicle.

	Returns
	-------
	List
		list of the nodes in the network describing the shortest route that connects all of them.
	"""

	tdf.sort_by_uid_and_datetime()

	# building a tdf with only the points belonging to the specified vehicle:
	try:
		tdf_one_id = tdf.loc[tdf['uid'] == vehicle_id]
	except KeyError:
		print('ID ' + str(vehicle_id) + ' not found.')
		return

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
