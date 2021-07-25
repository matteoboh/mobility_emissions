import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import spearmanr
from skmob.measures.individual import *
from skmob.preprocessing.clustering import cluster
from shapely.geometry import MultiPoint
from .emissions import *

###########################################################################################################
####################################### COMPUTE STATISTICS ################################################
###########################################################################################################


def compute_corrs_with_mobility_measures(tdf_original, tdf_with_emissions, set_of_pollutants={'CO_2', 'NO_x', 'PM', 'VOC'},
										 corr_coef='spearman', plot_scatter=False):
	"""Compute correlation coefficients between emissions and some mobility measures of the vehicles.
	The mobility measures are: radius of gyration, uncorrelated entropy, maximum distance travelled, and straight line distance travelled.

	Parameters
	----------
	tdf_original : TrajDataFrame
		TrajDataFrame with original trajectories (before the time filtering and computation of emissions)

	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	set_of_pollutants : set
		the set of pollutants for which one wants to compute the correlations.

	corr_coef : str
		if not 'spearman', then the Pearson's correlation coefficients are returned.

	plot_scatter : bool
		whether to show the scatter plot for each couple of attributes

	Returns
	-------
	DataFrame
		a DataFrame containing the computed coefficients for each couple (pollutant, mobility metric).
		a DataFrame containing the p-values returned by scipy.stats.spearmanr for each couple (pollutant, mobility metric). If corr_coef != 'spearman', then this DataFrame is empty.
		a DataFrame with all the mobility metrics and emissions for each user.
	"""

	# stops detection (taking first & end points of each trajectory as stop locations)
	tdf_start_points = tdf_original.groupby(['uid', 'tid']).head(1)
	tdf_end_points = tdf_original.groupby(['uid', 'tid']).tail(1)
	tdf_start_end_points = pd.concat([tdf_start_points, tdf_end_points]).sort_index()

	# clustering stops
	tdf_clustered_stops = cluster(tdf_start_end_points, cluster_radius_km=0.1)

	# assigning the lat/lng of the centroid of each cluster to its points:
	def assign_centroid(df):
		centroid = MultiPoint(np.array(df[['lat', 'lng']])).centroid
		df['lat'] = centroid.x
		df['lng'] = centroid.y
		return df

	tdf_stops = tdf_clustered_stops.groupby(['uid', 'cluster']).apply(assign_centroid)

	# computing the mobility measures:
	print('Computing radius of gyration...')
	rg_df = radius_of_gyration(tdf_original)
	print('Computing uncorrelated entropy...')
	ue_df = uncorrelated_entropy(tdf_stops, normalize=True)
	print('Computing maximum distance travelled...')
	md_df = maximum_distance(tdf_stops)
	print('Computing distance straight line...')
	ds_df = distance_straight_line(tdf_original)

	corr_coefs = pd.DataFrame(columns=set_of_pollutants, index=['r_gyr',
																'un_entropy',
																'max_dist',
																'dist_straight'])
	df_pvals = corr_coefs.copy()

	map__vehicle__CO2 = map_vehicle_to_emissions(tdf_with_emissions, 'CO_2')
	map__vehicle__NOx = map_vehicle_to_emissions(tdf_with_emissions, 'NO_x')
	map__vehicle__PM = map_vehicle_to_emissions(tdf_with_emissions, 'PM')
	map__vehicle__VOC = map_vehicle_to_emissions(tdf_with_emissions, 'VOC')

	df_rows = []
	for c_uid in map__vehicle__CO2.keys():
		c_rg = float(rg_df[rg_df['uid'] == c_uid]['radius_of_gyration'])
		c_ue = float(ue_df[ue_df['uid'] == c_uid]['norm_uncorrelated_entropy'])
		c_md = float(md_df[md_df['uid'] == c_uid]['maximum_distance'])
		c_ds = float(ds_df[ds_df['uid'] == c_uid]['distance_straight_line'])
		c_CO2 = np.sum(map__vehicle__CO2[c_uid])
		c_NOx = np.sum(map__vehicle__NOx[c_uid])
		c_PM = np.sum(map__vehicle__PM[c_uid])
		c_VOC = np.sum(map__vehicle__VOC[c_uid])
		c_row = [c_uid,
				 c_CO2, c_NOx, c_PM, c_VOC,
				 c_rg, c_ue, c_md, c_ds]
		df_rows.append(c_row)

	df = pd.DataFrame(df_rows, columns=['uid',
										'CO_2', 'NO_x', 'PM', 'VOC',
										'r_gyr', 'un_entropy', 'max_dist', 'dist_straight'])

	for c_pollutant in set_of_pollutants:
		if corr_coef == 'spearman':
			spearman_radius = spearmanr(df[c_pollutant], df['r_gyr'])
			spearman_entropy = spearmanr(df[c_pollutant], df['un_entropy'])
			spearman_dist = spearmanr(df[df['max_dist'].isnull() == False][c_pollutant],
									  df[df['max_dist'].isnull() == False]['max_dist'])
			spearman_dist_straight = spearmanr(df[df['dist_straight'].isnull() == False][c_pollutant],
											   df[df['dist_straight'].isnull() == False]['dist_straight'])

			corr_coefs.loc['r_gyr', c_pollutant] = spearman_radius[0]
			corr_coefs.loc['un_entropy', c_pollutant] = spearman_entropy[0]
			corr_coefs.loc['max_dist', c_pollutant] = spearman_dist[0]
			corr_coefs.loc['dist_straight', c_pollutant] = spearman_dist_straight[0]

			# p-values
			df_pvals.loc['r_gyr', c_pollutant] = spearman_radius[1]
			df_pvals.loc['un_entropy', c_pollutant] = spearman_entropy[1]
			df_pvals.loc['max_dist', c_pollutant] = spearman_dist[1]
			df_pvals.loc['dist_straight', c_pollutant] = spearman_dist_straight[1]

		else:
			corr_coefs.loc['r_gyr', c_pollutant] = np.corrcoef(df[c_pollutant], df['r_gyr'])[1][0]
			corr_coefs.loc['un_entropy', c_pollutant] = np.corrcoef(df[c_pollutant], df['un_entropy'])[1][0]
			corr_coefs.loc['max_dist', c_pollutant] = np.corrcoef(df[c_pollutant], df['max_dist'])[1][0]
			corr_coefs.loc['dist_straight', c_pollutant] = np.corrcoef(df[c_pollutant], df['dist_straight'])[1][0]

	print("%s's correlation coeffs:" % corr_coef.capitalize())
	print()
	print(corr_coefs)

	if plot_scatter == True:
		fig = scatter_matrix(df.drop(['uid'], axis=1), figsize=(10, 10))
		plt.show()

	return corr_coefs, df_pvals, df


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


def compute_corrs_with_edges_attributes(road_network, pollutant, list_attribute_names, corr_coef='spearman',
										plot_scatter=False):
	"""Compute correlation coefficients between the edges' attributes of a road network
	(for which a value of emissions has previously been estimated).

	Parameters
	----------
	road_network : networkx MultiDiGraph

	pollutant : str
		name of the pollutant for which one wants the correlations with the other attributes.

	list_attribute_names : list
		the list with the names of the edges' attributes.
		It must also comprehend the pollutant.

	corr_coef : str
		if 'spearman' returns the Spearman correlation matrix AND the p-values,
		else returns the Pearson correlation matrix.

	plot_scatter : bool
		whether to return the scatterplot matrix or not.

	Returns
	-------
	numpy.ndarray
		the correlation matrix.
	"""

	map__edge__pollutant = nx.get_edge_attributes(road_network, pollutant)

	list_all_dicts_of_edges_attributes_where_pollutant_isnot_None = [road_network.get_edge_data(u, v, key) for
																	 (u, v, key), poll in map__edge__pollutant.items()
																	 if poll != None]

	list_all_attributes = []
	for c_attr in list_attribute_names:
		c_list_attr = [np.float(edge_attr[c_attr]) if edge_attr[c_attr] != None else None for edge_attr in
					   list_all_dicts_of_edges_attributes_where_pollutant_isnot_None]
		list_all_attributes.append(c_list_attr)

	df = pd.DataFrame(list_all_attributes).T
	df.columns = list_attribute_names
	df_no_nan = df.dropna()
	list_all_attributes_no_nan = [list(df_no_nan[col]) for col in df_no_nan.columns]

	if plot_scatter == True:
		fig = scatter_matrix(df, figsize=(10, 10))
		plt.savefig('scatter_matrix_%s.png' % pollutant)

	if corr_coef == 'spearman':
		spearman_results = spearmanr(np.array(list_all_attributes_no_nan), axis=1)
		return spearman_results[0], spearman_results[1], df
	else:
		return np.corrcoef(np.array(list_all_attributes_no_nan)), df
