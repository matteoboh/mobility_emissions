import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import spearmanr
from skmob.measures.individual import *
from .emissions import *

###########################################################################################################
####################################### COMPUTE STATISTICS ################################################
###########################################################################################################

def compute_corrs(tdf_with_emissions, set_of_pollutants={'CO_2', 'NO_x', 'PM', 'VOC'}):
	"""Compute correlation coefficients between emissions and (1) radius of gyration, (2) uncorrelated entropy.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	set_of_pollutants : set
		the set of pollutants for which one wants to compute the correlations.

	Returns
	-------
	DataFrame
		a DataFrame containing the computed coefficients for each of the pollutants.
	"""

	print('Computing radius of gyration...')
	rg_df = radius_of_gyration(tdf_with_emissions)
	print('Computing uncorrelated entropy...')
	ue_df = uncorrelated_entropy(tdf_with_emissions, normalize=True)

	corr_coefs = pd.DataFrame(columns=set_of_pollutants, index=['r_gyr', 'un_entropy'])

	for c_pollutant in set_of_pollutants:

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


def compute_corrs_between_edges_attributes(road_network, pollutant, list_attribute_names, corr_coef='spearman',
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
	df_no_nan = df.dropna()
	list_all_attributes_no_nan = [list(df_no_nan[col]) for col in df_no_nan.columns]

	if plot_scatter == True:
		df.columns = list_attribute_names

		fig = scatter_matrix(df, figsize=(10, 10))
		plt.savefig('scatter_matrix_%s.png' % pollutant)

	if corr_coef == 'spearman':
		return spearmanr(np.array(list_all_attributes_no_nan), axis=1)
	else:
		return np.corrcoef(np.array(list_all_attributes_no_nan))
