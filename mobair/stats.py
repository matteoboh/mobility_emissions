import pandas as pd
import osmnx as ox
from skmob.measures.individual import *
from .emissions import *

###########################################################################################################
####################################### COMPUTE STATISTICS ################################################
###########################################################################################################

def compute_corrs(tdf_with_emissions, list_of_pollutants=['CO_2', 'NO_x', 'PM', 'VOC']):
	"""Compute correlation coefficients between emissions and (1) radius of gyration, (2) uncorrelated entropy.

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