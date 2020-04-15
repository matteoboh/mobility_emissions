import numpy as np
from speed_and_acceleration import *

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