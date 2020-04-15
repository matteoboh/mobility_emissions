from skmob.utils import gislib
from skmob.core.trajectorydataframe import *

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