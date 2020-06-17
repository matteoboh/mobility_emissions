import skmob
from skmob.core.trajectorydataframe import *
import pandas as pd


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
			new_tid = c_tid + max_tid + i #  the new 'tid' given to the sub-trajectory
			dict_subtraj_with_new_tid[new_tid] = c_subtraj
			i+= 1

		# Appending all the new sub-trajectories to a new tdf:
		df_filtered_with_new_tid = pd.DataFrame()
		for tid, subtraj in dict_subtraj_with_new_tid.items():
			slice_of_df_filtered_with_new_tid = tdf_filtered.loc[list(subtraj)]
			slice_of_df_filtered_with_new_tid['tid'] = tid
			df_filtered_with_new_tid = df_filtered_with_new_tid.append(slice_of_df_filtered_with_new_tid)

	tdf_filtered_with_new_tid = skmob.TrajDataFrame(df_filtered_with_new_tid, latitude='lat', longitude='lng',
													datetime='datetime', user_id='uid')

	return tdf_filtered_with_new_tid