import skmob
import pandas as pd
from datetime import timedelta
from skmob.utils import utils, constants
from skmob.core.trajectorydataframe import *

###########################################################################################################
##################################### FILTERING POINTS BY TIME ############################################
###########################################################################################################

def filter_on_time_interval(tdf, max_time=30):
	"""Time filtering.

	For each trajectory in a TrajDataFrame, retains only the subset of consecutive points
	(i.e. the sub-trajectories) that are distant no more than max_time (in seconds) from each other.
	Each sub-trajectory will have a new tid (that has nothing to do with the old one
	of the original trajectory).

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

	groupby = []
	if utils.is_multi_user(tdf):
		if len(set(tdf[constants.UID])) != 1:
			groupby.append(constants.UID)

	if len(groupby) > 0:
		tdf_filtered = tdf.groupby(groupby, group_keys=False, as_index=False).apply(filter_on_time_interval_for_1uid,
																					max_time)
	else:
		tdf_filtered = filter_on_time_interval_for_1uid(tdf, max_time)

	return tdf_filtered.sort_by_uid_and_datetime().reset_index(drop=True)


def filter_on_time_interval_for_1uid(tdf, max_time=30):
	tdf_1uid = tdf.copy().reset_index(drop=True)

	# creating DF with time differences between consecutive points
	df_timediffs = pd.DataFrame(tdf_1uid.datetime.diff())

	# Selecting the sub-trajectories, composed by points that are selected s.t.
	# an index of a point pi is saved if:
	# dist(pi, pi-1) < max_time OR
	# dist(pi+1, pi) < max_time
	list_indexes = [index for index in df_timediffs.index[:-1] if
					(df_timediffs.loc[index].datetime < timedelta(seconds=max_time)) or (
								df_timediffs.loc[index + 1].datetime < timedelta(seconds=max_time))]
	last_index = df_timediffs.index[-1]
	if df_timediffs.loc[last_index].datetime < timedelta(seconds=max_time):
		list_indexes.extend([last_index])
	tdf_1uid__filt = tdf_1uid.loc[list_indexes]
	tdf_1uid__filt.reset_index(inplace=True)

	# Creating the new tid for the selected sub-trajectories
	for index in tdf_1uid__filt.index[1:]:
		c_old_index = tdf_1uid__filt.loc[index]['index']
		previous_old_index = tdf_1uid__filt.loc[index - 1]['index']
		if c_old_index - previous_old_index > 1:
			tdf_1uid__filt.loc[index:, 'tid'] = tdf_1uid__filt.loc[index - 1, 'tid'] + 1
		else:
			if df_timediffs.loc[c_old_index].datetime > timedelta(seconds=max_time):
				tdf_1uid__filt.loc[index:, 'tid'] = tdf_1uid__filt.loc[index - 1, 'tid'] + 1

	# Selecting only sub-trajectories composed by more than one point
	tdf_1uid__filt__final = tdf_1uid__filt.groupby('tid').filter(lambda x: len(x) > 1).drop('index', axis=1)

	return tdf_1uid__filt__final