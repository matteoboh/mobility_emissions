from .util_funcs import *

### PARAMETERS
UID_LIMIT = None # limit of users to download
AREA = 'uk' #'italy'
start = '2017-01-01'
end = '2017-02-02'
PATH_TO_OUTPUT_FILE = './input_files/all_traj/'
###

### download trajectories
tdf = download_trajectories(area=AREA, uid_limit=UID_LIMIT,
							start_date=start, end_date=end)
print('Size of TrajDataFrame: %s' %len(tdf))
num_of_uid = len(tdf['uid'].unique())
print('Number of users: %s' %num_of_uid)

### Saving in CSV
tdf.to_csv(PATH_TO_OUTPUT_FILE+'%s_%susers_%s_%s.csv' % (AREA, num_of_uid, start, end), index=False)
