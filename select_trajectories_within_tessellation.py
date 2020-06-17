import json
import skmob
import pandas as pd
from .util_funcs import *

### PARAMETERS
CELL_SIZE = 1500 # size of cell in the spatial tessellation
REGION = "Florence" #"Greater London" #"Rome"
PATH_TO_INPUT_FILE = './input_files/all_traj/'
NAME_OF_INPUT_FILE = 'italy_27837users_2017-01-01_2017-02-02.csv'
PATH_TO_OUTPUT_FILE = './input_files/'
PATH_TO_TESSELLATION = './input_files/tessellations/'
###

### LOAD TRAJECTORIES
df = pd.read_csv(PATH_TO_INPUT_FILE+NAME_OF_INPUT_FILE, header=0)
tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lng', datetime='datetime', user_id='uid', trajectory_id='tid')
#tdf = skmob.read(PATH_TO_TDF+NAME_OF_TDF)

### DOWNLOAD TESSELLATION
tessellation = download_square_tessellation(cell_size=CELL_SIZE, region=REGION)

### SELECT TRAJECTORIES WITHIN TESSELLATION
grouped_tdf = select_trajectories_within_tessellation(tdf, tessellation)
num_of_uid = len(grouped_tdf['uid'].unique())

### Save the spatial tessellation and the trajectories into a file
def save_files(tdf, tessellation):
    # load the json configuration file
    config = json.load(open('config.json'))
    region_name = REGION.lower().replace(" ", "_")
    tessellation.to_file(PATH_TO_TESSELLATION + "%s_tessellation_%s.geojson" % (region_name, CELL_SIZE), driver='GeoJSON')

    tdf.to_csv(PATH_TO_OUTPUT_FILE + NAME_OF_INPUT_FILE[:-4] + '__%s_%susers.csv' % (region_name, num_of_uid), index=False)

save_files(grouped_tdf, tessellation)