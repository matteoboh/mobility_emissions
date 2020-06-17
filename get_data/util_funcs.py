import pandas as pd
import geopandas as gpd
import psycopg2
import json
import skmob
from skmob.tessellation import tilers
import numpy as np

def to_TrajDataFrame(gdf):
    """
    Convert the GeoDataFrame containing the trajectories into a TrajDataFrame
    
    Parameters
    ----------
    gdf : GeoDataFrame
        the GeoDataFrame containing the trajectories
        
    Returns
    -------
    TrajDataFrame
        the TrajDataFrame containing the trajectories
    """
    def get_points(row):
        """
        Create a DataFrame from each user and trajectory identifier, 
        extracting latitude, longitude and datetime from each trajectory

        Parameters
        ----------
        row: pandas DataFrame
            the DataFrame describing a trajectory for a user

        Returns
        -------
        pandas DataFrame
            a DataFrame containing latitude (lat), longitude (lng) and datetime
        """
        rows = []
        for point in row.traj.iloc[0].coords:
            lat, lon = point[1], point[0]
            time = pd.to_datetime(point[2] * 1000, unit='s') # time is UTC time divided by 1000
            rows.append([lat, lon, time])
        return pd.DataFrame(rows, columns=['lat', 'lng', 'datetime'])

    dis_gdf = gdf.groupby(['uid','tid']).apply(lambda row: get_points(row)).reset_index().drop('level_2', axis=1).astype({'tid': 'int32'})
    return skmob.TrajDataFrame(dis_gdf)


def download_trajectories(config_file='config.json', area='uk', uid_limit=100, 
                          start_date='2017-02-02', end_date='2017-03-02'):
    """
    Download the trajectories from the server
    
    Parameters
    ----------
    config_file : str
        the name of the configuration file. The default is 'config.json'.
        
    area : str
        the area from which to download the data. Possible values are "uk" and "italy".
        The default is "uk".
    
    uid_limit : int or None
        the number of users whose trajectories to download. If None, download data about 
        all the users. The default is 100.
        
    Returns
    -------
    TrajDataFrame
        the TrajDataFrame containing the trajectories
    """
    # load the json configuration file
    config = json.load(open('config.json'))
    # execute the query
    con = psycopg2.connect(database=config['database'], 
                           user=config['user'], 
                           password=config['password'],
                           host=config['host'], 
                           port=config['port'])
    table = config['table_uk'] if area == 'uk' else config['table_italy']
    if uid_limit is None:
        sql = "select * from %s where start_time >= '%s' and end_time <= '%s';" %(table, 
                                                                              start_date, 
                                                                             end_date)
    else:
        sql = "select * from %s where start_time >= '%s' and end_time <= '%s' and uid in (select distinct uid from %s limit %s);" %(table, start_date, end_date, table, uid_limit)

    return to_TrajDataFrame(gpd.GeoDataFrame.from_postgis(sql, con, geom_col='traj'))

def download_square_tessellation(cell_size=1500, region='Greater London'):
    """
    Download a square spatial tessellation of the `region` indicated in input.
    Each cell in the spatial tesselation has side of dimension `cell_size`.
    
    Parameters
    ----------
    cell_size : int
        the size of the side of each cell. Default: 1500
        
    region : str
        the name of the region. Default: 'Greater London'
        
    Returns
    -------
    GeoDataFrame
        a GeoDataFrame describing the spatial tessellation
    """
    # Build a tessellation over the city
    return tilers.tiler.get("squared", base_shape=region, meters=cell_size)

def select_trajectories_within_tessellation(tdf, tessellation):
    """
    Select only the trajectories that fall entirely within the spatial tessellation.
    
    Parameters
    ----------
    tdf : TrajDataFrame
        the TrajDataFrame the describe the trajectories
        
    tessellation : GeoDataFrame
        the GeoDataFrame describing the spatial tessellation
        
    Returns
    -------
    TrajDataFrame
        a TrajDataFrame that contains all and only the trajectories that fall
        entirely within the spatial tessellation
    """
    # map each point to the corresponding tile in the tessellation
    tdf_mapped = tdf.mapping(tessellation, remove_na=True)
    
    def check_nan(tdf):
        """
        Check if there is a NaN in a trajectory's TrajDataFrame. 

        Parameters
        ----------
        uid_tid_tdf: TrajDataFrame
            contains info about a user's trajectory

        Returns
        -------
        TrajDataFrame
            the a user's trajectory
        """
        if not tdf['tile_ID'].isna().values.any():
            return tdf
        else:
            tdf['tile_ID'] = -1
            return tdf
        
    if 'tid' in tdf_mapped.columns:
        grouped_tdf = tdf_mapped.groupby(['uid', 'tid']).apply(lambda uid_tid_tdf: check_nan(uid_tid_tdf))
    else:
        grouped_tdf = tdf_mapped.groupby('uid').apply(lambda uid_tdf: check_nan(uid_tdf))
    return grouped_tdf[grouped_tdf['tile_ID'] != -1]

def temporal_aggregation(tdf, n_minutes_in_slot=15):
    """
    Aggregate the trajectories using a temporal slot of duration `n_minutes_in_slot`.
    
    Parameters
    ----------
    tdf : TrajDataFrame
        the TrajDataFrame that describes the trajectories
        
    n_minutes_in_slot : int
        size of the time slot (in minutes). Default: 15
        
    Returns
    -------
    TrajDataFrame
        the trajectory data frame aggregated
    """
    freq = '%s%s' %(n_minutes_in_slot, 'min')

    def aggregation(uid_tdf, freq=freq):
        return uid_tdf.sort_values(by='datetime').set_index('datetime').groupby(pd.Grouper(freq=freq, label='right')).agg(
        {'lat': 'mean', 
        'lng':'mean'}).fillna(method='ffill')

    return skmob.TrajDataFrame(tdf.groupby('uid').apply(lambda uid_tdf: aggregation(uid_tdf)).reset_index())

def filter_by_n_weeks(aggr_tdf, n_minutes_in_slot=15, min_weeks=4):
    """
    Filter out people with less than `min_weeks` weeks of data.
    
    Parameters
    ----------
    aggr_tdf : TrajDataFrame
        the TrajDataFrame describing the aggregated trajectories
        
    n_minutes_in_slot : int
        size of the time slot (in minutes). Default: 15
        
    min_weeks : int
        the minimum number of weeks required for each user. Default: 4
        
    Returns
    -------
    TrajDataFrame
        the TrajDataFrame without the users with less than `min_weeks` weeks of data
    """
    # Filter out people with too short mobility (less than one week of data)
    MINUTES_IN_DAY, DAYS_IN_WEEK, WEEKS_IN_MONTH = 1440, 7, 4
    MIN_SLOTS = MINUTES_IN_DAY/n_minutes_in_slot * DAYS_IN_WEEK * min_weeks 
    filt_aggr_tdf = aggr_tdf.groupby('uid').filter(lambda uid_aggr_tdf: 
                                               len(uid_aggr_tdf) >= MIN_SLOTS)
    return filt_aggr_tdf

def fill_first_and_last_days(tdf, n_minutes_in_slot=15):
    """
    The first and last days of a user may have missing time slots. This function
    fills them.
    
    aggr_tdf : TrajDataFrame
        the TrajDataFrame describing the aggregated trajectories
        
    n_minutes_in_slot : int
        size of the time slot (in minutes). Default: 15
        
    Returns
    -------
    TrajDataFrame
        the TrajDataFrame with the first and last days filled
    """
    freq = '%s%s' %(n_minutes_in_slot, 'min')
    ## Fill the first and the last day of each individual
    def fill_trajs(uid_tdf):
        min_slot, max_slot = uid_tdf.datetime.min(), uid_tdf.datetime.max()
        
        # check if it already exists
        new_min_slot = min_slot.replace(hour=0, minute=0, second=0)

        new_max_slot = max_slot.replace(hour=23, minute=60 - n_minutes_in_slot, second=0)
        r_min = pd.date_range(freq=freq, start=new_min_slot, end=max_slot)
        uid_tdf = uid_tdf.set_index('datetime').reindex(r_min).fillna(method='bfill')
        r_max = pd.date_range(freq=freq, start=new_min_slot, end=new_max_slot)
        uid_tdf = uid_tdf.reindex(r_max).fillna(method='ffill')
        return uid_tdf.reset_index().rename(columns={'index': 'datetime'})

    return tdf.groupby('uid').apply(lambda uid_tdf: fill_trajs(uid_tdf)).reset_index(drop=True)

def split_trajectories(filled_aggr_tdf, n_minutes_in_slot=15, n_weeks=4):
    """
    Split the trajectories by week in groups of `n_weeks`.
    
    filled_aggr_tdf : TrajDataFrame
        the TrajDataFrame describing the aggregated trajectories
        
    n_minutes_in_slot : int
        size of the time slot (in minutes). Default: 15
        
    n_weeks : int
        groups of weeks. Default: 4
        
    Returns
    -------
    TrajDataFrame
    """
    day_of_weeks = filled_aggr_tdf.datetime.dt.dayofweek + 1
    week_of_year = filled_aggr_tdf.datetime.dt.weekofyear
    filled_aggr_tdf['day_of_week'] = day_of_weeks
    filled_aggr_tdf['week_of_year'] = week_of_year
    
    def find_subtrajs(uid_df):
        MINUTES_IN_DAY, DAYS_IN_WEEK = 60 * 24, 7

        # eliminate incomplete weeks
        slots_in_week = int(MINUTES_IN_DAY/n_minutes_in_slot * DAYS_IN_WEEK)
        slots_in_traj = n_weeks * slots_in_week
        filt_uid_df = uid_df.groupby('week_of_year').filter(lambda dw_df: len(dw_df) == slots_in_week)

        # generate tids
        user_n_weeks = len(filt_uid_df.week_of_year.unique())
        # number of sub-trajectory to split the history of the user in
        user_n_trajs = user_n_weeks // n_weeks 
        slots_to_take = user_n_trajs * n_weeks * slots_in_week
        filt_uid_df = filt_uid_df[:slots_to_take]
        tids = [tid + 1 for tid in range(user_n_trajs) for _ in range(slots_in_traj)]
        filt_uid_df['tid'] = tids
        filt_uid_df['uid'] = filt_uid_df.apply(lambda row: '%s_%s' %(row.uid, row.tid), axis=1)
        return filt_uid_df

    filled_aggr_tdf['tid'] = 0
    filled_aggr_tdf['uid_tid'] = 0
    w_filled_aggr_tdf = filled_aggr_tdf.groupby('uid').apply(lambda uid_df: find_subtrajs(uid_df)).reset_index(drop=True)
    return w_filled_aggr_tdf.drop(['uid_tid', 'tid', 'day_of_week', 'week_of_year'], axis=1)

def convert_coords(tdf, tessellation):
    """
    Convert real latitude and longitude of each point to the latitude and longitude
    corresponding to the centroid of each tile.
    
    Parameters
    ----------
    tdf : TrajDataFrame
    
    tessellation : GeoDataFrame
    
    Returns
    -------
    TrajDataFrame
    """
    def get_coordinate(row, c):    
        tile_id = row['tile_ID']
        coordinate = 0  

        if c == 'lat':
            coordinate = 0
        elif c == 'lng':
            coordinate = 1    
        return lats_lngs[int(tile_id)][coordinate]
    
    lats_lngs = tessellation.geometry.apply(skmob.utils.utils.get_geom_centroid, args=[True]).values
    tdf['lat'] = tdf.apply(lambda row: get_coordinate(row, 'lat'),axis=1)
    tdf['lng'] = tdf.apply(lambda row: get_coordinate(row, 'lng'),axis=1)
    
    return tdf
