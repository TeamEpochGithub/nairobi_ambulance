# Parallelization
import dask.dataframe as dd
from dask.dataframe import from_pandas
import multiprocessing as mp

# Geo-spatial data
import geopandas as gpd
import shapely
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import scipy
from scipy.spatial import cKDTree

# Classic
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import math
import itertools
from operator import itemgetter

# I/O
import glob

'''
Data cleaning of each speed data to reduce size of the file
'''
# Load the speed .csv data
# nairobi_speed = gpd.read_file('DataJupyter/uber_data/movement-speeds-hourly-nairobi-2018-1.csv')
# speed_processed = nairobi_speed.drop(['geometry', 'segment_id', 'start_junction_id', 'end_junction_id'], axis=1);
# speed_processed.to_csv("nairobi_speed_january_2018.csv")

'''
Loading the speed data
'''
# January - March 2018
nairobi_speed_january_2018 = pd.read_file('nairobi_speed_january_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_february_2018 = gpd.read_file('nairobi_speed_february_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_march_2018 = gpd.read_file('nairobi_speed_march_2018.csv', parse_dates=["utc_timestamp"])

# April - June 2018
nairobi_speed_april_2018 = gpd.read_file('nairobi_speed_april_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_may_2018 = gpd.read_file('nairobi_speed_may_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_june_2018 = gpd.read_file('nairobi_speed_june_2018.csv', parse_dates=["utc_timestamp"])

# July - September 2018
nairobi_speed_july_2018 = gpd.read_file('nairobi_speed_july_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_august_2018 = gpd.read_file('nairobi_speed_august_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_september_2018 = gpd.read_file('nairobi_speed_september_2018.csv', parse_dates=["utc_timestamp"])

# September - December 2018
nairobi_speed_october_2018 = gpd.read_file('nairobi_speed_october_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_november_2018 = gpd.read_file('nairobi_speed_november_2018.csv', parse_dates=["utc_timestamp"])
nairobi_speed_december_2018 = gpd.read_file('nairobi_speed_december_2018.csv', parse_dates=["utc_timestamp"])

############################# 2019 #############################

# January - March 2019
nairobi_speed_january_2019 = gpd.read_file('nairobi_speed_april_2019.csv', parse_dates=["utc_timestamp"])
nairobi_speed_february_2019 = gpd.read_file('nairobi_speed_february_2019.csv', parse_dates=["utc_timestamp"])
nairobi_speed_march_2019 = gpd.read_file('nairobi_speed_march_2019.csv', parse_dates=["utc_timestamp"])

# April - June 2019
nairobi_speed_april_2019 = gpd.read_file('nairobi_speed_april_2019.csv', parse_dates=["utc_timestamp"])
nairobi_speed_may_2019 = gpd.read_file('nairobi_speed_may_2019.csv', parse_dates=["utc_timestamp"])
nairobi_speed_june_2019 = gpd.read_file('nairobi_speed_june_2019.csv', parse_dates=["utc_timestamp"])

# July 2019
nairobi_speed_july_2019 = gpd.read_file('nairobi_speed_july_2019.csv', parse_dates=["utc_timestamp"])

# Creating one big speed dataframe
speed_data = [nairobi_speed_january_2018, nairobi_speed_february_2018, nairobi_speed_march_2018,
              nairobi_speed_april_2018, nairobi_speed_may_2018, nairobi_speed_june_2018, nairobi_speed_july_2018,
              nairobi_speed_august_2018, nairobi_speed_september_2018, nairobi_speed_october_2018,
              nairobi_speed_november_2018, nairobi_speed_december_2018, nairobi_speed_january_2019,
              nairobi_speed_february_2019, nairobi_speed_march_2019, nairobi_speed_april_2019,
              nairobi_speed_may_2019, nairobi_speed_june_2019, nairobi_speed_july_2019]  # List of speed dataframes

geo_speed_data = gpd.GeoDataFrame(pd.concat(speed_data, ignore_index=True))

'''
Ubre data : Quarterly Speeds Statistics by Hour of Day
'''
quarter_1_2018 = pd.read_csv('uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q1.csv')
quarter_2_2018 = pd.read_csv('uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q2.csv')
quarter_3_2018 = pd.read_csv('uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q3.csv')
quarter_4_2018 = pd.read_csv('uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q4.csv')

quarter_1_2019 = pd.read_csv('uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2019-Q1.csv')
quarter_2_2019 = pd.read_csv('uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2019-Q2.csv')

# Creating one big speed dataframe
quarters_prepare = [quarter_1_2018, quarter_2_2018, quarter_3_2018, quarter_4_2018, quarter_1_2019, quarter_2_2019]

quaterly = pd.concat(quarters_prepare)
quaterly = quaterly.sort_values(by=['hour_of_day', 'year', 'quarter'], ascending=True)

# Road data
nairobi_speed_only_visualization = gpd.read_file('nairobi_speed_only_visualization.geojson')
road_2019 = gpd.read_file("updated_road_2019.geojson")

'''
Loading data from Zindi
'''

# Training data
train = pd.read_csv('nairobi_data/data_zindi/Train.csv', parse_dates=['datetime'])

# Weather data
weather_data = pd.read_csv('nairobi_data/data_zindi/Weather_Nairobi_Daily_GFS.csv', parse_dates=['Date'])

# Segment survey data
segment_info = pd.read_csv('nairobi_data/data_zindi/Segment_info.csv')

# Segment geometry
segment_geometry = gpd.read_file('nairobi_data/data_zindi/segments_geometry.geojson')

#################################################################################################
####################################### Data Processing #########################################
#################################################################################################
####################################### NOT FINISHED ############################################
#################################################################################################

'''
Functions needed for operations and processing
'''


# Delete rows with NaN
def delete_rows_with_nan(dataframe):
    is_NaN = dataframe.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = dataframe[row_has_NaN]

    for index, row in rows_with_NaN.iterrows():
        dataframe.drop(index, inplace=True)


# Computes the distance between two points based on their coordinates
# The formula : https://en.wikipedia.org/wiki/Great-circle_distance
def haversine(coord1, coord2):
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1[0]
    lon2, lat2 = coord2[0]
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    # km = meters / 1000.0  # output distance in kilometers

    meters = round(meters)
    # km = round(km, 3)
    # print(f"Distance: {meters} m")
    # print(f"Distance: {km} km")
    return meters


# CKD-Tree to compute nearest road to all the points in Train.csv
def ckdtree_nearest_road_to_point(gdfA, gdfB, gdfB_cols=['road_id']):
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdfA.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]

    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)

    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


# Processing segment_info
segment_info = segment_info.append(pd.Series([-1], index=segment_info.columns[:len([-1])]), ignore_index=True)
segment_info = segment_info.fillna(0)
segment_info = segment_info.groupby(['segment_id']).sum().reset_index()

# Processing weather data
delete_rows_with_nan(weather_data)

# Creating a column with only date, and transforming it into Datetime object
train['date'] = pd.to_datetime([d.date() for d in train.datetime])

# Merge train with weather
train_with_weather = train.merge(weather_data, how='left', left_on='date', right_on='Date')
train_with_weather.drop('date', axis=1, inplace=True)
train_with_weather.drop('Date', axis=1, inplace=True)
train_with_weather.drop('uid', axis=1, inplace=True)

# Processing the road data
road_processed = nairobi_speed_only_visualization.drop(['speed_mean_kph', 'pct_from_freeflow',
                                                        'speed_freeflow_kph'], axis=1)
road_processed['road_id'] = nairobi_speed_only_visualization.index
road_2019['road_id'] = road_2019.index

# Creating a (geometry) point column and add it to the Train.csv
points = [Point(xy) for xy in zip(train_with_weather['longitude'], train_with_weather['latitude'])]

# Converting to GeoDataframe in order to have Points object as geometry dtype containing points
geo_points = GeoDataFrame(train_with_weather, crs="EPSG:4326", geometry=points)
geo_points['segment_id'] = None

'''
Closest point to the segment_geometry 
'''
for i, point in geo_points.iterrows():

    smallest_distance = sys.maxsize
    smallest_index = -1
    for j, road in segment_geometry.iterrows():

        road_geometry = road['geometry']
        nearest_road_point = nearest_points(road_geometry, point['geometry'])[0]

        current_smallest_distance = haversine(nearest_road_point.coords[:], point['geometry'].coords[:])
        if current_smallest_distance < smallest_distance:
            smallest_distance = current_smallest_distance
            smallest_index = j

    if smallest_distance < 150 and smallest_index != -1:
        geo_points.at[i, 'segment_id'] = segment_geometry.at[smallest_index, 'segment_id']
    else:
        geo_points.at[i, 'segment_id'] = -1

geo_points = geo_points.merge(segment_info, how='left', left_on='segment_id', right_on='segment_id')
geo_points.drop('segment_id', axis=1, inplace=True)
geo_points.drop('geometry', axis=1, inplace=True)

pandas_df_from_geo = pd.DataFrame(geo_points, copy=True)
pandas_df_from_geo.to_csv('data_without_speed_distance_*certain distance threshold*.csv')

'''
Assigning points to the particular road in particular time if the distance threshold is met.
Converted lat and lon to Point geo-object and using .distance() to measure distance between the road and
the point.
'''
nearest_road_to_point_any_distance = ckdtree_nearest_road_to_point(geo_points, road_2019)
data_with_road_distance_threshold = nearest_road_to_point_any_distance[
    nearest_road_to_point_any_distance["dist"] <= 0.03]

# Data with point connected to osm road
point_road_merged = data_with_road_distance_threshold.merge(road_processed, how='left', on='road_id')
point_road_merged.drop(['geometry_x', 'road_id', 'dist', 'geometry_y'], axis=1, inplace=True)
point_road_merged = pd.DataFrame(point_road_merged)

'''
Assigning GeoSeries (in this case LineString) to the speed data.
Iterate through both dataframes and if osm_way_id, osm_start_node_id and osm_end_node_id are matched,
assign to that row the road data.
'''
nairobi_speed_january_2018 = pd.read_csv('nairobi_speed_january_2018.csv', parse_dates=["utc_timestamp"])

nairobi_speed_january_2018['utc_timestamp'] = pd.to_datetime(nairobi_speed_january_2018['utc_timestamp'])
nairobi_speed_january_2018['end_timestamp'] = pd.to_datetime(nairobi_speed_january_2018
                                                             .year * 10000 +
                                                             nairobi_speed_january_2018
                                                             .month * 100 + nairobi_speed_january_2018
                                                             .day,
                                                             format='%Y%m%d') + nairobi_speed_january_2018.hour.astype(
    'timedelta64[h]')

nairobi_speed_january_2018['utc_timestamp'] = pd.to_datetime(nairobi_speed_january_2018['utc_timestamp'])
nairobi_speed_january_2018['end_timestamp'] = pd.to_datetime(nairobi_speed_january_2018['end_timestamp'])
train['datetime'] = pd.to_datetime(train['datetime'])
point_road_merged['datetime'] = pd.to_datetime(point_road_merged['datetime'])

nairobi_speed_january_2018['utc_timestamp'] = nairobi_speed_january_2018['utc_timestamp'].dt.tz_localize(None)
nairobi_speed_january_2018['end_timestamp'] = nairobi_speed_january_2018['end_timestamp'].dt.tz_localize(None)
train['datetime'] = train['datetime'].dt.tz_localize(None)
point_road_merged['datetime'] = point_road_merged['datetime'].dt.tz_localize(None)

point_road_merged['quarter'] = 0
for i, x in point_road_merged.iterrows():

    m = x.datetime.month

    if m == 1 or m == 2 or m == 3:
        point_road_merged.at[i, 'quarter'] = 1
    elif m == 4 or m == 5 or m == 6:
        point_road_merged.at[i, 'quarter'] = 2
    elif m == 7 or m == 8 or m == 9:
        point_road_merged.at[i, 'quarter'] = 3
    elif m == 10 or m == 11 or m == 12:
        point_road_merged.at[i, 'quarter'] = 4


def binary_search_speed(df, current_point):
    particular_osm = df[(df['osm_way_id'] == current_point.osmwayid) &
                        (df['quarter'] == current_point.quarter) &
                        (df['year'] == current_point.datetime.year)]
    start, end = 0, (len(particular_osm) - 1)
    while start <= end:
        mid = (start + end) // 2
        speed = particular_osm.iloc[mid]

        if speed.hour_of_day == current_point.datetime.hour:
            return pd.Series([speed.speed_kph_mean, speed.speed_kph_stddev,
                              speed.speed_kph_p50, speed.speed_kph_p85])

        if current_point.datetime.hour < speed.hour_of_day:
            end = mid - 1
        else:
            start = mid + 1

    return pd.Series([0, 0, 0, 0])


point_road_merged['speed_kph_mean'] = 0
point_road_merged['speed_kph_stddev'] = 0
point_road_merged['speed_kph_p50'] = 0
point_road_merged['speed_kph_p85'] = 0

# Works but cannot return multiple columns
# dask_train = dd.from_pandas(point_road_merged, npartitions=2*multiprocessing.cpu_count())
# dask_train['speed_kph_mean'] = dask_train.map_partitions(lambda df: df.apply(lambda x: binary_search_speed(
#     quaterly, x), axis=1)).compute(scheduler='processes')

point_road_merged[['speed_kph_mean', 'speed_kph_stddev', 'speed_kph_p50', 'speed_kph_p85']] = point_road_merged.apply(
    lambda x: binary_search_speed(
        quaterly, x), axis=1)

point_road_merged.drop(['quarter'], axis=1, inplace=True)
point_road_merged.to_csv('speed_data_dist_0.03_ckdtree.csv')
