# Parallelization
import dask.dataframe as dd

# Geo-spatial data
import geopandas as gpd
import math
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

# Classic
import numpy as np
import pandas as pd
import sys
# import matplotlib.pyplot as plt

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
nairobi_speed_january_2018 = gpd.read_file('nairobi_speed_january_2018.csv', parse_dates=["utc_timestamp"])
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
Dask parallelization
Dask Geopandas : https://github.com/jsignell/dask-geopandas (still experimental)
'''
# Speed data with parallelization
# df_speed = dd.read_csv("nairobi_speed_*_*.csv", parse_dates=["utc_timestamp"])

# Road data
nairobi_speed_only_visualization = gpd.read_file('nairobi_speed_only_visualization.geojson')

'''
Loading data from Zindi
'''

# Training data
train = pd.read_csv('nairobi_data/data_zindi/Train.csv')

# Weather data
weather_data = pd.read_csv('nairobi_data/data_zindi/Weather_Nairobi_Daily_GFS.csv')

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
        dataframe = dataframe.drop(index)
        df = df.drop(index)


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

# Creating a (geometry) point column and add it to the Train.csv
points = [Point(xy) for xy in zip(train['longitude'], train['latitude'])]

# Converting to GeoDataframe in order to have Points object as geometry dtype containing points
geo_points = GeoDataFrame(train, crs="EPSG:4326", geometry=points)
geo_speed_data['longitude'] = np.nan
geo_speed_data['latitude'] = np.nan

# Closest point to the segment_geometry : TESTING PURPOSES
for i, point in geo_points.iterrows():

    smallest_distance = 1000000000000
    for j, road in segment_geometry.iterrows():

        current_distance = point['geometry'].distance(road['geometry'])
        if (current_distance < smallest_distance):
            smallest_distance = current_distance
    print(point.geometry, ":", smallest_distance)

'''
Assigning GeoSeries (in this case LineString) to the speed data.
Iterate through both dataframes and if osm_way_id, osm_start_node_id and osm_end_node_id are matched,
assign to that row the road data
'''
# def assign_linestring_to_speed(road, speed):

# trying instead for nested loop to do lambda expressions + Dask parallelization
# geo_speed_data['geometry'] = geo_speed_data.apply(lambda row: assign_linestring_to_speed(road, row))
for i, road in road_processed.iterrows():

    for j, speed in geo_speed_data.iterrows():

        if (road['osmwayid'] == speed['osm_way_id'] and road['osmstartnodeid'] == speed['osm_start_node_id'] and
                road['osmendnodeid'] == speed['osm_end_node_id']):
            geo_speed_data.at[j, 'geometry'] = road['geometry']
            geo_speed_data.at[j, 'osmhighway'] = road['osmhighway']

'''
Assigning points to the particular road in particular time if the distance threshold is met.
Converted lat and lon to Point geo-object and using .distance() to measure distance between the road and
the point.
'''


# Assign points to the road/speed data
def point_to_road(point, road, road_index, distance):
    if distance >= 0 and distance < 20:
        geo_speed_data.at[smallest_index_road, 'longitude'] = point['longitude']
        geo_speed_data.at[smallest_index_road, 'latitude'] = point['latitude']


# Iterate through points and roads
for i, point in geo_points.iterrows():

    smallest_distance = sys.maxint;
    smallest_index_road = -1;
    for j, road in geo_speed_data.iterrows():
        # Getting the point of the road that is nearest to the
        road_geometry = road['geometry']
        nearest_road_point = nearest_points(road_geometry, point['geometry'])[0]

        current_smallest_distance = haversine(nearest_road_point, point['geometry'])
        # if current_smallest_distance < smallest_distance:
        #      if road['utc_timestamp'].hour < point['datetime'].hour:
        #         smallest_distance = current_smallest_distance
        #     smallest_index_road = j

    point_to_road(point, road, smallest_index_road, smallest_distance)

'''
Deleting redundant rows (roads) if no points were assigned.
'''

# Drop the speed rows if it contains points that are NaN

delete_rows_with_nan(geo_speed_data)
