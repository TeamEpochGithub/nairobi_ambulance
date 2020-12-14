# Parallelization
import dask.dataframe as dd

# Geo-spatial data
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString

# Classic
import numpy as np
import pandas as pd
import sys
# import matplotlib.pyplot as plt

# I/O
import glob

'''
Loading the data with geopandas in order for 'geometry' column to be created.
Second option is the load .csv as in panda and then convert it to GeoDataFrame.
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
Dask doesn't have configuration for geopandas, however this one has:
https://github.com/jsignell/dask-geopandas
It is still in the experimental stage though. 
It is a shame tho :(

Also cannot just work with Pandas and Dask because it doesn't recognize GeoSeries.
'''
# Speed data with parallelization
# df_speed = dd.read_csv("nairobi_speed_*_*.csv", parse_dates=["utc_timestamp"])

# Road data
nairobi_speed_only_visualization = gpd.read_file('nairobi_speed_only_visualization.geojson')

# Loading training data
train = pd.read_csv('DataJupyter/Train.csv')

# Processing the road data
road_processed = nairobi_speed_only_visualization.drop(['speed_mean_kph', 'pct_from_freeflow',
                                                        'speed_freeflow_kph'], axis=1)

# # Creating a (geometry) point column and add it to the Train.csv
points = [Point(xy) for xy in zip(train['longitude'], train['latitude'])]

# Converting to GeoDataframe in order to have Points object as geometry dtype containing points
geo_points = GeoDataFrame(train, crs="EPSG:4326", geometry=points)
geo_speed_data['longitude'] = np.nan
geo_speed_data['latitude'] = np.nan

#################################################################################################
####################################### Data Processing #########################################
#################################################################################################
####################################### NOT FINISHED ############################################
#################################################################################################
'''
Carefully read this : Maybe it's not good idea to do road/speed first but points/road and then (points/road)/speed.

Author : Bill Ton Hoang Nguyen
Last updated : 12/14/2020
Notes : gonna change the design of the data processing, also I am still trying to make it more efficient.
If not then we have to run it several hours.
'''

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
for i, point in train.iterrows():

    smallest_distance = sys.maxint;
    smallest_index_road = -1;
    for j, road in geo_speed_data.iterrows():
        current_distance = geo_points['geometry'].distance(road['geometry'])

        if current_distance < smallest_distance:

            if road['utc_timestamp'].hour < point['datetime'].hour:
                smallest_distance = current_distance
            smallest_index_road = j

    point_to_road(point, road, smallest_index_road, smallest_distance)

'''
Deleting redundant rows (roads) if no points were assigned.
'''


# Drop the speed rows if it contains points that are NaN
def delete_rows_with_nan(dataframe):
    is_NaN = dataframe.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = dataframe[row_has_NaN]

    for index, row in rows_with_NaN.iterrows():
        dataframe = dataframe.drop(index)
        df = df.drop(index)


delete_rows_with_nan(geo_speed_data)
