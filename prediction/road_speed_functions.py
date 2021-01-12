# Geo-spatial data
from scipy.spatial import cKDTree

# Classic
import numpy as np
import pandas as pd
import itertools
import time
from operator import itemgetter


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


# Assign speed to a certain road
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

def fill_speeds(df):
    start_time = time.time()
    # Uber data : Quarterly Speeds Statistics by Hour of Day
    quarter_1_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q1.csv')
    quarter_2_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q2.csv')
    quarter_3_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q3.csv')
    quarter_4_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q4.csv')

    quarter_1_2019 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2019-Q1.csv')
    quarter_2_2019 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2019-Q2.csv')

    for i, x in df.iterrows():
        # print(i)
        m = x.datetime.month
        h = x.datetime.hour

        if m == 1 or m == 2 or m == 3:
            df.at[i, 'quarter'] = 1
            speed_quarter_1_2018 = quarter_1_2018[
                (quarter_1_2018['hour_of_day'] == h) & (quarter_1_2018['year'] == 2018)]

            speed_quarter_1_2019 = quarter_1_2019[
                (quarter_1_2019['hour_of_day'] == h) & (quarter_1_2019['year'] == 2019)]

            mean_kph_2019 = speed_quarter_1_2019['speed_kph_mean'].mean()
            mean_kph_2018 = speed_quarter_1_2018['speed_kph_mean'].mean()

            std_kph_2019 = speed_quarter_1_2019['speed_kph_stddev'].mean()
            std_kph_2018 = speed_quarter_1_2018['speed_kph_stddev'].mean()

            mean_kph = (mean_kph_2018 + mean_kph_2019) / 2
            std_kph = (std_kph_2019 + std_kph_2018) / 2

            low_speed_bias = std_kph
            low_speed = mean_kph - std_kph - low_speed_bias
            high_speed = mean_kph + std_kph
            random_speed = np.random.uniform(low_speed, high_speed)

            df.at[i, 'speed_kph_mean'] = random_speed

        elif m == 4 or m == 5 or m == 6:
            df.at[i, 'quarter'] = 2
            speed_quarter_2_2018 = quarter_2_2018[
                (quarter_2_2018['hour_of_day'] == h) & (quarter_2_2018['year'] == 2018)]

            speed_quarter_2_2019 = quarter_2_2019[
                (quarter_2_2019['hour_of_day'] == h) & (quarter_2_2019['year'] == 2019)]

            mean_kph_2019 = speed_quarter_2_2019['speed_kph_mean'].mean()
            mean_kph_2018 = speed_quarter_2_2018['speed_kph_mean'].mean()

            std_kph_2019 = speed_quarter_2_2019['speed_kph_stddev'].mean()
            std_kph_2018 = speed_quarter_2_2018['speed_kph_stddev'].mean()

            mean_kph = (mean_kph_2018 + mean_kph_2019) / 2
            std_kph = (std_kph_2019 + std_kph_2018) / 2

            low_speed_bias = std_kph
            low_speed = mean_kph - std_kph - low_speed_bias
            high_speed = mean_kph + std_kph
            random_speed = np.random.uniform(low_speed, high_speed)

            df.at[i, 'speed_kph_mean'] = random_speed

        elif m == 7 or m == 8 or m == 9:
            df.at[i, 'quarter'] = 3
            speed_quarter_3_2018 = quarter_3_2018[
                (quarter_3_2018['hour_of_day'] == h) & (quarter_3_2018['year'] == 2018)]

            mean_kph = speed_quarter_3_2018['speed_kph_mean'].mean()
            std_kph = speed_quarter_3_2018['speed_kph_stddev'].mean()

            low_speed_bias = std_kph
            low_speed = mean_kph - std_kph - low_speed_bias
            high_speed = mean_kph + std_kph
            random_speed = np.random.uniform(low_speed, high_speed)

            df.at[i, 'speed_kph_mean'] = random_speed

        elif m == 10 or m == 11 or m == 12:
            df.at[i, 'quarter'] = 4
            speed_quarter_4_2018 = quarter_4_2018[
                (quarter_4_2018['hour_of_day'] == h) & (quarter_4_2018['year'] == 2018)]

            mean_kph = speed_quarter_4_2018['speed_kph_mean'].mean()
            std_kph = speed_quarter_4_2018['speed_kph_stddev'].mean()

            low_speed_bias = std_kph
            low_speed = mean_kph - std_kph - low_speed_bias
            high_speed = mean_kph + std_kph
            random_speed = np.random.uniform(low_speed, high_speed)

            df.at[i, 'speed_kph_mean'] = random_speed

    print("--- %s seconds for adding speeds ---" % (time.time() - start_time))
