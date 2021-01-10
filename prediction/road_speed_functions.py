# Geo-spatial data
from scipy.spatial import cKDTree

# Classic
import numpy as np
import pandas as pd
import itertools
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
