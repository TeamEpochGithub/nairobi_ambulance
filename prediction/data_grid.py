# Classic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Geo-spatial libraries
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point

# .py scripts
from time_aux_functions import roundDateTime3h, gen_datetime
from road_speed_functions import ckdtree_nearest_road_to_point,  binary_search_speed

# Classifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance, XGBClassifier

# Load the training data
dfCrashes = pd.read_csv('nairobi_data/data_zindi/Train.csv', parse_dates=['datetime'])

# Load the road data
# road_2019 = gpd.read_file("nairobi_data/updated_road_2019.geojson")
# road_2019['road_id'] = road_2019.index


#Uber data : Quarterly Speeds Statistics by Hour of Day
quarter_1_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q1.csv')
quarter_2_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q2.csv')
quarter_3_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q3.csv')
quarter_4_2018 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2018-Q4.csv')

quarter_1_2019 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2019-Q1.csv')
quarter_2_2019 = pd.read_csv('nairobi_data/uber_quaterly/movement-speeds-quarterly-by-hod-nairobi-2019-Q2.csv')

# Creating one big speed dataframe
# quarters_prepare = [quarter_1_2018, quarter_2_2018, quarter_3_2018, quarter_4_2018, quarter_1_2019, quarter_2_2019]
#
# quaterly = pd.concat(quarters_prepare)
# quaterly = quaterly.sort_values(by=['hour_of_day', 'year', 'quarter'], ascending=True)

# Remove the useless columns
dfCrashes = dfCrashes.drop(columns='uid')

# Round to 1 decimal
dfCrashes = dfCrashes.round(1)

# Round the datetimes to the nearest 3h span
dfCrashes['datetime'] = dfCrashes['datetime'].apply(roundDateTime3h)

# Count the number of crashes in each cell
dfCrashes = dfCrashes.groupby(dfCrashes.columns.tolist(), as_index=False).size()
dfCrashes.rename(columns={'size': 'n_crashes'}, inplace=True)

# Generate 10000 random dates
n_negative_samples = 10000
dates = []
for i in range(n_negative_samples):
    dates.append(gen_datetime(min_year=2018, max_year=2019))

# Generate 6000 random latitudes and longitudes
randLat = np.random.uniform(-0.5, -3.1, n_negative_samples)
randLong = np.random.uniform(36, 38, n_negative_samples)
no_crashes = np.zeros(n_negative_samples)

dfNegativeSamples = pd.DataFrame(columns=['datetime', 'latitude', 'longitude', 'n_crashes'])
dfNegativeSamples['datetime'] = dates
dfNegativeSamples['latitude'] = randLat
dfNegativeSamples['longitude'] = randLong
dfNegativeSamples['n_crashes'] = 0

# Round to 1 decimal
dfNegativeSamples = dfNegativeSamples.round(1)

dfNegativeSamples['quarter'] = 0
dfNegativeSamples['speed_kph_mean'] = 0.0

for i, x in dfNegativeSamples.iterrows():
    print(i)
    m = x.datetime.month
    h = x.datetime.hour

    if m == 1 or m == 2 or m == 3:
        dfNegativeSamples.at[i, 'quarter'] = 1
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

        dfNegativeSamples.at[i, 'speed_kph_mean'] = random_speed

    elif m == 4 or m == 5 or m == 6:
        dfNegativeSamples.at[i, 'quarter'] = 2
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

        dfNegativeSamples.at[i, 'speed_kph_mean'] = random_speed

    elif m == 7 or m == 8 or m == 9:
        dfNegativeSamples.at[i, 'quarter'] = 3
        speed_quarter_3_2018 = quarter_3_2018[
            (quarter_3_2018['hour_of_day'] == h) & (quarter_3_2018['year'] == 2018)]

        mean_kph = speed_quarter_3_2018['speed_kph_mean'].mean()
        std_kph = speed_quarter_3_2018['speed_kph_stddev'].mean()

        low_speed_bias = std_kph
        low_speed = mean_kph - std_kph - low_speed_bias
        high_speed = mean_kph + std_kph
        random_speed = np.random.uniform(low_speed, high_speed)

        dfNegativeSamples.at[i, 'speed_kph_mean'] = random_speed

    elif m == 10 or m == 11 or m == 12:
        dfNegativeSamples.at[i, 'quarter'] = 4
        speed_quarter_4_2018 = quarter_4_2018[
            (quarter_4_2018['hour_of_day'] == h) & (quarter_4_2018['year'] == 2018)]

        mean_kph = speed_quarter_4_2018['speed_kph_mean'].mean()
        std_kph = speed_quarter_4_2018['speed_kph_stddev'].mean()

        low_speed_bias = std_kph
        low_speed = mean_kph - std_kph - low_speed_bias
        high_speed = mean_kph + std_kph
        random_speed = np.random.uniform(low_speed, high_speed)

        dfNegativeSamples.at[i, 'speed_kph_mean'] = random_speed

dfNegativeSamples = dfNegativeSamples.drop(columns=['quarter'])
print(dfNegativeSamples)

# # Load the Speed data
dfSpeed = pd.read_csv('final_data/speed_data/speed_data_dist_0.1_ckdtree.csv', parse_dates=['datetime'], index_col=0)

# # Remove columns
dfSpeed = dfSpeed.drop(columns=['uid', 'osmstartnodeid', 'osmhighway',
                                'osmendnodeid', 'osmwayid', 'osmname', 'dist', 'quarter', 'speed_kph_stddev',
                                'speed_kph_p50', 'speed_kph_p85'])
#
# # Round to 1 decimal
dfSpeed['latitude'] = dfSpeed['latitude'].round(1)
dfSpeed['longitude'] = dfSpeed['longitude'].round(1)
# # Round the datetimes to the nearest 3h span
dfSpeed['datetime'] = dfSpeed['datetime'].apply(roundDateTime3h)
#
# # # ToDo: Compute average if two rows have the same lat,long and datetime
#
dfCrashes = dfCrashes.merge(dfSpeed, how='left', left_on=['datetime', 'latitude', 'longitude'],
                             right_on=['datetime', 'latitude', 'longitude'])

# Round the datetimes to the nearest 3h span
dfNegativeSamples['datetime'] = dfNegativeSamples['datetime'].apply(roundDateTime3h)

# # Add negative samples
dfCrashes = dfCrashes.append(dfNegativeSamples)
dfCrashes = dfCrashes.drop_duplicates(subset=['datetime', 'latitude', 'longitude'])

# Add date column for merging
dfCrashes['date'] = pd.to_datetime([d.date() for d in dfCrashes.datetime])
dfWeather = pd.read_csv('nairobi_data/data_zindi/Weather_Nairobi_Daily_GFS.csv', parse_dates=['Date'])

dfCrashes = dfCrashes.merge(dfWeather, how='left', left_on='date', right_on='Date')
dfCrashes = dfCrashes.drop(columns=['date', 'Date'])
#
# # ToDo: Check why there are NaN. All the rows should have speed
dfCrashes = dfCrashes.fillna(0)
#
# Change datetime for Hour, Day of the week and month
dfCrashes['dayofweek'] = dfCrashes['datetime'].dt.dayofweek
dfCrashes['month'] = dfCrashes['datetime'].dt.month
dfCrashes['hour'] = dfCrashes['datetime'].dt.hour
dfCrashes = dfCrashes.drop(columns='datetime')

# XGBoost
X = dfCrashes.drop(columns='n_crashes')
y = dfCrashes['n_crashes']
y.loc[y >= 1] = 1
y = y.astype(int)
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit model training data
model = XGBClassifier(n_estimators=100, max_depth=8, use_label_encoder=False, objective='binary:logistic')
model.fit(X_train, y_train)
# Make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

plot_importance(model)
print(plt.show())

# ----------------------- Submission file -----------------------
# submission = pd.read_csv('nairobi_data/data_zindi/SampleSubmission.csv', parse_dates=['date'])
#
# submission = submission['date']
#
# lat = np.arange(-3.1, -0.4, 0.1)
# long = np.arange(36, 38, 0.1)
# grid = np.zeros((len(lat) * len(long), 2))
#
# for i in range(len(lat)):
#     for j in range(len(long)):
#         grid[i * len(long) + j, 0] = lat[i]
#         grid[i * len(long) + j, 1] = long[j]
#
# for date in submission.values:
#     df = pd.DataFrame(grid, columns=['latitude', 'longitude'])
#     df['datetime'] = date
#
#     df = df.merge(dfWeather, how='left', left_on='datetime', right_on='Date')
#     df = df.drop(columns='Date')
#
#     # df = df.merge(dfSpeed, how='left', left_on=['datetime', 'latitude', 'longitude'],
#     #                             right_on=['datetime', 'latitude', 'longitude'])
#     #
#     # # ToDo: Check why there are NaN. All the rows should have speed
#     # df = df.fillna(0)
#
#     # Change datetime for Hour, Day of the week and month
#     df['dayofweek'] = df['datetime'].dt.dayofweek
#     df['month'] = df['datetime'].dt.month
#     df['hour'] = df['datetime'].dt.hour
#     df = df.drop(columns='datetime')
#
#     y_pred = model.predict(df)
#
#     auxDf = df.loc[y_pred == 1]
#     crashes = auxDf[['latitude', 'longitude']].values
#     print(y_pred)
