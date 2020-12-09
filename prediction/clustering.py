import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from os import path
from kneed import KneeLocator

sys.path.append(path.abspath('../optimization'))

from genetic_algorithm import GeneticAlgorithm

# Load the data
train = pd.read_csv('nairobi_data/data_zindi/Train.csv', parse_dates=['datetime'])
submission = pd.read_csv('nairobi_data/data_zindi/SampleSubmission.csv', parse_dates=['date'])
weather = pd.read_csv('nairobi_data/data_zindi/Weather_Nairobi_Daily_GFS.csv', parse_dates=['Date'])

# Split day and time for merging with weather
train['date'] = pd.to_datetime([d.date() for d in train.datetime])
submission['datetime'] = submission['date']
submission['date'] = pd.to_datetime([d.date() for d in submission.date])

# Drop the UID column
train = train.drop(columns='uid')

# Merge datasets
trainMerged = train.merge(weather, how='left', left_on='date', right_on='Date')
submissionMerged = submission.merge(weather, how='left', left_on='date', right_on='Date')

# Fill NAs
trainMerged = trainMerged.fillna(method='ffill')
submissionMerged = submissionMerged.fillna(method='ffill')

# Delete Date column and set datetime as index
trainMerged = trainMerged.drop(columns='date')
trainMerged = trainMerged.drop(columns='Date')
trainMerged = trainMerged.set_index('datetime')
submissionMerged = submissionMerged.drop(columns='date')
submissionMerged = submissionMerged.drop(columns='Date')
submissionMerged = submissionMerged.set_index('datetime')

# Add columns dayofweek and month
trainMerged['dayofweek'] = trainMerged.index.dayofweek
trainMerged['month'] = trainMerged.index.month
submissionMerged['dayofweek'] = submissionMerged.index.dayofweek
submissionMerged['month'] = submissionMerged.index.month

# Split in chunk of 3 hours
train1 = trainMerged.between_time('00:00', "02:59:59")
train2 = trainMerged.between_time('03:00', "05:59:59")
train3 = trainMerged.between_time('06:00', "08:59:59")
train4 = trainMerged.between_time('09:00', "11:59:59")
train5 = trainMerged.between_time('12:00', "14:59:59")
train6 = trainMerged.between_time('15:00', "17:59:59")
train7 = trainMerged.between_time('18:00', "20:59:59")
train8 = trainMerged.between_time('21:00', "23:59:59")

submission1 = submissionMerged.between_time('00:00', "02:59:59")
submission2 = submissionMerged.between_time('03:00', "05:59:59")
submission3 = submissionMerged.between_time('06:00', "08:59:59")
submission4 = submissionMerged.between_time('09:00', "11:59:59")
submission5 = submissionMerged.between_time('12:00', "14:59:59")
submission6 = submissionMerged.between_time('15:00', "17:59:59")
submission7 = submissionMerged.between_time('18:00', "20:59:59")
submission8 = submissionMerged.between_time('21:00', "23:59:59")

fullTrain = [train1, train2, train3, train4, train5, train6, train7, train8]
fullSubmission = [submission1, submission2, submission3, submission4, submission5, submission6, submission7,
                  submission8]
kmeansFitted = []
scalers = []
ambulancesPosition = []

# Iterate over each dataframe
for i in range(len(fullTrain)):

    # First drop datetime, we have already cluster in time
    fullTrain[i].index = range(len(fullTrain[i]))

    # Get Lat Long for each crash and remove them from the DF. We do not want to cluster the position
    crashes = fullTrain[i][['longitude', 'latitude']].values
    fullTrain[i] = fullTrain[i].drop(columns=['longitude', 'latitude'])

    # Scale the features
    scaler = StandardScaler()
    features = fullTrain[i].to_numpy(dtype=float)
    scaledFeatures = scaler.fit_transform(features)
    scalers.append(scaler)

    # Initialise Kmeans args
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

    sse = []

    # Iterate over different cluster numbers to get the best one
    kMax = 20
    for k in range(1, kMax):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaledFeatures)
        sse.append(kmeans.inertia_)

    # Plot
    # plt.style.use("fivethirtyeight")
    # plt.plot(range(1, kMax), sse)
    # plt.xticks(range(1, kMax))
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("SSE")
    # plt.show()

    # Use Knee detector to get the best number of clusters
    kl = KneeLocator(range(1, kMax), sse, curve="convex", direction="decreasing")

    # Fit the final kmeans for the knee point
    finalKmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs)
    finalKmeans.fit(scaledFeatures)
    kmeansFitted.append(finalKmeans)

    ambulancesPositionsClusters = []
    n_iter = 3000
    # For each cluster, compute the optimal ambulance position
    for nCluster in range(kl.elbow):
        points = crashes[finalKmeans.labels_ == nCluster, :]
        ga = GeneticAlgorithm(crashes, n_iter)
        result = ga.run()
        ambulancesPositionsClusters.append(result)

    ambulancesPosition.append(ambulancesPositionsClusters)

# Suppress warning for copy an slide of a dataframe
pd.options.mode.chained_assignment = None  # default='warn'

for i in range(len(fullSubmission)):
    # First drop datetime, we have already cluster in time

    auxDf = fullSubmission[i].copy()
    auxDf.index = range(len(auxDf))

    # Drop columns for ambulances positions in the aux dataframe
    auxDf = auxDf.drop(columns='A0_Latitude')
    auxDf = auxDf.drop(columns='A0_Longitude')
    auxDf = auxDf.drop(columns='A1_Latitude')
    auxDf = auxDf.drop(columns='A1_Longitude')
    auxDf = auxDf.drop(columns='A2_Latitude')
    auxDf = auxDf.drop(columns='A2_Longitude')
    auxDf = auxDf.drop(columns='A3_Latitude')
    auxDf = auxDf.drop(columns='A3_Longitude')
    auxDf = auxDf.drop(columns='A4_Latitude')
    auxDf = auxDf.drop(columns='A4_Longitude')
    auxDf = auxDf.drop(columns='A5_Latitude')
    auxDf = auxDf.drop(columns='A5_Longitude')

    # Scale features
    features = auxDf.to_numpy(dtype=float)
    scaledFeatures = scalers[i].transform(features)
    labels = kmeansFitted[i].predict(scaledFeatures)

    for nCluster in range(kmeansFitted[i].n_clusters):
        indices = labels == nCluster
        fullSubmission[i].loc[indices, ['A0_Longitude', 'A0_Latitude', 'A1_Longitude', 'A1_Latitude',
                                        'A2_Longitude', 'A2_Latitude', 'A3_Longitude', 'A3_Latitude',
                                        'A4_Longitude', 'A4_Latitude', 'A5_Longitude', 'A5_Latitude']] = \
        ambulancesPosition[i][nCluster]

finalResult = pd.concat(
    [fullSubmission[0], fullSubmission[1], fullSubmission[2], fullSubmission[3], fullSubmission[4], fullSubmission[5],
     fullSubmission[6], fullSubmission[7]])
finalResult = finalResult.sort_index()
finalResult = finalResult.drop(columns=['precipitable_water_entire_atmosphere', 'relative_humidity_2m_above_ground',
                                        'specific_humidity_2m_above_ground', 'temperature_2m_above_ground',
                                        'u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground',
                                        'dayofweek', 'month'])

finalResult.to_csv('ClusteringSubmission.csv', date_format='%m/%d/%Y %H:%M')
