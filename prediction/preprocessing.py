import pandas as pd
import numpy as np
from time_aux_functions import roundDateTime3h
from time_aux_functions import gen_datetime

# Load the training data
dfCrashes = pd.read_csv('nairobi_data/data_zindi/Train.csv', parse_dates=['datetime'])

# Remove the useless columns
dfCrashes = dfCrashes.drop(columns='uid')

# Add column that indicates that have been a crash. It will be our objective to predict.
# In this dataframe all the rows means a crash
dfCrashes['crash'] = 1

# Generate 6000 random dates
n_negative_samples = 6000
dates = []
for i in range(n_negative_samples):
    dates.append(gen_datetime(min_year=2018, max_year=2019))

# Generate 6000 random latitudes and longitudes
randLat = np.random.uniform(-0.5, -3.1, n_negative_samples)
randLong = np.random.uniform(36, 38, n_negative_samples)
no_crashes = np.zeros(n_negative_samples)

dfNegativeSamples = pd.DataFrame(columns=['datetime', 'latitude', 'longitude', 'crash'])
dfNegativeSamples['datetime'] = dates
dfNegativeSamples['latitude'] = randLat
dfNegativeSamples['longitude'] = randLong
dfNegativeSamples['crash'] = no_crashes


# Add negative samples
dfFull = dfCrashes.append(dfNegativeSamples)

# Round the latitude and longitude to 0.01. This is the grid width and height
dfFull = dfFull.round(2)

# Round the datetimes to the nearest 3h span
dfFull['datetime'] = dfFull['datetime'].apply(roundDateTime3h)

# Print to check
print(dfCrashes.head(1000))
print(dfCrashes.describe())

# ToDo: Add more features (columns) such as if it was raining, temperature, characteristics of the closest segment,
#  uber data, etc

