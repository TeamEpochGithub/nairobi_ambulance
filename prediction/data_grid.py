import pandas as pd
import numpy as np
from time_aux_functions import roundDateTime3h, gen_datetime

# Load the training data
dfCrashes = pd.read_csv('nairobi_data/data_zindi/Train.csv', parse_dates=['datetime'])

# Remove the useless columns
dfCrashes = dfCrashes.drop(columns='uid')

# Round to 1 decimal
dfCrashes = dfCrashes.round(1)

# Round the datetimes to the nearest 3h span
dfCrashes['datetime'] = dfCrashes['datetime'].apply(roundDateTime3h)

# Count the number of crashes in each cell
dfCrashes = dfCrashes.groupby(dfCrashes.columns.tolist(), as_index=False).size()
dfCrashes.rename(columns = {'size':'n_crashes'}, inplace = True)

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

# Round the datetimes to the nearest 3h span
dfNegativeSamples['datetime'] = dfNegativeSamples['datetime'].apply(roundDateTime3h)

# Add negative samples
dfCrashes = dfCrashes.append(dfNegativeSamples)
dfCrashes = dfCrashes.drop_duplicates(subset=['datetime', 'latitude', 'longitude'])

# Add date column for merging
dfCrashes['date'] = pd.to_datetime([d.date() for d in dfCrashes.datetime])
dfWeather = pd.read_csv('nairobi_data/data_zindi/Weather_Nairobi_Daily_GFS.csv', parse_dates=['Date'])

dfCrashes = dfCrashes.merge(dfWeather, how='left', left_on='date', right_on='Date')
dfCrashes = dfCrashes.drop(columns=['date', 'Date'])

# ToDo: Add more speed features


# ToDo: Train regressor



