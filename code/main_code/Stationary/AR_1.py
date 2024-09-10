import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Parameters for AR(1)
np.random.seed(0)
n = 10000  # number of time points
alpha = 0.7  # autoregressive coefficient (|alpha| < 1 for stationarity)

# Generate AR(1) process
ar_data = np.zeros(n)
ar_data[0] = np.random.normal()

for t in range(1, n):
    ar_data[t] = alpha * ar_data[t - 1] + np.random.normal()

# Plot the AR(1) time series
plt.plot(ar_data)
plt.title("AR(1) Process")
plt.show()

# Save to csv
np.savetxt('AR_1_data.csv', ar_data, delimiter=',', fmt='%.5f', header='data', comments='')


# split into train, test, validation and save to csv
seed = 6501

df = pd.read_csv('AR_1_data.csv')
train_df, remain_df = train_test_split(df, train_size=0.7, random_state=seed)
val_df, test_df = train_test_split(remain_df, test_size=0.5, random_state=seed)

train_df.to_csv("train_stationary.csv", index=False)
test_df.to_csv("test_stationary.csv", index=False)
val_df.to_csv("val_stationary.csv", index=False)

