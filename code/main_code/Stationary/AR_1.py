import numpy as np
import matplotlib.pyplot as plt

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
