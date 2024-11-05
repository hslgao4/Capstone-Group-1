from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.holtwinters as ets
import numpy as np

weather_path = '../../data/weather.csv'
df = pd.read_csv(weather_path, parse_dates=['date'])

# Raw - add
y = df["temperature"]
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

# holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add').fit()
holt_t = ets.ExponentialSmoothing(yt, trend='add', damped=True, seasonal='add').fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

# Raw - multiplicative
y = df["temperature"] + 12
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=144).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)
holt_f = holt_f -12

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Raw(add)-720')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()


