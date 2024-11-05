import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import sys


register_matplotlib_converters()

df = pd.read_csv("AirPassengers.csv",
                 index_col="Month",
                 parse_dates=True)
y = df["#Passengers"]


lags = 38
ACF_PACF_Plot(y, lags)
# acf = sm.tsa.stattools.acf(y,nlags = lags)
# pacf = sm.tsa.stattools.pacf(y, nlags = lags)
# plt.figure()
# plt.subplot(211)
# plot_acf(y, ax=plt.gca(), lags = lags)
# plt.subplot(212)
# plot_pacf(y, ax=plt.gca(),lags=lags)
# plt.show()



yt, yf = train_test_split(y, shuffle= False, test_size=0.2)

holtt = ets.ExponentialSmoothing(yt,trend=None,damped=False,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)


MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for simple exponential smoothing is ", MSE)

# y_ave = yt.mean()
# index = pd.date_range(start= '1958-07-31',end='1960-12-01',freq='M')
# y_ave = pd.Series(y_ave)
# y_ave = y_ave.repeat(len(yf))

#-----
# fig, ax = plt.subplots()
# ax.plot(yt,label= "Train Data")
# ax.plot(y_ave,label= "Test Data")
# ax.plot(holtf,label= "Simple Exponential Smoothing")
# plt.show()
# #----







fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Simple Exponential Smoothing")

plt.legend(loc='upper left')
plt.title('Simple Exponential Smoothing- Air Passengers')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.show()


holtt = ets.ExponentialSmoothing(yt,trend='multiplicative',damped=True,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for double exponential smoothing is ", MSE)

fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Holt's Linear Trend Method")

plt.legend(loc='upper left')
plt.title("Holt's Linear Trend Method")
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.show()


holtt = ets.ExponentialSmoothing(yt,trend='mul',damped=True,seasonal='mul').fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for holt-winter method is ", MSE)
fig, ax = plt.subplots()
ax.plot(yt,label= "train")
ax.plot(yf,label= "test")
ax.plot(holtf,label= "Holt-Winter Method")

plt.legend(loc='upper left')
plt.title('Holt-Winter Method')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.show()