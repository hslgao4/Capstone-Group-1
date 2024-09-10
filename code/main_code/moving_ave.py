import os, sys
import pandas as pd
utils_dir = '../component'
sys.path.insert(0, os.path.abspath(utils_dir))
from classes import *
import matplotlib.pyplot as plt


# Moving average for stationary dataset
df1_train = pd.read_csv('./Stationary/train_stationary.csv')
df1_test = pd.read_csv('./Stationary/test_stationary.csv')

train_1 = df1_train.iloc[:, 0].tolist()
test_1 = df1_test.iloc[:, 0].tolist()

window_size = 10
train_RMSE = []
test_RMSE =[]
for i in range(1, window_size):
    model = MovingAverage(window_size=i)
    model.fit(train_1)
    model.forecast(test_1)

    train_forecasts = model.get_train_forecasts()
    test_forecasts = model.get_test_forecasts()
    train_rmse = model.get_train_rmse()
    test_rmse = model.get_test_rmse()

    train_RMSE.append(train_rmse)
    test_RMSE.append(test_rmse)

plt.plot(range(1, window_size), train_RMSE, label='Train RMSE', marker='o')
plt.plot(range(1, window_size), test_RMSE, label='Test RMSE', marker='o')

plt.xlabel('Window Size')
plt.ylabel('RMSE')
plt.title('Train vs. Test RMSE')
plt.show()