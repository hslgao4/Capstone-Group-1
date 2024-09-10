import numpy as np
from math import sqrt

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = None
        self.train_forecasts = []
        self.test_forecasts = []
        self.train_data = None
        self.test_data = None

    def fit(self, train_data):
        self.train_data = train_data
        self.window = train_data[:self.window_size]

        for t in range(self.window_size, len(train_data)):
            forecast = np.mean(self.window)
            self.train_forecasts.append(forecast)
            self.window = self.window[1:] + [train_data[t]]

    def forecast(self, test_data):
        self.test_data = test_data
        for t in range(len(test_data)):
            forecast = np.mean(self.window)
            self.test_forecasts.append(forecast)
            self.window = self.window[1:] + [test_data[t]]

    def get_train_forecasts(self):
        return np.array(self.train_forecasts)

    def get_test_forecasts(self):
        return np.array(self.test_forecasts)

    def get_train_rmse(self):
        squared_errors = [(actual - forecast) ** 2 for actual, forecast in
                          zip(self.train_data[self.window_size:], self.train_forecasts)]
        mse = np.mean(squared_errors)
        return round(sqrt(mse), 2)

    def get_test_rmse(self):
        squared_errors = [(actual - forecast) ** 2 for actual, forecast in zip(self.test_data, self.test_forecasts)]
        mse = np.mean(squared_errors)
        return round(sqrt(mse), 2)

