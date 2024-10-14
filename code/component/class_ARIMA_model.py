from utils import *

class ARIMA_model(BaseEstimator):
    def __init__(self, AR_order=1, MA_order=0, inte_order=0):
        self.AR_order = AR_order
        self.MA_order = MA_order
        self.inte_order = inte_order
        self.model = None


    def fit(self, X, y=None):
        self.model = ARIMA(X, order=(self.AR_order, self.inte_order, self.MA_order)).fit()
        return self

    def predict(self, X):
        return self.model.predict(start=0, end=len(X)-1)

    def forecast(self, steps):
        return self.model.forecast(steps=steps)
