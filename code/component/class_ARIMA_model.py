from utils import *

class ARIMA_model(BaseEstimator):
    def __init__(self, AR_order=1, MA_order=0, Inte_order=0):
        self.AR_order = AR_order
        self.MA_order = MA_order
        self.Inte_order = Inte_order
        self.model = None

    def fit(self, X, y=None):
        self.model = ARIMA(X, order=(self.AR_order, self.Inte_order, self.MA_order)).fit(method_kwargs = {'epsilon': 1e-8,
                                                                                                          'maxfun': 500,})
        return self

    def predict(self, X):
        return self.model.predict(start=0, end=len(X)-1)

    def forecast(self, steps):
        return self.model.forecast(steps=steps)

# result = model.fit(ftol=1e-9, maxiter=200)