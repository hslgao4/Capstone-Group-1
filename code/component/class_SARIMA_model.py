from utils import *

class SARIMA_Model(BaseEstimator):
    def __init__(self,
                 model_name,
                 AR_order=1, MA_order=0, Inte_order=0,
                 AR_s=1, MA_s=0, Inte_s=0, Seas_s=12):
        self.model_name = model_name
        self.AR_order = AR_order
        self.MA_order = MA_order
        self.inte_order = Inte_order
        self.AR_s = AR_s
        self.MA_s = MA_s
        self.inte_s = Inte_s
        self.seas_s = Seas_s
        self.model = None


    def fit(self, X, y=None):
        self.model = SARIMAX(
            X,
            order=(self.AR_order, self.inte_order, self.MA_order),
            seasonal_order=(self.AR_s, self.inte_s, self.MA_s, self.seas_s)
        ).fit()
        return self

    def predict(self, X):
        return self.model.predict(start=0, end=len(X)-1)

    def forecast(self, steps):
        return self.model.forecast(steps=steps)